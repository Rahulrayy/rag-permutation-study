"""GPU smoke test. Run this before burning an overnight on the main grid.

Checks, in order, the things that actually break:
  1. CUDA is visible and the 3060 is the device torch picks;
  2. the checkpoint loads in 4-bit and fits in VRAM;
  3. greedy decoding is genuinely deterministic (same prompt twice, same answer),
     including when the prompt sits in a different batch position;
  4. permuting a real context changes nothing about the content -- and whether it
     changes the answer, which is the entire premise in miniature;
  5. `score()` returns a usable answer log-prob, which the LOO oracle depends on.

Step 5 is the one worth being paranoid about: it is the reason the primary
generator is local rather than hosted, and it is not exercised anywhere else
until week 3.

    python -m src.smoke
    python -m src.smoke --model Qwen/Qwen2.5-0.5B-Instruct --n 2
"""

from __future__ import annotations

import argparse
import time

from .chunks import permutation_set
from .data import load_dataset
from .generate import DecodeParams, LocalGenerator, build_prompt
from .metrics import token_f1

STRATEGIES = ["rank", "reverse", "random", "random", "random"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--quantization", default="4bit")
    parser.add_argument("--n", type=int, default=3, help="queries to try")
    args = parser.parse_args()

    print("=" * 62)
    print("1. CUDA")
    print("=" * 62)
    import torch

    print(f"   torch                {torch.__version__}")
    print(f"   cuda available       {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"   device               {props.name}")
        print(f"   VRAM                 {props.total_memory / 1e9:.1f} GB")
    else:
        print("   WARNING: no CUDA. A 3B model on CPU will make the main run")
        print("   take days rather than hours.")

    print()
    print("=" * 62)
    print(f"2. Loading {args.model} ({args.quantization})")
    print("=" * 62)
    gen = LocalGenerator(model=args.model, quantization=args.quantization, batch_size=4)
    t0 = time.time()
    tok, model = gen._load()
    print(f"   loaded in            {time.time() - t0:.1f}s")
    if torch.cuda.is_available():
        print(f"   VRAM allocated       {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"   chat template        {'yes' if getattr(tok, 'chat_template', None) else 'no'}")

    examples = load_dataset("hotpotqa_distractor", n_queries=args.n, stratify_by="hop_type")
    params = DecodeParams()

    print()
    print("=" * 62)
    print("3. Determinism under greedy decoding")
    print("=" * 62)
    probe = build_prompt(examples[0].question, examples[0].chunks)
    a = gen.generate(probe, params).text
    b = gen.generate(probe, params).text
    print(f"   run 1                {a!r}")
    print(f"   run 2                {b!r}")
    if a != b:
        print("   FAIL: greedy decoding is not deterministic. Every number in")
        print("   this study assumes it is -- stop and fix this first.")
    else:
        print("   OK: identical.")

    # Repeating one prompt only proves batch-of-one determinism. Real runs batch
    # variable-length prompts with left padding, and the cache is keyed on the
    # prompt alone -- so if batch composition perturbs the logits, the answer
    # stored for a prompt depends on which batch it happened to land in, and
    # reruns with a different grid order silently disagree.
    if len(examples) > 1:
        others = [build_prompt(e.question, e.chunks) for e in examples[1:]]
        padded = gen.generate_batch([probe] + others, params)[0].text
        trailing = gen.generate_batch(others + [probe], params)[-1].text
        print(f"   batched, first slot  {padded!r}")
        print(f"   batched, last slot   {trailing!r}")
        if padded == trailing == a:
            print("   OK: batch position does not change the answer.")
        else:
            print("   WARNING: the answer depends on batch composition. Padding")
            print("   numerics, most likely. The cache keys on the prompt alone,")
            print("   so results become order-of-execution dependent. Drop")
            print("   batch_size (2, then 1) and re-check before a long run.")

    print()
    print("=" * 62)
    print("4. Permutation sensitivity (the premise, in miniature)")
    print("=" * 62)
    for ex in examples:
        orderings = permutation_set(ex.chunks, STRATEGIES, seed=20260828, key=ex.qid)
        prompts = [build_prompt(ex.question, o) for o in orderings]
        # Content must be identical across permutations; only order differs.
        assert len({"".join(sorted(p)) for p in prompts}) == 1, "content changed!"

        t0 = time.time()
        answers = [g.text for g in gen.generate_batch(prompts, params)]
        f1s = [token_f1(t, ex.answer) for t in answers]

        print(f"   Q: {ex.question[:64]}")
        print(f"   gold: {ex.answer!r}  (gold paragraphs at {ex.gold_chunk_ids})")
        for strategy, ans, f1 in zip(STRATEGIES, answers, f1s):
            print(f"      {strategy:8s} f1={f1:.2f}  {ans[:56]!r}")
        spread = max(f1s) - min(f1s)
        print(f"   f1 spread across permutations: {spread:.3f}"
              f"   ({time.time() - t0:.1f}s for {len(prompts)} generations)")
        print()

    print("=" * 62)
    print("5. Answer log-prob (what the LOO oracle needs)")
    print("=" * 62)
    ex = examples[0]
    full = build_prompt(ex.question, ex.chunks)
    lp_full = gen.score(full, ex.answer)
    # Drop the gold paragraphs: the log-prob should get worse, or the LOO oracle
    # has nothing to rank on.
    without_gold = [c for c in ex.chunks if not c.is_gold]
    lp_nogold = gen.score(build_prompt(ex.question, without_gold), ex.answer)
    print(f"   logP(answer | all 10 paragraphs)   {lp_full:8.3f}")
    print(f"   logP(answer | gold removed)        {lp_nogold:8.3f}")
    print(f"   leave-gold-out drop                {lp_full - lp_nogold:8.3f}")
    if lp_full <= lp_nogold:
        print("   WARNING: removing the gold paragraphs did not hurt the answer's")
        print("   log-prob. On a single example that may just be memorization")
        print("   (see the nocontext arm), but if it holds broadly the oracle is")
        print("   ranking noise.")
    else:
        print("   OK: gold paragraphs measurably help. Oracle has signal to rank on.")

    print()
    print("=" * 62)
    print("Smoke test complete. If all five sections look right, run the pilot:")
    print("   python -m src.run  --config configs/pilot.yaml")
    print("   python -m src.gate results/pilot_w1/generations.csv")
    print("=" * 62)


if __name__ == "__main__":
    main()
