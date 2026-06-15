"""Paired bootstrap significance test between two systems' BLEU.

Consumes the ``predictions_labels.txt`` files that ``multimodalhugs-generate``
writes (lines ``L [idx] \\t ref`` and ``P [idx] \\t hyp``), one per system, both
generated from the same test set. Reports each system's BLEU, the difference,
a bootstrap confidence interval for the difference, and a p-value.

    python bleu_significance.py \\
        --baseline   baseline/predictions_labels.txt \\
        --proposed   proposed/predictions_labels.txt \\
        [--metric bleu|chrf] [--n_samples 1000] [--seed 12345]
"""

import argparse
import re
import sys

import numpy as np

LINE = re.compile(r"([LP])\s*\[(\d+)\]\s*\t(.*)")


def parse_predictions_labels(path):
    """Return {idx: (reference, hypothesis)} from a predictions_labels.txt file."""
    refs, hyps = {}, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = LINE.match(line.rstrip("\n"))
            if not m:
                continue
            tag, idx, text = m.group(1), int(m.group(2)), m.group(3)
            (refs if tag == "L" else hyps)[idx] = text
    common = sorted(refs.keys() & hyps.keys())
    return [refs[i] for i in common], [hyps[i] for i in common]


def load_aligned(baseline_path, proposed_path):
    base_refs, base_hyps = parse_predictions_labels(baseline_path)
    prop_refs, prop_hyps = parse_predictions_labels(proposed_path)
    if base_refs != prop_refs:
        raise ValueError(
            "References differ between the two files — the systems must be generated "
            "from the same test set in the same order."
        )
    if not base_refs:
        raise ValueError("No aligned sentences found. Check the file format.")
    return base_refs, base_hyps, prop_hyps


def make_scorer(metric):
    if metric == "bleu":
        from sacrebleu.metrics import BLEU
        scorer = BLEU(effective_order=False)
    elif metric == "chrf":
        from sacrebleu.metrics import CHRF
        scorer = CHRF()
    else:
        raise ValueError(f"unknown metric: {metric}")
    return lambda hyps, refs: scorer.corpus_score(hyps, [refs]).score


def paired_bootstrap(refs, hyps_a, hyps_b, score_fn, n_samples, seed):
    rng = np.random.default_rng(seed)
    n = len(refs)
    refs = np.array(refs, dtype=object)
    hyps_a = np.array(hyps_a, dtype=object)
    hyps_b = np.array(hyps_b, dtype=object)

    deltas = np.empty(n_samples)
    for b in range(n_samples):
        idx = rng.integers(0, n, n)              # same resample for both systems (paired)
        r = refs[idx].tolist()
        deltas[b] = score_fn(hyps_b[idx].tolist(), r) - score_fn(hyps_a[idx].tolist(), r)
        if (b + 1) % 100 == 0:
            print(f"  bootstrap {b + 1}/{n_samples}", file=sys.stderr)

    return deltas


def main():
    p = argparse.ArgumentParser(description="Paired bootstrap BLEU significance")
    p.add_argument("--baseline", required=True, help="predictions_labels.txt of the baseline system")
    p.add_argument("--proposed", required=True, help="predictions_labels.txt of the proposed system")
    p.add_argument("--metric", default="bleu", choices=["bleu", "chrf"])
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()

    refs, base_hyps, prop_hyps = load_aligned(args.baseline, args.proposed)
    score = make_scorer(args.metric)

    base_score = score(base_hyps, refs)
    prop_score = score(prop_hyps, refs)
    observed_delta = prop_score - base_score

    print(f"Comparing on {len(refs)} sentences  (metric={args.metric}, n_samples={args.n_samples})")
    deltas = paired_bootstrap(refs, base_hyps, prop_hyps, score, args.n_samples, args.seed)

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    # two-sided p-value: how often the resampled difference flips sign vs the observed one
    p_value = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())

    print(f"\nBaseline {args.metric.upper()}   {base_score:.2f}")
    print(f"Proposed {args.metric.upper()}   {prop_score:.2f}")
    print(f"Difference        {observed_delta:+.2f}")
    print(f"Bootstrap mean Δ  {deltas.mean():+.2f}")
    print(f"95% CI of Δ       [{lo:+.2f}, {hi:+.2f}]")
    print(f"p-value           {p_value:.4f}")
    print("significant at 0.05" if p_value < 0.05 else "not significant at 0.05")


if __name__ == "__main__":
    main()
