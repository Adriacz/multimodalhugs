"""
bootstrap_significance.py

Paired bootstrap resampling significance test for BLEU/chrF.
Tests whether system B (your model) is significantly better than system A (baseline).

Usage:
    python bootstrap_significance.py \\
        --hyp_a   /path/to/baseline_hypotheses.txt \\
        --hyp_b   /path/to/your_model_hypotheses.txt \\
        --ref     /path/to/references.txt \\
        [--metric bleu]   # bleu or chrf
        [--n_samples 1000] \\
        [--seed 42]

Each file has one sentence per line, same order.
Hypothesis files come from the model generate output (predictions.txt).

Reference (method): Koehn (2004) "Statistical Significance Tests for
Machine Translation Evaluation", EMNLP.
"""

# Standard Library Imports
import argparse
import logging
import sys
from typing import List, Tuple

# Third-Party Imports
import numpy as np
from sacrebleu.metrics import BLEU, CHRF

logger = logging.getLogger(__name__)


def _read_lines(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f]
    return lines


def _score(metric, hyps: List[str], refs: List[str]) -> float:
    return metric.corpus_score(hyps, [refs]).score


def bootstrap_paired_test(
    hyp_a: List[str],
    hyp_b: List[str],
    refs: List[str],
    metric,
    n_samples: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, List[float]]:
    """
    Paired bootstrap resampling significance test.

    For each bootstrap iteration, samples N sentences with replacement and
    computes metric(A) and metric(B) on that sample. Returns:
      - p_value : fraction of samples where A >= B  (H0: B not better than A)
      - win_rate: fraction of samples where B > A
      - deltas  : list of (score_B - score_A) for each sample (for CI)
    """
    rng = np.random.default_rng(seed)
    N = len(hyp_a)

    count_b_wins = 0
    deltas: List[float] = []

    for _ in range(n_samples):
        idx = rng.integers(0, N, size=N)
        sample_a = [hyp_a[i] for i in idx]
        sample_b = [hyp_b[i] for i in idx]
        sample_r = [refs[i]  for i in idx]

        score_a = _score(metric, sample_a, sample_r)
        score_b = _score(metric, sample_b, sample_r)
        delta   = score_b - score_a
        deltas.append(delta)

        if score_b > score_a:
            count_b_wins += 1

    p_value  = 1.0 - count_b_wins / n_samples
    win_rate = count_b_wins / n_samples
    return p_value, win_rate, deltas


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="Bootstrap significance test")
    parser.add_argument("--hyp_a",      required=True, help="Baseline hypotheses (one per line)")
    parser.add_argument("--hyp_b",      required=True, help="Your model hypotheses (one per line)")
    parser.add_argument("--ref",        required=True, help="References (one per line)")
    parser.add_argument("--metric",     default="bleu", choices=["bleu", "chrf"])
    parser.add_argument("--n_samples",  type=int, default=1000)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    hyp_a = _read_lines(args.hyp_a)
    hyp_b = _read_lines(args.hyp_b)
    refs  = _read_lines(args.ref)

    if not (len(hyp_a) == len(hyp_b) == len(refs)):
        logger.error("Files have different number of lines: A=%d B=%d ref=%d", len(hyp_a), len(hyp_b), len(refs))
        sys.exit(1)

    N = len(refs)
    logger.info("Loaded %d sentence pairs", N)

    metric = BLEU(effective_order=True) if args.metric == "bleu" else CHRF()
    metric_name = args.metric.upper()

    # Scores on the full test set
    score_a = _score(metric, hyp_a, refs)
    score_b = _score(metric, hyp_b, refs)

    logger.info("Running %d bootstrap samples...", args.n_samples)
    p_value, win_rate, deltas = bootstrap_paired_test(
        hyp_a, hyp_b, refs, metric, n_samples=args.n_samples, seed=args.seed
    )

    deltas_np = np.array(deltas)
    ci_low  = float(np.percentile(deltas_np, 2.5))
    ci_high = float(np.percentile(deltas_np, 97.5))

    width = 46
    print(f"\n{'═' * width}")
    print(f"  Bootstrap Significance Test  ({metric_name})")
    print(f"{'═' * width}")
    print(f"  N sentences          {N}")
    print(f"  Bootstrap samples    {args.n_samples}")
    print()
    print(f"  {metric_name} system A (baseline)   {score_a:.2f}")
    print(f"  {metric_name} system B (your model) {score_b:.2f}")
    print(f"  Δ (B − A)            {score_b - score_a:+.2f}")
    print()
    print(f"  95% CI of Δ          [{ci_low:+.2f}, {ci_high:+.2f}]")
    print(f"  B wins in bootstrap  {win_rate:.1%}  ({int(win_rate * args.n_samples)}/{args.n_samples})")
    print(f"  p-value              {p_value:.4f}")
    print()

    if p_value < 0.01:
        print(f"  ✓  p < 0.01 — improvement is HIGHLY significant.")
    elif p_value < 0.05:
        print(f"  ✓  p < 0.05 — improvement is statistically significant.")
    elif p_value < 0.10:
        print(f"  ~  p < 0.10 — marginal significance (borderline).")
    else:
        print(f"  ✗  p = {p_value:.3f} — improvement is NOT statistically significant.")

    if ci_low > 0:
        print(f"  ✓  95% CI entirely above 0 — B is consistently better.")
    elif ci_high < 0:
        print(f"  ✗  95% CI entirely below 0 — B is consistently worse.")
    else:
        print(f"  ~  95% CI crosses 0 — improvement is inconsistent across samples.")

    print(f"{'═' * width}\n")


if __name__ == "__main__":
    main()
