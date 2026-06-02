"""
eval_space_alignment.py

Space alignment evaluation for SiameseMultiModalEmbedderModel.
Runs three complementary analyses on the video/audio mapper-level embeddings:

  1. Cross-modal retrieval  — R@1/5/10, MRR, median rank
  2. Hubness analysis       — k-skewness, hub/anti-hub rates
  3. CCA (PLS canonical)    — per-component Pearson correlations

Usage:
    python eval_space_alignment.py \\
        --checkpoint_path /path/to/checkpoint \\
        --metadata_file   /path/to/validation.tsv \\
        [--skip_frames_stride 3] \\
        [--n_cca_components 20] \\
        [--hubness_k 5] \\
        [--device cuda] \\
        [--max_samples 500]
"""

# Standard Library Imports
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-Party Imports
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import skew

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model / processor loading
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: str, device: torch.device):
    from multimodalhugs.models.siamese_multimodal_embedder.modeling_siamese_multimodal_embedder import (
        SiameseMultiModalEmbedderModel,
    )
    model = SiameseMultiModalEmbedderModel.from_pretrained(checkpoint_path)
    model.eval()
    model.to(device)
    return model


def _make_video_processor(preprocessor_path: str, skip_frames_stride: Optional[int]):
    from multimodalhugs.processors.video_modality_processor import VideoModalityProcessor
    return VideoModalityProcessor(
        custom_preprocessor_path=preprocessor_path,
        skip_frames_stride=skip_frames_stride,
        join_chw=False,
    )


def _make_audio_processor(preprocessor_path: str):
    from multimodalhugs.processors.speech_modality_processor import SpeechModalityProcessor
    return SpeechModalityProcessor(custom_preprocessor_path=preprocessor_path)


def _get_preprocessor_paths(
    checkpoint_path: str,
    video_override: Optional[str],
    audio_override: Optional[str],
) -> Tuple[str, str]:
    import json
    config_file = Path(checkpoint_path) / "config.json"
    video_path = video_override
    audio_path = audio_override

    if config_file.exists() and (video_path is None or audio_path is None):
        with open(config_file) as f:
            cfg = json.load(f)
        if video_path is None:
            video_path = cfg.get("pretrained_feature_extractor", "openai/clip-vit-base-patch32")
            logger.info("Video preprocessor from config: %s", video_path)
        if audio_path is None:
            audio_path = cfg.get("audio_pretrained_feature_extractor", "openai/whisper-medium")
            logger.info("Audio preprocessor from config: %s", audio_path)

    return video_path or "openai/clip-vit-base-patch32", audio_path or "openai/whisper-medium"


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def _extract_all_embeddings(
    model,
    df: pd.DataFrame,
    video_proc,
    audio_proc,
    device: torch.device,
    max_samples: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    video_embs: List[torch.Tensor] = []
    audio_embs: List[torch.Tensor] = []

    n = len(df) if max_samples is None else min(max_samples, len(df))

    for i, row in enumerate(df.itertuples(index=False)):
        if i >= n:
            break

        signal_path  = row.signal
        signal_start = getattr(row, "signal_start", 0) or 0
        signal_end   = getattr(row, "signal_end",   0) or 0
        audio_start  = getattr(row, "audio_start",  signal_start) or 0
        audio_end    = getattr(row, "audio_end",    signal_end)   or 0

        try:
            video_tensor = video_proc.process_sample({
                "signal": signal_path, "signal_start": signal_start, "signal_end": signal_end,
            })
            audio_tensor = audio_proc.process_sample({
                "signal": signal_path, "signal_start": audio_start, "signal_end": audio_end,
            })
        except Exception as exc:
            logger.warning("Skipping sample %d (%s): %s", i, signal_path, exc)
            continue

        video_emb, audio_emb = model.extract_embeddings(
            video_tensor.unsqueeze(0).to(device),
            audio_tensor.unsqueeze(0).to(device),
        )
        video_embs.append(video_emb.cpu())
        audio_embs.append(audio_emb.cpu())

        if (i + 1) % 50 == 0:
            logger.info("  processed %d / %d samples", i + 1, n)

    return torch.cat(video_embs, dim=0), torch.cat(audio_embs, dim=0)


# ---------------------------------------------------------------------------
# Analysis 1 — Cross-modal retrieval
# ---------------------------------------------------------------------------

def retrieval_analysis(
    video_embs: torch.Tensor,
    audio_embs: torch.Tensor,
    ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    from multimodalhugs.modules.retrieval_metrics import cross_modal_retrieval
    return cross_modal_retrieval(video_embs, audio_embs, ks=ks)


# ---------------------------------------------------------------------------
# Analysis 2 — Hubness
# ---------------------------------------------------------------------------

def hubness_analysis(
    video_embs: torch.Tensor,
    audio_embs: torch.Tensor,
    k: int = 5,
) -> Dict[str, float]:
    """
    Compute hubness statistics on the audio embedding space.

    N_k(i) = number of video embeddings for which audio i is a top-k NN.
    A uniform space has E[N_k] = k. Hubs have N_k >> k.

    Returns k-skewness Sk (skewness of the N_k distribution), hub rate,
    and anti-hub rate.
    """
    N = video_embs.shape[0]
    video_norm = F.normalize(video_embs, dim=-1)
    audio_norm = F.normalize(audio_embs, dim=-1)

    sim = video_norm @ audio_norm.T  # [N, N]
    top_k = sim.argsort(dim=1, descending=True)[:, :k]  # [N, k]

    # Count how many times each audio index is retrieved
    counts = torch.zeros(N, dtype=torch.long)
    for idx in top_k.flatten().tolist():
        counts[idx] += 1

    counts_np = counts.numpy().astype(float)
    sk = float(skew(counts_np))

    # Hub: appears more than 2k times (monopolises retrieval)
    hub_rate  = float((counts > 2 * k).float().mean().item())
    # Anti-hub: never retrieved by any video
    antihub_rate = float((counts == 0).float().mean().item())

    return {
        "k_skewness":    sk,
        "hub_rate":      hub_rate,
        "antihub_rate":  antihub_rate,
        "max_N_k":       float(counts.max().item()),
        "mean_N_k":      float(counts.float().mean().item()),
    }


# ---------------------------------------------------------------------------
# Analysis 3 — CCA (via PLSCanonical)
# ---------------------------------------------------------------------------

def cca_analysis(
    video_embs: torch.Tensor,
    audio_embs: torch.Tensor,
    n_components: int = 20,
) -> Dict[str, float]:
    """
    Fit PLSCanonical on (video, audio) embeddings and compute per-component
    Pearson correlations between the projected spaces.

    PLSCanonical is used instead of sklearn CCA because it is numerically
    stable when N is of the same order of magnitude as D.
    """
    from sklearn.cross_decomposition import PLSCanonical

    V = video_embs.numpy()
    A = audio_embs.numpy()
    N = V.shape[0]

    # Safe cap: need at least 5 samples per component
    n_comp = min(n_components, N // 5, V.shape[1], A.shape[1])
    if n_comp < 1:
        logger.warning("Too few samples for CCA (%d). Skipping.", N)
        return {}

    pls = PLSCanonical(n_components=n_comp, max_iter=1000)
    pls.fit(V, A)
    V_c, A_c = pls.transform(V, A)

    correlations = [
        float(np.corrcoef(V_c[:, i], A_c[:, i])[0, 1])
        for i in range(n_comp)
    ]

    return {
        "mean_canonical_corr": float(np.mean(correlations)),
        "corr_dim_1":          correlations[0],
        "corr_dim_2":          correlations[1] if n_comp > 1 else float("nan"),
        "corr_dim_5":          correlations[4] if n_comp > 4 else float("nan"),
        "corr_dim_10":         correlations[9] if n_comp > 9 else float("nan"),
        "corr_dim_20":         correlations[19] if n_comp > 19 else float("nan"),
        "n_components_used":   n_comp,
        "all_correlations":    correlations,
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _print_section(title: str, metrics: Dict[str, float]):
    width = 42
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")
    for key, val in metrics.items():
        if key == "all_correlations":
            continue
        if isinstance(val, float):
            if key in ("median_rank", "max_N_k", "mean_N_k", "n_components_used"):
                print(f"  {key:<26}{val:.1f}")
            elif key in ("hub_rate", "antihub_rate"):
                print(f"  {key:<26}{val:.1%}")
            else:
                print(f"  {key:<26}{val:.4f}")
        else:
            print(f"  {key:<26}{val}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="Space alignment evaluation")
    parser.add_argument("--checkpoint_path",    required=True)
    parser.add_argument("--metadata_file",      required=True)
    parser.add_argument("--video_preprocessor", default=None)
    parser.add_argument("--audio_preprocessor", default=None)
    parser.add_argument("--skip_frames_stride", type=int,   default=3)
    parser.add_argument("--n_cca_components",   type=int,   default=20)
    parser.add_argument("--hubness_k",          type=int,   default=5)
    parser.add_argument("--device",             default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_samples",        type=int,   default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info("Device: %s", device)

    video_pp, audio_pp = _get_preprocessor_paths(
        args.checkpoint_path, args.video_preprocessor, args.audio_preprocessor
    )

    logger.info("Loading model from %s", args.checkpoint_path)
    model = _load_model(args.checkpoint_path, device)

    video_proc = _make_video_processor(video_pp, args.skip_frames_stride)
    audio_proc = _make_audio_processor(audio_pp)

    logger.info("Reading metadata from %s", args.metadata_file)
    df = pd.read_csv(args.metadata_file, sep="\t")
    logger.info("  %d rows found", len(df))

    logger.info("Extracting embeddings...")
    video_embs, audio_embs = _extract_all_embeddings(
        model, df, video_proc, audio_proc, device, args.max_samples
    )
    N = video_embs.shape[0]
    logger.info("Collected %d embedding pairs  D=%d", N, video_embs.shape[1])

    print(f"\n{'═' * 42}")
    print(f"  Space Alignment Report  (N={N})")
    print(f"{'═' * 42}")

    _print_section(
        "1. Cross-modal Retrieval  (video → audio)",
        retrieval_analysis(video_embs, audio_embs),
    )

    hub_metrics = hubness_analysis(video_embs, audio_embs, k=args.hubness_k)
    _print_section(
        f"2. Hubness  (k={args.hubness_k})",
        hub_metrics,
    )
    if hub_metrics.get("k_skewness", 0) > 1.2:
        print("  ⚠  k-skewness > 1.2 — hubness detected.")
    else:
        print("  ✓  k-skewness ≤ 1.2 — no significant hubness.")

    cca_metrics = cca_analysis(video_embs, audio_embs, n_components=args.n_cca_components)
    if cca_metrics:
        _print_section(
            f"3. CCA / PLS  ({cca_metrics['n_components_used']} components)",
            cca_metrics,
        )
        all_corr = cca_metrics.get("all_correlations", [])
        if all_corr:
            print(f"\n  Per-component correlations:")
            for i, c in enumerate(all_corr, 1):
                bar = "█" * int(max(c, 0) * 20)
                print(f"  dim {i:>3}  {c:+.4f}  {bar}")

    print(f"\n{'═' * 42}\n")


if __name__ == "__main__":
    main()
