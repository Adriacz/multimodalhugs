"""
eval_space_alignment.py

Cross-modal space alignment evaluation for SiameseMultiModalEmbedderModel.

Measures how well the video and audio embedding spaces are aligned after training
by computing cross-modal retrieval accuracy: for each video embedding, the paired
audio (same sentence) should be the nearest neighbour in the audio embedding space.

Usage:
    python eval_space_alignment.py \\
        --checkpoint_path /path/to/checkpoint \\
        --metadata_file   /path/to/validation.tsv \\
        [--video_preprocessor openai/clip-vit-base-patch32] \\
        [--audio_preprocessor openai/whisper-medium] \\
        [--skip_frames_stride 3] \\
        [--device cuda] \\
        [--max_samples 500]

The script reads the model config from the checkpoint to infer the preprocessor
paths automatically; --video_preprocessor and --audio_preprocessor only need to
be set when you want to override them.

Output example:
    Processed 423 samples
    R@1:          0.412
    R@5:          0.731
    R@10:         0.853
    MRR:          0.532
    median_rank:  3.0
"""

# Standard Library Imports
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Third-Party Imports
import pandas as pd
import torch

logger = logging.getLogger(__name__)


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


def _get_preprocessor_paths(checkpoint_path: str, video_override: Optional[str], audio_override: Optional[str]):
    """Read preprocessor HF paths from the checkpoint config.json when not overridden."""
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

    video_path = video_path or "openai/clip-vit-base-patch32"
    audio_path = audio_path or "openai/whisper-medium"
    return video_path, audio_path


def _extract_all_embeddings(
    model,
    df: pd.DataFrame,
    video_proc,
    audio_proc,
    device: torch.device,
    max_samples: Optional[int],
) -> tuple:
    """
    Process every TSV row and collect mean-pooled video + audio embeddings.

    Returns:
        (video_embs, audio_embs): two [N, D] CPU tensors.
    """
    video_embs: List[torch.Tensor] = []
    audio_embs: List[torch.Tensor] = []

    n = len(df) if max_samples is None else min(max_samples, len(df))

    for i, row in enumerate(df.itertuples(index=False)):
        if i >= n:
            break

        signal_path = row.signal
        signal_start = getattr(row, "signal_start", 0) or 0
        signal_end   = getattr(row, "signal_end",   0) or 0
        audio_start  = getattr(row, "audio_start",  signal_start) or 0
        audio_end    = getattr(row, "audio_end",    signal_end)   or 0

        try:
            video_tensor = video_proc.process_sample({
                "signal":       signal_path,
                "signal_start": signal_start,
                "signal_end":   signal_end,
            })
            audio_tensor = audio_proc.process_sample({
                "signal":       signal_path,
                "signal_start": audio_start,
                "signal_end":   audio_end,
            })
        except Exception as exc:
            logger.warning("Skipping sample %d (%s): %s", i, signal_path, exc)
            continue

        # video_tensor: [T_v, C, H, W]  → add batch dim → [1, T_v, C, H, W]
        # audio_tensor: [n_mels, T_a]   → add batch dim → [1, n_mels, T_a]
        input_frames = video_tensor.unsqueeze(0).to(device)
        input_audio  = audio_tensor.unsqueeze(0).to(device)

        video_emb, audio_emb = model.extract_embeddings(input_frames, input_audio)

        video_embs.append(video_emb.cpu())
        audio_embs.append(audio_emb.cpu())

        if (i + 1) % 50 == 0:
            logger.info("  processed %d / %d samples", i + 1, n)

    return torch.cat(video_embs, dim=0), torch.cat(audio_embs, dim=0)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="Cross-modal space alignment evaluation")
    parser.add_argument("--checkpoint_path",     required=True,  help="Path to trained SiameseMultiModalEmbedderModel checkpoint")
    parser.add_argument("--metadata_file",       required=True,  help="TSV file (validation or test split)")
    parser.add_argument("--video_preprocessor",  default=None,   help="HF path for CLIP (overrides checkpoint config)")
    parser.add_argument("--audio_preprocessor",  default=None,   help="HF path for Whisper (overrides checkpoint config)")
    parser.add_argument("--skip_frames_stride",  type=int, default=3, help="Frame downsampling stride (default: 3)")
    parser.add_argument("--device",              default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_samples",         type=int, default=None, help="Cap number of samples (useful for quick tests)")
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info("Device: %s", device)

    video_pp, audio_pp = _get_preprocessor_paths(
        args.checkpoint_path, args.video_preprocessor, args.audio_preprocessor
    )

    logger.info("Loading model from %s", args.checkpoint_path)
    model = _load_model(args.checkpoint_path, device)

    logger.info("Initialising video processor (%s, stride=%s)", video_pp, args.skip_frames_stride)
    video_proc = _make_video_processor(video_pp, args.skip_frames_stride)

    logger.info("Initialising audio processor (%s)", audio_pp)
    audio_proc = _make_audio_processor(audio_pp)

    logger.info("Reading metadata from %s", args.metadata_file)
    df = pd.read_csv(args.metadata_file, sep="\t")
    logger.info("  %d rows found", len(df))

    logger.info("Extracting embeddings...")
    video_embs, audio_embs = _extract_all_embeddings(
        model, df, video_proc, audio_proc, device, args.max_samples
    )
    N = video_embs.shape[0]
    logger.info("Collected %d embedding pairs (D=%d)", N, video_embs.shape[1])

    from multimodalhugs.modules.retrieval_metrics import cross_modal_retrieval

    metrics = cross_modal_retrieval(video_embs, audio_embs, ks=(1, 5, 10))

    print(f"\nProcessed {N} samples")
    for key, val in metrics.items():
        if key == "median_rank":
            print(f"  {key:<18}{val:.1f}")
        else:
            print(f"  {key:<18}{val:.4f}")


if __name__ == "__main__":
    main()
