"""
Convert a Phase-1 audio-only MultiModalEmbedderModel checkpoint into a
SiameseMultiModalEmbedderModel checkpoint ready for Phase-2 video fine-tuning.

What this script does
---------------------
Phase 1 trains an audio model (MultiModalEmbedderModel with Whisper as
feature_extractor).  Its checkpoint has these weight keys:

    feature_extractor.*     ← Whisper encoder
    multimodal_mapper.*     ← audio-to-backbone mapper
    backbone.*              ← M2M-100 (domain-adapted on LSC-Parlament)

Phase 2 needs a SiameseMultiModalEmbedderModel where:

    audio_feature_extractor.*  ← Phase-1 Whisper  (frozen reference)
    audio_mapper.*             ← Phase-1 mapper   (frozen reference)
    backbone.*                 ← Phase-1 M2M      (frozen backbone)
    feature_extractor.*        ← fresh CLIP       (to be trained)
    multimodal_mapper.*        ← fresh VideoMapper (to be trained)

The script renames the Phase-1 keys and loads them into an initialised
siamese model, then saves the result with save_pretrained().

Usage
-----
    mmhugs-convert-phase1 \\
        --phase1_checkpoint /path/to/phase1/checkpoint-best \\
        --output_dir       /path/to/phase2/siamese_init \\
        [--feat_dim 512] \\
        [--pretrained_feature_extractor openai/clip-vit-base-patch32] \\
        [--multimodal_mapper_type linear] \\
        [--multimodal_mapper_layer_norm_before]
"""

import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoConfig

import multimodalhugs.models  # noqa: F401 — registers all model types

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Key renaming
# ---------------------------------------------------------------------------

def _rename_keys(state_dict: dict) -> dict:
    """
    Rename Phase-1 MultiModalEmbedderModel keys to siamese audio-branch keys.

        feature_extractor.*  →  audio_feature_extractor.*
        multimodal_mapper.*  →  audio_mapper.*
        backbone.*           →  backbone.*  (unchanged)
    """
    renamed = {}
    for k, v in state_dict.items():
        if k.startswith("feature_extractor."):
            renamed["audio_feature_extractor." + k[len("feature_extractor."):]] = v
        elif k.startswith("multimodal_mapper."):
            renamed["audio_mapper." + k[len("multimodal_mapper."):]] = v
        else:
            renamed[k] = v
    return renamed


# ---------------------------------------------------------------------------
# State-dict loading (supports both pytorch_model.bin and safetensors)
# ---------------------------------------------------------------------------

def _load_state_dict(checkpoint_dir: Path) -> dict:
    safetensors_path = checkpoint_dir / "model.safetensors"
    bin_path = checkpoint_dir / "pytorch_model.bin"

    if safetensors_path.exists():
        from safetensors.torch import load_file
        return load_file(str(safetensors_path), device="cpu")

    if bin_path.exists():
        return torch.load(str(bin_path), map_location="cpu")

    raise FileNotFoundError(
        f"No model weights found in {checkpoint_dir}. "
        "Expected model.safetensors or pytorch_model.bin."
    )


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(
    phase1_checkpoint: str,
    output_dir: str,
    # Video branch overrides (defaults match config_video_ft.yaml)
    feat_dim: int = 512,
    pretrained_feature_extractor: str = "openai/clip-vit-base-patch32",
    multimodal_mapper_type: str = "linear",
    multimodal_mapper_layer_norm_before: bool = True,
    multimodal_mapper_layer_norm: bool = False,
    multimodal_mapper_activation: bool = False,
    multimodal_mapper_dropout: float = 0.1,
):
    from multimodalhugs.models.siamese_multimodal_embedder.configuration_siamese_multimodal_embedder import (
        SiameseMultiModalEmbedderConfig,
    )
    from multimodalhugs.models.siamese_multimodal_embedder.modeling_siamese_multimodal_embedder import (
        SiameseMultiModalEmbedderModel,
    )

    phase1_dir = Path(phase1_checkpoint)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load Phase-1 config
    # ------------------------------------------------------------------
    phase1_config = AutoConfig.from_pretrained(str(phase1_dir))
    logger.info("Loaded Phase-1 config: %s", phase1_config.model_type)

    # ------------------------------------------------------------------
    # 2. Build siamese config
    #    - Audio branch:  derived from Phase-1 feature_extractor settings
    #    - Video branch:  CLIP (fresh)
    #    - Backbone:      Phase-1 M2M (will be loaded from state dict)
    #    - OT disabled, video-only training, dual eval enabled
    # ------------------------------------------------------------------
    phase1_dict = phase1_config.to_dict()
    # Remove keys that conflict with SiameseMultiModalEmbedderConfig.__init__
    for drop in ("model_type", "_name_or_path", "transformers_version"):
        phase1_dict.pop(drop, None)

    siamese_config = SiameseMultiModalEmbedderConfig(
        # ── backbone (from Phase 1) ──────────────────────────────────
        **{k: v for k, v in phase1_dict.items()
           if k.startswith("backbone") or k in (
               "d_model", "vocab_size", "pad_token_id", "eos_token_id",
               "decoder_start_token_id", "forced_eos_token_id", "max_length",
           )},
        # ── video branch (fresh CLIP) ────────────────────────────────
        feature_extractor_type="clip",
        pretrained_feature_extractor=pretrained_feature_extractor,
        feat_dim=feat_dim,
        freeze_feature_extractor=False,
        multimodal_mapper_type=multimodal_mapper_type,
        multimodal_mapper_layer_norm_before=multimodal_mapper_layer_norm_before,
        multimodal_mapper_layer_norm=multimodal_mapper_layer_norm,
        multimodal_mapper_activation=multimodal_mapper_activation,
        multimodal_mapper_dropout=multimodal_mapper_dropout,
        freeze_multimodal_mapper=False,
        # ── audio branch (Phase-1 reference, frozen) ─────────────────
        audio_feature_extractor_type=getattr(phase1_config, "feature_extractor_type", "whisper"),
        audio_pretrained_feature_extractor=getattr(
            phase1_config, "pretrained_feature_extractor", "openai/whisper-medium"
        ),
        audio_feat_dim=getattr(phase1_config, "feat_dim", 1024),
        audio_freeze_feature_extractor=True,
        audio_multimodal_mapper_type=getattr(phase1_config, "multimodal_mapper_type", "linear"),
        audio_multimodal_mapper_layer_norm_before=getattr(
            phase1_config, "multimodal_mapper_layer_norm_before", True
        ),
        audio_multimodal_mapper_layer_norm=getattr(phase1_config, "multimodal_mapper_layer_norm", False),
        audio_multimodal_mapper_activation=getattr(phase1_config, "multimodal_mapper_activation", False),
        audio_multimodal_mapper_dropout=getattr(phase1_config, "multimodal_mapper_dropout", 0.1),
        audio_freeze_multimodal_mapper=True,
        # ── backbone freeze (Phase-2: backbone is a fixed reference) ────
        # Override whatever Phase-1 had — Phase-1 trained the backbone,
        # but Phase-2 keeps it frozen so only CLIP+VideoMapper are updated.
        freeze_backbone=True,
        freeze_decoder_embed_tokens=True,
        freeze_encoder_embed_tokens=True,
        freeze_lm_head=True,
        # ── OT disabled; video-only training; dual eval ───────────────
        ot_lambda=0.0,
        sinkhorn_epsilon=0.1,
        sinkhorn_max_iter=100,
        train_with_audio=False,
        eval_audio=True,
    )

    # ------------------------------------------------------------------
    # 3. Initialise the siamese model
    #    The __init__ loads CLIP from pretrained_feature_extractor and
    #    Whisper from audio_pretrained_feature_extractor.  Backbone and
    #    mappers start randomly / from pretrained HF weights.
    # ------------------------------------------------------------------
    logger.info("Initialising SiameseMultiModalEmbedderModel...")
    model = SiameseMultiModalEmbedderModel(siamese_config)

    # ------------------------------------------------------------------
    # 4. Load Phase-1 weights with renamed keys (strict=False so that
    #    CLIP and VideoMapper missing keys are silently skipped)
    # ------------------------------------------------------------------
    logger.info("Loading Phase-1 state dict from %s", phase1_dir)
    phase1_state = _load_state_dict(phase1_dir)
    renamed_state = _rename_keys(phase1_state)

    missing, unexpected = model.load_state_dict(renamed_state, strict=False)

    # Keys expected to be missing:
    #   - fresh video branch (CLIP + VideoMapper initialised from pretrained)
    #   - M2M-100 tied embedding weights (saved once in safetensors, restored
    #     automatically via the tie mechanism — not a real gap)
    _tied_suffixes = (
        "backbone.model.encoder.embed_tokens.weight",
        "backbone.model.decoder.embed_tokens.weight",
        "backbone.lm_head.weight",
    )
    expected_missing_prefixes = ("feature_extractor.", "multimodal_mapper.")
    truly_missing = [
        k for k in missing
        if not any(k.startswith(p) for p in expected_missing_prefixes)
        and k not in _tied_suffixes
    ]
    if truly_missing:
        logger.warning("Unexpected missing keys after loading Phase-1 weights:\n%s",
                       "\n".join(truly_missing))
    if unexpected:
        logger.warning("Unexpected keys in Phase-1 state dict (ignored):\n%s",
                       "\n".join(unexpected))

    logger.info(
        "Loaded Phase-1 weights — %d keys renamed to audio branch, "
        "%d backbone keys transferred.",
        sum(1 for k in renamed_state if k.startswith("audio_")),
        sum(1 for k in renamed_state if k.startswith("backbone.")),
    )

    # ------------------------------------------------------------------
    # 5. Save
    #    Some CLIP tensors are non-contiguous views; safetensors requires
    #    contiguous memory layout before serialising.
    # ------------------------------------------------------------------
    for param in model.parameters():
        param.data = param.data.contiguous()
    model.save_pretrained(str(output_path))
    logger.info("Siamese checkpoint saved to %s", output_path)
    logger.info(
        "\nNext steps:\n"
        "  1. Point 'setup.output_dir' in config_video_ft.yaml to: %s\n"
        "  2. Run: mmhugs-train --task translation --config_path config_video_ft.yaml",
        output_path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(
        description="Convert a Phase-1 audio MultiModalEmbedderModel checkpoint "
                    "into a Phase-2 SiameseMultiModalEmbedderModel checkpoint.",
    )
    parser.add_argument("--phase1_checkpoint", required=True,
                        help="Path to the Phase-1 checkpoint directory (checkpoint-best).")
    parser.add_argument("--output_dir", required=True,
                        help="Where to save the converted siamese checkpoint.")
    parser.add_argument("--feat_dim", type=int, default=512,
                        help="Video feature dim (default: 512 for CLIP-ViT-B/32).")
    parser.add_argument("--pretrained_feature_extractor",
                        default="openai/clip-vit-base-patch32",
                        help="HF ID or path for the video feature extractor.")
    parser.add_argument("--multimodal_mapper_type", default="linear")
    parser.add_argument("--multimodal_mapper_layer_norm_before",
                        action="store_true", default=True)
    parser.add_argument("--multimodal_mapper_layer_norm", action="store_true", default=False)
    parser.add_argument("--multimodal_mapper_activation", action="store_true", default=False)
    parser.add_argument("--multimodal_mapper_dropout", type=float, default=0.1)

    args = parser.parse_args()
    convert(
        phase1_checkpoint=args.phase1_checkpoint,
        output_dir=args.output_dir,
        feat_dim=args.feat_dim,
        pretrained_feature_extractor=args.pretrained_feature_extractor,
        multimodal_mapper_type=args.multimodal_mapper_type,
        multimodal_mapper_layer_norm_before=args.multimodal_mapper_layer_norm_before,
        multimodal_mapper_layer_norm=args.multimodal_mapper_layer_norm,
        multimodal_mapper_activation=args.multimodal_mapper_activation,
        multimodal_mapper_dropout=args.multimodal_mapper_dropout,
    )


if __name__ == "__main__":
    main()
