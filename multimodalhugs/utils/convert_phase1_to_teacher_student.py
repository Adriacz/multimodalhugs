"""
Convert a Phase-1 audio MultiModalEmbedderModel checkpoint into an
AudioTeacherVideoStudentModel checkpoint ready for teacher-student OT training.

What this script does
---------------------
Phase 1 trains an audio model (MultiModalEmbedderModel with Whisper):

    feature_extractor.*   ← Whisper encoder
    multimodal_mapper.*   ← AudioMapper (audio→d_model)
    backbone.*            ← M2M-100 (domain-adapted on LSC-Parlament)

The new model needs:

    backbone.*            ← Phase-1 M2M  (initialised from Phase 1)
    feature_extractor.*   ← fresh CLIP   (Stage-1: frozen; Stage-2: unfrozen)
    multimodal_mapper.*   ← fresh VideoMapper (always trainable)
    [teacher_model]       ← Phase-1 checkpoint loaded at runtime (NOT saved)

The script:
  1. Reads the Phase-1 config.
  2. Builds an AudioTeacherVideoStudentConfig pointing teacher_model_path to the
     Phase-1 checkpoint directory.
  3. Initialises AudioTeacherVideoStudentModel (CLIP + VideoMapper fresh,
     backbone from Phase-1 via config).
  4. Loads only backbone.* weights from the Phase-1 state dict into the student.
  5. Saves the student checkpoint (teacher weights are excluded automatically).

Usage
-----
    mmhugs-convert-phase1-ts \\
        --phase1_checkpoint /path/to/phase1/checkpoint-best \\
        --output_dir        /path/to/teacher_student_init \\
        [--feat_dim 512] \\
        [--pretrained_feature_extractor openai/clip-vit-base-patch32] \\
        [--multimodal_mapper_type linear] \\
        [--ot_lambda_mapper 1.0] \\
        [--ot_lambda_encoder 1.0] \\
        [--freeze_feature_extractor]   # Stage 1: CLIP frozen (default: True)
        [--no_freeze_feature_extractor] # Stage 2: CLIP unfrozen
"""

import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoConfig

import multimodalhugs.models  # noqa: F401 — registers all model types

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State-dict loading (supports safetensors and .bin)
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
    # Video branch
    feat_dim: int = 512,
    pretrained_feature_extractor: str = "openai/clip-vit-base-patch32",
    multimodal_mapper_type: str = "linear",
    multimodal_mapper_layer_norm_before: bool = True,
    multimodal_mapper_layer_norm: bool = False,
    multimodal_mapper_activation: bool = False,
    multimodal_mapper_dropout: float = 0.1,
    freeze_feature_extractor: bool = True,  # Stage 1: CLIP frozen
    # OT
    ot_lambda_mapper: float = 1.0,
    ot_lambda_encoder: float = 1.0,
    sinkhorn_epsilon: float = 0.1,
    sinkhorn_max_iter: int = 100,
):
    from multimodalhugs.models.audio_teacher_video_student.configuration_audio_teacher_video_student import (
        AudioTeacherVideoStudentConfig,
    )
    from multimodalhugs.models.audio_teacher_video_student.modeling_audio_teacher_video_student import (
        AudioTeacherVideoStudentModel,
    )

    phase1_dir = Path(phase1_checkpoint)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load Phase-1 config
    # ------------------------------------------------------------------
    phase1_config = AutoConfig.from_pretrained(str(phase1_dir))
    logger.info("Loaded Phase-1 config: %s", phase1_config.model_type)

    phase1_dict = phase1_config.to_dict()
    for drop in ("model_type", "_name_or_path", "transformers_version"):
        phase1_dict.pop(drop, None)

    # ------------------------------------------------------------------
    # 2. Build AudioTeacherVideoStudentConfig
    #    - Video branch: fresh CLIP + VideoMapper
    #    - Backbone: same arch as Phase 1 (weights loaded below)
    #    - Teacher: Phase-1 checkpoint path (loaded at runtime, not saved)
    #    - Backbone frozen: True (student backbone is pretrained, only CLIP+mapper train)
    # ------------------------------------------------------------------
    student_config = AudioTeacherVideoStudentConfig(
        # ── backbone (same arch as Phase 1) ─────────────────────────────
        **{k: v for k, v in phase1_dict.items()
           if k.startswith("backbone") or k in (
               "d_model", "vocab_size", "pad_token_id", "eos_token_id",
               "bos_token_id", "decoder_start_token_id", "forced_eos_token_id",
               "max_length",
           )},
        # ── video branch (fresh) ─────────────────────────────────────────
        feature_extractor_type="clip",
        pretrained_feature_extractor=pretrained_feature_extractor,
        feat_dim=feat_dim,
        freeze_feature_extractor=freeze_feature_extractor,
        multimodal_mapper_type=multimodal_mapper_type,
        multimodal_mapper_layer_norm_before=multimodal_mapper_layer_norm_before,
        multimodal_mapper_layer_norm=multimodal_mapper_layer_norm,
        multimodal_mapper_activation=multimodal_mapper_activation,
        multimodal_mapper_dropout=multimodal_mapper_dropout,
        freeze_multimodal_mapper=False,
        # ── backbone frozen (Phase-1 backbone anchors the representation space) ─
        freeze_backbone=True,
        freeze_decoder_embed_tokens=True,
        freeze_encoder_embed_tokens=True,
        freeze_lm_head=True,
        # ── teacher ──────────────────────────────────────────────────────
        teacher_model_path=str(phase1_dir.resolve()),
        # ── OT ───────────────────────────────────────────────────────────
        ot_lambda_mapper=ot_lambda_mapper,
        ot_lambda_encoder=ot_lambda_encoder,
        sinkhorn_epsilon=sinkhorn_epsilon,
        sinkhorn_max_iter=sinkhorn_max_iter,
    )

    # ------------------------------------------------------------------
    # 3. Initialise student model
    #    __init__ loads CLIP from pretrained_feature_extractor and the
    #    teacher from teacher_model_path.  Backbone starts from scratch
    #    here but we overwrite with Phase-1 weights below.
    # ------------------------------------------------------------------
    logger.info("Initialising AudioTeacherVideoStudentModel ...")
    model = AudioTeacherVideoStudentModel(student_config)

    # ------------------------------------------------------------------
    # 4. Load Phase-1 backbone weights into the student
    #    Only backbone.* keys are transferred; feature_extractor.* and
    #    multimodal_mapper.* from Phase 1 are ignored (student uses CLIP).
    # ------------------------------------------------------------------
    logger.info("Loading Phase-1 backbone weights from %s ...", phase1_dir)
    phase1_state = _load_state_dict(phase1_dir)
    backbone_state = {k: v for k, v in phase1_state.items() if k.startswith("backbone.")}

    missing, unexpected = model.load_state_dict(backbone_state, strict=False)

    # Expected missing: CLIP (feature_extractor.*), VideoMapper (multimodal_mapper.*),
    # and tied embedding keys (restored automatically by HF tying mechanism).
    _tied_keys = {
        "backbone.model.encoder.embed_tokens.weight",
        "backbone.model.decoder.embed_tokens.weight",
        "backbone.lm_head.weight",
    }
    truly_missing = [
        k for k in missing
        if not k.startswith("feature_extractor.")
        and not k.startswith("multimodal_mapper.")
        and k not in _tied_keys
    ]
    if truly_missing:
        logger.warning(
            "Unexpected missing backbone keys:\n%s", "\n".join(truly_missing)
        )
    if unexpected:
        logger.warning(
            "Unexpected keys in Phase-1 state dict (ignored):\n%s",
            "\n".join(unexpected),
        )

    logger.info(
        "Transferred %d backbone keys from Phase 1.", len(backbone_state)
    )

    # ------------------------------------------------------------------
    # 5. Save student checkpoint (teacher weights excluded automatically)
    # ------------------------------------------------------------------
    for param in model.parameters():
        param.data = param.data.contiguous()

    model.save_pretrained(str(output_path))
    logger.info("Student checkpoint saved to %s", output_path)
    logger.info(
        "\nNext steps:\n"
        "  1. Run mmhugs-setup --config_path config_teacher_student.yaml "
        "--do_dataset --do_processor\n"
        "  2. Run mmhugs-train --task translation "
        "--config_path config_teacher_student.yaml\n"
        "  Stage 2: set freeze_feature_extractor: false in config.json and restart."
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
        description=(
            "Convert a Phase-1 audio MultiModalEmbedderModel checkpoint "
            "into an AudioTeacherVideoStudentModel checkpoint."
        )
    )
    parser.add_argument(
        "--phase1_checkpoint", required=True,
        help="Path to the Phase-1 checkpoint directory.",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Where to save the converted student checkpoint.",
    )
    parser.add_argument("--feat_dim", type=int, default=512)
    parser.add_argument(
        "--pretrained_feature_extractor", default="openai/clip-vit-base-patch32",
    )
    parser.add_argument("--multimodal_mapper_type", default="linear")
    parser.add_argument(
        "--multimodal_mapper_layer_norm_before", action="store_true", default=True,
    )
    parser.add_argument(
        "--multimodal_mapper_layer_norm", action="store_true", default=False,
    )
    parser.add_argument(
        "--multimodal_mapper_activation", action="store_true", default=False,
    )
    parser.add_argument("--multimodal_mapper_dropout", type=float, default=0.1)
    parser.add_argument(
        "--freeze_feature_extractor", dest="freeze_feature_extractor",
        action="store_true", default=True,
        help="Stage 1: freeze CLIP (default).",
    )
    parser.add_argument(
        "--no_freeze_feature_extractor", dest="freeze_feature_extractor",
        action="store_false",
        help="Stage 2: unfreeze CLIP.",
    )
    parser.add_argument("--ot_lambda_mapper", type=float, default=1.0)
    parser.add_argument("--ot_lambda_encoder", type=float, default=1.0)
    parser.add_argument("--sinkhorn_epsilon", type=float, default=0.1)
    parser.add_argument("--sinkhorn_max_iter", type=int, default=100)

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
        freeze_feature_extractor=args.freeze_feature_extractor,
        ot_lambda_mapper=args.ot_lambda_mapper,
        ot_lambda_encoder=args.ot_lambda_encoder,
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        sinkhorn_max_iter=args.sinkhorn_max_iter,
    )


if __name__ == "__main__":
    main()
