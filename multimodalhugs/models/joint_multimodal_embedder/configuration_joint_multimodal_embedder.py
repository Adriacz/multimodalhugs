from typing import Optional

from multimodalhugs.models.siamese_multimodal_embedder.configuration_siamese_multimodal_embedder import (
    SiameseMultiModalEmbedderConfig,
)


class JointMultiModalEmbedderConfig(SiameseMultiModalEmbedderConfig):
    """
    Configuration for JointMultiModalEmbedderModel.

    Extends SiameseMultiModalEmbedderConfig with an asymmetric multi-task
    sampler. Each training step (when both modalities are present) is
    stochastically routed to one of three modes:

        1. Video + OT   (default, dominant) — video → M2M (MT loss) and
                          OT(video, audio) as an auxiliary alignment loss.
                          The audio branch is computed only for OT; the
                          backbone sees video only. This is the path whose
                          weights are used at video-only inference.
        2. Audio-only    (rehearsal, minority) — audio → M2M (ASR MT loss).
                          Keeps the warm-started decoder/audio path from
                          drifting as the model adapts to video ("no s'oblidi").
                          No OT (no video in the backbone).
        3. Joint fusion  (optional) — cat([video, audio]) → M2M. The decoder
                          sees both. Off by default (audio-crutch risk for the
                          video-only goal).

    Routing probabilities (sampled per training micro-step):
        p_audio_only        → mode 2
        p_joint_fusion      → mode 3
        1 - the above       → mode 1 (video + OT)
    With gradient_accumulation_steps > 1 each micro-step samples
    independently, so an optimizer step sees a mixture.

    Since the audio path starts from a strong speech checkpoint, the intended
    setting is a LOW p_audio_only (e.g. 0.15-0.2) and p_joint_fusion = 0:
    mostly video learning, a little audio rehearsal so the decoder doesn't
    forget.

    Args:
        p_audio_only: Probability a training step is audio-only ASR rehearsal.
        p_joint_fusion: Probability a training step fuses video+audio into M2M.
        generation_mode: Modality used at generation/eval — "video" (default,
            so eval_sacrebleu measures the video-only goal), "joint", or "audio".
    """

    model_type = "joint_multimodal_embedder"

    def __init__(
        self,
        model_type: str = "joint_multimodal_embedder",
        p_audio_only: float = 0.0,
        p_joint_fusion: float = 0.0,
        generation_mode: str = "video",
        **kwargs,
    ):
        super().__init__(model_type=model_type, **kwargs)
        self.p_audio_only = p_audio_only
        self.p_joint_fusion = p_joint_fusion
        self.generation_mode = generation_mode
