from typing import Optional

from multimodalhugs.models.siamese_multimodal_embedder.configuration_siamese_multimodal_embedder import (
    SiameseMultiModalEmbedderConfig,
)


class JointMultiModalEmbedderConfig(SiameseMultiModalEmbedderConfig):
    """
    Configuration for JointMultiModalEmbedderModel.

    Extends SiameseMultiModalEmbedderConfig with no new parameters — the
    architecture is identical but the forward pass feeds BOTH video and audio
    to the M2M backbone instead of video only.

    Training forward:
        Video:   FeatureExtractor → MultimodalMapper → video_repr [B, T_v, D]
        Audio:   AudioFeatureExtractor → AudioMultimodalMapper → audio_repr [B, T_a, D]
        OT:      batch_sinkhorn_loss(video_repr, audio_repr) → ot_loss
        Fusion:  cat([video_repr, audio_repr], dim=1) → fused_repr [B, T_v+T_a, D]
        Backbone: merge_modalities(fused_repr) → M2M encoder+decoder → mt_loss
        Loss:    mt_loss + ot_lambda * ot_loss

    Inference modes (same as SiameseMultiModalEmbedderModel):
        - Video-only  (input_audio=None): identical to MultiModalEmbedderModel
        - Audio-only  (input_frames=None): AudioFE → AudioMapper → M2M
        - Both inputs: fused sequence sent to M2M (same as training)
    """

    model_type = "joint_multimodal_embedder"

    def __init__(self, model_type: str = "joint_multimodal_embedder", **kwargs):
        super().__init__(model_type=model_type, **kwargs)
