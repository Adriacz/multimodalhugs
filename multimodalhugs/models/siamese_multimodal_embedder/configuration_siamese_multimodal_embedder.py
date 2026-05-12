from typing import Optional, Dict, Any

from multimodalhugs.models.multimodal_embedder.configuration_multimodal_embedder import MultiModalEmbedderConfig
from multimodalhugs.modules import get_feature_extractor_class


class SiameseMultiModalEmbedderConfig(MultiModalEmbedderConfig):
    """
    Configuration for SiameseMultiModalEmbedderModel.

    Extends MultiModalEmbedderConfig with a second (audio) feature extractor
    and Optimal Transport alignment parameters.

    The video pathway is unchanged: FeatureExtractor → MultimodalMapper → Backbone.
    The audio pathway adds:  AudioFeatureExtractor (frozen) → LinearProjection → OT loss.

    Phase 1 — Siamese pretraining:
        Train with audio+video samples.  The OT loss aligns audio representations
        (from a frozen Whisper encoder) to the video encoder's output space,
        effectively bootstrapping the video encoder with phonetic signal.

    Phase 2 — SLT fine-tuning:
        Fine-tune on video-only samples using the standard translation loss.
        Pass ``input_audio=None`` to the forward() method (or simply use a
        VideoAudio2TextDataset without the audio slot) to disable the OT branch.

    Args:
        audio_feature_extractor_type (str, optional):
            Type string recognized by ``get_feature_extractor_class``.
            Typically ``"whisper"``.  When None, the audio branch is disabled.
        pretrained_audio_feature_extractor (str, optional):
            HuggingFace model ID or local path for the pretrained audio encoder,
            e.g. ``"openai/whisper-medium"``.
        audio_feat_dim (int):
            Output dimensionality of the audio feature extractor.
            For Whisper-medium this is 1024.  Must match the encoder's
            ``d_model`` / ``last_hidden_state`` dimension.
        freeze_audio_feature_extractor (bool):
            Whether to freeze the audio feature extractor.  Defaults to True
            (the audio encoder is used as a fixed teacher signal).
        ot_lambda (float):
            Weight applied to the Sinkhorn OT loss.  Set to 0.0 to disable.
        sinkhorn_epsilon (float):
            Entropy regularization coefficient for Sinkhorn iterations.
            Smaller values produce a sharper transport plan but may be numerically
            less stable.
        sinkhorn_max_iter (int):
            Number of Sinkhorn iterations.
    """

    model_type = "siamese_multimodal_embedder"

    def __init__(
        self,
        model_type: str = "siamese_multimodal_embedder",
        audio_feature_extractor_type: Optional[str] = None,
        pretrained_audio_feature_extractor: Optional[str] = None,
        audio_feat_dim: int = 1024,
        freeze_audio_feature_extractor: bool = True,
        ot_lambda: float = 1.0,
        sinkhorn_epsilon: float = 0.1,
        sinkhorn_max_iter: int = 100,
        **kwargs,
    ):
        super().__init__(model_type=model_type, **kwargs)

        self.audio_feature_extractor_type = audio_feature_extractor_type
        self.pretrained_audio_feature_extractor = pretrained_audio_feature_extractor
        self.audio_feat_dim = audio_feat_dim
        self.freeze_audio_feature_extractor = freeze_audio_feature_extractor
        self.ot_lambda = ot_lambda
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_max_iter = sinkhorn_max_iter
