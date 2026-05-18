from typing import Optional

from multimodalhugs.models.multimodal_embedder.configuration_multimodal_embedder import MultiModalEmbedderConfig


class SiameseMultiModalEmbedderConfig(MultiModalEmbedderConfig):
    """
    Configuration for SiameseMultiModalEmbedderModel.

    Extends MultiModalEmbedderConfig with a second (audio) feature extractor,
    an audio-specific MultimodalMapper, and Optimal Transport alignment params.

    Video pathway (unchanged from parent):
        FeatureExtractor → MultimodalMapper → merge_modalities → Backbone

    Audio pathway (added):
        AudioFeatureExtractor → AudioMultimodalMapper → OT loss (then discarded)

    Args:
        audio_feature_extractor_type: Type string for the audio encoder,
            e.g. ``"whisper"``.  When None the audio branch is disabled.
        audio_pretrained_feature_extractor: HF model ID or local path,
            e.g. ``"openai/whisper-medium"``.
        audio_feat_dim: Output dim of the audio feature extractor.
            Whisper-medium → 1024.
        audio_freeze_feature_extractor: Freeze the audio encoder weights.
        audio_multimodal_mapper_type: Mapper type for the audio branch —
            one of ``{"linear", "adapter", "cnn_adapter"}``.
        audio_multimodal_mapper_layer_norm_before: LayerNorm before mapper.
        audio_multimodal_mapper_layer_norm: LayerNorm inside mapper.
        audio_multimodal_mapper_activation: ReLU at mapper output.
        audio_multimodal_mapper_factor: Overparameterization factor (adapter).
        audio_multimodal_mapper_dropout: Dropout in the audio mapper.
        audio_freeze_multimodal_mapper: Freeze audio mapper weights.
        ot_lambda: Weight for the Sinkhorn OT loss.  0.0 disables it.
        sinkhorn_epsilon: Entropy regularization for Sinkhorn.
        sinkhorn_max_iter: Number of Sinkhorn iterations.
    """

    model_type = "siamese_multimodal_embedder"

    def __init__(
        self,
        model_type: str = "siamese_multimodal_embedder",
        # --- Audio feature extractor ---
        audio_feature_extractor_type: Optional[str] = None,
        audio_pretrained_feature_extractor: Optional[str] = None,
        audio_feat_dim: int = 1024,
        audio_freeze_feature_extractor: bool = True,
        # --- Audio multimodal mapper ---
        audio_multimodal_mapper_type: Optional[str] = None,
        audio_multimodal_mapper_layer_norm_before: bool = False,
        audio_multimodal_mapper_layer_norm: bool = False,
        audio_multimodal_mapper_activation: bool = False,
        audio_multimodal_mapper_factor: Optional[int] = None,
        audio_multimodal_mapper_dropout: Optional[float] = None,
        audio_freeze_multimodal_mapper: bool = False,
        # --- OT alignment ---
        ot_lambda: float = 1.0,
        sinkhorn_epsilon: float = 0.1,
        sinkhorn_max_iter: int = 100,
        # --- Eval ---
        eval_audio: bool = True,
        **kwargs,
    ):
        super().__init__(model_type=model_type, **kwargs)

        self.audio_feature_extractor_type = audio_feature_extractor_type
        self.audio_pretrained_feature_extractor = audio_pretrained_feature_extractor
        self.audio_feat_dim = audio_feat_dim
        self.audio_freeze_feature_extractor = audio_freeze_feature_extractor

        self.audio_multimodal_mapper_type = audio_multimodal_mapper_type
        self.audio_multimodal_mapper_layer_norm_before = audio_multimodal_mapper_layer_norm_before
        self.audio_multimodal_mapper_layer_norm = audio_multimodal_mapper_layer_norm
        self.audio_multimodal_mapper_activation = audio_multimodal_mapper_activation
        self.audio_multimodal_mapper_factor = audio_multimodal_mapper_factor
        self.audio_multimodal_mapper_dropout = audio_multimodal_mapper_dropout
        self.audio_freeze_multimodal_mapper = audio_freeze_multimodal_mapper

        self.ot_lambda = ot_lambda
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_max_iter = sinkhorn_max_iter
        self.eval_audio = eval_audio
