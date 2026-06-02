# Standard Library Imports
from typing import Optional

# Local Application Imports
from multimodalhugs.models.multimodal_embedder.configuration_multimodal_embedder import MultiModalEmbedderConfig


class SiameseMultiModalEmbedderConfig(MultiModalEmbedderConfig):
    """
    **SiameseMultiModalEmbedderConfig: Configuration for SiameseMultiModalEmbedderModel.**

    Extends ``MultiModalEmbedderConfig`` with a second (audio) feature extractor,
    an audio-specific MultimodalMapper, and Optimal Transport alignment parameters.

    Video pathway (unchanged from parent):
        FeatureExtractor → MultimodalMapper → merge_modalities → Backbone

    Audio pathway (added):
        AudioFeatureExtractor → AudioMultimodalMapper → OT loss

    OT position is controlled by ``ot_position``:
        ``"mapper"``  — OT between mapper outputs, before merge_modalities.
        ``"encoder"`` — OT between M2M encoder outputs (audio pass uses no_grad).
        ``"both"``    — OT at mapper output and encoder output simultaneously.

    **Args:**
    - `pretrained_speech_checkpoint` (Optional[str]): Path to a speech2text checkpoint.
      When set, backbone + audio branch weights are loaded from it (warm-start for Phase 2).
    - `audio_feature_extractor_type` (Optional[str]): Type string for the audio encoder, e.g. ``"whisper"``.
    - `audio_pretrained_feature_extractor` (Optional[str]): HF model ID or local path, e.g. ``"openai/whisper-medium"``.
    - `audio_feat_dim` (int): Output dim of the audio feature extractor. Whisper-medium → 1024.
    - `audio_freeze_feature_extractor` (bool): Freeze the audio encoder weights.
    - `audio_multimodal_mapper_type` (Optional[str]): Mapper type — one of ``{"linear", "adapter", "cnn_adapter"}``.
    - `audio_multimodal_mapper_layer_norm_before` (bool): LayerNorm before mapper.
    - `audio_multimodal_mapper_layer_norm` (bool): LayerNorm inside mapper.
    - `audio_multimodal_mapper_activation` (bool): ReLU at mapper output.
    - `audio_multimodal_mapper_factor` (Optional[int]): Overparameterization factor (adapter).
    - `audio_multimodal_mapper_dropout` (Optional[float]): Dropout in the audio mapper.
    - `audio_freeze_multimodal_mapper` (bool): Freeze audio mapper weights.
    - `ot_lambda` (float): Weight for the OT loss at mapper output. Default: 1.0.
    - `ot_lambda_encoder` (float): Weight for the OT loss at encoder output (only for ``ot_position="both"``).
    - `sinkhorn_epsilon` (float): Entropy regularization for Sinkhorn.
    - `sinkhorn_max_iter` (int): Number of Sinkhorn iterations.
    - `ot_position` (str): Where OT is applied — ``"mapper"`` (default), ``"encoder"``, or ``"both"``.
    - `eval_audio` (bool): Run an audio-only eval pass during training.
    """

    model_type = "siamese_multimodal_embedder"

    def __init__(
        self,
        model_type: str = "siamese_multimodal_embedder",
        # --- Speech checkpoint warm-start ---
        pretrained_speech_checkpoint: Optional[str] = None,
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
        ot_lambda_encoder: float = 1.0,
        sinkhorn_epsilon: float = 0.1,
        sinkhorn_max_iter: int = 100,
        ot_position: str = "mapper",
        # --- Eval ---
        eval_audio: bool = True,
        **kwargs,
    ):
        super().__init__(model_type=model_type, **kwargs)

        self.pretrained_speech_checkpoint = pretrained_speech_checkpoint

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
        self.ot_lambda_encoder = ot_lambda_encoder
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_max_iter = sinkhorn_max_iter
        self.ot_position = ot_position
        self.eval_audio = eval_audio
