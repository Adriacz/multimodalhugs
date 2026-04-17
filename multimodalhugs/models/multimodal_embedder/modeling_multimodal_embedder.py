# Standard Library Imports
import logging
import math
import importlib
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, Union

# Third-Party Imports
import torch
from transformers.models.auto.modeling_auto import MODEL_WITH_LM_HEAD_MAPPING_NAMES
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers import (
    M2M100ForConditionalGeneration,
    PreTrainedModel,
    PretrainedConfig,
    AutoConfig,
)
from transformers.modeling_outputs import Seq2SeqLMOutput
from accelerate.utils import find_tied_parameters
from ruamel.yaml import YAML

# Local Application Imports
from multimodalhugs.models.utils import get_backbone_config_class, get_backbone_model_class
from multimodalhugs.utils.registry import register_model
from multimodalhugs.modules import MultimodalMapper, FeatureExtractor, get_feature_extractor_class
from multimodalhugs.modules.utils import set_module_parameters, extend_all_embeddings_and_lm_head, merge_modalities, merge_modalities_mask_correction
from multimodalhugs.utils import serialize_config
from multimodalhugs.models.multimodal_embedder.configuration_multimodal_embedder import MultiModalEmbedderConfig
from multimodalhugs.models.utils import EncoderWrapper

logger = logging.getLogger(__name__)

# Define the custom model class
@register_model("multimodal_embedder")
class MultiModalEmbedderModel(PreTrainedModel):
    """
    **MultiModalEmbedderModel: A Transformer-based multimodal model.**

    This model extends `transformers.PreTrainedModel`, integrating visual and textual 
    inputs using a feature extractor, a Multimodal Mapper (Multimodal Mapper), and 
    a backbone Transformer model.

    When both `audio_feature_extractor_type` and `feature_extractor_type` are set in
    the config, the model operates in multimodal mode: it processes video frames and
    audio mel-spectrograms through independent feature extractors and mappers, then
    concatenates their embeddings along the temporal dimension before passing them to
    the backbone encoder.
    """
    config_class = MultiModalEmbedderConfig
    base_model_prefix = "multimodal_embedder"
    is_parallelizable = True
    _keep_in_fp32_modules = []
    _no_split_modules = []

    def __init__(self, config):
        """
        **Initialize the MultiModalEmbedderModel.**

        **Args:**
        - `config` (MultiModalEmbedderConfig): Model configuration.
        """
        super().__init__(config)
        self._init_feature_extractor(config)
        self._init_multimodal_mapper(config)
        self._init_audio_feature_extractor(config)
        self._init_audio_multimodal_mapper(config)
        self.decoder_start_token_id = config.decoder_start_token_id
        self.pad_token_id = config.pad_token_id
        self.eos_token_id = config.eos_token_id
        self.bos_token_id = config.bos_token_id if config.bos_token_id is not None else config.decoder_start_token_id
        self._init_backbone(config)
        self.max_length = config.max_length
        self.post_init()

    def _init_feature_extractor(self, config):
        """
        **Initialize the video feature extractor.**

        **Args:**
        - `config` (MultiModalEmbedderConfig): Model configuration.
        """
        if config.feature_extractor_type:
            self.feature_extractor = FeatureExtractor(
                feature_extractor_type=config.feature_extractor_type,
                pretrained_module=config.pretrained_feature_extractor,
                config=config.feature_extractor_config,
            )
            set_module_parameters(self.feature_extractor, freeze=config.freeze_feature_extractor)
        else:
            self.feature_extractor = None

        if self.feature_extractor is not None:
            self.is_parallelizable = self.is_parallelizable and getattr(self.feature_extractor, "is_parallelizable", True)
            self._no_split_modules = self._no_split_modules + (getattr(self.feature_extractor, "_no_split_modules", []) or [])
            self._keep_in_fp32_modules = self._keep_in_fp32_modules + (getattr(self.feature_extractor, "_keep_in_fp32_modules", []) or [])

    def _init_multimodal_mapper(self, config):
        """
        **Initialize the video multimodal mapper.**

        **Args:**
        - `config` (MultiModalEmbedderConfig): Model configuration.
        """
        if config.multimodal_mapper_type is not None:
            self.multimodal_mapper = MultimodalMapper(
                feat_dim=config.feat_dim,
                output_dim=config.d_model,
                mapping_layer_type=config.multimodal_mapper_type,
                layer_norm_before=config.multimodal_mapper_layer_norm_before,
                adapter_factor=config.multimodal_mapper_factor,
                p_dropout=config.multimodal_mapper_dropout,
                layer_norm=config.multimodal_mapper_layer_norm,
                activation=config.multimodal_mapper_activation,
                adapter_ksize=config.adapter_ksize,
                adapter_stride=config.adapter_stride
            )
            set_module_parameters(self.multimodal_mapper, freeze=config.freeze_multimodal_mapper)
        else:
            self.multimodal_mapper = None

    def _init_audio_feature_extractor(self, config):
        """
        **Initialize the audio feature extractor.**

        **Args:**
        - `config` (MultiModalEmbedderConfig): Model configuration.
        """
        if config.audio_feature_extractor_type:
            self.audio_feature_extractor = FeatureExtractor(
                feature_extractor_type=config.audio_feature_extractor_type,
                pretrained_module=config.audio_pretrained_feature_extractor,
                config=config.audio_feature_extractor_config,
            )
            set_module_parameters(self.audio_feature_extractor, freeze=config.audio_freeze_feature_extractor)
        else:
            self.audio_feature_extractor = None

        if self.audio_feature_extractor is not None:
            self.is_parallelizable = self.is_parallelizable and getattr(self.audio_feature_extractor, "is_parallelizable", True)
            self._no_split_modules = self._no_split_modules + (getattr(self.audio_feature_extractor, "_no_split_modules", []) or [])
            self._keep_in_fp32_modules = self._keep_in_fp32_modules + (getattr(self.audio_feature_extractor, "_keep_in_fp32_modules", []) or [])

    def _init_audio_multimodal_mapper(self, config):
        """
        **Initialize the audio multimodal mapper.**

        **Args:**
        - `config` (MultiModalEmbedderConfig): Model configuration.
        """
        if config.audio_multimodal_mapper_type is not None:
            self.audio_multimodal_mapper = MultimodalMapper(
                feat_dim=config.audio_feat_dim,
                output_dim=config.d_model,
                mapping_layer_type=config.audio_multimodal_mapper_type,
                layer_norm_before=config.audio_multimodal_mapper_layer_norm_before,
                adapter_factor=config.audio_multimodal_mapper_factor,
                p_dropout=config.audio_multimodal_mapper_dropout,
                layer_norm=config.audio_multimodal_mapper_layer_norm,
                activation=config.audio_multimodal_mapper_activation,
            )
            set_module_parameters(self.audio_multimodal_mapper, freeze=config.audio_freeze_multimodal_mapper)
        else:
            self.audio_multimodal_mapper = None

    def _init_backbone(self, config):
        """
        **Initialize the Transformer backbone model.**

        **Args:**
        - `config` (MultiModalEmbedderConfig): Model configuration.
        """
        BackboneModelClass = get_backbone_model_class(config.backbone_type)
        if config.backbone_config is not None:
            self.backbone = BackboneModelClass(config.backbone_config)
        else:
            self.backbone = BackboneModelClass.from_pretrained(config.pretrained_backbone)
        
        if isinstance(config.backbone_tied_weights_keys, list):
            self.backbone._tied_weights_keys = config.backbone_tied_weights_keys

        set_module_parameters(self.backbone, freeze=config.freeze_backbone)
        set_module_parameters(self.get_backbone_encoder.embed_tokens, freeze=config.freeze_encoder_embed_tokens)
        set_module_parameters(self.get_backbone_decoder.embed_tokens, freeze=config.freeze_decoder_embed_tokens)
        set_module_parameters(self.lm_head, freeze=config.freeze_lm_head)
        
        freeze_shared = (
            config.freeze_decoder_embed_tokens or 
            config.freeze_encoder_embed_tokens or 
            config.freeze_lm_head
        )
        set_module_parameters(self.get_shared, freeze=freeze_shared, verbose=False)
        self.is_parallelizable = self.is_parallelizable and getattr(self.backbone, "is_parallelizable", True)
        self._no_split_modules = self._no_split_modules + (getattr(self.backbone, "_no_split_modules", []) or [])
        self._keep_in_fp32_modules = self._keep_in_fp32_modules + (getattr(self.backbone, "_keep_in_fp32_modules", []) or [])

    def _process_video_branch(self, input_frames, frames_padding_mask):
        """
        **Run the video branch: feature extractor + mapper.**

        **Args:**
        - `input_frames` (torch.Tensor): Video tensor of shape `(B, T_v, *)`.
        - `frames_padding_mask` (Optional[torch.Tensor]): Mask of shape `(B, T_v)`, 1=real, 0=padding.

        **Returns:**
        - Tuple of (embeddings `(B, T_v', d_model)`, mask `(B, T_v')`).
        """
        if self.feature_extractor is not None:
            video_embeds = self.feature_extractor(input_frames)
            if frames_padding_mask is not None:
                B, T = video_embeds.shape[:2]
                frames_padding_mask = torch.ones((B, T), dtype=frames_padding_mask.dtype, device=frames_padding_mask.device)
        else:
            video_embeds = input_frames

        if self.multimodal_mapper is not None:
            video_embeds, frames_padding_mask = self.multimodal_mapper(video_embeds, frames_padding_mask)

        return video_embeds, frames_padding_mask

    def _process_audio_branch(self, input_audio, audio_padding_mask):
        """
        **Run the audio branch: feature extractor + mapper.**

        `input_audio` arrives as `(B, n_mels, T_a)` from the collator.
        Whisper's encoder expects `(B, n_mels, T)`, so no transposition is needed.
        After the encoder the output is `(B, T_a', d_model)`.

        **Args:**
        - `input_audio` (torch.Tensor): Audio mel tensor of shape `(B, n_mels, T_a)`.
        - `audio_padding_mask` (Optional[torch.Tensor]): Mask of shape `(B, T_a)`, 1=real, 0=padding.

        **Returns:**
        - Tuple of (embeddings `(B, T_a', d_model)`, mask `(B, T_a')`).
        """
        if self.audio_feature_extractor is not None:
            # Whisper encoder expects [B, n_mels, T] — matches collator output directly
            audio_embeds = self.audio_feature_extractor(input_audio)
            if audio_padding_mask is not None:
                B, T = audio_embeds.shape[:2]
                audio_padding_mask = torch.ones((B, T), dtype=audio_padding_mask.dtype, device=audio_padding_mask.device)
        else:
            audio_embeds = input_audio

        if self.audio_multimodal_mapper is not None:
            audio_embeds, audio_padding_mask = self.audio_multimodal_mapper(audio_embeds, audio_padding_mask)

        return audio_embeds, audio_padding_mask

    def _fuse_modalities(self, video_embeds, video_mask, audio_embeds, audio_mask):
        """
        **Fuse video and audio embeddings by temporal concatenation.**

        Both tensors are `(B, T, d_model)` at this point. The result is
        `(B, T_v' + T_a', d_model)`. Samples where one modality is all-padding
        contribute nothing — the attention mask ensures the encoder ignores them.

        **Args:**
        - `video_embeds` (Optional[torch.Tensor]): Shape `(B, T_v', d_model)`.
        - `video_mask` (Optional[torch.Tensor]): Shape `(B, T_v')`.
        - `audio_embeds` (Optional[torch.Tensor]): Shape `(B, T_a', d_model)`.
        - `audio_mask` (Optional[torch.Tensor]): Shape `(B, T_a')`.

        **Returns:**
        - Tuple of (fused embeddings `(B, T_v'+T_a', d_model)`, fused mask `(B, T_v'+T_a')`).
        """
        if video_embeds is not None and audio_embeds is not None:
            fused_embeds = torch.cat([video_embeds, audio_embeds], dim=1)
            if video_mask is not None and audio_mask is not None:
                fused_mask = torch.cat([video_mask, audio_mask], dim=1)
            else:
                fused_mask = None
            return fused_embeds, fused_mask

        if video_embeds is not None:
            return video_embeds, video_mask

        if audio_embeds is not None:
            return audio_embeds, audio_mask

        raise ValueError("Both video_embeds and audio_embeds are None — cannot fuse.")

    def get_input_embeddings(self):
        """
        **Retrieve the input embeddings.**

        **Returns:**
        - `torch.nn.Module`: Input embedding layer.
        """
        if hasattr(self.backbone, 'shared'):
            return self.backbone.shared
        elif hasattr(self.backbone, 'model'):
            if hasattr(self.backbone.model, 'shared'):
                return getattr(self.backbone.model, 'shared', None) 
        else:
            return None

    def set_input_embeddings(self, value):
        """
        **Set new input embeddings.**

        **Args:**
        - `value` (torch.nn.Module): New embedding module.
        """
        # Set 'shared' attribute
        if hasattr(self.backbone, 'shared'):
            self.backbone.shared = value
        elif hasattr(self.backbone, 'model') and hasattr(self.backbone.model, 'shared'):
            self.backbone.model.shared = value
        else:
            raise AttributeError("Neither 'shared' nor 'model.shared' exists in the backbone.")
        
        # Set 'encoder.embed_tokens'
        encoder = (
            getattr(self.backbone, 'encoder', None) or
            getattr(self.backbone.model, 'encoder', None)
        )
        if encoder and hasattr(encoder, 'embed_tokens'):
            encoder.embed_tokens = value

        # Set 'decoder.embed_tokens'
        decoder = (
            getattr(self.backbone, 'decoder', None) or
            getattr(self.backbone.model, 'decoder', None)
        )
        if decoder and hasattr(decoder, 'embed_tokens'):
            decoder.embed_tokens = value

    def get_output_embeddings(self):
        """
        **Retrieve the output embedding layer (LM Head).**

        **Returns:**
        - `torch.nn.Module`: LM head layer.
        """
        return self.lm_head

    @property
    def lm_head(self):
        """
        **Retrieve the language modeling head.**

        **Returns:**
        - `torch.nn.Module`: LM head module.
        """
        if hasattr(self.backbone, 'lm_head'):
            return self.backbone.lm_head
        return None

    @property
    def get_backbone_encoder(self):
        """
        **Retrieve the encoder module from the backbone.**

        This method checks if the backbone model has a direct `encoder` attribute.
        If not, it assumes the backbone has a `.model` submodule containing the encoder.

        **Returns:**
        - `torch.nn.Module`: The encoder module of the backbone model.
        """
        return self.backbone.encoder if hasattr(self.backbone, 'encoder') else self.backbone.model.encoder

    @property
    def get_backbone_decoder(self):
        """
        **Retrieve the decoder module from the backbone.**

        Similar to `get_backbone_encoder`, this method checks if the backbone model has a 
        direct `decoder` attribute. If not, it assumes the backbone has a `.model` submodule 
        containing the decoder.

        **Returns:**
        - `torch.nn.Module`: The decoder module of the backbone model.
        """
        return self.backbone.decoder if hasattr(self.backbone, 'decoder') else self.backbone.model.decoder
    
    @property
    def get_shared(self):
        """
        **Retrieve the shared embedding layer of the backbone.**

        This method returns the shared embedding layer, if present, which is commonly used 
        to tie input and output embeddings in Transformer-based architectures.

        **Returns:**
        - `torch.nn.Module` or `None`: The shared embedding layer, if available.
        """
        if hasattr(self.backbone, 'shared'):
            return self.backbone.shared
        elif hasattr(self.backbone, 'model'):
            if hasattr(self.backbone.model, 'shared'):
                return getattr(self.backbone.model, 'shared', None) 
        else:
            return None

    @classmethod
    def build_model(cls, **kwargs):
        """
        Build the model instance using provided configuration parameters.
        
        Expected common keys:
          - src_tokenizer: the source tokenizer.
          - tgt_tokenizer: the target tokenizer.
          - config_path: the path to the YAML configuration file.
          - new_vocab_tokens: a list of new vocabulary tokens.
        
        Plus any extra keys defined in the config.
        """
        src_tokenizer = kwargs.pop("src_tokenizer")
        tgt_tokenizer = kwargs.pop("tgt_tokenizer")
        config_path = kwargs.pop("config_path", None)
        new_vocab_tokens = kwargs.pop("new_vocab_tokens", [])

        if src_tokenizer is None or tgt_tokenizer is None:
            raise ValueError("Please provide the src_tokenizer and the tgt_tokenizer in case the dataset used does not have these as a parameter.")
            
        cfg = kwargs
        if not isinstance(cfg, PretrainedConfig):
            cfg = cls.config_class.from_dict(serialize_config(cfg))
        else:
            cfg = cfg

        # Load backbone model & config
        BackboneModelClass = get_backbone_model_class(cfg.backbone_type)
        if cfg.pretrained_backbone is not None:
            backbone = BackboneModelClass.from_pretrained(cfg.pretrained_backbone)
            cfg.backbone_config = AutoConfig.from_pretrained(cfg.pretrained_backbone)
        else:
            backbone = BackboneModelClass(cfg.backbone_config)
        
        cfg.d_model = cfg.d_model or cfg.backbone_config.d_model
        cfg.decoder_start_token_id = cfg.decoder_start_token_id or cfg.backbone_config.decoder_start_token_id
        cfg.backbone_tied_weights_keys = cfg.backbone_tied_weights_keys or find_tied_parameters(backbone)[0]
        backbone._tied_weights_keys = cfg.backbone_tied_weights_keys

        # Determine EOS and PAD token indices
        pad_token_id = src_tokenizer.convert_tokens_to_ids(src_tokenizer.pad_token)
        bos_token_id = src_tokenizer.convert_tokens_to_ids(src_tokenizer.bos_token)
        eos_token_id = src_tokenizer.convert_tokens_to_ids(src_tokenizer.eos_token)

        # Update configuration with these values if not already defined
        cfg.pad_token_id = cfg.pad_token_id or pad_token_id
        cfg.bos_token_id = cfg.bos_token_id or bos_token_id
        cfg.eos_token_id = cfg.eos_token_id or eos_token_id
        tokenizer_vocab_size = getattr(src_tokenizer, "total_vocab_size", src_tokenizer.vocab_size)

        backbone, new_vocab_size = extend_all_embeddings_and_lm_head(backbone=backbone, num_new_tokens=len(new_vocab_tokens), verbose=True)
        cfg.backbone_config.vocab_size = new_vocab_size

        # Create an instance of the model
        model = cls(config=cfg)

        # Copy the weights from the backbone instance to the model.backbone
        model.backbone.load_state_dict(backbone.state_dict())

        # Converts all tensors in the model to contiguous
        for param in model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        return model

    def forward(
        self,
        input_frames: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        frames_padding_mask: Optional[torch.Tensor] = None,
        input_audio: Optional[torch.Tensor] = None,
        audio_padding_mask: Optional[torch.Tensor] = None,
        encoder_prompt: Optional[torch.LongTensor] = None,
        encoder_prompt_length_padding_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        """
        **Forward pass of the MultiModalEmbedderModel.**

        This method performs the forward propagation of the model, processing multimodal 
        inputs including textual and video-based features. The method integrates visual 
        embeddings, applies the Multimodal Mapper (Multimodal Mapper), and processes text 
        tokens through the Transformer backbone.

        ### **Args:**
        - `input_frames` (Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]):  
            Video frames of shape `(B, T_v, *)`. Passed through the video feature extractor and mapper.

        - `frames_padding_mask` (Optional[torch.Tensor], shape: `(B, T_v)`):
            1=real frame, 0=padding for the video modality.

        - `input_audio` (Optional[torch.Tensor], shape: `(B, n_mels, T_a)`):
            Audio mel-spectrogram. Passed through the audio feature extractor and mapper.

        - `audio_padding_mask` (Optional[torch.Tensor], shape: `(B, T_a)`):
            1=real frame, 0=padding for the audio modality.

        - `encoder_prompt` (Optional[torch.LongTensor], shape: `(B, prompt_n_tokens)`):  
        A prompt consisting of tokenized text that is prepended to the model's input.

        - `encoder_prompt_length_padding_mask` (Optional[torch.LongTensor], shape: `(B, prompt_n_tokens)`):  
        Mask to indicate padding tokens in the encoder prompt.

        - `input_ids` (Optional[torch.LongTensor], shape: `(B, S_text)`):  
        Tokenized input sequence. Padding tokens will be ignored.

        - `attention_mask` (Optional[torch.Tensor], shape: `(B, N_frames)`):  
        A mask that indicates which tokens or frames should be attended to (`1`) and 
        which should be ignored (`0`).

        - `decoder_input_ids` (Optional[torch.LongTensor], shape: `(B, T_text)`):  
        Input IDs for the decoder during training or inference.

        - `decoder_attention_mask` (Optional[torch.LongTensor], shape: `(B, T_text)`):  
        Mask for decoder inputs, where `0` indicates padding elements.

        - `head_mask`, `decoder_head_mask`, `cross_attn_head_mask`, `encoder_outputs`,
          `past_key_values`, `inputs_embeds`, `decoder_inputs_embeds`, `labels`,
          `use_cache`, `output_attentions`, `output_hidden_states`, `return_dict`:
          Same semantics as the original model.

        ### **Returns:**
        - `Union[Tuple[torch.Tensor], Seq2SeqLMOutput]`

        ### **Processing Steps:**
        1. **Video branch:** `input_frames` → `feature_extractor` → `multimodal_mapper` → `(B, T_v', d_model)`
        2. **Audio branch:** `input_audio` → `audio_feature_extractor` → `audio_multimodal_mapper` → `(B, T_a', d_model)`
        3. **Fusion:** temporal concatenation → `(B, T_v'+T_a', d_model)`
        4. **Backbone:** fused embeddings + encoder prompt → M2M-100 encoder → decoder → logits
        """

        if encoder_outputs is None:
            if labels is not None:
                # Normal training: Use backbone method to create decoder_input_ids from labels (If model has a method called "prepare_decoder_input_ids_from_labels()", this step is done by the collator)
                decoder_input_ids = None
                decoder_attention_mask = None

            if inputs_embeds is None:
                video_embeds, video_mask = None, None
                audio_embeds, audio_mask = None, None

                # Process video branch if input_frames is provided and not all-padding
                if input_frames is not None and frames_padding_mask is not None and frames_padding_mask.any():
                    video_embeds, video_mask = self._process_video_branch(input_frames, frames_padding_mask)
                elif input_frames is not None and frames_padding_mask is None:
                    video_embeds, video_mask = self._process_video_branch(input_frames, None)

                # Process audio branch if input_audio is provided and not all-padding
                if input_audio is not None and audio_padding_mask is not None and audio_padding_mask.any():
                    audio_embeds, audio_mask = self._process_audio_branch(input_audio, audio_padding_mask)
                elif input_audio is not None and audio_padding_mask is None:
                    audio_embeds, audio_mask = self._process_audio_branch(input_audio, None)

                if video_embeds is not None or audio_embeds is not None:
                    # Fuse both modalities (or use whichever is available)
                    inputs_embeds, attention_mask = self._fuse_modalities(
                        video_embeds, video_mask, audio_embeds, audio_mask
                    )
                elif input_ids is not None:
                    inputs_embeds = self.get_backbone_encoder.embed_tokens(input_ids)
                    input_ids = None

            if inputs_embeds is None:
                inputs_embeds = self.get_backbone_encoder.embed_tokens(input_ids)
                input_ids = None

            # Fallback: If we have embeds but no mask, we assume no padding
            if attention_mask is None and inputs_embeds is not None:
                attention_mask = torch.ones(
                    inputs_embeds.shape[:2],
                    dtype=torch.long, device=inputs_embeds.device,
                )

            inputs_embeds, attention_mask = merge_modalities(
                x=inputs_embeds, 
                padding_mask=attention_mask, 
                prompt=encoder_prompt, 
                prompt_length_padding_mask=encoder_prompt_length_padding_mask,
                embeddings_module=self.get_backbone_encoder.embed_tokens, 
                pad_idx=self.pad_token_id, 
                eos_idx=self.eos_token_id, 
            )
        else:
            if self.multimodal_mapper is not None:
                attention_mask = self.multimodal_mapper.mask_correction(attention_mask)

            # When encoder_outputs is not None, we still have to correct the mask
            attention_mask = merge_modalities_mask_correction(
                padding_mask=attention_mask, 
                prompt=encoder_prompt, 
                prompt_length_padding_mask=encoder_prompt_length_padding_mask,
                embeddings_module=self.get_backbone_encoder.embed_tokens, 
                pad_idx=self.pad_token_id, 
                eos_idx=self.eos_token_id, 
            )
            # Fix mask size for Whisper: encoder_outputs already has the correct size
            if attention_mask is not None and self.audio_feature_extractor is not None:
                if hasattr(self.config, 'audio_feature_extractor_type') and self.config.audio_feature_extractor_type == 'whisper':
                    B = attention_mask.shape[0]
                    T = encoder_outputs[0].shape[1]
                    attention_mask = torch.ones((B, T), dtype=attention_mask.dtype, device=attention_mask.device)

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds if encoder_outputs is None else None,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs

    def input_to_encoder_outputs(
        self,
        input_frames: Optional[torch.LongTensor] = None,
        frames_padding_mask: Optional[torch.Tensor] = None,
        input_audio: Optional[torch.Tensor] = None,
        audio_padding_mask: Optional[torch.Tensor] = None,
        encoder_prompt: Optional[torch.LongTensor] = None,
        encoder_prompt_length_padding_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        **Encodes the multimodal input and returns encoder outputs.**

        Processes video and/or audio inputs through their respective branches,
        fuses the resulting embeddings, and passes them to the backbone encoder.
        Used during `model.generate()` to retrieve encoder representations before decoding.

        **Args:**
        - `input_frames` (Optional[torch.Tensor]): Video tensor of shape `(B, T_v, *)`.
        - `frames_padding_mask` (Optional[torch.Tensor]): Mask of shape `(B, T_v)`.
        - `input_audio` (Optional[torch.Tensor]): Audio mel tensor of shape `(B, n_mels, T_a)`.
        - `audio_padding_mask` (Optional[torch.Tensor]): Mask of shape `(B, T_a)`.
        - `encoder_prompt` (Optional[torch.LongTensor]): Shape `(B, prompt_n_tokens)`.
        - `encoder_prompt_length_padding_mask` (Optional[torch.LongTensor]): Shape `(B, prompt_n_tokens)`.
        - `input_ids` (Optional[torch.Tensor]): Tokenized text input IDs.
        - `attention_mask` (Optional[torch.Tensor]): Attention mask.
        - `head_mask`, `inputs_embeds`, `output_attentions`, `output_hidden_states`, `return_dict`:
          Same semantics as the original model.

        **Returns:**
        - `BaseModelOutput` or `Tuple`: Encoder outputs.
        """
        if inputs_embeds is None:
            video_embeds, video_mask = None, None
            audio_embeds, audio_mask = None, None

            if input_frames is not None and frames_padding_mask is not None and frames_padding_mask.any():
                video_embeds, video_mask = self._process_video_branch(input_frames, frames_padding_mask)
            elif input_frames is not None and frames_padding_mask is None:
                video_embeds, video_mask = self._process_video_branch(input_frames, None)

            if input_audio is not None and audio_padding_mask is not None and audio_padding_mask.any():
                audio_embeds, audio_mask = self._process_audio_branch(input_audio, audio_padding_mask)
            elif input_audio is not None and audio_padding_mask is None:
                audio_embeds, audio_mask = self._process_audio_branch(input_audio, None)

            if video_embeds is not None or audio_embeds is not None:
                inputs_embeds, attention_mask = self._fuse_modalities(
                    video_embeds, video_mask, audio_embeds, audio_mask
                )
            elif input_ids is not None:
                inputs_embeds = self.get_backbone_encoder.embed_tokens(input_ids)
                input_ids = None

        if inputs_embeds is None:
            inputs_embeds = self.get_backbone_encoder.embed_tokens(input_ids)
            input_ids = None
        
        # Fallback: If we have embeds but no mask, we assume no padding
        if attention_mask is None and inputs_embeds is not None:
            attention_mask = torch.ones(
                inputs_embeds.shape[:2],
                dtype=torch.long, device=inputs_embeds.device,
            )

        inputs_embeds, attention_mask = merge_modalities(
            x=inputs_embeds, 
            padding_mask=attention_mask, 
            prompt=encoder_prompt, 
            prompt_length_padding_mask=encoder_prompt_length_padding_mask,
            embeddings_module=self.get_backbone_encoder.embed_tokens, 
            pad_idx=self.pad_token_id, 
            eos_idx=self.eos_token_id, 
        )

        return self.get_backbone_encoder(
            input_ids=None,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """
        **Prepares model inputs for autoregressive text generation.**

        This method adapts the inputs before passing them to the `backbone` model 
        during text generation (e.g., beam search or greedy decoding). It ensures 
        stability by handling empty `past_key_values` and properly structuring the 
        inputs for multimodal generation.

        **Args:**
        - `*args`: Positional arguments passed to the backbone model.
        - `**kwargs`: Keyword arguments containing `past_key_values`, `input_frames`,
          `frames_padding_mask`, `input_audio`, `audio_padding_mask`, `inputs_embeds`,
          `encoder_prompt`, and `encoder_prompt_length_padding_mask`.

        **Returns:**
        - `dict`: A dictionary containing all required inputs for the `backbone.generate()` function.
        """
        if kwargs.get('past_key_values', ()) == ():
            kwargs['past_key_values'] = None

        model_inputs = self.backbone.prepare_inputs_for_generation(*args, **kwargs)

        for key in ('input_frames', 'frames_padding_mask', 'input_audio', 'audio_padding_mask',
                    'inputs_embeds', 'encoder_prompt', 'encoder_prompt_length_padding_mask'):
            val = kwargs.get(key, None)
            if val is not None:
                model_inputs[key] = val

        return model_inputs

    def get_encoder(self):
        """
        **Retrieves the encoder component of the model.**

        **Returns:**
        - `EncoderWrapper`: The encoder module of the model.
        """
        return EncoderWrapper(self)
        
    def _reorder_cache(self, past_key_values, beam_idx):
        """
        **Reorders the past key-value cache for beam search decoding.**

        **Args:**
        - `past_key_values` (Tuple[Tuple[torch.FloatTensor]]): Cached key-value pairs.
        - `beam_idx` (torch.LongTensor, shape `(num_beams,)`): Indices of surviving beams.

        **Returns:**
        - `Tuple[Tuple[torch.FloatTensor]]`: Reordered past key-value states.
        """
        return self.backbone._reorder_cache(past_key_values, beam_idx)

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Unified interface for preparing decoder_input_ids from labels.
        Tries to detect the right method on the backbone or in its module.
        """
        # 1. Direct standard method (Bart, MBart, Marian…)
        if hasattr(self.backbone, "prepare_decoder_input_ids_from_labels"):
            return self.backbone.prepare_decoder_input_ids_from_labels(labels)

        # 2. Private method (_shift_right) (T5, ByT5…)
        if hasattr(self.backbone, "_shift_right"):
            return self.backbone._shift_right(labels)

        # 3. Public method (shift_tokens_right) on the class
        if hasattr(self.backbone, "shift_tokens_right"):
            return self.backbone.shift_tokens_right(labels)

        # 4. Fallback: scan all methods to find one with "shift" + "right" in the name
        for attr_name in dir(self.backbone):
            if "shift" in attr_name and "right" in attr_name:
                candidate = getattr(self.backbone, attr_name)
                if callable(candidate):
                    return candidate(labels)

        # 5. Dynamic import of the backbone's module
        module_name = self.backbone.__class__.__module__
        try:
            modeling_module = importlib.import_module(module_name)

            # Priority order in the module
            if hasattr(modeling_module, "prepare_decoder_input_ids_from_labels"):
                return modeling_module.prepare_decoder_input_ids_from_labels(labels)

            if hasattr(modeling_module, "_shift_right"):
                return modeling_module._shift_right(labels)

            if hasattr(modeling_module, "shift_tokens_right"):
                return modeling_module.shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

            # Last resort: search for any function with "shift" and "right" in its name
            for name in dir(modeling_module):
                if "shift" in name and "right" in name:
                    candidate = getattr(modeling_module, name)
                    if callable(candidate):
                        return candidate(labels)

        except Exception as e:
            print(f"[WARNING] Could not import {module_name}: {e}")

        # 6. Nothing found
        raise NotImplementedError(
            f"{self.__class__.__name__} could not infer how to prepare_decoder_input_ids_from_labels "
            f"for backbone {self.backbone.__class__.__name__}. "
            "Please implement manually."
        )
