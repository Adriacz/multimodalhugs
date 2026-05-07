import torch
import random

from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
from transformers.utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def create_seq2seq_labels_from_samples(
    samples: List[Dict[str, Union[str, List[int]]]],
    tokenizer: PreTrainedTokenizerBase,
    label_pad_token_id: int = -100,
    padding: Union[bool, str, PaddingStrategy] = True,
    max_length: Optional[int] = None,
    pad_to_multiple_of: Optional[int] = None,
    return_tensors: str = "pt"
) -> Optional[Dict[str, Union[torch.Tensor, Any]]]:

    """
    Tokenizes and formats decoder labels from a batch of multimodal samples.

    This function processes the 'decoder_prompt' and 'output' fields from each sample by
    tokenizing them, concatenating the resulting token IDs, appending an EOS token,
    and optionally padding them to a common length.

    Args:
        samples (List[Dict[str, Union[str, List[int]]]]): A list of samples, each containing
            the fields 'decoder_prompt' and 'output' (both strings expected).
        tokenizer (PreTrainedTokenizerBase): The tokenizer used for tokenizing text fields and
            converting tokens to IDs.
        label_pad_token_id (int, optional): The ID used to pad sequences. This is typically
            set to -100 so that the loss function ignores padding tokens. Defaults to -100.
        padding (Union[bool, str, PaddingStrategy], optional): Padding strategy to apply. If False,
            no padding is applied. If 'max_length' or True, padding is applied based on `max_length`
            or the longest sequence. Defaults to True.
        max_length (Optional[int], optional): The maximum length to pad/truncate the label sequences to.
            Required if padding is 'max_length'. Defaults to None.
        pad_to_multiple_of (Optional[int], optional): If provided, the sequences will be padded to a multiple
            of this value. Useful for hardware optimization. Defaults to None.
        return_tensors (str, optional): The tensor format to return: 'pt' (PyTorch), 'tf' (TensorFlow), or 'np' (NumPy).
            Defaults to 'pt'.

    Returns:
        Optional[Dict[str, Union[torch.Tensor, Any]]]: A dictionary containing:
            - 'labels': A tensor or array of padded token ID sequences.
        If any sample is missing an 'output', returns None.
    """

    # Check for missing outputs
    if any(sample.get('output') is None for sample in samples):
        return None

    # Build raw label sequences
    labels = []
    for sample in samples:
        # Tokenize decoder prompt and output, append EOS token
        prompt_ids = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(sample['decoder_prompt'])
        )
        output_ids = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(sample['output'])
        )
        combined = prompt_ids + output_ids + [tokenizer.eos_token_id]
        labels.append(combined)

    # Determine if padding is disabled
    no_padding = padding is False or padding == PaddingStrategy.DO_NOT_PAD

    if no_padding:
        return {'labels': labels}

    # Determine target max length
    if padding == PaddingStrategy.MAX_LENGTH and max_length:
        max_len = max_length
    else:
        max_len = max(len(seq) for seq in labels)

    # Adjust to pad_to_multiple_of
    if pad_to_multiple_of:
        multiple = pad_to_multiple_of
        max_len = ((max_len + multiple - 1) // multiple) * multiple

    # Pad sequences on correct side
    padded = []
    side = tokenizer.padding_side
    for seq in labels:
        pad_len = max_len - len(seq)
        if side == 'right':
            padded_seq = seq + [label_pad_token_id] * pad_len
        else:
            padded_seq = [label_pad_token_id] * pad_len + seq
        padded.append(padded_seq)
    final_labels = padded

    # Convert lists to tensors based on return format
    if return_tensors == 'pt':
        return {'labels': torch.tensor(final_labels, dtype=torch.int64)}
    elif return_tensors == 'tf':
        import tensorflow as tf
        return {'labels': tf.constant(final_labels, dtype=tf.int64)}
    else:
        import numpy as np
        return {'labels': np.array(final_labels, dtype=np.int64)}


@dataclass
class DataCollatorMultimodalSeq2Seq:
    """
    Data collator for multimodal sequence-to-sequence tasks.

    This collator prepares batches that include text labels (e.g., for decoder outputs),
    optionally pads them, and can use the model to generate decoder input IDs.
    It also delegates the multimodal input construction to a provided processor.

    Args:
        processor (Any): A ProcessorMixin object that processes multimodal and related inputs.
        tokenizer (PreTrainedTokenizerBase): Tokenizer used to tokenize and convert labels to IDs.
        model (Optional[Any], optional): The model used to optionally prepare decoder input IDs
            from labels. Defaults to None.
        padding (Union[bool, str, PaddingStrategy], optional): Padding strategy, can be True, False,
            'longest', 'max_length', or PaddingStrategy enum. Defaults to True.
        max_length (Optional[int], optional): If set, truncates/pads sequences to this length. Required if
            padding is set to 'max_length'. Defaults to None.
        pad_to_multiple_of (Optional[int], optional): If set, pad sequences to a multiple of this value.
            Useful for hardware optimization (e.g., XLA or Tensor Cores). Defaults to None.
        label_pad_token_id (int, optional): Token ID used to pad the labels. Ignored by loss functions
            like cross-entropy. Defaults to -100.
        return_tensors (str, optional): The format in which to return the batch ('pt', 'tf', or 'np').
            Defaults to 'pt'.
    """
    processor: Any
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __init__(
        self,
        processor: Any,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model: Optional[Any] = None,  # The model (optional, used to create decoder input IDs if needed)
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
        return_tensors: str = "pt",
    ):
        """
        Initialize the multimodal sequence-to-sequence data collator.

        Args:
            processor (Any): A ProcessorMixin object that processes multimodal inputs and merges them with text fields.
            tokenizer (PreTrainedTokenizerBase, optional): Tokenizer to tokenize text inputs and outputs.
                If None, defaults to `processor.tokenizer`.
            model (Optional[Any], optional): The model used to prepare decoder_input_ids from labels.
            padding (Union[bool, str, PaddingStrategy], optional): Strategy for padding labels.
                Can be True, False, 'longest', 'max_length', or a PaddingStrategy.
            max_length (Optional[int], optional): Max length for padding/truncating labels.
            pad_to_multiple_of (Optional[int], optional): Pads to a multiple of this number, if set.
            label_pad_token_id (int, optional): Token ID used to pad label sequences.
            return_tensors (str, optional): Format of output tensors: 'pt', 'tf', or 'np'. Defaults to 'pt'.
        """
        self.processor = processor
        self.tokenizer = tokenizer if tokenizer is not None else processor.tokenizer
        self.model = model
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors

    def _obtain_labels_and_decoder_input_ids(
        self,
        samples: List[Dict[str, Union[List[int], torch.Tensor]]],
        return_tensors: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize, pad, and optionally create decoder inputs from text labels.

        Args:
            samples: List of dicts with 'decoder_prompt' and 'output'.
            return_tensors: Override for return_tensors attribute.

        Returns:
            A dict with:
              - 'labels': Padded token ID sequences.
              - 'decoder_input_ids': Prepared IDs if model supports it.
        """
        rt = return_tensors or self.return_tensors
        batch = create_seq2seq_labels_from_samples(
            samples=samples,
            tokenizer=self.tokenizer,
            label_pad_token_id=self.label_pad_token_id,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=rt
        ) or {}
        # Use model to prepare decoder inputs if available
        if 'labels' in batch and self.model and hasattr(self.model, 'prepare_decoder_input_ids_from_labels') and self.model.training:
            batch['decoder_input_ids'] = self.model.prepare_decoder_input_ids_from_labels(
                labels=batch['labels']
            )
        return batch

    def __call__(
        self,
        samples: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of multimodal samples for model input.

        Args:
            samples: Each item contains multimodal data and text fields.

        Returns:
            A dict ready for model.forward(), including inputs and labels.
        """
        # Process text side: tokenization, padding, decoder inputs
        text_batch = self._obtain_labels_and_decoder_input_ids(samples)
        full_batch = self.processor(
            batch=samples,
            batch_dict=text_batch,
            return_tensors=self.return_tensors,
        )
        return full_batch


# Modality constants
MODALITY_VIDEO = "video"
MODALITY_AUDIO = "audio"
MODALITY_BOTH  = "both"


@dataclass
class DataCollatorMultimodalVideoAudio(DataCollatorMultimodalSeq2Seq):
    """
    Data collator for multimodal sequence-to-sequence tasks with video and audio inputs.

    Extends DataCollatorMultimodalSeq2Seq to handle two input modalities simultaneously.
    For each sample in the batch, the collator randomly assigns a modality (video or audio),
    or uses both. The inactive modality for each sample is represented as an all-zeros tensor
    with an all-zeros mask, so the model always receives tensors of consistent shape.

    The randomness is fully contained here. The model and processors are unaware of the
    assignment — they only see tensors and masks.

    Args:
        video_processor (Any): Processor for the video modality. Must implement
            _video_file_to_tensor(path, signal_start, signal_end).
        audio_processor (Any): Processor for the audio modality. Must implement
            _audio_to_tensor(path, audio_start, audio_end).
        tokenizer (PreTrainedTokenizerBase): Tokenizer used to tokenize and convert labels to IDs.
        model (Optional[Any], optional): The model used to optionally prepare decoder input IDs
            from labels. Defaults to None.
        modality_sampling (str, optional): Controls which modalities are active per sample.
            - 'random': each sample gets video OR audio, chosen independently at random.
            - 'video':  all samples use video only.
            - 'audio':  all samples use audio only.
            - 'both':   all samples use both modalities simultaneously.
            Defaults to 'random'.
        video_prob (float, optional): Probability of assigning video when
            modality_sampling='random'. Defaults to 0.5.
        padding (Union[bool, str, PaddingStrategy], optional): Padding strategy for labels.
            Defaults to True.
        max_length (Optional[int], optional): If set, truncates/pads label sequences to this length.
            Defaults to None.
        pad_to_multiple_of (Optional[int], optional): If set, pad sequences to a multiple of this value.
            Defaults to None.
        label_pad_token_id (int, optional): Token ID used to pad the labels. Defaults to -100.
        return_tensors (str, optional): The format in which to return the batch ('pt', 'tf', or 'np').
            Defaults to 'pt'.
    """
    video_processor: Any = None
    audio_processor: Any = None
    modality_sampling: str = "random"
    video_prob: float = 0.7

    def __init__(
        self,
        video_processor: Any,
        audio_processor: Any,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model: Optional[Any] = None,  # The model (optional, used to create decoder input IDs if needed)
        modality_sampling: str = "random",
        video_prob: float = 0.5,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
        return_tensors: str = "pt",
    ):
        # We pass video_processor as the generic 'processor' to the parent,
        # but __call__ is fully overridden so it is not used for signal processing.
        super().__init__(
            processor=video_processor,
            tokenizer=tokenizer if tokenizer is not None else video_processor.tokenizer,
            model=model,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            label_pad_token_id=label_pad_token_id,
            return_tensors=return_tensors,
        )
        self.video_processor = video_processor
        self.audio_processor = audio_processor
        self.modality_sampling = modality_sampling
        self.video_prob = video_prob

    def _assign_modalities(self, n: int) -> List[str]:
        """
        Returns a list of length n with a modality label per sample.

        Args:
            n: Number of samples in the batch.

        Returns:
            List of modality strings, one per sample. Each value is one of
            MODALITY_VIDEO, MODALITY_AUDIO, or MODALITY_BOTH.
        """
        if self.modality_sampling == MODALITY_VIDEO:
            return [MODALITY_VIDEO] * n
        if self.modality_sampling == MODALITY_AUDIO:
            return [MODALITY_AUDIO] * n
        if self.modality_sampling == MODALITY_BOTH:
            return [MODALITY_BOTH] * n
        # 'random': each sample independently gets video or audio
        return [
            MODALITY_VIDEO if random.random() < self.video_prob else MODALITY_AUDIO
            for _ in range(n)
        ]

    def _get_video_tensor(self, sample: Dict) -> torch.Tensor:
        """
        Extracts a video tensor [T, C, H, W] for a single sample.

        Args:
            sample: Dict containing 'signal', 'signal_start', 'signal_end'.

        Returns:
            Video tensor of shape [T, C, H, W] (or [T, *] depending on processor).
        """
        return self.video_processor._video_file_to_tensor(
            sample["signal"],
            sample.get("signal_start", 0.0),
            sample.get("signal_end",   0.0),
        )

    def _get_audio_tensor(self, sample: Dict) -> torch.Tensor:
        """
        Extracts an audio tensor [T, n_mels] for a single sample.

        The audio processor natively produces [n_mels, T], so we transpose
        here to make the temporal dimension consistent (dim 0) with the video
        tensor and with pad_and_create_mask expectations.

        Args:
            sample: Dict containing 'signal', 'audio_start', 'audio_end'.

        Returns:
            Audio tensor of shape [n_mels, T].
        """
        mel = self.audio_processor._audio_to_tensor(
            sample["signal"],
            sample.get("audio_start", 0.0),
            sample.get("audio_end",   0.0),
        )
        return mel # [n_mels, T]

    def _build_modality_tensors(
        self,
        samples: List[Dict],
        modalities: List[str],
    ):
        """
        Builds padded tensors and masks for both modalities across the batch.

        For each sample, video and/or audio tensors are extracted according to
        the assigned modality. Samples whose modality is inactive receive an
        all-zeros tensor and an all-zeros mask.

        Args:
            samples: List of sample dicts.
            modalities: List of modality labels, one per sample.

        Returns:
            input_frames        Tensor [B, T_v, ...]    padded video frames
            frames_padding_mask Tensor [B, T_v]          1=real frame, 0=padding
            input_audio         Tensor [B, n_mels , T_a]  padded audio frames
            audio_padding_mask  Tensor [B, T_a]          1=real frame, 0=padding
        """
        video_tensors = []
        audio_tensors = []

        for sample, modality in zip(samples, modalities):
            use_video = modality in (MODALITY_VIDEO, MODALITY_BOTH)
            use_audio = modality in (MODALITY_AUDIO, MODALITY_BOTH)
            video_tensors.append(self._get_video_tensor(sample) if use_video else None)
            audio_tensors.append(self._get_audio_tensor(sample) if use_audio else None)

        # Determine reference shapes and max lengths for padding
        # Use defaults when all tensors of a modality are None (entire batch is the other modality)
        _ref_video = next((t for t in video_tensors if t is not None), None)
        _ref_audio = next((t for t in audio_tensors if t is not None), None)
        ref_video_shape = _ref_video.shape[1:] if _ref_video is not None else (3, 224, 224)
        ref_n_mels = _ref_audio.shape[0] if _ref_audio is not None else 80
        max_T_v = max(t.shape[0] if t is not None else 0 for t in video_tensors)
        max_T_a = max(t.shape[1] if t is not None else 0 for t in audio_tensors)

        # If all tensors of a modality are None, use size 1 to avoid zero-dim tensors
        max_T_v = max_T_v if max_T_v > 0 else 1
        max_T_a = max_T_a if max_T_a > 0 else 1

        B = len(samples)

        # Allocate output tensors (zeros serve as padding for inactive modalities)
        input_frames        = torch.zeros((B, max_T_v) + ref_video_shape)
        frames_padding_mask = torch.zeros((B, max_T_v), dtype=torch.long)
        input_audio = torch.zeros((B, ref_n_mels, max_T_a))
        audio_padding_mask  = torch.zeros((B, max_T_a), dtype=torch.long)

        for i, (vt, at) in enumerate(zip(video_tensors, audio_tensors)):
            if vt is not None:
                T_v = vt.shape[0]
                input_frames[i, :T_v]        = vt
                frames_padding_mask[i, :T_v] = 1
            if at is not None:
                T_a = at.shape[1]
                input_audio[i, :, :T_a]   = at
                audio_padding_mask[i, :T_a] = 1

        return input_frames, frames_padding_mask, input_audio, audio_padding_mask

    def __call__(
        self,
        samples: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of multimodal samples for model input.

        Args:
            samples: Each item contains multimodal data and text fields.

        Returns:
            A dict ready for model.forward(), including inputs and labels.
        """
        # Assign a modality to each sample in the batch
        modalities = self._assign_modalities(len(samples))

        # Build video and audio tensors with their respective masks
        input_frames, frames_padding_mask, input_audio, audio_padding_mask = \
            self._build_modality_tensors(samples, modalities)

        # Process text side: tokenization, padding, decoder inputs
        text_batch = self._obtain_labels_and_decoder_input_ids(samples)

        # Process encoder prompt if present
        # (both processors share the same tokenizer so either works)
        if any(s.get("encoder_prompt") for s in samples):
            padded_prompts, prompt_mask = self.video_processor.process_prompts(
                [s.get("encoder_prompt", "") for s in samples]
            )
            text_batch["encoder_prompt"] = padded_prompts
            text_batch["encoder_prompt_length_padding_mask"] = prompt_mask

        full_batch = {
            "input_frames":        input_frames,
            "frames_padding_mask": frames_padding_mask,
            "input_audio":         input_audio,
            "audio_padding_mask":  audio_padding_mask,
            **text_batch,
        }

        return full_batch
