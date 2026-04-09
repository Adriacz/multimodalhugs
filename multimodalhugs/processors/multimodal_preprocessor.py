# Standard library
import logging
import os
from typing import Any, Optional

# Local application imports
from multimodalhugs.processors.multimodal_sequence2sequence_processor import MultimodalSequence2SequenceProcessor
from multimodalhugs.processors.video2text_preprocessor import Video2TextTranslationProcessor
from multimodalhugs.processors.speech2text_preprocessor import Speech2TextTranslationProcessor


logger = logging.getLogger(__name__)


class MultimodalVideoAudioProcessor(MultimodalSequence2SequenceProcessor):
    """
    Wrapper processor for joint video+audio multimodal training.

    Holds a Video2TextTranslationProcessor and a Speech2TextTranslationProcessor
    as sub-processors. Signal extraction (video frames and audio mel-spectrograms)
    is delegated to DataCollatorMultimodalVideoAudio at collation time, so
    `_transform_get_items_output` is a no-op here.

    This processor is registered with AutoProcessor so that the trainer can load
    it from disk via `AutoProcessor.from_pretrained()`. The trainer then detects
    this class and constructs DataCollatorMultimodalVideoAudio instead of
    DataCollatorMultimodalSeq2Seq.

    The two sub-processors are saved in subdirectories `video_processor/` and
    `audio_processor/` inside the wrapper's save directory, and reloaded from
    there by `from_pretrained()`.
    """
    name = "multimodal_video_audio_processor"
    attributes = ["tokenizer"]
    model_input_names = ["input_frames", "frames_padding_mask", "input_audio", "audio_padding_mask", "attention_mask"]
    tokenizer_class = "AutoTokenizer"

    # Subdirectory names for the two sub-processors
    VIDEO_PROCESSOR_DIR = "video_processor"
    AUDIO_PROCESSOR_DIR = "audio_processor"

    def __init__(
        self,
        video_processor: Optional[Video2TextTranslationProcessor] = None,
        audio_processor: Optional[Speech2TextTranslationProcessor] = None,
        tokenizer: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initializes the MultimodalVideoAudioProcessor.

        Args:
            video_processor (Optional[Video2TextTranslationProcessor]): Processor for the video modality.
            audio_processor (Optional[Speech2TextTranslationProcessor]): Processor for the audio modality.
            tokenizer (Optional[Any]): Shared tokenizer for both modalities.
                If None, defaults to `video_processor.tokenizer`.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        # Use the shared tokenizer from video_processor if not provided
        if tokenizer is None and video_processor is not None:
            tokenizer = video_processor.tokenizer

        super().__init__(tokenizer=tokenizer, **kwargs)

        self.video_processor = video_processor
        self.audio_processor = audio_processor

    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Saves the wrapper processor and its two sub-processors to disk.

        The tokenizer is saved by the parent class. The two sub-processors
        are saved in subdirectories `video_processor/` and `audio_processor/`
        inside `save_directory`.

        Args:
            save_directory (str): Directory where the processor will be saved.
            **kwargs: Forwarded to the parent `save_pretrained`.
        """
        # Save tokenizer and processor_config.json via parent
        super().save_pretrained(save_directory=save_directory, **kwargs)

        # Save sub-processors in subdirectories
        if self.video_processor is not None:
            video_dir = os.path.join(save_directory, self.VIDEO_PROCESSOR_DIR)
            self.video_processor.save_pretrained(save_directory=video_dir, push_to_hub=False)

        if self.audio_processor is not None:
            audio_dir = os.path.join(save_directory, self.AUDIO_PROCESSOR_DIR)
            self.audio_processor.save_pretrained(save_directory=audio_dir, push_to_hub=False)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """
        Loads the wrapper processor and its two sub-processors from disk.

        Expects `video_processor/` and `audio_processor/` subdirectories
        inside `pretrained_model_name_or_path`.

        Args:
            pretrained_model_name_or_path (str): Path to the saved processor directory.
            **kwargs: Forwarded to the parent `from_pretrained`.

        Returns:
            MultimodalVideoAudioProcessor: The loaded processor.
        """
        # Load tokenizer via parent
        instance = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Load sub-processors from subdirectories
        video_dir = os.path.join(pretrained_model_name_or_path, cls.VIDEO_PROCESSOR_DIR)
        audio_dir = os.path.join(pretrained_model_name_or_path, cls.AUDIO_PROCESSOR_DIR)

        if os.path.isdir(video_dir):
            instance.video_processor = Video2TextTranslationProcessor.from_pretrained(video_dir)
        else:
            logger.warning(f"Video processor directory not found at {video_dir}")
            instance.video_processor = None

        if os.path.isdir(audio_dir):
            instance.audio_processor = Speech2TextTranslationProcessor.from_pretrained(audio_dir)
        else:
            logger.warning(f"Audio processor directory not found at {audio_dir}")
            instance.audio_processor = None

        return instance

    def _transform_get_items_output(self, batch):
        """
        Dataset-level transform applied during iteration (via `with_transform`).

        Signal extraction is handled by DataCollatorMultimodalVideoAudio at
        collation time, so the batch is returned unchanged here.

        Args:
            batch (Dict[str, List[Any]]): A batch from the HuggingFace dataset.

        Returns:
            The same batch unchanged.
        """
        return batch

    def _obtain_multimodal_input_and_masks(self, batch, **kwargs):
        """
        Not used in multimodal mode — signal processing is handled by the collator.

        Args:
            batch: List of sample dicts.

        Returns:
            Empty dict and unchanged kwargs.
        """
        return {}, kwargs
