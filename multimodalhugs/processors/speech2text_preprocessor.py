# Standard library
import logging
import os

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Union

# Third-party libraries
import psutil
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from transformers import AutoProcessor
from transformers.feature_extraction_utils import BatchFeature

# Local application imports
from multimodalhugs.data import pad_and_create_mask
from multimodalhugs.processors import MultimodalSequence2SequenceProcessor


logger = logging.getLogger(__name__)


def get_dynamic_cache_size():
    """
    Computes a dynamic LRU cache size based on available RAM.
    Tries to read cluster memory from SLURM env vars first,
    falls back to system RAM. Allocates 5% of total memory,
    assuming ~10MB per cached audio tensor.
    """
    cluster_mem = None
    for var in ("SLURM_MEM_PER_GPU", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU"):
        if os.getenv(var):
            cluster_mem = int(os.getenv(var)) * 1e6
            break
    total = cluster_mem or psutil.virtual_memory().total
    size = int((total * 0.05) / 10e6)  # 5% of RAM, ~10MB per audio (vs ~50MB for video)  PREGUNTAR SI ESTÀ BÉ
    return max(10, size)

class Speech2TextTranslationProcessor(MultimodalSequence2SequenceProcessor):
    """
    Reads a .wav audio file (path / ndarray / tensor) and converts it to a
    Mel-spectrogram of shape [n_mels, T], then pads & masks across the batch.

    Two modes:
    - With `custom_preprocessor_path`: delegates feature extraction to a
      HuggingFace AutoProcessor (e.g. WhisperProcessor). Output shape
      depends on the processor (typically [n_mels, T] or [1, n_mels, T]).
    - Without: uses `torchaudio.transforms.MelSpectrogram` with configurable
      parameters. Output shape: [n_mels, T].

    In both cases the final tensor fed to the model has shape [n_mels, T]
    and the batch is returned as [B, n_mels, T] with its attention mask [B, T].

    """
    name = "speech2text_processor"
    attributes = ["tokenizer"]
    model_input_names = ["inputs_embeds", "attention_mask"]
    tokenizer_class = "AutoTokenizer"

    TARGET_SAMPLE_RATE = 16_000  # Hz — resample any audio that differs

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        custom_preprocessor_path: Optional[str] = None,
        # MelSpectrogram parameters (only used when custom_preprocessor_path is None)
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        use_cache: bool = False,
        **kwargs,
    ):
        """
        Initializes the Speech2TextTranslationProcessor.

        Args:
            tokenizer: HuggingFace tokenizer for processing the text output.
            custom_preprocessor_path: Optional path/name of a HuggingFace
                AutoProcessor (e.g. "openai/whisper-small") used for feature
                extraction. When set, n_mels/n_fft/hop_length are ignored.
            n_mels (int): Number of Mel filterbanks. Default: 80.
            n_fft (int): FFT window size in samples. Default: 400
                (= 25 ms at 16 kHz).
            hop_length (int): Hop size in samples between STFT frames.
                Default: 160 (= 10 ms at 16 kHz).
            use_cache (bool): If True, wraps `_audio_to_tensor` with an
                LRU cache so repeated accesses to the same file are free.
            **kwargs: Forwarded to `MultimodalSequence2SequenceProcessor`.
        """
        super().__init__(tokenizer=tokenizer, **kwargs)

        self.custom_preprocessor_path = custom_preprocessor_path
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.use_cache = use_cache
        
        # Build the Hugging Face preprocessor  
        self.custom_preprocessor = AutoProcessor.from_pretrained(self.custom_preprocessor_path) if self.custom_preprocessor_path is not None else None

        # Build the torchaudio MelSpectrogram transform (used when no custom preprocessor)
        # This transform expects a waveform of shape [1, T] and returns [n_mels, T_frames].
        self._mel_transform = T.MelSpectrogram(
            sample_rate=self.TARGET_SAMPLE_RATE,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        if self.use_cache:
            cache_size = get_dynamic_cache_size()
            logger.info(f" Audio cache size: {cache_size}")
            self._audio_to_tensor = lru_cache(maxsize=cache_size)(self._audio_to_tensor)

      
    def _load_waveform(self, audio_path: Union[str, Path], start_sec: float, end_sec: Optional[float]) -> torch.Tensor:
        """
        Loads a slice of a .wav file as a waveform tensor [1, T] at
        TARGET_SAMPLE_RATE. Resamples if the file's native rate differs.

        Args:
            audio_path: Path to the .wav file.
            start_sec: Start of the clip in seconds.
            end_sec: End of the clip in seconds, or None to read until EOF.

        Returns:
            Waveform tensor of shape [1, T] (mono, float32, 16 kHz).
        """

        info = torchaudio.info(str(audio_path))
        native_sr = info.sample_rate

        # Convert seconds → sample indices for torchaudio.load()
        frame_offset = int(start_sec * native_sr)
        if end_sec is not None:
            num_frames = int((end_sec - start_sec) * native_sr)
        else:
            num_frames = -1  # -1 = read until end of file

        waveform, sr = torchaudio.load(
            str(audio_path),
            frame_offset=frame_offset,
            num_frames=num_frames,
        )

        # Mix down to mono if needed: [C, T] → [1, T]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to TARGET_SAMPLE_RATE if necessary
        if sr != self.TARGET_SAMPLE_RATE:
            resampler = T.Resample(orig_freq=sr, new_freq=self.TARGET_SAMPLE_RATE)
            waveform = resampler(waveform)

        return waveform.to(torch.float32)  # [1, T]


    def _audio_to_tensor(self, audio_input: Union[str, Path, np.ndarray, torch.Tensor], signal_start: float = 0.0, signal_end: float = 0.0) -> torch.Tensor:
        """
        Converts an audio input to a Mel-spectrogram tensor [n_mels, T].

        Args:
            audio_input: Path to a .wav file, a numpy array [T] or [C, T],
                or a pre-computed torch.Tensor.
            signal_start: Start of the clip in milliseconds.
            signal_end: End of the clip in milliseconds (0 = until EOF).

        Returns:
            Mel-spectrogram tensor of shape [n_mels, T_frames].
        """
        # Already a tensor 
        if isinstance(audio_input, torch.Tensor):
            return audio_input  # assume already [n_mels, T] 

        # numpy array -> tensor waveform [1, T]
        if isinstance(audio_input, np.ndarray):
            waveform = torch.from_numpy(audio_input)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)  # [T] → [1, T]
            waveform = waveform.to(torch.float32)
            mel = self._mel_transform(waveform).squeeze(0)  # [n_mels, T_frames]
            return mel

        # File path: parse timestamps and load 
        start_sec = (signal_start or 0.0) / 1000.0
        end_sec   = (signal_end / 1000.0) if signal_end else None

        # Branch A: delegate entirely to a HuggingFace AutoProcessor
        if self.custom_preprocessor is not None:
            waveform = self._load_waveform(audio_input, start_sec, end_sec)
            # AutoProcessor expects a 1-D numpy array at TARGET_SAMPLE_RATE
            features = self.custom_preprocessor(
                waveform.squeeze(0).numpy(),
                sampling_rate=self.TARGET_SAMPLE_RATE,
                return_tensors="pt",
            )
            # Most HF audio processors return `input_features` [1, n_mels, T]
            mel = features["input_features"].squeeze(0)  # -> [n_mels, T]
            return mel  # [n_mels, T]

        # Branch B: torchaudio MelSpectrogram
        waveform = self._load_waveform(audio_input, start_sec, end_sec)  # [1, T]
        mel = self._mel_transform(waveform).squeeze(0)                   # [n_mels, T_frames]
        return mel  # [n_mels, T_frames]


    # MultimodalSeq2SeqProcessor interface

    def _obtain_multimodal_input_and_masks(self, batch: BatchFeature, **kwargs):
        """
        Converts a batch of samples to padded Mel-spectrograms + attention masks.

        Each sample produces a [n_mels, T_i] tensor (variable T_i). After
        padding, the batch has shape [B, n_mels, T_max] and the mask [B, T_max].

        Returns:
            {
                "input_audio":    Tensor [B, n_mels, T_max],
                "attention_mask": Tensor [B, T_max],  (1 = real, 0 = pad)
            }
        """
        sequences = [
            self._audio_to_tensor(
                sample["signal"],
                sample.get("signal_start", 0.0),
                sample.get("signal_end",   0.0),
            )
            for sample in batch
        ]
        # pad_and_create_mask works on the last dim (time) of each tensor,
        # so [n_mels, T_i] tensors are padded to [n_mels, T_max].
        padded, masks = pad_and_create_mask(sequences)
        return {
            "input_audio":    padded,  # [B, n_mels, T_max]
            "attention_mask": masks,   # [B, T_max]
        }, kwargs

    def _transform_get_items_output(self, batch):
        """
        Dataset-level transform applied during iteration (via `with_transform`).

        Runs `_audio_to_tensor` on each sample in the batch before
        collation, allowing the DataLoader to parallelise the (potentially
        expensive) mel computation across workers.

        Args:
            batch: Dict[str, List[Any]] — a batch from the HuggingFace dataset.

        Returns:
            The same dict with `signal` replaced by a list of
            [n_mels, T_frames] tensors.
        """
        tensor_signals = [
            self._audio_to_tensor(
                batch["signal"][i],
                batch["signal_start"][i],
                batch["signal_end"][i],
            )
            for i in range(len(batch["signal"]))
        ]
        batch["signal"] = tensor_signals
        return batch
