import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import AutoProcessor

# Standard Library Imports (av optional — required only for .mp4 containers)
try:
    import av as _av
    _AV_AVAILABLE = True
except ImportError:
    _AV_AVAILABLE = False

from multimodalhugs.data import pad_and_create_mask
from multimodalhugs.processors.modality_processor import ModalityProcessor, ProcessBatchOutput
from multimodalhugs.processors.utils import get_dynamic_cache_size, SignalUnit

logger = logging.getLogger(__name__)


class SpeechModalityProcessor(ModalityProcessor):
    """
    Loads and preprocesses audio sequences.

    process_sample — reads a .wav file, optionally applies a custom audio
                     preprocessor (e.g. WhisperProcessor), returns [n_mels, T].
    process_batch  — pads a list of [n_mels, T_i] tensors to [B, n_mels, T_max]
                     and returns a [B, T_max] attention mask.

    The dataset TSV (ca_filtered) stores clip boundaries as ``audio_start`` /
    ``audio_end`` in milliseconds.  The column_map of the ProcessorSlot (or of
    the ``speech2text`` pipeline preset) renames them to ``signal_start`` /
    ``signal_end`` before calling ``process_sample``, so this processor always
    receives the standard internal names and is schema-agnostic.
    """

    TARGET_SAMPLE_RATE = 16_000  # Hz — resample any audio that differs

    def __init__(
        self,
        custom_preprocessor_path: Optional[str] = None,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        use_cache: bool = False,
        io_max_retries: int = 3,
        signal_start_end_unit: SignalUnit = SignalUnit.MILLISECONDS,
    ):
        """
        Args:
            custom_preprocessor_path: HuggingFace model ID or local path to an
                audio preprocessor (e.g. ``"openai/whisper-medium"``).
                When provided, the waveform is passed through the HF feature
                extractor (e.g. WhisperFeatureExtractor). When None, a
                torchaudio MelSpectrogram is computed from n_mels/n_fft/hop_length.
            n_mels: Number of Mel filterbanks. Only used when
                ``custom_preprocessor_path`` is None. Default: 80.
            n_fft: FFT window size in samples. Only used when
                ``custom_preprocessor_path`` is None. Default: 400
                (= 25 ms at 16 kHz).
            hop_length: Hop size in samples between STFT frames. Only used when
                ``custom_preprocessor_path`` is None. Default: 160
                (= 10 ms at 16 kHz).
            use_cache: If True, wraps ``_load_audio`` with an LRU cache whose
                size is derived from available system (or SLURM) memory,
                assuming ~10 MB per cached audio. Useful when the same audio
                clips are repeated across epochs. Default: False.
            io_max_retries: Maximum number of attempts when opening an audio
                file fails with an I/O error. Retries use exponential backoff
                (1 s, 2 s, 4 s, …) to tolerate transient NFS slowness on
                shared clusters. Default: 3.
            signal_start_end_unit: Unit for ``signal_start`` / ``signal_end``
                values received by ``process_sample`` (after the column_map
                renaming from ``audio_start`` / ``audio_end`` in the TSV).
                Only ``SignalUnit.MILLISECONDS`` is supported; the parameter
                is kept for consistency with ``VideoModalityProcessor`` and to
                facilitate future extensions.
                When ``signal_start=0`` and ``signal_end=0`` the full file is
                always loaded regardless of this setting.
        """
        try:
            signal_start_end_unit = SignalUnit(signal_start_end_unit)
        except ValueError:
            raise ValueError(
                f"Invalid signal_start_end_unit '{signal_start_end_unit}'. "
                f"Must be one of: {[u.value for u in SignalUnit]}."
            )
        self.custom_preprocessor_path = custom_preprocessor_path
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.use_cache = use_cache
        self.io_max_retries = io_max_retries
        self.signal_start_end_unit = signal_start_end_unit
        self.custom_preprocessor = (
            AutoProcessor.from_pretrained(custom_preprocessor_path)
            if custom_preprocessor_path is not None
            else None
        )
        self._mel_transform = T.MelSpectrogram(
            sample_rate=self.TARGET_SAMPLE_RATE,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        if use_cache:
            cache_size = get_dynamic_cache_size(avg_item_size_bytes=10e6)  # ~10 MB per audio
            logger.info(f"Audio cache size: {cache_size}")
            self._load_audio = lru_cache(maxsize=cache_size)(self._load_audio)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_waveform_from_mp4(
        self,
        audio_path: Union[str, Path],
        signal_start: float = 0.0,
        signal_end: float = 0.0,
    ) -> torch.Tensor:
        """
        **Load an audio slice from an .mp4 container using PyAV.**

        Uses PyAV for accurate seek and decode of compressed audio (AAC) inside
        video containers, avoiding the timestamp drift that torchaudio produces
        when seeking compressed streams.

        **Args:**
        - `audio_path` (Union[str, Path]): Path to the .mp4 (or other container) file.
        - `signal_start` (float): Clip start in milliseconds. 0.0 = file start.
        - `signal_end` (float): Clip end in milliseconds. 0.0 = file end.

        **Returns:**
        - `torch.Tensor`: Waveform `[1, T]` (mono, float32, TARGET_SAMPLE_RATE).
        """
        if not _AV_AVAILABLE:
            raise ImportError(
                "PyAV is required to load audio from .mp4 files. "
                "Install it with: pip install av"
            )

        start_sec = (signal_start or 0.0) / 1000.0
        end_sec   = (signal_end / 1000.0) if signal_end else None

        container = _av.open(str(audio_path))
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if audio_stream is None:
            container.close()
            raise ValueError(f"No audio stream found in '{audio_path}'")

        native_sr = audio_stream.codec_context.sample_rate

        # Seek to slightly before the target start to avoid decoder artifacts.
        if start_sec > 0:
            seek_pts = max(0, int((start_sec - 0.1) * _av.time_base))
            container.seek(seek_pts)

        chunks = []
        chunk_start_sec = None  # PTS of the first included frame
        for packet in container.demux(audio_stream):
            for frame in packet.decode():
                t = float(frame.pts * audio_stream.time_base)
                if t + frame.samples / native_sr < start_sec:
                    continue
                if end_sec is not None and t > end_sec:
                    break
                # frame.to_ndarray(): [channels, samples] or [samples] depending on layout
                arr = frame.to_ndarray()
                if arr.ndim == 1:
                    arr = arr[None, :]  # [1, samples]
                chunks.append(torch.from_numpy(arr.copy()).float())
                if chunk_start_sec is None:
                    chunk_start_sec = t

        container.close()

        if not chunks:
            raise ValueError(
                f"No audio frames decoded from '{audio_path}' "
                f"in [{signal_start}, {signal_end}] ms."
            )

        waveform = torch.cat(chunks, dim=-1)  # [C, T]

        # Trim relative to the first decoded frame's PTS.
        # Pre-cut clips preserve original session timestamps, so start_sec is not 0.
        _chunk_start = chunk_start_sec if chunk_start_sec is not None else start_sec
        rel_start = max(0, int((start_sec - _chunk_start) * native_sr))
        rel_end = (
            int((end_sec - _chunk_start) * native_sr) if end_sec is not None else waveform.shape[-1]
        )
        rel_end = min(rel_end, waveform.shape[-1])
        waveform = waveform[:, rel_start:rel_end]

        if waveform.shape[-1] == 0:
            raise ValueError(
                f"Audio segment has zero samples after trimming '{audio_path}' "
                f"to [{signal_start}, {signal_end}] ms "
                f"(chunk_start={_chunk_start:.4f}s, rel_start={rel_start}, rel_end={rel_end})."
            )

        # Mix down to mono: [C, T] → [1, T]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to TARGET_SAMPLE_RATE if necessary.
        if native_sr != self.TARGET_SAMPLE_RATE:
            resampler = T.Resample(orig_freq=native_sr, new_freq=self.TARGET_SAMPLE_RATE)
            waveform = resampler(waveform)

        return waveform.to(torch.float32)  # [1, T]

    def _load_waveform(
        self,
        audio_path: Union[str, Path],
        signal_start: float = 0.0,
        signal_end: float = 0.0,
    ) -> torch.Tensor:
        """
        **Load a slice of an audio file as a waveform tensor.**

        Dispatches to ``_load_waveform_from_mp4`` for .mp4/.mkv containers (PyAV)
        and falls back to torchaudio for .wav and other lossless formats.

        **Args:**
        - `audio_path` (Union[str, Path]): Path to the audio/video file.
        - `signal_start` (float): Clip start in milliseconds. 0.0 = start of file.
        - `signal_end` (float): Clip end in milliseconds. 0.0 = end of file.

        **Returns:**
        - `torch.Tensor`: Waveform `[1, T]` (mono, float32, 16 kHz).
        """
        ext = Path(str(audio_path)).suffix.lower()
        _CONTAINER_FORMATS = {".mp4", ".mkv", ".mov", ".avi", ".mp3", ".m4a", ".aac"}
        if ext in _CONTAINER_FORMATS:
            return self._load_waveform_from_mp4(audio_path, signal_start, signal_end)

        # --- torchaudio path (wav, flac, ogg, …) ---
        info = torchaudio.info(str(audio_path))
        native_sr = info.sample_rate

        start_sec = (signal_start or 0.0) / 1000.0
        end_sec   = (signal_end / 1000.0) if signal_end else None

        frame_offset = int(start_sec * native_sr)
        num_frames   = (
            int((end_sec - start_sec) * native_sr)
            if end_sec is not None
            else -1  # -1 = read until end of file
        )

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

    def _load_audio(
        self,
        audio_path: Union[str, Path],
        signal_start: float = 0.0,
        signal_end: float = 0.0,
    ) -> torch.Tensor:
        """
        Load an audio clip from disk and apply the preprocessing pipeline.

        When ``custom_preprocessor`` is set, the waveform is passed through
        the HF audio preprocessor (e.g. WhisperFeatureExtractor).
        Otherwise, torchaudio's MelSpectrogram is computed directly.
        Retries up to ``io_max_retries`` times with exponential backoff on
        transient I/O failures.

        Args:
            audio_path: Path to a .wav audio file.
            signal_start: Clip start in milliseconds (mapped from ``audio_start``
                in the TSV via the ProcessorSlot column_map). 0.0 means start of file.
            signal_end: Clip end in milliseconds (mapped from ``audio_end``
                in the TSV via the ProcessorSlot column_map). 0.0 means end of file.

        Returns:
            Float tensor of shape [n_mels, T] where T is the number of
            mel-spectrogram frames.

        Raises:
            IOError: If the audio cannot be opened after ``io_max_retries``
                attempts.
        """
        last_exc: Exception = IOError(f"Cannot open audio {audio_path}")
        for attempt in range(max(1, self.io_max_retries)):
            try:
                waveform = self._load_waveform(audio_path, signal_start, signal_end)  # [1, T]

                if self.custom_preprocessor is not None:
                    # HF AutoProcessor (e.g. WhisperFeatureExtractor) expects a
                    # 1-D numpy array at TARGET_SAMPLE_RATE.
                    features = self.custom_preprocessor(
                        waveform.squeeze(0).numpy(),
                        sampling_rate=self.TARGET_SAMPLE_RATE,
                        return_tensors="pt",
                    )
                    # Most HF audio processors return `input_features` [1, n_mels, T]
                    result = features["input_features"].squeeze(0)  # → [n_mels, T]
                else:
                    result = self._mel_transform(waveform).squeeze(0)  # → [n_mels, T]

                return result

            except (IOError, OSError, RuntimeError) as exc:
                last_exc = exc
                if attempt < self.io_max_retries - 1:
                    delay = 2 ** attempt
                    logger.warning(
                        "Failed to load audio '%s' (attempt %d/%d): %s. Retrying in %ds.",
                        audio_path, attempt + 1, self.io_max_retries, exc, delay,
                    )
                    time.sleep(delay)

        raise IOError(
            f"Cannot open audio '{audio_path}' after {self.io_max_retries} attempts."
        ) from last_exc

    # ------------------------------------------------------------------
    # ModalityProcessor interface
    # ------------------------------------------------------------------

    def process_sample(
        self,
        values: Union[Any, Dict[str, Any]],
        **kwargs,
    ) -> torch.Tensor:
        """
        Load and preprocess a single audio sample. Called at dataset-transform time.

        Args:
            values: One of:
                - str or Path — path to a .wav file; loaded and preprocessed.
                - torch.Tensor — returned unchanged (already preprocessed).
                - np.ndarray — waveform array [T] or [C, T]; mel-spectrogram
                  is computed directly without loading from disk.
                - dict — mapping with keys:
                    ``"signal"`` (str/Path, required): path to the .wav file.
                    ``"signal_start"`` (float, optional): clip start in ms,
                    mapped from ``audio_start`` in the TSV. Default 0.0.
                    ``"signal_end"`` (float, optional): clip end in ms,
                    mapped from ``audio_end`` in the TSV. Default 0.0.

        Returns:
            Float tensor of shape [n_mels, T].
        """
        if isinstance(values, dict):
            signal = values["signal"]
            signal_start = values.get("signal_start", 0.0)
            signal_end = values.get("signal_end", 0.0)
        else:
            signal = values
            signal_start = kwargs.get("signal_start", 0.0)
            signal_end = kwargs.get("signal_end", 0.0)

        if isinstance(signal, torch.Tensor):
            return signal
        if isinstance(signal, np.ndarray):
            waveform = torch.from_numpy(signal).to(torch.float32)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)  # [T] → [1, T]
            if self.custom_preprocessor is not None:
                features = self.custom_preprocessor(
                    waveform.squeeze(0).numpy(),
                    sampling_rate=self.TARGET_SAMPLE_RATE,
                    return_tensors="pt",
                )
                return features["input_features"].squeeze(0)
            return self._mel_transform(waveform).squeeze(0)

        return self._load_audio(signal, signal_start, signal_end)

    def process_batch(
        self,
        samples: List[torch.Tensor],
        **kwargs,
    ) -> ProcessBatchOutput:
        """
        Pad a batch of audio tensors to a common length. Called at collation time.

        Args:
            samples: List of B tensors, each of shape [n_mels, T_i], as returned
                by ``process_sample``.

        Returns:
            ProcessBatchOutput where:
                - data: Float tensor of shape [B, n_mels, T_max], zero-padded.
                - mask: Bool tensor of shape [B, T_max], True for valid frames.
        """
        padded, mask = pad_and_create_mask(samples)
        return ProcessBatchOutput(data=padded, mask=mask)
