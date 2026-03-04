# Speech2text script 

import os
import torch
import datasets
from pathlib import Path
from omegaconf import ListConfig
from typing import Any, Union, Optional, Dict, Tuple
from dataclasses import dataclass, field
import torchaudio

# from torchvision.io import read_video
from datasets import DatasetInfo, SplitGenerator
from datasets import load_dataset

from multimodalhugs.data import (
    MultimodalDataConfig,
    file_exists_filter,
    duration_filter,
    resolve_and_update_config,
    gather_appropriate_data_cfg,
    get_all_dataclass_fields, 
    build_merged_omegaconf_config
)

from multimodalhugs.utils.utils import get_num_proc
from multimodalhugs.utils.registry import register_dataset


# ---- Passar de mp4 a WAV ----
#from pydub import AudioSegment

#audio = AudioSegment.from_file("video.mp4", format="mp4") fer for per totes els videos de la carpeta

# Convertir a los parámetros ideales: 16kHz, Mono, WAV
#audio = audio.set_frame_rate(16000).set_channels(1)

# Exportar
# audio.export("audio_listo.wav", format="wav")
# -------------------------

@dataclass
class Speech2TextDataConfig(MultimodalDataConfig):
    """
    Configuration for Speech-to-Text dataset.

    Args:
        name (str): Identifier for this config class.
        max_duration (Optional[float]): Filter out audios longer than this many seconds.
        min_duration (Optional[float]): Filter out audios shorter than this many seconds.
    """
    name: str = "Speech2TextDataConfig"
    max_duration: Optional[int] = field(
        default=None,
        metadata={"help": "Filter out audios longer than this (in seconds)."}
    )
    min_duration: Optional[int] = field(
        default=None, 
        metadata={"help": "Filter out audios shorter than this value (in seconds)"}
    )
    def __init__(self, cfg=None, **kwargs):
        data_cfg = gather_appropriate_data_cfg(cfg)
        valid_config, extra_args, cfg_for_super = build_merged_omegaconf_config(type(self), data_cfg, **kwargs)
        super().__init__(cfg=cfg_for_super, **extra_args)
        # pull from OmegaConf yaml (or leave defaults)
        self.max_duration = valid_config.get("max_duration", self.max_duration)
        self.min_duration = valid_config.get("min_duration", self.min_duration)

@register_dataset("Speech2text")
class Speech2TextDataset(datasets.GeneratorBasedBuilder):
    """
    **Speech2TextDataset: A dataset class for Speech-to-Text tasks.**

    Reads .wav audio files referenced in a TSV metadata file and filters
    them by duration (in seconds). The TSV is expected to have the same
    structure as Video2TextDataset:
        signal, signal_start, signal_end, encoder_prompt, decoder_prompt, output

    Where signal_start and signal_end are in milliseconds.
    """
    def __init__(
        self, 
        config: Optional[Speech2TextDataConfig] = None,
        *args, 
        **kwargs
    ):
        """
        Initialize the Speech2TextDataset.

        You can pass either:
        - a config object (`Speech2TextDataConfig`), or
        - keyword arguments that match its fields.

        If both are provided, keyword arguments take priority.
        """
        config, kwargs = resolve_and_update_config(Speech2TextDataConfig, config, kwargs)
        dataset_info = DatasetInfo(description="Dataset class for Speech2Text.")
        super().__init__(info=dataset_info, *args, **kwargs)

        self.name = "Speech2text"
        self.config = config
        self.max_frames = config.max_duration
        self.max_duration = config.max_duration
        self.min_duration = config.min_duration
        self.min_frames = config.min_duration

    def _info(self):
        features = {
            "signal": str, # path to the wav
            "signal_start": Optional[int],
            "signal_end": Optional[int],
            "encoder_prompt": Optional[str],
            "decoder_prompt": Optional[str],
            "output": Optional[str], # transcroption
        }
        return DatasetInfo(
            description="Speech2TextDataset for multimodal sequence-to-text",
            features=datasets.Features(features),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager) -> list:
        splits = []
        if self.config.train_metadata_file:
            splits.append(
                SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "metafile_path": self.config.train_metadata_file,
                        "split": "train",
                    },
                )
            )
        if self.config.validation_metadata_file:
            splits.append(
                SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "metafile_path": self.config.validation_metadata_file,
                        "split": "validation",
                    },
                )
            )
        if self.config.test_metadata_file:
            splits.append(
                SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "metafile_path": self.config.test_metadata_file,
                        "split": "test",
                    },
                )
            )
        return splits

    def _generate_examples(self, metafile_path: str, split: str):
        # Load the metadata CSV/TSV
        dataset = load_dataset(
            "csv",
            data_files=[str(metafile_path)],
            split="train",
            delimiter="\t",
            num_proc=get_num_proc(),
        )

        # Filter missing files
        dataset = dataset.filter(lambda ex: file_exists_filter("signal", ex), num_proc=get_num_proc())

        def mapping_function(sample: Dict[str, Any]) -> Dict[str, Any]:
            audio_path = sample["signal"]

            # Convert millisecond timestamps to seconds
            start_ms = sample.get("signal_start", 0) or 0
            end_ms   = sample.get("signal_end",   0) or 0
            start_sec = start_ms / 1000.0
            end_sec   = end_ms   / 1000.0 if end_ms > 0 else None

           
            # Primary method: torchaudio 
            try: 
                info = torchaudio.info(str(audio_path))
            except Exception:
                sample["_invalid"] = True
                sample["DURATION"] = 0.0
                return sample

            sample_rate = info.sample_rate 
            total_frames = info.num_frames  # total samples in file  
            total_duration_sec = total_frames / sample_rate # total duration in seconds 

            # compute clip duration timestamps 
            clip_start = start_sec
            clip_end   = min(end_sec, total_duration_sec) if end_sec is not None else total_duration_sec
            clip_duration = clip_end - clip_start

            if clip_duration <= 0:
                sample["_invalid"] = True
                sample["DURATION"] = 0.0
                return sample

            # If the computed duration is very close to max_duration threshold (±0.05s),
            # fall back to soundfile for a more precise measurement by actually
            # reading the audio slice and counting its samples.
            max_d = self.max_duration
            if max_d is not None and abs(clip_duration - max_d) <= 0.05:
                # --- Fallback: soundfile ---
                # soundfile reads actual audio samples, giving an exact frame count.
                try:
                    with sf.SoundFile(str(audio_path)) as f:
                        f.seek(int(clip_start * f.samplerate))
                        frames_to_read = (
                            int((clip_end - clip_start) * f.samplerate)
                            if end_sec is not None
                            else -1  # -1 means read until end of file
                        )
                        audio_data = f.read(frames=frames_to_read)
                    clip_duration = len(audio_data) / f.samplerate
                except Exception:
                    sample["_invalid"] = True
                    sample["DURATION"] = 0.0
                    return sample

            sample["DURATION"] = clip_duration
            sample["_invalid"] = False
            return sample


        # Map to extract duration
        dataset = dataset.map(mapping_function, num_proc=get_num_proc())
        dataset = dataset.filter(lambda ex: not ex.get("_invalid", False), num_proc=get_num_proc())

        # Filter by max_frames if set
        if self.max_frames is not None or self.min_frames:
            dataset = dataset.filter(
                lambda ex: duration_filter(
                    ex,
                    min_frames=self.min_frames,
                    max_frames=self.max_frames, 
                ),
                num_proc=get_num_proc(),
            )

        # Yield examples
        for idx, item in enumerate(dataset):
            yield idx, {
                "signal": item["signal"],
                "signal_start": item.get("signal_start", 0),
                "signal_end": item.get("signal_end", 0),
                "encoder_prompt": item.get("encoder_prompt", "") or "",
                "decoder_prompt": item.get("decoder_prompt", "") or "",
                "output": item.get("output", ""),
            }