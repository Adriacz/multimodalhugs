import torch
import datasets
import torchaudio
from pathlib import Path
from typing import Any, Optional, Dict
from dataclasses import dataclass, field
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


@dataclass
class MultimodalDatasetConfig(MultimodalDataConfig):
    """
    Configuration for the Multimodal (video + audio) dataset.

    Each sample is a single .mp4 file that contains both the video stream
    (sign language interpreter) and the audio stream (speaker).
    Temporal boundaries for each modality are stored separately:
        - signal_start / signal_end : signing interval (ms)
        - audio_start  / audio_end  : speech interval  (ms)

    Args:
        name (str): Identifier for this config class.
        max_frames (Optional[int]): Filter out videos with more frames than this.
        min_frames (Optional[int]): Filter out videos with fewer frames than this.
        max_audio_duration (Optional[float]): Filter out clips whose audio is
            longer than this many seconds.
        min_audio_duration (Optional[float]): Filter out clips whose audio is
            shorter than this many seconds.
    """
    name: str = "MultimodalDatasetConfig"
    max_frames: Optional[int] = field(
        default=None,
        metadata={"help": "Filter out videos longer than this (in frames)."}
    )
    min_frames: Optional[int] = field(
        default=None,
        metadata={"help": "Filter out videos shorter than this (in frames)."}
    )
    max_audio_duration: Optional[float] = field(
        default=None,
        metadata={"help": "Filter out audio clips longer than this (in seconds)."}
    )
    min_audio_duration: Optional[float] = field(
        default=None,
        metadata={"help": "Filter out audio clips shorter than this (in seconds)."}
    )

    def __init__(self, cfg=None, **kwargs):
        data_cfg = gather_appropriate_data_cfg(cfg)
        valid_config, extra_args, cfg_for_super = build_merged_omegaconf_config(type(self), data_cfg, **kwargs)
        super().__init__(cfg=cfg_for_super, **extra_args)
        # pull from OmegaConf yaml (or leave defaults)
        self.max_frames = valid_config.get("max_frames", self.max_frames)
        self.min_frames = valid_config.get("min_frames", self.min_frames)
        self.max_audio_duration = valid_config.get("max_audio_duration", self.max_audio_duration)
        self.min_audio_duration = valid_config.get("min_audio_duration", self.min_audio_duration)


@register_dataset("multimodal")
class MultimodalDataset(datasets.GeneratorBasedBuilder):
    """
    Multimodal dataset for joint Sign Language Translation + ASR.

    Reads a single TSV with columns:
        signal, signal_start, signal_end,
        audio_start, audio_end,
        encoder_prompt, decoder_prompt, output

    Each .mp4 file contains both the video stream (signing) and the
    audio stream (speech). The two modalities are extracted independently
    using their respective timestamp columns.
    """

    def __init__(
        self,
        config: Optional[MultimodalDatasetConfig] = None,
        *args,
        **kwargs
    ):
        config, kwargs = resolve_and_update_config(MultimodalDatasetConfig, config, kwargs)
        dataset_info = DatasetInfo(description="Multimodal dataset for video+audio to text.")
        super().__init__(info=dataset_info, *args, **kwargs)

        self.name = "multimodal"
        self.config = config
        self.max_frames = config.max_frames
        self.min_frames = config.min_frames
        self.max_audio_duration = config.max_audio_duration
        self.min_audio_duration = config.min_audio_duration

    def _info(self):
        features = {
            "signal":          str,
            "signal_start":    Optional[float],
            "signal_end":      Optional[float],
            "audio_start":     Optional[float],
            "audio_end":       Optional[float],
            "encoder_prompt":  Optional[str],
            "decoder_prompt":  Optional[str],
            "output":          Optional[str],
        }
        return DatasetInfo(
            description="MultimodalDataset for joint video+audio sequence-to-text",
            features=datasets.Features(features),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager) -> list:
        splits = []
        if self.config.train_metadata_file:
            splits.append(SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"metafile_path": self.config.train_metadata_file, "split": "train"},
            ))
        if self.config.validation_metadata_file:
            splits.append(SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"metafile_path": self.config.validation_metadata_file, "split": "validation"},
            ))
        if self.config.test_metadata_file:
            splits.append(SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"metafile_path": self.config.test_metadata_file, "split": "test"},
            ))
        return splits

    def _generate_examples(self, metafile_path: str, split: str):
        dataset = load_dataset(
            "csv",
            data_files=[str(metafile_path)],
            split="train",
            delimiter="\t",
            num_proc=get_num_proc(),
        )

        # Filter samples whose .mp4 file does not exist
        dataset = dataset.filter(
            lambda ex: file_exists_filter("signal", ex),
            num_proc=get_num_proc()
        )

        def mapping_function(sample: Dict[str, Any]) -> Dict[str, Any]:
            """
            Computes AUDIO_DURATION (seconds) and VIDEO_FRAMES (frame count)
            for filtering. Does not load the actual pixel/waveform data.
            """
            signal_path = sample["signal"]

            # --- Audio duration (from the .mp4 audio stream via torchaudio) ---
            a_start_ms = sample.get("audio_start", 0) or 0
            a_end_ms   = sample.get("audio_end",   0) or 0
            a_start_sec = a_start_ms / 1000.0
            a_end_sec   = a_end_ms   / 1000.0 if a_end_ms > 0 else None

            try:
                info = torchaudio.info(str(signal_path))
                sr = info.sample_rate
                total_sec = info.num_frames / sr
                clip_end = min(a_end_sec, total_sec) if a_end_sec is not None else total_sec
                audio_duration = clip_end - a_start_sec
                if audio_duration <= 0:
                    sample["_invalid"] = True
                    sample["AUDIO_DURATION"] = 0.0
                    sample["VIDEO_FRAMES"] = 0
                    return sample
            except Exception:
                sample["_invalid"] = True
                sample["AUDIO_DURATION"] = 0.0
                sample["VIDEO_FRAMES"] = 0
                return sample

            # --- Video frame count (from the .mp4 video stream via av) ---
            import av
            v_start_ms = sample.get("signal_start", 0) or 0
            v_end_ms   = sample.get("signal_end",   0) or 0
            v_start_sec = v_start_ms / 1000.0
            v_end_sec   = v_end_ms   / 1000.0 if v_end_ms > 0 else None

            try:
                container = av.open(str(signal_path))
                if not container.streams.video:
                    sample["_invalid"] = True
                    sample["AUDIO_DURATION"] = 0.0
                    sample["VIDEO_FRAMES"] = 0
                    return sample

                stream = container.streams.video[0]
                start_pts = int(v_start_sec / float(stream.time_base))
                container.seek(start_pts, any_frame=False, backward=True, stream=stream)

                frame_count = 0
                for frame in container.decode(video=0):
                    ts = float(frame.pts * stream.time_base)
                    if ts < v_start_sec:
                        continue
                    if v_end_sec is not None and ts > v_end_sec:
                        break
                    frame_count += 1
                container.close()
            except Exception:
                sample["_invalid"] = True
                sample["AUDIO_DURATION"] = 0.0
                sample["VIDEO_FRAMES"] = 0
                return sample

            sample["AUDIO_DURATION"] = audio_duration
            sample["VIDEO_FRAMES"]   = frame_count
            sample["_invalid"]       = False
            return sample

        dataset = dataset.map(mapping_function, num_proc=get_num_proc())
        dataset = dataset.filter(
            lambda ex: not ex.get("_invalid", False),
            num_proc=get_num_proc()
        )

        # Filter by video frame count
        if self.max_frames is not None or self.min_frames is not None:
            dataset = dataset.filter(
                lambda ex: duration_filter(
                    ex,
                    min_frames=self.min_frames,
                    max_frames=self.max_frames,
                    duration_key="VIDEO_FRAMES",
                ),
                num_proc=get_num_proc(),
            )

        # Filter by audio duration
        if self.max_audio_duration is not None or self.min_audio_duration is not None:
            dataset = dataset.filter(
                lambda ex: duration_filter(
                    ex,
                    min_frames=self.min_audio_duration,
                    max_frames=self.max_audio_duration,
                    duration_key="AUDIO_DURATION",
                ),
                num_proc=get_num_proc(),
            )

        for idx, item in enumerate(dataset):
            yield idx, {
                "signal":         item["signal"],
                "signal_start":   item.get("signal_start", 0),
                "signal_end":     item.get("signal_end",   0),
                "audio_start":    item.get("audio_start",  0),
                "audio_end":      item.get("audio_end",    0),
                "encoder_prompt": item.get("encoder_prompt", "") or "",
                "decoder_prompt": item.get("decoder_prompt", "") or "",
                "output":         item.get("output", ""),
            }
