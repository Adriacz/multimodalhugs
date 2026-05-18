import os
import torch
import datasets
from pathlib import Path
from typing import Any, Optional, Dict
from dataclasses import dataclass, field

try:
    import av
    _AV_AVAILABLE = True
except ImportError:
    _AV_AVAILABLE = False

try:
    from torchvision.io import read_video
    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False

from datasets import DatasetInfo, SplitGenerator
from datasets import load_dataset

from multimodalhugs.data import (
    MultimodalDataConfig,
    file_exists_filter,
    duration_filter,
    resolve_and_update_config,
    gather_appropriate_data_cfg,
    get_all_dataclass_fields,
    build_merged_omegaconf_config,
)
from multimodalhugs.utils.utils import get_num_proc
from multimodalhugs.utils.registry import register_dataset
from multimodalhugs.processors.utils import SignalUnit


@dataclass
class VideoAudio2TextDataConfig(MultimodalDataConfig):
    """
    Configuration for VideoAudio-to-Text dataset (LSC-Parlament style).

    The TSV must contain both video and audio timestamps for the same .mp4 file:
        signal, signal_start, signal_end, audio_start, audio_end,
        encoder_prompt, decoder_prompt, output

    Filtering is based on the video clip length (in frames).
    """
    name: str = "VideoAudio2TextDataConfig"
    max_frames: Optional[int] = field(
        default=None,
        metadata={"help": "Filter out clips with more video frames than this."}
    )
    min_frames: Optional[int] = field(
        default=None,
        metadata={"help": "Filter out clips with fewer video frames than this."}
    )
    signal_start_end_unit: SignalUnit = field(
        default=SignalUnit.MILLISECONDS,
        metadata={"help": "Unit for signal_start/signal_end. MILLISECONDS or FRAMES."}
    )

    def __init__(self, cfg=None, **kwargs):
        data_cfg = gather_appropriate_data_cfg(cfg)
        valid_config, extra_args, cfg_for_super = build_merged_omegaconf_config(type(self), data_cfg, **kwargs)
        super().__init__(cfg=cfg_for_super, **extra_args)
        self.max_frames = valid_config.get("max_frames", self.max_frames)
        self.min_frames = valid_config.get("min_frames", self.min_frames)
        self.signal_start_end_unit = SignalUnit(
            valid_config.get("signal_start_end_unit", self.signal_start_end_unit)
        )


@register_dataset("videoaudio2text")
class VideoAudio2TextDataset(datasets.GeneratorBasedBuilder):
    """
    Dataset for audio+video-to-text tasks using LSC-Parlament style TSV files.

    Each row references a single .mp4 file containing both a video track
    (the signer) and an audio track (the speaker). Timestamps are in ms:
        signal_start/signal_end — video clip of the signer
        audio_start/audio_end  — audio clip of the speaker

    Because both modalities come from the same file but use different column
    names, two ProcessorSlots can map them independently without collision:
        VideoModalityProcessor: column_map={"video_signal": "signal", ...}
        SpeechModalityProcessor: column_map={"audio_signal": "signal", ...}
    """

    def __init__(
        self,
        config: Optional[VideoAudio2TextDataConfig] = None,
        *args,
        **kwargs,
    ):
        if not _AV_AVAILABLE or not _TORCHVISION_AVAILABLE:
            missing = [p for p, ok in [("av", _AV_AVAILABLE), ("torchvision", _TORCHVISION_AVAILABLE)] if not ok]
            raise ImportError(
                f"VideoAudio2TextDataset requires: {', '.join(missing)}. "
                f'Install with: pip install {" ".join(missing)}'
            )
        config, kwargs = resolve_and_update_config(VideoAudio2TextDataConfig, config, kwargs)
        dataset_info = DatasetInfo(description="Dataset class for VideoAudio2Text.")
        super().__init__(info=dataset_info, *args, **kwargs)

        self.name = "videoaudio2text"
        self.config = config
        self.max_frames = config.max_frames
        self.min_frames = config.min_frames
        self.signal_start_end_unit = config.signal_start_end_unit

    def _info(self):
        features = {
            "video_signal": str,         # path to the .mp4 (video)
            "signal_start": Optional[int],
            "signal_end": Optional[int],
            "audio_signal": str,         # same path, separate key for ProcessorSlot
            "audio_start": Optional[int],
            "audio_end": Optional[int],
            "encoder_prompt": Optional[str],
            "decoder_prompt": Optional[str],
            "output": Optional[str],
        }
        return DatasetInfo(
            description="VideoAudio2TextDataset for multimodal sequence-to-text",
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
        dataset = dataset.filter(lambda ex: file_exists_filter("signal", ex), num_proc=get_num_proc())

        signal_start_end_unit = self.signal_start_end_unit

        def mapping_function(sample: Dict[str, Any]) -> Dict[str, Any]:
            video_path = sample["signal"]
            signal_start = sample.get("signal_start", 0) or 0
            signal_end   = sample.get("signal_end",   0) or 0
            full_signal  = signal_start == 0 and signal_end == 0

            container = av.open(str(video_path))
            if not container.streams.video:
                container.close()
                sample["_invalid"] = True
                sample["DURATION"] = 0
                return sample

            if signal_start_end_unit == SignalUnit.FRAMES and not full_signal:
                container.close()
                sample["DURATION"] = int(signal_end) - int(signal_start)
                sample["_invalid"] = False
                return sample

            if signal_start_end_unit == SignalUnit.MILLISECONDS:
                start_sec = signal_start / 1000.0
                end_sec   = signal_end / 1000.0 if signal_end > 0 else None
            else:
                start_sec = 0.0
                end_sec   = None

            stream = container.streams.video[0]
            start_pts = int(start_sec / float(stream.time_base))
            container.seek(start_pts, any_frame=False, backward=True, stream=stream)

            count = 0
            for frame in container.decode(video=0):
                timestamp = float(frame.pts * stream.time_base)
                if timestamp < start_sec:
                    continue
                if end_sec is not None and timestamp > end_sec:
                    break
                count += 1
            container.close()

            maxf = self.max_frames
            if signal_start_end_unit == SignalUnit.MILLISECONDS and maxf is not None and abs(count - maxf) <= 2:
                frames, _, _ = read_video(
                    str(video_path),
                    start_pts=start_sec,
                    end_pts=end_sec,
                    pts_unit="sec",
                )
                sample["DURATION"] = frames.shape[0]
            else:
                sample["DURATION"] = count

            sample["_invalid"] = False
            return sample

        dataset = dataset.map(mapping_function, num_proc=get_num_proc())
        dataset = dataset.filter(lambda ex: not ex.get("_invalid", False), num_proc=get_num_proc())

        if self.max_frames is not None or self.min_frames is not None:
            dataset = dataset.filter(
                lambda ex: duration_filter(ex, min_frames=self.min_frames, max_frames=self.max_frames),
                num_proc=get_num_proc(),
            )

        for idx, item in enumerate(dataset):
            path = item["signal"]
            yield idx, {
                "video_signal": path,
                "signal_start": item.get("signal_start", 0),
                "signal_end":   item.get("signal_end",   0),
                "audio_signal": path,
                "audio_start":  item.get("audio_start",  0),
                "audio_end":    item.get("audio_end",    0),
                "encoder_prompt": item.get("encoder_prompt", "") or "",
                "decoder_prompt": item.get("decoder_prompt", "") or "",
                "output":         item.get("output", ""),
            }
