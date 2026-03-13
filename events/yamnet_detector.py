"""Audio event detection using pretrained YAMNet."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import librosa
import numpy as np

logger = logging.getLogger(__name__)

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
TARGET_EVENT_FRAGMENTS: dict[str, tuple[str, ...]] = {
    "crying": ("crying", "sobbing", "whimper", "wail"),
    "shouting": ("shout", "yell", "scream"),
    "fall": ("fall", "thud", "thump", "crash", "bang"),
    "heavy_breathing": ("breathing", "wheeze", "pant", "gasp", "snort"),
    "glass_breaking": ("glass", "shatter", "breaking"),
}


class YAMNetDetector:
    """Run YAMNet and aggregate target event probabilities."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._class_names: list[str] | None = None

    def _load_model(self) -> None:
        if self._model is not None and self._class_names is not None:
            return

        try:
            import tensorflow as tf
            import tensorflow_hub as hub
        except ImportError as exc:
            raise RuntimeError(
                "tensorflow and tensorflow-hub are required for YAMNet event detection."
            ) from exc

        logger.info("Loading YAMNet model from %s", YAMNET_URL)
        self._model = hub.load(YAMNET_URL)
        class_map_path = self._model.class_map_path().numpy()

        class_names: list[str] = []
        with tf.io.gfile.GFile(class_map_path) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                class_names.append(row["display_name"])

        self._class_names = class_names

    @staticmethod
    def _prepare_waveform(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)
        if sample_rate != 16000:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
        return waveform.astype(np.float32)

    def detect_from_waveform(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> dict[str, float]:
        """Return target audio event probabilities without thresholding."""
        self._load_model()
        assert self._model is not None
        assert self._class_names is not None

        prepared_waveform = self._prepare_waveform(waveform, sample_rate)
        scores, _, _ = self._model(prepared_waveform)
        frame_scores = scores.numpy()
        lower_names = [name.lower() for name in self._class_names]

        aggregated: dict[str, float] = {}
        for event_name, fragments in TARGET_EVENT_FRAGMENTS.items():
            indices = [
                index
                for index, class_name in enumerate(lower_names)
                if any(fragment in class_name for fragment in fragments)
            ]
            if not indices:
                aggregated[event_name] = 0.0
                continue
            aggregated[event_name] = round(float(np.max(frame_scores[:, indices])), 4)

        logger.info("Detected event probabilities: %s", aggregated)
        return aggregated

    def detect_from_file(self, audio_path: str | Path) -> dict[str, float]:
        """Load a WAV/MP3/etc. file and run YAMNet on it."""
        waveform, sample_rate = librosa.load(Path(audio_path).as_posix(), sr=None, mono=True)
        return self.detect_from_waveform(waveform=waveform, sample_rate=int(sample_rate))

