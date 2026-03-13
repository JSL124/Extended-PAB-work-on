"""Audio loading, normalization, resampling, and optional denoising."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


@dataclass(slots=True)
class PreprocessedAudio:
    """Normalized mono audio persisted to disk for downstream modules."""

    waveform: np.ndarray
    sample_rate: int
    output_path: Path
    duration_seconds: float


def load_audio(audio_path: PathLike) -> tuple[np.ndarray, int]:
    """Load audio as mono float32 waveform."""
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, sample_rate = librosa.load(path.as_posix(), sr=None, mono=True)
    logger.info("Loaded audio from %s at %s Hz", path, sample_rate)
    return waveform.astype(np.float32), int(sample_rate)


def normalize_audio(waveform: np.ndarray) -> np.ndarray:
    """Peak-normalize audio to a safe amplitude range."""
    peak = np.max(np.abs(waveform))
    if peak <= 1e-8:
        return waveform.astype(np.float32)
    normalized = waveform / peak
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def resample_audio(waveform: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    """Resample the waveform to the target sample rate."""
    if source_sr == target_sr:
        return waveform.astype(np.float32)
    logger.info("Resampling audio from %s Hz to %s Hz", source_sr, target_sr)
    resampled = librosa.resample(waveform, orig_sr=source_sr, target_sr=target_sr)
    return resampled.astype(np.float32)


def reduce_noise(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply broadband stationary noise reduction."""
    logger.info("Applying noise reduction")
    reduced = nr.reduce_noise(y=waveform, sr=sample_rate)
    return reduced.astype(np.float32)


def write_waveform(output_path: PathLike, waveform: np.ndarray, sample_rate: int) -> Path:
    """Persist waveform to a PCM16 WAV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path.as_posix(), waveform, sample_rate, subtype="PCM_16")
    return path


def preprocess_audio(
    audio_path: PathLike,
    output_path: PathLike,
    target_sr: int = 16000,
    apply_noise_reduction: bool = False,
) -> PreprocessedAudio:
    """Preprocess an input audio recording for all downstream tasks."""
    waveform, source_sr = load_audio(audio_path)
    waveform = normalize_audio(waveform)
    waveform = resample_audio(waveform, source_sr, target_sr)

    if apply_noise_reduction:
        waveform = reduce_noise(waveform, target_sr)
        waveform = normalize_audio(waveform)

    saved_path = write_waveform(output_path, waveform, target_sr)
    duration_seconds = float(len(waveform) / target_sr) if len(waveform) else 0.0

    logger.info(
        "Preprocessed audio saved to %s (duration=%.2fs)",
        saved_path,
        duration_seconds,
    )
    return PreprocessedAudio(
        waveform=waveform,
        sample_rate=target_sr,
        output_path=saved_path,
        duration_seconds=duration_seconds,
    )

