"""Voice activity detection based on WebRTC VAD."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import webrtcvad

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Frame:
    """PCM16 frame used by WebRTC VAD."""

    payload: bytes
    timestamp: float
    duration: float


def _waveform_to_pcm16(waveform: np.ndarray) -> bytes:
    clipped = np.clip(waveform, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    return pcm16.tobytes()


def _generate_frames(
    waveform: np.ndarray,
    sample_rate: int,
    frame_duration_ms: int,
) -> list[Frame]:
    pcm_bytes = _waveform_to_pcm16(waveform)
    frame_size = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    frame_duration = frame_duration_ms / 1000.0

    frames: list[Frame] = []
    offset = 0
    timestamp = 0.0

    while offset + frame_size <= len(pcm_bytes):
        frames.append(
            Frame(
                payload=pcm_bytes[offset : offset + frame_size],
                timestamp=timestamp,
                duration=frame_duration,
            )
        )
        offset += frame_size
        timestamp += frame_duration

    return frames


def detect_speech_segments(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    aggressiveness: int = 2,
    frame_duration_ms: int = 30,
    padding_duration_ms: int = 300,
) -> list[tuple[float, float]]:
    """Return contiguous speech regions as `(start_time, end_time)` tuples."""
    if sample_rate != 16000:
        raise ValueError("WebRTC VAD expects 16 kHz audio in this implementation.")
    if frame_duration_ms not in {10, 20, 30}:
        raise ValueError("frame_duration_ms must be 10, 20, or 30 for WebRTC VAD.")
    if len(waveform) == 0:
        return []

    vad = webrtcvad.Vad(aggressiveness)
    frames = _generate_frames(waveform, sample_rate, frame_duration_ms)
    if not frames:
        return []

    segments: list[tuple[float, float]] = []
    padding_seconds = padding_duration_ms / 1000.0
    in_speech = False
    speech_start = 0.0
    speech_end = 0.0
    silence_accumulator = 0.0

    for frame in frames:
        is_speech = vad.is_speech(frame.payload, sample_rate)

        if is_speech and not in_speech:
            speech_start = frame.timestamp
            in_speech = True

        if in_speech:
            if is_speech:
                speech_end = frame.timestamp + frame.duration
                silence_accumulator = 0.0
            else:
                silence_accumulator += frame.duration
                if silence_accumulator >= padding_seconds:
                    segments.append((round(speech_start, 3), round(speech_end, 3)))
                    in_speech = False
                    silence_accumulator = 0.0

    if in_speech:
        segments.append((round(speech_start, 3), round(speech_end, 3)))

    logger.info("Detected %s speech segment(s)", len(segments))
    return segments


def concatenate_speech_segments(
    waveform: np.ndarray,
    sample_rate: int,
    speech_segments: list[tuple[float, float]],
) -> np.ndarray:
    """Concatenate speech-only waveform for speaker identification."""
    if not speech_segments:
        return waveform

    chunks: list[np.ndarray] = []
    for start_time, end_time in speech_segments:
        start_index = max(int(start_time * sample_rate), 0)
        end_index = min(int(end_time * sample_rate), len(waveform))
        if end_index > start_index:
            chunks.append(waveform[start_index:end_index])

    if not chunks:
        return waveform
    return np.concatenate(chunks).astype(np.float32)

