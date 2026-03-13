"""Speaker enrollment CLI for storing resident voice embeddings."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

from audio.preprocess import preprocess_audio, write_waveform
from audio.vad import concatenate_speech_segments, detect_speech_segments
from common.config import Settings
from common.logging_utils import configure_logging
from speaker.identify import SpeakerIdentifier

logger = logging.getLogger(__name__)


def _sanitize_speaker_name(speaker_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", speaker_name).strip("_") or "resident"


def enroll_speaker(
    audio_path: str | Path,
    speaker_name: str,
    settings: Settings | None = None,
) -> Path:
    """Enroll a resident by extracting and saving a pyannote embedding."""
    config = settings or Settings.from_env()
    config.ensure_runtime_dirs()

    sanitized_name = _sanitize_speaker_name(speaker_name)
    preprocessed = preprocess_audio(
        audio_path=audio_path,
        output_path=config.temp_dir / f"{sanitized_name}_enrollment.wav",
        target_sr=config.sample_rate,
        apply_noise_reduction=config.apply_noise_reduction,
    )
    speech_segments = detect_speech_segments(
        waveform=preprocessed.waveform,
        sample_rate=preprocessed.sample_rate,
        aggressiveness=config.vad_aggressiveness,
    )
    speech_only_waveform = concatenate_speech_segments(
        waveform=preprocessed.waveform,
        sample_rate=preprocessed.sample_rate,
        speech_segments=speech_segments,
    )
    speech_only_path = write_waveform(
        config.temp_dir / f"{sanitized_name}_speech_only.wav",
        speech_only_waveform,
        preprocessed.sample_rate,
    )

    identifier = SpeakerIdentifier(config)
    embedding = identifier.extract_embedding(speech_only_path)
    target_path = config.voice_db_dir / f"{sanitized_name}.npy"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np

    np.save(target_path, embedding)
    logger.info("Saved enrolled speaker embedding to %s", target_path)
    return target_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enroll a resident voice sample.")
    parser.add_argument("--audio", required=True, help="Path to the enrollment audio file.")
    parser.add_argument("--speaker", required=True, help="Resident name or identifier.")
    return parser


def main() -> None:
    configure_logging()
    args = build_arg_parser().parse_args()
    saved_path = enroll_speaker(audio_path=args.audio, speaker_name=args.speaker)
    print(saved_path)


if __name__ == "__main__":
    main()

