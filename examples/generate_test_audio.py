"""Generate a lightweight synthetic WAV file for local pipeline smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def generate_test_audio(
    output_path: str | Path = "examples/generated/test_alert.wav",
    duration_seconds: float = 8.0,
    sample_rate: int = 16000,
) -> Path:
    """Create a simple synthetic alert-like waveform."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    time_axis = np.linspace(0.0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    tone = 0.08 * np.sin(2 * np.pi * 220 * time_axis)
    pulse = np.where((time_axis > 1.0) & (time_axis < 1.4), 0.18 * np.sin(2 * np.pi * 880 * time_axis), 0.0)
    breath = np.where(
        (time_axis > 4.0) & (time_axis < 7.0),
        0.05 * np.sin(2 * np.pi * 50 * time_axis),
        0.0,
    )
    noise = 0.015 * np.random.default_rng(7).standard_normal(len(time_axis))
    waveform = np.clip(tone + pulse + breath + noise, -1.0, 1.0).astype(np.float32)

    sf.write(output.as_posix(), waveform, sample_rate, subtype="PCM_16")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic audio for smoke tests.")
    parser.add_argument(
        "--output",
        default="examples/generated/test_alert.wav",
        help="Output WAV path.",
    )
    args = parser.parse_args()

    generated_path = generate_test_audio(args.output)
    print(generated_path)


if __name__ == "__main__":
    main()

