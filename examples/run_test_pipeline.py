"""Example script for running the PAB triage pipeline locally."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from common.logging_utils import configure_logging
from examples.generate_test_audio import generate_test_audio
from pipeline.main_pipeline import EmergencyTriagePipeline, save_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local example of the PAB AI triage pipeline.")
    parser.add_argument(
        "--audio",
        help="Path to an existing audio file. If omitted, a synthetic WAV file is generated.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/reports/example_report.json",
        help="Path to save the generated JSON report.",
    )
    args = parser.parse_args()

    configure_logging()
    audio_path = Path(args.audio) if args.audio else generate_test_audio()

    pipeline = EmergencyTriagePipeline()
    report = pipeline.run(audio_path)
    saved_path = save_report(report, args.output)

    print(report.model_dump_json(indent=2))
    print(f"\nSaved example report to {saved_path}")


if __name__ == "__main__":
    main()
