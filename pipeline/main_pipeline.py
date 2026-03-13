"""End-to-end PAB AI emergency triage pipeline."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from audio.preprocess import preprocess_audio, write_waveform
from audio.vad import concatenate_speech_segments, detect_speech_segments
from common.config import Settings
from common.logging_utils import configure_logging
from common.schemas import (
    EmergencyReport,
    TranscriptAnalysisResult,
    SpeakerIdentificationResult,
    TriageResult,
    TranscriptResult,
    UrgencyLevel,
)
from speaker.identify import SpeakerIdentifier
from speech.transcribe import OpenAITranscriber
from speech.transcript_analysis import analyze_transcript
from triage.context_builder import build_audio_context
from triage.false_alarm_detector import FalseAlarmDetector
from triage.llm_triage import LLMTriageEngine

logger = logging.getLogger(__name__)


class EmergencyTriagePipeline:
    """Coordinates the full AI pipeline from audio input to structured report."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.settings.ensure_runtime_dirs()
        self.speaker_identifier = SpeakerIdentifier(self.settings)
        self.transcriber = OpenAITranscriber(self.settings)
        self.false_alarm_detector = FalseAlarmDetector()
        self.llm_triage = LLMTriageEngine(self.settings)

    def run(self, audio_path: str | Path) -> EmergencyReport:
        """Execute the full triage pipeline for a PAB audio recording."""
        request_id = str(uuid4())
        audio_path = Path(audio_path).resolve()
        errors: list[str] = []

        processed_path = self.settings.processed_audio_dir / f"{audio_path.stem}_{request_id}.wav"
        preprocessed = preprocess_audio(
            audio_path=audio_path,
            output_path=processed_path,
            target_sr=self.settings.sample_rate,
            apply_noise_reduction=self.settings.apply_noise_reduction,
        )

        speech_segments = detect_speech_segments(
            waveform=preprocessed.waveform,
            sample_rate=preprocessed.sample_rate,
            aggressiveness=self.settings.vad_aggressiveness,
        )
        speech_only_waveform = concatenate_speech_segments(
            waveform=preprocessed.waveform,
            sample_rate=preprocessed.sample_rate,
            speech_segments=speech_segments,
        )
        speech_only_path = write_waveform(
            self.settings.temp_dir / f"{audio_path.stem}_{request_id}_speech_only.wav",
            speech_only_waveform,
            preprocessed.sample_rate,
        )

        speaker_result = SpeakerIdentificationResult()
        try:
            speaker_result = self.speaker_identifier.identify(speech_only_path)
        except Exception as exc:
            logger.exception("Speaker identification failed")
            errors.append(f"speaker_identification_failed: {exc}")

        transcript_result = TranscriptResult()
        try:
            transcript_result = self.transcriber.transcribe(preprocessed.output_path)
        except Exception as exc:
            logger.exception("Transcription failed")
            errors.append(f"transcription_failed: {exc}")

        transcript_analysis: TranscriptAnalysisResult = analyze_transcript(
            transcript_result.analysis_text,
            settings=self.settings,
        )
        audio_context = build_audio_context(
            transcript=transcript_result,
            transcript_analysis=transcript_analysis,
            speaker=speaker_result,
            speech_segments=speech_segments,
            audio_duration_seconds=preprocessed.duration_seconds,
        )

        false_alarm_result = self.false_alarm_detector.detect(
            transcript=transcript_result,
            transcript_analysis=transcript_analysis,
            audio_context=audio_context,
            speaker=speaker_result,
            speech_segments=speech_segments,
            audio_duration_seconds=preprocessed.duration_seconds,
        )

        triage_result: TriageResult | None = None
        if false_alarm_result.false_alarm:
            triage_result = TriageResult(
                incident="false_alarm",
                urgency=UrgencyLevel.FALSE_ALARM,
                confidence=false_alarm_result.confidence,
                recommended_action="log the alert, confirm resident safety, and close the case",
                rationale=false_alarm_result.reason,
            )
        else:
            try:
                triage_result = self.llm_triage.triage(
                    speaker=speaker_result,
                    audio_context=audio_context,
                    transcript=transcript_result,
                    transcript_analysis=transcript_analysis,
                )
            except Exception as exc:
                logger.exception("LLM triage failed")
                errors.append(f"llm_triage_failed: {exc}")
                triage_result = TriageResult(
                    incident=transcript_analysis.incident,
                    urgency=UrgencyLevel.UNCERTAIN,
                    confidence=0.35,
                    recommended_action="operator should call the resident back and assess manually",
                    rationale="LLM triage unavailable; fallback generated from transcript analysis.",
                )

        report = EmergencyReport(
            request_id=request_id,
            audio_path=str(audio_path),
            processed_audio_path=str(preprocessed.output_path),
            audio_duration_seconds=round(preprocessed.duration_seconds, 4),
            speech_segments=speech_segments,
            speaker=speaker_result,
            audio_events={},
            audio_context=audio_context,
            transcript=transcript_result,
            transcript_analysis=transcript_analysis,
            false_alarm=false_alarm_result,
            triage=triage_result,
            errors=errors,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        return report


def save_report(report: EmergencyReport, output_path: str | Path) -> Path:
    """Persist a report as pretty-printed JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PAB AI triage pipeline.")
    parser.add_argument("--audio", required=True, help="Path to input audio recording.")
    parser.add_argument(
        "--output",
        help="Optional path to write the JSON report. Defaults to artifacts/reports/<request>.json",
    )
    return parser


def main() -> None:
    configure_logging()
    args = build_arg_parser().parse_args()

    pipeline = EmergencyTriagePipeline()
    report = pipeline.run(args.audio)

    output_path = (
        Path(args.output)
        if args.output
        else pipeline.settings.reports_dir / f"{report.request_id}.json"
    )
    saved_path = save_report(report, output_path)

    print(report.model_dump_json(indent=2))
    print(f"\nSaved report to {saved_path}")


if __name__ == "__main__":
    main()
