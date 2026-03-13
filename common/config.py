"""Runtime configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    """Centralized project configuration loaded from environment variables."""

    project_root: Path
    sample_rate: int
    apply_noise_reduction: bool
    vad_aggressiveness: int
    voice_db_dir: Path
    processed_audio_dir: Path
    reports_dir: Path
    temp_dir: Path
    pyannote_model_name: str
    huggingface_token: str | None
    openai_api_key: str | None
    openai_transcription_model: str
    openai_translation_model: str
    openai_transcript_analysis_model: str
    openai_triage_model: str
    transcription_prompt: str

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from the current environment."""
        load_dotenv()
        project_root = Path(
            os.getenv(
                "PAB_PROJECT_ROOT",
                Path(__file__).resolve().parents[1],
            )
        ).resolve()

        return cls(
            project_root=project_root,
            sample_rate=int(os.getenv("PAB_SAMPLE_RATE", "16000")),
            apply_noise_reduction=_get_bool("PAB_NOISE_REDUCTION", False),
            vad_aggressiveness=int(os.getenv("PAB_VAD_AGGRESSIVENESS", "2")),
            voice_db_dir=(project_root / os.getenv("VOICE_DB_DIR", "voice_db")).resolve(),
            processed_audio_dir=(
                project_root / os.getenv("PAB_PROCESSED_AUDIO_DIR", "artifacts/processed")
            ).resolve(),
            reports_dir=(project_root / os.getenv("PAB_REPORTS_DIR", "artifacts/reports")).resolve(),
            temp_dir=(project_root / os.getenv("PAB_TEMP_DIR", "artifacts/tmp")).resolve(),
            pyannote_model_name=os.getenv("PAB_PYANNOTE_MODEL", "pyannote/embedding"),
            huggingface_token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_transcription_model=os.getenv(
                "OPENAI_TRANSCRIPTION_MODEL",
                "gpt-4o-mini-transcribe",
            ),
            openai_translation_model=os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-4.1-mini"),
            openai_transcript_analysis_model=os.getenv("OPENAI_TRANSCRIPT_ANALYSIS_MODEL", "gpt-4.1-mini"),
            openai_triage_model=os.getenv("OPENAI_TRIAGE_MODEL", "gpt-5-mini"),
            transcription_prompt=os.getenv(
                "OPENAI_TRANSCRIPTION_PROMPT",
                (
                    "This is an elderly-resident personal alert button emergency call. "
                    "Transcribe speech faithfully and keep short utterances such as pain, help, "
                    "breathing difficulty, and mentions of falls."
                ),
            ),
        )

    def ensure_runtime_dirs(self) -> None:
        """Create the runtime directories used by the project."""
        for directory in (
            self.voice_db_dir,
            self.processed_audio_dir,
            self.reports_dir,
            self.temp_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
