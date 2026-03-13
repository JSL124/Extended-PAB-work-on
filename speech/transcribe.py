"""Speech transcription using the OpenAI Audio API."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from common.config import Settings
from common.schemas import TranscriptResult

logger = logging.getLogger(__name__)


class OpenAITranscriber:
    """Wrapper around the OpenAI transcription API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for transcription.") from exc

        self._client = OpenAI(api_key=self.settings.openai_api_key)

    @staticmethod
    def _detect_language_from_text(text: str, reported_language: str) -> str:
        """Fallback language detection from transcript text when API returns unknown."""
        normalized_reported = (reported_language or "").strip().lower()
        if normalized_reported and normalized_reported != "unknown":
            return reported_language

        stripped = text.strip()
        if not stripped:
            return "unknown"

        cjk_count = sum(1 for char in stripped if "\u4e00" <= char <= "\u9fff")
        hangul_count = sum(1 for char in stripped if "\uac00" <= char <= "\ud7a3")
        hiragana_katakana_count = sum(
            1 for char in stripped if ("\u3040" <= char <= "\u309f") or ("\u30a0" <= char <= "\u30ff")
        )
        latin_count = sum(1 for char in stripped if char.isascii() and char.isalpha())

        if cjk_count >= 2:
            return "chinese"
        if hangul_count >= 2:
            return "korean"
        if hiragana_katakana_count >= 2:
            return "japanese"
        if latin_count >= 3:
            return "english"

        return "unknown"

    def transcribe(self, audio_path: str | Path) -> TranscriptResult:
        """Transcribe a PAB alert recording into text and detected language."""
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        logger.info(
            "Transcribing audio with OpenAI model %s",
            self.settings.openai_transcription_model,
        )
        with path.open("rb") as audio_file:
            response = self._client.audio.transcriptions.create(
                model=self.settings.openai_transcription_model,
                file=audio_file,
                prompt=self.settings.transcription_prompt,
                response_format="json",
                temperature=0,
            )

        payload = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        original_text = payload.get("text", "").strip()
        language = self._detect_language_from_text(
            text=original_text,
            reported_language=payload.get("language", "unknown"),
        )
        translated_text = self._translate_to_english(text=original_text, language=language)
        analysis_text = translated_text or original_text
        analysis_language = "english" if translated_text else language
        transcript = TranscriptResult(
            text=original_text,
            translated_text=translated_text,
            language=language,
            analysis_text=analysis_text,
            analysis_language=analysis_language,
        )
        logger.info("Transcription complete (language=%s)", transcript.language)
        return transcript

    def _translate_to_english(self, *, text: str, language: str) -> str:
        """Translate transcript text into English for downstream analysis."""
        normalized_text = text.strip()
        normalized_language = (language or "unknown").strip().lower()
        if not normalized_text:
            return ""
        if normalized_language in {"english", "en", "en-us", "en-gb"}:
            return normalized_text

        logger.info(
            "Translating transcript to English with model %s",
            self.settings.openai_translation_model,
        )
        response = self._client.chat.completions.create(
            model=self.settings.openai_translation_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Translate the provided emergency-call transcript into English. "
                        "Preserve urgency, symptoms, and meaning. Return only the translation."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Source language: {language or 'unknown'}\n"
                        f"Transcript:\n{normalized_text}"
                    ),
                },
            ],
        )
        translated = response.choices[0].message.content or ""
        return translated.strip()
