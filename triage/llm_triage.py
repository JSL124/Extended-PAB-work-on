"""LLM-based emergency triage using OpenAI structured outputs."""

from __future__ import annotations

import json
import logging

from common.config import Settings
from common.schemas import (
    AudioContextResult,
    SpeakerIdentificationResult,
    TriageResult,
    TranscriptAnalysisResult,
    TranscriptResult,
)

logger = logging.getLogger(__name__)


class LLMTriageEngine:
    """Use an OpenAI GPT model to reason over multimodal alert evidence."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for LLM triage.") from exc

        self._client = OpenAI(api_key=self.settings.openai_api_key)

    def triage(
        self,
        speaker: SpeakerIdentificationResult,
        audio_context: AudioContextResult,
        transcript: TranscriptResult,
        transcript_analysis: TranscriptAnalysisResult,
    ) -> TriageResult:
        """Return structured triage reasoning for a non-false-alarm alert."""
        evidence = {
            "speaker": speaker.model_dump(),
            "audio_context": audio_context.model_dump(),
            "transcript": transcript.model_dump(),
            "transcript_analysis": transcript_analysis.model_dump(),
        }

        system_prompt = (
            "You are an emergency triage assistant for elderly residents using a personal alert button. "
            "Reason conservatively, do not overstate certainty, and return only structured output. "
            "Infer the most likely incident from transcript and resident context, assign one urgency label "
            "from FALSE_ALARM, NON_URGENT, UNCERTAIN, URGENT, and recommend the next operator action. "
            "Use transcript.analysis_text as the primary language-normalized basis for reasoning."
        )

        logger.info("Running OpenAI triage with model %s", self.settings.openai_triage_model)
        completion = self._client.chat.completions.parse(
            model=self.settings.openai_triage_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Analyze the following emergency-alert evidence as JSON.\n"
                        f"{json.dumps(evidence, ensure_ascii=False, indent=2)}"
                    ),
                },
            ],
            response_format=TriageResult,
        )

        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError("OpenAI triage did not return a structured response.")
        return parsed
