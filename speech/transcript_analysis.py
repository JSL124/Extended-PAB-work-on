"""Rule-based transcript signal extraction."""

from __future__ import annotations

import logging
import re
from collections import Counter

from common.config import Settings
from common.schemas import TranscriptAnalysisResult

logger = logging.getLogger(__name__)

INCIDENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "fall": (
        r"\bfall(?:en|ing)?\b",
        r"\bslip(?:ped|ping)?\b",
        r"\bon the floor\b",
        r"\bcan't get up\b",
        r"\bcannot get up\b",
        r"\btripped\b",
    ),
    "breathing_issue": (
        r"\bcan'?t breathe\b",
        r"\bshort of breath\b",
        r"\bbreath(?:ing)? problem\b",
        r"\bwheez(?:e|ing)\b",
        r"\bgasp(?:ing)?\b",
    ),
    "pain": (
        r"\bpain\b",
        r"\bhurts?\b",
        r"\bsore\b",
    ),
    "medical_distress": (
        r"\bchest pain\b",
        r"\bdizzy\b",
        r"\bfaint\b",
        r"\bbleeding\b",
        r"\bhelp me\b",
    ),
    "fire_or_hazard": (
        r"\bfire\b",
        r"\bsmoke\b",
        r"\bglass\b",
        r"\bbroke\b",
    ),
}

SYMPTOM_PATTERNS: dict[str, tuple[str, ...]] = {
    "leg pain": (r"\bleg pain\b", r"\bmy leg\b", r"\bleg hurts\b"),
    "hip pain": (r"\bhip pain\b", r"\bmy hip\b"),
    "chest pain": (r"\bchest pain\b",),
    "shortness of breath": (r"\bshort of breath\b", r"\bcan'?t breathe\b"),
    "bleeding": (r"\bbleeding\b", r"\bblood\b"),
    "dizziness": (r"\bdizzy\b", r"\blightheaded\b"),
}

KEYWORD_TERMS: tuple[str, ...] = (
    "fall",
    "fell",
    "slipped",
    "tripped",
    "pain",
    "breathing",
    "breathe",
    "help",
    "blood",
    "bleeding",
    "dizzy",
    "bathroom",
    "floor",
    "ambulance",
    "chest",
)


def _find_matches(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def analyze_transcript_rule_based(text: str) -> TranscriptAnalysisResult:
    """Extract incident category, symptoms, and salient keywords from text."""
    normalized = text.lower().strip()
    if not normalized:
        return TranscriptAnalysisResult()

    incident_scores: Counter[str] = Counter()
    for incident, patterns in INCIDENT_PATTERNS.items():
        if _find_matches(normalized, patterns):
            incident_scores[incident] += 1

    symptoms = [
        symptom
        for symptom, patterns in SYMPTOM_PATTERNS.items()
        if _find_matches(normalized, patterns)
    ]

    keywords = [term for term in KEYWORD_TERMS if term in normalized]
    incident = incident_scores.most_common(1)[0][0] if incident_scores else "unknown"

    return TranscriptAnalysisResult(
        incident=incident,
        symptoms=symptoms,
        keywords=keywords,
    )


class TranscriptAnalyzer:
    """LLM-first transcript analyzer with deterministic fallback."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self._client = None

        if not self.settings.openai_api_key:
            return

        try:
            from openai import OpenAI
        except Exception:
            logger.warning("OpenAI package unavailable; transcript analysis will use rule-based fallback.")
            return

        self._client = OpenAI(api_key=self.settings.openai_api_key)

    def analyze(self, text: str) -> TranscriptAnalysisResult:
        """Analyze transcript text, preferring LLM extraction and falling back to rules."""
        fallback = analyze_transcript_rule_based(text)
        normalized = text.strip()
        if not normalized or self._client is None:
            return fallback

        try:
            completion = self._client.chat.completions.parse(
                model=self.settings.openai_transcript_analysis_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You extract structured emergency-call signals from English transcript text. "
                            "Return concise JSON with incident, symptoms, and keywords only. "
                            "Use short English symptom phrases such as chest pain, head pain, generalized pain, "
                            "shortness of breath, dizziness, bleeding, weakness, confusion. "
                            "If the incident is unclear, set incident to 'unknown'."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Transcript:\n{normalized}",
                    },
                ],
                response_format=TranscriptAnalysisResult,
            )
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                return fallback

            incident = parsed.incident if parsed.incident != "unknown" else fallback.incident
            symptoms = list(dict.fromkeys([*parsed.symptoms, *fallback.symptoms]))
            keywords = list(dict.fromkeys([*parsed.keywords, *fallback.keywords]))

            return TranscriptAnalysisResult(
                incident=incident,
                symptoms=symptoms,
                keywords=keywords,
            )
        except Exception as exc:
            logger.warning("LLM transcript analysis failed; using rule-based fallback: %s", exc)
            return fallback


def analyze_transcript(text: str, settings: Settings | None = None) -> TranscriptAnalysisResult:
    """Public analyzer entry point used by the pipeline."""
    return TranscriptAnalyzer(settings).analyze(text)
