"""LLM-backed transcript analysis with deterministic symptom normalization."""

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

CANONICAL_SYMPTOM_RULES: dict[str, tuple[str, ...]] = {
    "chest_discomfort": (
        r"\bheart (?:feels |is )?(?:uncomfortable|not okay|not right)\b",
        r"\bheart discomfort\b",
        r"\bheart problem\b",
        r"\bchest discomfort\b",
        r"\bchest feels uncomfortable\b",
        r"\bchest tight(?:ness)?\b",
        r"\bchest pressure\b",
    ),
    "chest_pain": (
        r"\bchest pain\b",
        r"\bpain in (?:my )?chest\b",
    ),
    "generalized_pain": (
        r"\beverywhere hurts\b",
        r"\bwhole body hurts\b",
        r"\ball over pain\b",
        r"\bbody aches?\b",
        r"\bgeneral(?:ized)? pain\b",
        r"\beverything hurts\b",
        r"\bmy whole body\b",
    ),
    "head_pain": (
        r"\bhead pain\b",
        r"\bhead hurts\b",
        r"\bheadache\b",
        r"\bhead (?:is )?(?:uncomfortable|not right)\b",
        r"\bpain in (?:my )?head\b",
    ),
    "shortness_of_breath": (
        r"\bshortness of breath\b",
        r"\bshort of breath\b",
        r"\bcan'?t breathe\b",
        r"\bbreathing problem\b",
        r"\btrouble breathing\b",
    ),
    "dizziness": (
        r"\bdizz(?:y|iness)\b",
        r"\blightheaded(?:ness)?\b",
        r"\bfeel faint\b",
        r"\bfaint\b",
    ),
    "bleeding": (
        r"\bbleeding\b",
        r"\bblood\b",
    ),
    "leg_pain": (
        r"\bleg pain\b",
        r"\bleg hurts\b",
        r"\bpain in (?:my )?leg\b",
    ),
    "hip_pain": (
        r"\bhip pain\b",
        r"\bmy hip\b",
        r"\bpain in (?:my )?hip\b",
    ),
    "weakness": (
        r"\bweak(?:ness)?\b",
        r"\bno strength\b",
    ),
    "confusion": (
        r"\bconfus(?:ed|ion)\b",
        r"\bnot thinking clearly\b",
        r"\bdisoriented\b",
    ),
}


def _find_matches(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def normalize_symptoms(phrases: list[str], text: str = "") -> list[str]:
    """Map free-text symptom phrases to canonical labels."""
    normalized_labels: list[str] = []
    candidate_texts = [value.strip().lower() for value in [*phrases, text] if value and value.strip()]

    for canonical, patterns in CANONICAL_SYMPTOM_RULES.items():
        if any(_find_matches(candidate, patterns) for candidate in candidate_texts):
            normalized_labels.append(canonical)

    return list(dict.fromkeys(normalized_labels))


def analyze_transcript_rule_based(text: str) -> TranscriptAnalysisResult:
    """Extract incident category, symptoms, salient keywords, and canonical symptoms."""
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
    canonical_symptoms = normalize_symptoms([*symptoms, *keywords], normalized)

    return TranscriptAnalysisResult(
        incident=incident,
        symptoms=symptoms,
        normalized_symptoms=canonical_symptoms,
        keywords=keywords,
    )


class TranscriptAnalyzer:
    """LLM-first transcript analyzer with deterministic fallback and normalization."""

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
        normalized_text = text.strip()
        if not normalized_text or self._client is None:
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
                            "shortness of breath, dizziness, bleeding, weakness, confusion, heart discomfort, "
                            "and chest discomfort. If the incident is unclear, set incident to 'unknown'."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Transcript:\n{normalized_text}",
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
            normalized_symptoms = normalize_symptoms([*symptoms, *keywords], normalized_text)
            if not normalized_symptoms:
                normalized_symptoms = fallback.normalized_symptoms

            return TranscriptAnalysisResult(
                incident=incident,
                symptoms=symptoms,
                normalized_symptoms=normalized_symptoms,
                keywords=keywords,
            )
        except Exception as exc:
            logger.warning("LLM transcript analysis failed; using rule-based fallback: %s", exc)
            return fallback


def analyze_transcript(text: str, settings: Settings | None = None) -> TranscriptAnalysisResult:
    """Public analyzer entry point used by the pipeline."""
    return TranscriptAnalyzer(settings).analyze(text)
