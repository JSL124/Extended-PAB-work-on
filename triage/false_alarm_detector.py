"""Heuristic false alarm detector that compares false-alarm and emergency evidence."""

from __future__ import annotations

import re

from common.schemas import (
    AudioContextResult,
    FalseAlarmResult,
    SpeakerIdentificationResult,
    TranscriptAnalysisResult,
    TranscriptResult,
)

FALSE_ALARM_PATTERNS: dict[str, tuple[str, ...]] = {
    "accidental press detected": (
        r"\bby mistake\b",
        r"\baccident(?:al|ally)?\b",
        r"\bwrong button\b",
        r"\boops\b",
        r"\bsorry\b.*\bpress",
    ),
    "device test detected": (
        r"\btesting\b",
        r"\btest call\b",
        r"\bchecking\b.*\bbutton\b",
        r"\btrying out\b",
    ),
    "loneliness or check-in call detected": (
        r"\banyone there\b",
        r"\bjust wanted to talk\b",
        r"\bfeeling lonely\b",
        r"\bcan someone talk to me\b",
        r"\bhello\??\b",
    ),
}


class FalseAlarmDetector:
    """Assess whether the alert is likely a non-emergency false alarm."""

    def detect(
        self,
        transcript: TranscriptResult,
        transcript_analysis: TranscriptAnalysisResult,
        audio_context: AudioContextResult,
        speaker: SpeakerIdentificationResult,
        speech_segments: list[tuple[float, float]],
        audio_duration_seconds: float,
    ) -> FalseAlarmResult:
        """Compare false-alarm evidence against emergency evidence."""
        text = transcript.analysis_text.lower().strip()

        cue_scores: dict[str, float] = {}
        for reason, patterns in FALSE_ALARM_PATTERNS.items():
            matches = sum(1 for pattern in patterns if re.search(pattern, text))
            if matches:
                cue_scores[reason] = matches * 1.4

        if not text and not speech_segments:
            cue_scores["no speech or silence detected"] = 1.8

        if text and speaker.speaker != "UNKNOWN":
            cue_scores = {
                reason: score + (0.1 * speaker.confidence)
                for reason, score in cue_scores.items()
            }

        emergency_score = 0.0
        if transcript_analysis.incident != "unknown":
            emergency_score += 1.5
        emergency_score += 0.45 * len(transcript_analysis.symptoms)
        emergency_score += 0.15 * len(transcript_analysis.keywords)
        emergency_score += 0.6 * audio_context.speech_ratio
        emergency_score += 0.12 * len(audio_context.distress_cues)
        if audio_context.transcript_present:
            emergency_score += 0.25

        if speech_segments and audio_duration_seconds > 0:
            speech_ratio = sum(end - start for start, end in speech_segments) / audio_duration_seconds
            emergency_score += max(speech_ratio, 0.0) * 0.4

        false_alarm_score = sum(cue_scores.values())
        total_score = false_alarm_score + emergency_score

        if total_score <= 1e-6:
            return FalseAlarmResult(
                false_alarm=False,
                confidence=0.5,
                reason="insufficient evidence for false alarm or emergency",
            )

        prior = 0.25
        false_alarm_probability = (false_alarm_score + prior) / (total_score + (2 * prior))
        emergency_probability = (emergency_score + prior) / (total_score + (2 * prior))
        false_alarm = false_alarm_score > 0 and false_alarm_probability >= emergency_probability

        if false_alarm:
            reason = max(cue_scores.items(), key=lambda item: item[1])[0]
            confidence = false_alarm_probability
        else:
            confidence = emergency_probability
            reason = "emergency indicators outweigh false-alarm cues"

        confidence = round(float(max(0.0, min(1.0, confidence))), 4)
        return FalseAlarmResult(
            false_alarm=false_alarm,
            confidence=confidence,
            reason=reason,
        )
