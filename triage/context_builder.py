"""Context builder for transcript-first triage."""

from __future__ import annotations

from common.schemas import (
    AudioContextResult,
    SpeakerIdentificationResult,
    TranscriptAnalysisResult,
    TranscriptResult,
)


def build_audio_context(
    *,
    transcript: TranscriptResult,
    transcript_analysis: TranscriptAnalysisResult,
    speaker: SpeakerIdentificationResult,
    speech_segments: list[tuple[float, float]],
    audio_duration_seconds: float,
) -> AudioContextResult:
    """Build a compact context payload without non-verbal audio event detection."""
    speech_duration = max(0.0, sum(end - start for start, end in speech_segments))
    if audio_duration_seconds > 0:
        speech_ratio = min(1.0, speech_duration / audio_duration_seconds)
    else:
        speech_ratio = 0.0

    distress_cues: list[str] = []
    if transcript_analysis.incident != "unknown":
        distress_cues.append(f"incident:{transcript_analysis.incident}")
    distress_cues.extend(f"symptom:{symptom}" for symptom in transcript_analysis.symptoms)
    distress_cues.extend(
        f"normalized_symptom:{symptom}" for symptom in transcript_analysis.normalized_symptoms
    )
    distress_cues.extend(f"keyword:{keyword}" for keyword in transcript_analysis.keywords)
    if transcript.analysis_text.strip():
        distress_cues.append("speech_present")
    if speaker.speaker != "UNKNOWN":
        distress_cues.append("speaker_identified")

    return AudioContextResult(
        speech_duration_seconds=round(speech_duration, 4),
        speech_ratio=round(speech_ratio, 4),
        silence_ratio=round(max(0.0, 1.0 - speech_ratio), 4),
        transcript_present=bool(transcript.analysis_text.strip()),
        speaker_known=speaker.speaker != "UNKNOWN",
        distress_cues=distress_cues,
    )
