"""Typed JSON schemas shared across the system."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class UrgencyLevel(str, Enum):
    """Allowed urgency labels for emergency triage."""

    FALSE_ALARM = "FALSE_ALARM"
    NON_URGENT = "NON_URGENT"
    UNCERTAIN = "UNCERTAIN"
    URGENT = "URGENT"


class SpeakerIdentificationResult(BaseModel):
    """Output schema for speaker identification."""

    speaker: str = "UNKNOWN"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    similarities: Dict[str, float] = Field(default_factory=dict)


class TranscriptResult(BaseModel):
    """Output schema for speech transcription."""

    text: str = ""
    translated_text: str = ""
    language: str = "unknown"
    analysis_text: str = ""
    analysis_language: str = "unknown"


class TranscriptAnalysisResult(BaseModel):
    """Signals extracted from the transcript."""

    incident: str = "unknown"
    symptoms: List[str] = Field(default_factory=list)
    normalized_symptoms: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)


class AudioContextResult(BaseModel):
    """Context summary derived from speech presence and transcript cues."""

    speech_duration_seconds: float = Field(default=0.0, ge=0.0)
    speech_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    silence_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    transcript_present: bool = False
    speaker_known: bool = False
    distress_cues: List[str] = Field(default_factory=list)


class FalseAlarmResult(BaseModel):
    """Output schema for false alarm detection."""

    false_alarm: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class TriageResult(BaseModel):
    """Structured result returned by the LLM triage module."""

    incident: str
    urgency: UrgencyLevel
    confidence: float = Field(ge=0.0, le=1.0)
    recommended_action: str
    rationale: str = ""


class EmergencyReport(BaseModel):
    """Top-level operator-facing report for a single PAB alert."""

    request_id: str
    audio_path: str
    processed_audio_path: str
    audio_duration_seconds: float = Field(ge=0.0)
    speech_segments: List[Tuple[float, float]] = Field(default_factory=list)
    speaker: SpeakerIdentificationResult
    audio_events: Dict[str, float] = Field(default_factory=dict)
    audio_context: AudioContextResult
    transcript: TranscriptResult
    transcript_analysis: TranscriptAnalysisResult
    false_alarm: FalseAlarmResult
    triage: Optional[TriageResult] = None
    errors: List[str] = Field(default_factory=list)
    generated_at: str
