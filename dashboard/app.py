"""Streamlit operator dashboard for the PAB AI triage pipeline."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from common.logging_utils import configure_logging
from pipeline.main_pipeline import EmergencyTriagePipeline


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

        :root {
            --bg: #f6f1e8;
            --surface: rgba(255, 255, 255, 0.84);
            --ink: #102a43;
            --muted: #486581;
            --accent: #c05621;
            --accent-soft: #fbd38d;
            --success: #2f855a;
            --danger: #c53030;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(251, 211, 141, 0.55), transparent 32%),
                radial-gradient(circle at top right, rgba(144, 205, 244, 0.32), transparent 28%),
                linear-gradient(180deg, #f8f5ef 0%, #edf2f7 100%);
            color: var(--ink);
            font-family: "IBM Plex Sans", sans-serif;
        }

        h1, h2, h3, h4 {
            font-family: "Space Grotesk", sans-serif !important;
            letter-spacing: -0.02em;
        }

        .hero {
            border: 1px solid rgba(16, 42, 67, 0.08);
            background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(255,245,235,0.92));
            border-radius: 24px;
            padding: 1.4rem 1.6rem;
            box-shadow: 0 18px 40px rgba(16, 42, 67, 0.08);
            margin-bottom: 1rem;
        }

        .card {
            border: 1px solid rgba(16, 42, 67, 0.08);
            background: var(--surface);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 28px rgba(16, 42, 67, 0.05);
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .metric-value {
            color: var(--ink);
            font-size: 1.45rem;
            font-weight: 700;
        }

        .badge {
            display: inline-block;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            font-size: 0.84rem;
            font-weight: 600;
            margin-top: 0.35rem;
        }

        .badge-danger {
            color: white;
            background: linear-gradient(135deg, #c53030, #9b2c2c);
        }

        .badge-success {
            color: white;
            background: linear-gradient(135deg, #2f855a, #276749);
        }

        .badge-neutral {
            color: #744210;
            background: linear-gradient(135deg, #f6e05e, #ecc94b);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _urgency_badge(urgency: str) -> str:
    if urgency == "URGENT":
        return '<span class="badge badge-danger">URGENT</span>'
    if urgency == "FALSE_ALARM":
        return '<span class="badge badge-success">FALSE ALARM</span>'
    return '<span class="badge badge-neutral">{}</span>'.format(urgency.replace("_", " "))


def _render_metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _stringify_dataframe_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert mixed-value display tables to Arrow-safe strings for Streamlit."""
    safe = frame.copy()
    safe["value"] = safe["value"].map(lambda value: str(value))
    return safe


def main() -> None:
    configure_logging()
    st.set_page_config(page_title="PAB AI Triage Dashboard", layout="wide")
    _inject_styles()

    st.markdown(
        """
        <div class="hero">
            <h1>PAB AI Triage Dashboard</h1>
            <p>Upload a personal alert button recording to generate an operator-ready incident report with speaker identity, event signals, false-alarm classification, and urgency reasoning.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Run Analysis")
        uploaded_file = st.file_uploader(
            "Audio recording",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
        )
        run_analysis = st.button("Analyze Alert", type="primary", use_container_width=True)

        st.caption("Required env vars: `OPENAI_API_KEY` and optional `HF_TOKEN` for pyannote models.")

    if not uploaded_file:
        st.info("Upload an audio file to run the pipeline.")
        return

    st.audio(uploaded_file)

    if not run_analysis:
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as temp_audio:
        temp_audio.write(uploaded_file.getvalue())
        temp_audio_path = Path(temp_audio.name)

    pipeline = EmergencyTriagePipeline()
    with st.spinner("Running PAB AI triage pipeline..."):
        report = pipeline.run(temp_audio_path)
        report_payload = report.model_dump(mode="json")

    triage = report_payload.get("triage") or {}
    false_alarm = report_payload["false_alarm"]
    transcript = report_payload["transcript"]
    speaker = report_payload["speaker"]

    col1, col2, col3 = st.columns(3)
    with col1:
        _render_metric_card("Speaker", speaker["speaker"])
    with col2:
        _render_metric_card("Speaker Confidence", f'{speaker["confidence"]:.2f}')
    with col3:
        urgency = triage.get("urgency", "UNCERTAIN")
        st.markdown(
            f"""
            <div class="card">
                <div class="metric-label">Urgency</div>
                <div class="metric-value">{urgency.replace("_", " ")}</div>
                {_urgency_badge(urgency)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    left, right = st.columns([1.05, 0.95])

    with left:
        st.subheader("Transcript")
        st.markdown(f'<div class="card">{transcript["text"] or "No transcript detected."}</div>', unsafe_allow_html=True)

        st.subheader("Translated Transcript (English)")
        st.markdown(
            f'<div class="card">{transcript["translated_text"] or "No English translation available."}</div>',
            unsafe_allow_html=True,
        )

        st.subheader("Recommended Action")
        st.markdown(
            f"""
            <div class="card">
                <strong>Incident:</strong> {triage.get("incident", "unknown")}<br/>
                <strong>Action:</strong> {triage.get("recommended_action", "manual review required")}<br/>
                <strong>Rationale:</strong> {triage.get("rationale", "No rationale available.")}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("False Alarm Status")
        badge = "badge-success" if false_alarm["false_alarm"] else "badge-danger"
        status = "Likely False Alarm" if false_alarm["false_alarm"] else "Emergency Signal Retained"
        st.markdown(
            f"""
            <div class="card">
                <span class="badge {badge}">{status}</span><br/>
                <strong>Confidence:</strong> {false_alarm["confidence"]:.2f}<br/>
                <strong>Reason:</strong> {false_alarm["reason"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.subheader("Audio Context")
        audio_context = report_payload["audio_context"]
        context_df = pd.DataFrame(
            {
                "field": [
                    "speech_duration_seconds",
                    "speech_ratio",
                    "silence_ratio",
                    "transcript_present",
                    "speaker_known",
                    "distress_cues",
                ],
                "value": [
                    audio_context["speech_duration_seconds"],
                    audio_context["speech_ratio"],
                    audio_context["silence_ratio"],
                    audio_context["transcript_present"],
                    audio_context["speaker_known"],
                    ", ".join(audio_context["distress_cues"]) or "-",
                ],
            }
        )
        st.dataframe(_stringify_dataframe_values(context_df), width="stretch", hide_index=True)

        st.subheader("Transcript Signals")
        signals_df = pd.DataFrame(
            {
                "field": [
                    "incident",
                    "symptoms",
                    "normalized_symptoms",
                    "keywords",
                    "language",
                    "analysis_language",
                ],
                "value": [
                    report_payload["transcript_analysis"]["incident"],
                    ", ".join(report_payload["transcript_analysis"]["symptoms"]) or "-",
                    ", ".join(report_payload["transcript_analysis"]["normalized_symptoms"]) or "-",
                    ", ".join(report_payload["transcript_analysis"]["keywords"]) or "-",
                    transcript["language"],
                    transcript["analysis_language"],
                ],
            }
        )
        st.dataframe(_stringify_dataframe_values(signals_df), width="stretch", hide_index=True)

    if report_payload["errors"]:
        st.warning("\n".join(report_payload["errors"]))

    with st.expander("Structured JSON Report", expanded=False):
        st.code(json.dumps(report_payload, indent=2), language="json")


if __name__ == "__main__":
    main()
