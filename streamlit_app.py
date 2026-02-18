import io
import re
import wave

import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Speech Metrics", page_icon="🎤")

st.title("🎤 Speech Metrics (Prototype)")
st.caption("Build: v1.4 – transcript + duration + WPM + filler%")

st.write("Upload a WAV file and we’ll transcribe it + calculate a few basic metrics.")

audio_file = st.file_uploader("Upload WAV audio", type=["wav"])

# Keep filler list small + obvious (you can expand later)
FILLER_SINGLE = {
    "um", "uh", "erm", "er", "ah", "like", "basically", "actually", "literally"
}
FILLER_PHRASES = [
    "you know",
    "sort of",
    "kind of",
]

def count_fillers(text: str) -> tuple[int, dict]:
    """
    Returns:
      total_filler_hits, breakdown_dict
    """
    t = text.lower()

    # Tokenise into words (simple, robust)
    words = re.findall(r"[a-z']+", t)

    breakdown = {w: 0 for w in sorted(FILLER_SINGLE)}
    total = 0

    # Single-word fillers
    for w in words:
        if w in FILLER_SINGLE:
            breakdown[w] += 1
            total += 1

    # Multi-word phrases (count occurrences in raw string)
    for phrase in FILLER_PHRASES:
        hits = len(re.findall(rf"\b{re.escape(phrase)}\b", t))
        breakdown[phrase] = hits
        total += hits

    # Remove zeros in breakdown for tidy display
    breakdown = {k: v for k, v in breakdown.items() if v > 0}
    return total, breakdown

def wav_duration_seconds(wav_bytes: bytes) -> float:
    """
    Reads WAV header to calculate duration (no ffmpeg, no extra deps).
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)

if audio_file is not None:
    st.success("File uploaded successfully!")

    audio_bytes = audio_file.read()

    # Duration
    try:
        duration_sec = wav_duration_seconds(audio_bytes)
    except Exception:
        st.error("Could not read WAV duration. Is this a valid WAV file?")
        st.stop()

    # OpenAI key check
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("Missing OPENAI_API_KEY in Streamlit Secrets (Manage app → Settings → Secrets).")
        st.stop()

    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    # Transcription
    try:
        with st.spinner("Transcribing audio..."):
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=("audio.wav", audio_bytes),
                response_format="json",
            )

        # IMPORTANT: transcript is a Pydantic object in newer SDKs, so use .text
        text = getattr(transcript, "text", None)
        if not text:
            # fallback if ever returned dict-like
            try:
                text = transcript["text"]
            except Exception:
                text = ""

        if not text.strip():
            st.error("Transcript came back empty.")
            st.stop()

    except Exception as e:
        st.error("Transcription failed.")
        st.exception(e)
        st.stop()

    # Metrics from transcript
    words = re.findall(r"[a-z']+", text.lower())
    word_count = len(words)

    minutes = max(duration_sec / 60.0, 1e-9)
    wpm = word_count / minutes

    filler_count, filler_breakdown = count_fillers(text)
    filler_pct = (filler_count / max(word_count, 1)) * 100

    # Output metrics
    st.subheader("Key metrics")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Duration", f"{duration_sec:.1f}s")
    col2.metric("Words", f"{word_count}")
    col3.metric("WPM", f"{round(wpm):d}")
    col4.metric("Filler %", f"{filler_pct:.1f}%")

    if filler_breakdown:
        st.caption("Filler breakdown (counts)")
        st.json(filler_breakdown)

    st.subheader("Transcript")
    st.write(text)
