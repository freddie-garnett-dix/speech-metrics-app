import io
import re
import wave

import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Speech Metrics (Prototype)", page_icon="🎤")

st.title("🎤 Speech Metrics (Prototype)")
st.caption("Build: v1.3 – transcript + duration + WPM (no pauses yet)")

st.write("Upload a **WAV** file and we’ll transcribe it + calculate a few basic metrics.")

audio_file = st.file_uploader("Upload WAV audio", type=["wav"])

def count_words(text: str) -> int:
    # Simple word count (letters/numbers/apostrophes)
    return len(re.findall(r"[A-Za-z0-9']+", text))

def wav_duration_seconds(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)

if audio_file is not None:
    st.success("File uploaded successfully!")

    audio_bytes = audio_file.read()

    # Duration (WAV-only, no ffmpeg needed)
    try:
        duration_sec = wav_duration_seconds(audio_bytes)
    except Exception as e:
        st.error("Could not read WAV duration. Please confirm the file is a valid WAV.")
        st.exception(e)
        st.stop()

    # OpenAI key check
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("Missing OPENAI_API_KEY in Streamlit Secrets.")
        st.stop()

    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    # Transcribe
    try:
        with st.spinner("Transcribing audio..."):
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=("audio.wav", audio_bytes),
                response_format="text",
            )

        text = transcript  # because response_format="text" returns a plain string

    except Exception as e:
        st.error("Transcription failed.")
        st.exception(e)
        st.stop()

    # Metrics
    words = count_words(text)
    minutes = max(duration_sec / 60.0, 1e-9)
    wpm = words / minutes

    st.subheader("Key metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Duration", f"{duration_sec:.1f}s")
    col2.metric("Words", f"{words}")
    col3.metric("WPM", f"{wpm:.0f}")

    st.subheader("Transcript")
    st.write(text)
