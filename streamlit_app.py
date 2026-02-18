import streamlit as st
from openai import OpenAI
from pydub import AudioSegment
from pydub.silence import detect_silence
import io

st.set_page_config(page_title="PowerLoom Speech Metrics", page_icon="🎤")

st.title("🎤 PowerLoom Speech Metrics App")

st.write("Upload a WAV audio file to analyse speaking performance and pauses.")

st.caption("⚠️ For pause detection, please upload a WAV file (MP3/M4A need ffmpeg which isn't available on Streamlit Cloud).")

# Upload audio file
audio_file = st.file_uploader("Upload WAV audio", type=["wav"])

if audio_file is not None:

    st.success("File uploaded successfully!")

    # Read bytes once
    audio_bytes = audio_file.read()

    # ================
    # TRANSCRIPTION
    # ================

    if "OPENAI_API_KEY" not in st.secrets:
        st.error("Missing OPENAI_API_KEY in Streamlit Secrets")
        st.stop()

    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    try:
        with st.spinner("Transcribing audio..."):
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=("audio.wav", audio_bytes),
                response_format="json",
            )

        st.subheader("Transcript")
        st.write(transcript["text"])

    except Exception as e:
        st.error("Transcription failed")
        st.exception(e)

    # ================
    # PAUSE ANALYSIS
    # ================

    try:
        with st.spinner("Analysing pauses..."):

            # Load WAV
            audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))

            duration_sec = len(audio) / 1000

            # Detect silence
            silences = detect_silence(
                audio,
                min_silence_len=400,   # pause threshold (ms)
                silence_thresh=audio.dBFS - 16
            )

            # Convert to seconds
            pauses = [(start/1000, end/1000) for start, end in silences]
            pause_lengths = [(end-start) for start, end in pauses]

            total_pause = sum(pause_lengths)
            long_pauses = [p for p in pause_lengths if p > 1.5]

        st.subheader("Pause Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total duration", f"{duration_sec:.1f}s")
        col2.metric("Total silence", f"{total_pause:.1f}s")
        col3.metric("Number of pauses", len(pauses))

        st.metric("Long pauses (>1.5s)", len(long_pauses))

        if pauses:
            st.subheader("Pause Breakdown")
            st.table([
                {"start": round(s,2), "end": round(e,2), "length": round(e-s,2)}
                for s, e in pauses[:20]
            ])
        else:
            st.info("No pauses detected.")

    except Exception as e:
        st.error("Pause analysis failed")
        st.exception(e)
