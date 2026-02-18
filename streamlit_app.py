import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="PowerLoom Speech Metrics", page_icon="🎤")

st.title("🎤 PowerLoom Speech Metrics App")
st.write("Upload an audio file to analyse your speaking performance.")

# Upload audio file
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    st.success("File uploaded successfully!")
    st.write("File name:", audio_file.name)

    # Check secret exists
    if "OPENAI_API_KEY" not in st.secrets:
        st.error(
            "Missing OPENAI_API_KEY in Streamlit Secrets. "
            "Go to Manage app → Settings → Secrets and add:\n\n"
            'OPENAI_API_KEY="sk-..."'
        )
        st.stop()

    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    try:
        with st.spinner("Transcribing audio..."):
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )

        st.subheader("Transcript")
        st.write(transcript.text)

        # Show a small proof that word timestamps exist
        words = getattr(transcript, "words", None)
        if words:
            st.subheader("First 20 timed words (proof of timestamps)")
            preview = [
                {
                    "word": w.word,
                    "start_s": w.start,
                    "end_s": w.end,
                }
                for w in words[:20]
            ]
            st.table(preview)
        else:
            st.warning(
                "No word-level timestamps returned. "
                "If this persists, we can switch approach (or check model/params)."
            )

    except Exception as e:
        st.error("Transcription failed.")
        st.exception(e)
