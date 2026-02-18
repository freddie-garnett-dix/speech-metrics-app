import io
import streamlit as st
from openai import OpenAI
from pydub import AudioSegment, silence

st.set_page_config(page_title="PowerLoom Speech Metrics", page_icon="🎤")

st.title("🎤 PowerLoom Speech Metrics App")
st.write("Upload an audio file to analyse your speaking performance.")

audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

# ---- Helpers ----
def seconds(ms: int) -> float:
    return ms / 1000.0

def analyse_pauses(
    audio: AudioSegment,
    min_silence_len_ms: int = 500,
    silence_thresh_dbfs_offset: int = 16,
):
    """
    Detect silent sections in the audio and return pause list + summary metrics.
    - min_silence_len_ms: minimum silence duration to count as a pause
    - silence_thresh_dbfs_offset: how much quieter than average to count as silence
    """
    # Dynamic threshold: "silence" is (average loudness - offset)
    silence_thresh = audio.dBFS - silence_thresh_dbfs_offset

    silent_ranges = silence.detect_silence(
        audio,
        min_silence_len=min_silence_len_ms,
        silence_thresh=silence_thresh,
    )

    # Convert to structured list
    pauses = []
    for start_ms, end_ms in silent_ranges:
        pauses.append(
            {
                "start_s": round(seconds(start_ms), 2),
                "end_s": round(seconds(end_ms), 2),
                "duration_s": round(seconds(end_ms - start_ms), 2),
            }
        )

    total_audio_s = seconds(len(audio))
    total_silence_s = round(sum(p["duration_s"] for p in pauses), 2)
    silence_pct = round((total_silence_s / total_audio_s) * 100, 1) if total_audio_s > 0 else 0.0

    def count_over(x: float) -> int:
        return sum(1 for p in pauses if p["duration_s"] >= x)

    longest_pause = round(max((p["duration_s"] for p in pauses), default=0.0), 2)
    avg_pause = round((total_silence_s / len(pauses)), 2) if pauses else 0.0

    summary = {
        "total_audio_s": round(total_audio_s, 2),
        "total_silence_s": total_silence_s,
        "silence_pct": silence_pct,
        "num_pauses": len(pauses),
        "avg_pause_s": avg_pause,
        "longest_pause_s": longest_pause,
        "pauses_ge_0_5s": count_over(0.5),
        "pauses_ge_1_0s": count_over(1.0),
        "pauses_ge_2_0s": count_over(2.0),
    }

    return pauses, summary

# ---- Main flow ----
if audio_file is not None:
    st.success("File uploaded successfully!")
    st.write("File name:", audio_file.name)

    # Read bytes once
    audio_bytes = audio_file.read()

    # ---- Pause analysis (audio-based, accurate) ----
    st.subheader("Pause detection settings")
    min_silence_len_ms = st.slider("Minimum pause length (ms)", 200, 3000, 500, 100)
    silence_offset = st.slider("Silence threshold sensitivity (dB below average)", 8, 30, 16, 1)
    st.caption("Tip: if it misses pauses, reduce the dB offset. If it detects too many, increase it.")

    with st.spinner("Analysing pauses from audio..."):
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        pauses, summary = analyse_pauses(
            audio,
            min_silence_len_ms=min_silence_len_ms,
            silence_thresh_dbfs_offset=silence_offset,
        )

    st.subheader("Pause metrics (audio-based – accurate)")
    st.write(
        {
            "Total audio length (s)": summary["total_audio_s"],
            "Total silence (s)": summary["total_silence_s"],
            "Silence %": summary["silence_pct"],
            "Number of pauses": summary["num_pauses"],
            "Average pause (s)": summary["avg_pause_s"],
            "Longest pause (s)": summary["longest_pause_s"],
            "Pauses ≥ 0.5s": summary["pauses_ge_0_5s"],
            "Pauses ≥ 1.0s": summary["pauses_ge_1_0s"],
            "Pauses ≥ 2.0s": summary["pauses_ge_2_0s"],
        }
    )

    if pauses:
        st.subheader("Pause list (each detected pause)")
        st.dataframe(pauses, use_container_width=True)
    else:
        st.info("No pauses detected with current settings. Try lowering the dB offset or reducing minimum pause length.")

    # ---- Transcription (language-based) ----
    st.divider()
    st.subheader("Transcription")

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
                file=("audio.m4a", audio_bytes),  # name doesn't matter much; bytes do
                response_format="json",
            )

        st.subheader("Transcript")
        st.write(transcript["text"])

    except Exception as e:
        st.error("Transcription failed.")
        st.exception(e)
