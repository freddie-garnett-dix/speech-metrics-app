import io
import re
import wave
import audioop
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Speech Metrics", page_icon="🎤")

st.title("🎤 Speech Metrics (Prototype)")
# Change this line each deploy so you can see the update instantly
st.caption("Build: v1.2 – core metrics only (WPM, pauses%, fillers%)")

st.write("Upload a **WAV** file. (WAV only keeps pause detection reliable on Streamlit Cloud.)")

audio_file = st.file_uploader("Upload WAV audio", type=["wav"])

FILLER_WORDS = [
    "um", "uh", "er", "ah", "like", "basically", "actually", "literally", "right"
]
FILLER_PHRASES = [
    "you know", "sort of", "kind of"
]

def tokenise_words(text: str):
    return re.findall(r"[a-z']+", text.lower())

def count_fillers(text: str):
    t = text.lower()
    words = tokenise_words(t)
    total_words = len(words)

    filler_count = 0

    # single-word fillers
    filler_set = set(FILLER_WORDS)
    filler_count += sum(1 for w in words if w in filler_set)

    # phrase fillers (count occurrences with word boundaries)
    for phrase in FILLER_PHRASES:
        filler_count += len(re.findall(r"\b" + re.escape(phrase) + r"\b", t))

    filler_pct = (filler_count / total_words * 100) if total_words else 0.0
    return total_words, filler_count, filler_pct

def analyse_pauses_from_wav_bytes(
    wav_bytes: bytes,
    window_ms: int = 20,
    silence_rms_threshold: int = 200,
    min_pause_ms: int = 200,
    long_pause_ms: int = 2000,
):
    """
    Pure-Python pause detection from WAV (no ffmpeg).
    Uses RMS energy per window; windows below threshold are treated as silence.

    Returns:
      duration_s, pause_s, pause_pct, long_pause_count
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()

        duration_s = nframes / float(framerate)

        # frames per analysis window
        frames_per_window = int(framerate * (window_ms / 1000.0))
        if frames_per_window <= 0:
            frames_per_window = 1

        silent_windows = []
        total_windows = 0

        wf.rewind()
        while True:
            frames = wf.readframes(frames_per_window)
            if not frames:
                break
            total_windows += 1

            # Convert to mono RMS if stereo: audioop.rms works on raw bytes regardless of channels,
            # but stereo RMS tends to be higher; we keep a single threshold for simplicity.
            rms = audioop.rms(frames, sampwidth)

            silent_windows.append(rms < silence_rms_threshold)

        # Convert silent windows into contiguous pause segments
        window_s = window_ms / 1000.0
        min_pause_windows = max(1, int(min_pause_ms / window_ms))
        long_pause_windows = max(1, int(long_pause_ms / window_ms))

        pause_s = 0.0
        long_pause_count = 0

        run = 0
        for is_silent in silent_windows + [False]:  # sentinel to flush last run
            if is_silent:
                run += 1
            else:
                if run >= min_pause_windows:
                    pause_len_s = run * window_s
                    pause_s += pause_len_s
                    if run >= long_pause_windows:
                        long_pause_count += 1
                run = 0

        pause_pct = (pause_s / duration_s * 100) if duration_s > 0 else 0.0
        return duration_s, pause_s, pause_pct, long_pause_count

if audio_file is not None:
    wav_bytes = audio_file.read()

    # --- Pause metrics (audio-based, reliable on WAV) ---
    with st.spinner("Calculating pause metrics..."):
        duration_s, pause_s, pause_pct, long_pause_count = analyse_pauses_from_wav_bytes(
            wav_bytes,
            window_ms=20,
            silence_rms_threshold=200,  # tweak later if needed
            min_pause_ms=200,
            long_pause_ms=2000,
        )

    # --- Transcription (text-based) ---
    if "OPENAI_API_KEY" not in st.secrets:
        st.error('Missing OPENAI_API_KEY in Streamlit Secrets (Manage app → Settings → Secrets).')
        st.stop()

    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    with st.spinner("Transcribing audio..."):
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=("audio.wav", wav_bytes),
            response_format="json",
        )

    text = transcript.get("text", "")
    total_words, filler_count, filler_pct = count_fillers(text)

    # --- WPM ---
    minutes = duration_s / 60.0 if duration_s > 0 else 0
    wpm = (total_words / minutes) if minutes > 0 else 0.0

    # --- Output (metrics only, no interpretation) ---
    st.subheader("Key metrics")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Duration (s)", f"{duration_s:.1f}")
    c2.metric("Total words", f"{total_words}")
    c3.metric("WPM", f"{wpm:.0f}")
    c4.metric("Pause %", f"{pause_pct:.1f}%")
    c5.metric("Filler %", f"{filler_pct:.1f}%")

    st.caption("Pause % is calculated from the audio waveform (silence windows). Fillers are counted from the transcript.")

    st.subheader("Extra (small)")
    c6, c7 = st.columns(2)
    c6.metric("Total pause time (s)", f"{pause_s:.1f}")
    c7.metric("Long pauses (≥2.0s)", f"{long_pause_count}")

    st.subheader("Transcript")
    st.write(text)
