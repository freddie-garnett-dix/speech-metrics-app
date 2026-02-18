import io
import re
import wave

import streamlit as st
from openai import OpenAI


# -----------------------------
# Page + header
# -----------------------------
st.set_page_config(page_title="Speech Metrics", page_icon="🎤")

st.title("🎤 Speech Metrics (Prototype)")
st.caption("Build: v1.4 – transcript + duration + WPM + fillers% + repetition% (WAV only)")
st.write("Upload a **WAV** file and we’ll transcribe it + calculate a few basic metrics.")


# -----------------------------
# Helpers
# -----------------------------
def wav_duration_seconds(wav_bytes: bytes) -> float:
    """Compute duration from WAV headers (no ffmpeg needed)."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        if rate == 0:
            return 0.0
        return frames / float(rate)


def tokenise_words(text: str) -> list[str]:
    # Simple word tokeniser (letters + apostrophes)
    return re.findall(r"[a-zA-Z']+", text.lower())


FILLER_SINGLE_WORDS = {
    "um", "uh", "erm", "er", "emm", "err",  # added emm/err
    "ah", "like", "basically", "actually", "literally",
}

FILLER_PHRASES = [
    "you know",
    "sort of",
    "kind of",
]


def count_fillers(text: str) -> tuple[int, int, int]:
    """
    Returns:
      filler_word_equivalents, single_word_fillers, phrase_hits_word_equivalents
    """
    lower = text.lower()
    words = tokenise_words(lower)

    single_hits = sum(1 for w in words if w in FILLER_SINGLE_WORDS)

    # Phrase hits – count occurrences and convert to "word equivalents"
    phrase_word_equiv = 0
    for phrase in FILLER_PHRASES:
        # Count non-overlapping occurrences
        hits = len(re.findall(rf"\b{re.escape(phrase)}\b", lower))
        phrase_word_equiv += hits * len(phrase.split())

    filler_word_equiv_total = single_hits + phrase_word_equiv
    return filler_word_equiv_total, single_hits, phrase_word_equiv


def count_immediate_repetitions(text: str) -> int:
    """Counts immediate repeated words: 'because because'."""
    words = tokenise_words(text)
    repeats = 0
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            repeats += 1
    return repeats


def get_transcript_text(transcript_obj) -> str:
    """Works whether SDK returns a dict-like object or a model with .text."""
    # Most common: transcript.text
    t = getattr(transcript_obj, "text", None)
    if isinstance(t, str) and t.strip():
        return t

    # Sometimes dict-like
    try:
        t2 = transcript_obj.get("text", "")
        if isinstance(t2, str):
            return t2
    except Exception:
        pass

    return ""


# -----------------------------
# Upload
# -----------------------------
audio_file = st.file_uploader("Upload WAV audio", type=["wav"])

if audio_file is not None:
    st.success("File uploaded successfully!")

    wav_bytes = audio_file.read()
    duration_sec = wav_duration_seconds(wav_bytes)

    # -----------------------------
    # Transcription
    # -----------------------------
    if "OPENAI_API_KEY" not in st.secrets:
        st.error(
            "Missing OPENAI_API_KEY in Streamlit Secrets.\n\n"
            "Go to **Manage app → Settings → Secrets** and add:\n"
            'OPENAI_API_KEY="sk-..."'
        )
        st.stop()

    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    with st.spinner("Transcribing audio..."):
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=("audio.wav", wav_bytes),
            response_format="json",
        )

    text = get_transcript_text(transcript)

    # -----------------------------
    # Metrics (no interpretation)
    # -----------------------------
    words = tokenise_words(text)
    word_count = len(words)
    minutes = duration_sec / 60.0 if duration_sec > 0 else 0.0
    wpm = (word_count / minutes) if minutes > 0 else 0.0

    filler_word_equiv, filler_single_hits, filler_phrase_word_equiv = count_fillers(text)
    filler_pct = (filler_word_equiv / word_count * 100.0) if word_count > 0 else 0.0

    repetition_count = count_immediate_repetitions(text)
    repetition_pct = (repetition_count / word_count * 100.0) if word_count > 0 else 0.0

    st.subheader("Key metrics")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Duration", f"{duration_sec:.1f}s")
    c2.metric("Words", f"{word_count}")
    c3.metric("WPM", f"{wpm:.0f}")
    c4.metric("Filler %", f"{filler_pct:.1f}%")
    c5.metric("Repetition %", f"{repetition_pct:.1f}%")

    with st.expander("Metric breakdown (counts)"):
        st.write(
            {
                "filler_word_equivalents_total": filler_word_equiv,
                "single_word_fillers_count": filler_single_hits,
                "phrase_filler_word_equivalents": filler_phrase_word_equiv,
                "immediate_repetitions_count": repetition_count,
            }
        )

    st.subheader("Transcript")
    st.write(text if text else "(No transcript text returned.)")
