import re
from collections import Counter
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="PowerLoom Speech Metrics", page_icon="🎤")
st.title("🎤 PowerLoom Speech Metrics App")
st.write("Upload an audio file to analyse speaking performance (v1 – transcript metrics).")

audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])

FILLERS = [
    "um", "uh", "er", "ah", "like", "you know", "sort of", "kind of",
    "basically", "actually", "literally", "right", "so"
]

STOPWORDS = set([
    "the","a","an","and","or","but","if","then","so","because","as","of","to","in",
    "on","for","with","at","by","from","up","down","out","about","into","over",
    "after","before","between","through","during","without","within",
    "i","you","we","they","he","she","it","me","him","her","us","them",
    "my","your","our","their","this","that","these","those","is","are","was","were",
    "be","been","being","do","does","did","have","has","had","will","would","can",
    "could","should","may","might","must"
])

def normalise_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t

def tokenise_words(t: str):
    return re.findall(r"[a-z']+", t.lower())

def count_fillers(text: str):
    t = normalise_text(text)
    counts = {}
    for f in FILLERS:
        # phrase-aware counting
        pattern = r"\b" + re.escape(f) + r"\b"
        counts[f] = len(re.findall(pattern, t))
    total = sum(counts.values())
    return total, {k:v for k,v in counts.items() if v > 0}

def repetition(words):
    content = [w for w in words if w not in STOPWORDS]
    c = Counter(content)
    top = c.most_common(15)
    # repeated words = count beyond first occurrence
    repeated_excess = sum(max(0, n - 1) for _, n in c.items())
    return repeated_excess, top

def estimate_duration_seconds(uploaded_file):
    # Streamlit doesn't always provide duration. We'll show "Unknown" unless available.
    # For WAV we could parse headers, but let's keep v1 simple.
    return None

if audio_file is not None:
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("Missing OPENAI_API_KEY in Streamlit Secrets")
        st.stop()

    audio_bytes = audio_file.read()
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    with st.spinner("Transcribing audio..."):
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=(audio_file.name, audio_bytes),
            response_format="json",
            # OPTIONAL: try word timestamps (if supported). If not supported, it will just ignore / error.
            # timestamp_granularities=["word"]
        )

    text = transcript.get("text", "")
    words = tokenise_words(text)

    total_words = len(words)
    duration_sec = estimate_duration_seconds(audio_file)

    # Basic pace based on total audio duration (if known)
    wpm = None
    if duration_sec and duration_sec > 0:
        wpm = (total_words / (duration_sec / 60.0))

    filler_total, filler_breakdown = count_fillers(text)
    filler_pct = (filler_total / total_words * 100) if total_words else 0

    repeated_excess, top_repeated = repetition(words)

    # Simple sentence stats
    sentences = [s for s in re.split(r"[.!?]+", text.strip()) if s.strip()]
    sentence_count = len(sentences)
    avg_sentence_len = (total_words / sentence_count) if sentence_count else 0
    longest_sentence_len = max((len(tokenise_words(s)) for s in sentences), default=0)
    questions = text.count("?")

    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total words", f"{total_words}")
    c2.metric("Filler %", f"{filler_pct:.1f}%")
    c3.metric("Filler count", f"{filler_total}")
    if wpm is None:
        c4.metric("WPM", "—")
        st.caption("WPM requires audio duration. We can add duration extraction next.")
    else:
        c4.metric("WPM", f"{wpm:.0f}")

    st.subheader("Delivery metrics")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Sentences", f"{sentence_count}")
    d2.metric("Avg sentence length", f"{avg_sentence_len:.1f} words")
    d3.metric("Longest sentence", f"{longest_sentence_len} words")
    d4.metric("Questions", f"{questions}")

    st.subheader("Filler words breakdown")
    if filler_breakdown:
        st.table([{"filler": k, "count": v} for k, v in sorted(filler_breakdown.items(), key=lambda x: -x[1])])
    else:
        st.info("No filler words detected (based on the current filler list).")

    st.subheader("Repetition (top words, excluding common stopwords)")
    st.table([{"word": w, "count": n} for w, n in top_repeated])

    st.subheader("Transcript")
    st.write(text)
