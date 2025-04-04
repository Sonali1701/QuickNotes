import streamlit as st
import subprocess
import os
from faster_whisper import WhisperModel
from transformers import pipeline

# -------------- Load Models ---------------- #
@st.cache_resource
def load_models():
    st.info("⏳ Loading models (first time may take a minute)...")
    whisper_model = WhisperModel("base")
    summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")
    qa_generator = pipeline("text2text-generation", model="valhalla/t5-small-qa-qg-hl")
    return whisper_model, summarizer, qa_generator

whisper_model, summarizer, qa_generator = load_models()

# -------------- Utility Functions ---------------- #

def convert_video_to_audio(video_path):
    audio_path = "temp_audio.wav"
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ]
    subprocess.run(command, check=True)
    return audio_path

def transcribe_audio(path):
    segments, _ = whisper_model.transcribe(path)
    text = ""
    for segment in segments:
        text += segment.text + " "
    return text.strip()

def summarize_text(text):
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    full_summary = ""
    for chunk in chunks:
        result = summarizer(chunk, max_length=60, min_length=20, do_sample=False)[0]['summary_text']
        full_summary += "- " + result.strip() + "\n"
    return full_summary.strip()

def generate_questions(text):
    highlight_prompt = "highlight: " + text.strip()
    result = qa_generator(highlight_prompt, max_length=64)
    questions = [qa['generated_text'] for qa in result]
    return "\n\n".join(questions)

# -------------- Streamlit UI ---------------- #

st.set_page_config(page_title="LectureLens", layout="centered")
st.title("🎓 LectureLens – AI-Powered Lecture & Meeting Summarizer")
st.markdown("Upload your lecture or meeting recording and get a **concise summary and Q&A** without using any paid APIs!")

uploaded_file = st.file_uploader("Upload Audio/Video File", type=["mp3", "wav", "m4a", "mp4", "mov"])

if uploaded_file is not None:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ File uploaded successfully!")

    ext = uploaded_file.name.split('.')[-1]
    path = uploaded_file.name

    if ext in ["mp4", "mov"]:
        st.info("🎞 Extracting audio from video using FFmpeg...")
        path = convert_video_to_audio(uploaded_file.name)

    st.info("🔊 Transcribing audio using Whisper...")
    transcript = transcribe_audio(path)
    st.text_area("📝 Transcript", transcript, height=300)

    if st.button("🚀 Generate Summary & Q&A"):
        st.info("✍️ Generating summary...")
        summary = summarize_text(transcript)
        st.success("✅ Summary ready!")
        st.text_area("📌 Summary", summary, height=250)

        st.info("❓ Generating Q&A...")
        questions = generate_questions(transcript)
        st.success("✅ Questions ready!")
        st.text_area("🧠 Quiz / Q&A", questions, height=250)
