import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from faster_whisper import WhisperModel
from transformers import pipeline
from pydub import AudioSegment
import os
import subprocess

# Load models once
whisper_model = WhisperModel("base")
summarizer = pipeline("summarization", model="t5-small")
qa_generator = pipeline("text2text-generation", model="valhalla/t5-small-qa-qg-hl")

def extract_audio_with_pydub(video_path):
    audio_path = "temp_audio.wav"
    AudioSegment.from_file(video_path).export(audio_path, format="wav")
    return audio_path

def transcribe_audio(path):
    segments, _ = whisper_model.transcribe(path)
    text = " ".join([segment.text for segment in segments])
    return text

def summarize_text(text):
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    summary = ""
    for chunk in chunks:
        result = summarizer(chunk, max_length=60, min_length=20, do_sample=False)
        summary += "- " + result[0]['summary_text'] + "\n"
    return summary

def generate_questions(text):
    highlight_prompt = "highlight: " + text.strip()
    result = qa_generator(highlight_prompt, max_length=64)
    return "\n".join([r['generated_text'] for r in result])

def process_file():
    file_path = filedialog.askopenfilename(filetypes=[
        ("Audio/Video Files", "*.mp3 *.wav *.m4a *.mp4 *.mov")
    ])
    if not file_path:
        return

    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, "🔄 Processing...\n")

    ext = file_path.split('.')[-1].lower()
    if ext in ["mp4", "mov", "m4a"]:
        output_text.insert(tk.END, "🎞 Extracting audio from video...\n")
        file_path = extract_audio_with_pydub(file_path)

    output_text.insert(tk.END, "🔊 Transcribing audio...\n")
    transcript = transcribe_audio(file_path)
    output_text.insert(tk.END, "📝 Transcript:\n" + transcript + "\n\n")

    output_text.insert(tk.END, "✍️ Generating Summary...\n")
    summary = summarize_text(transcript)
    output_text.insert(tk.END, "📌 Summary:\n" + summary + "\n\n")

    output_text.insert(tk.END, "❓ Generating Q&A...\n")
    questions = generate_questions(transcript)
    output_text.insert(tk.END, "🧠 Q&A:\n" + questions + "\n")

    messagebox.showinfo("Done", "✅ Lecture processed successfully!")

# -------- GUI -------- #
root = tk.Tk()
root.title("LectureLens - Lecture Summarizer")
root.geometry("800x600")

frame = tk.Frame(root)
frame.pack(pady=10)

tk.Button(frame, text="📁 Upload Lecture", font=("Arial", 14), command=process_file).pack()

output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=30, font=("Courier", 10))
output_text.pack(pady=10)

root.mainloop()
