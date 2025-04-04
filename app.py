from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import torch
import whisper
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Whisper model for audio transcription
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper model loaded.")

# Load T5 QA/QG model with correct tokenizer
print("Loading QA Generator model...")
model_name = "valhalla/t5-small-qa-qg-hl"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
qa_generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
print("QA Generator model loaded.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Transcribe using Whisper
    print("Transcribing audio...")
    result = whisper_model.transcribe(file_path)
    transcription = result['text']
    print("Transcription completed.")

    return jsonify({'transcription': transcription})

@app.route('/generate_qa', methods=['POST'])
def generate_qa():
    data = request.get_json()
    text = data.get('text')

    if not text:
        return jsonify({'error': 'No text provided'})

    print("Generating QA pairs...")
    input_text = f"highlight: {text}"
    result = qa_generator(input_text, max_length=128, do_sample=False)
    output_text = result[0]['generated_text']
    print("QA generation completed.")

    return jsonify({'qa': output_text})

# 🔥 NEW route that handles everything together
@app.route('/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Step 1: Transcribe
    print("Transcribing audio...")
    result = whisper_model.transcribe(file_path)
    transcription = result['text']
    print("Transcription done.")

    # Step 2: Generate QA
    print("Generating QA pairs...")
    input_text = f"highlight: {transcription}"
    qa_result = qa_generator(input_text, max_length=128, do_sample=False)
    qa_output = qa_result[0]['generated_text']
    print("QA generation done.")

    return jsonify({
        'transcription': transcription,
        'qa': qa_output
    })

if __name__ == '__main__':
    app.run(debug=True)
