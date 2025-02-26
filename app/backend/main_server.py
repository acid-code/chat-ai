import torch
from flask import Flask, request, jsonify, Response
import time
from flask_cors import CORS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import intel_npu_acceleration_library as npu
from intel_npu_acceleration_library.compiler import CompilerConfig
import soundfile as sf
from pydub import AudioSegment
import re
import os
import tempfile
import librosa
import numpy as np
import whisper
from faster_whisper import WhisperModel
from transformers import pipeline
from huggingface_hub import login
import torch
from deep_translator import GoogleTranslator


hugging_face_token = "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"


login(hugging_face_token)

from transformers import pipeline, TextStreamer, set_seed
import torch
import os


app = Flask(__name__)
CORS(app)

chat_model = "acid-code/finetuned_tinyllama"

print("Loading chat model...")
pipe = pipeline(
    "text-generation",
    model=chat_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_cache=False,  # Ensure the model is not loaded from cache
)
print("Creating streamer")
streamer = TextStreamer(pipe.tokenizer, skip_special_tokens=True, skip_prompt=True)
set_seed(42)
messages = [
    {
        "role": "system",
        "content": "You are a professional psychiatrist conducting a therapy session. Be empathetic, logical, and structured in your response.",
    },
]

# # Load fine-tuned model
# model_path = r"C:\Users\asaf2\Documents\Projects\Psych Model\models\hebrew_counseling_model_finetuned"
# model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained(model_path)

# # Optimize model for Intel NPU
# compiler_conf = CompilerConfig(dtype=torch.float32)
# model = npu.compile(model, compiler_conf)

# Load Whisper models
print("Loading Whisper models...")
general_model_name = "base"
device = "cuda" if torch.cuda.is_available() else "cpu"
general_model = whisper.load_model(general_model_name, device=device)
hebrew_model = WhisperModel("sivan22/faster-whisper-ivrit-ai-whisper-large-v2-tuned")
print("done!")

print("Creating Translators")
en_translator = GoogleTranslator(source="en", target="hebrew")
he_translator = GoogleTranslator(source="hebrew", target="en")
print("Done")


@app.route("/chat", methods=["POST"])
def chat():
    print("Received")
    data = request.json
    print(data)
    user_input = data.get("input", "")
    print(user_input)
    tranlated = False
    if is_hebrew(user_input):
        tranlated = True
        user_input = he_translator.translate(user_input)
    messages.append({"role": "user", "content": user_input})

    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    def generate():
        out = pipe(
            prompt,
            max_new_tokens=256,  # ✅ Lower max tokens for speed
            do_sample=True,
            temperature=0.7,
            top_k=40,  # ✅ Reduce k for faster decoding
            top_p=0.9,  # ✅ Lower top_p for better performance
            repetition_penalty=1.02,  # ✅ Avoid repetition
            use_cache=True,  # ✅ Enable KV caching
        )
        response = out[0]["generated_text"].split("<|assistant|>")[-1].strip()
        messages.append({"role": "assistant", "content": response})
        if tranlated:
            response = en_translator.translate(response)
        for char in response:  # ✅ Stream by words
            yield char
            time.sleep(0.02)  # ✅ Less delay for smoother response

    print("returning response object")
    return Response(generate(), content_type="text/plain")


def is_hebrew(text):
    return bool(re.search(r"[\u0590-\u05FF]", text))


def transcribe_with_general_model(audio_file_path):
    audio = AudioSegment.from_file(audio_file_path)
    audio_numpy = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    audio_numpy = librosa.resample(
        audio_numpy, orig_sr=audio.frame_rate, target_sr=16000
    )
    print("refactored file")
    temp_file_name = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            temp_file_name = tmpfile.name
            sf.write(temp_file_name, audio_numpy, 16000)
        print("sending for model")
        transcription_result = general_model.transcribe(temp_file_name)
        print("got result: ", transcription_result)
        return transcription_result["text"]
    finally:
        if temp_file_name:
            os.remove(temp_file_name)


def transcribe_with_hebrew_model(audio_file_path):
    audio = AudioSegment.from_file(audio_file_path)
    audio_numpy = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    audio_numpy = librosa.resample(
        audio_numpy, orig_sr=audio.frame_rate, target_sr=16000
    )
    print("refactored file")
    temp_file_name = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            temp_file_name = tmpfile.name
            sf.write(temp_file_name, audio_numpy, 16000)
        print("sending for model")
        segemnts, _ = hebrew_model.transcribe(temp_file_name, language="he")
        print("got result: ", segemnts)
        return ". ".join([segment.text for segment in segemnts])
    finally:
        if temp_file_name:
            os.remove(temp_file_name)


@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    try:
        # Get the audio file
        audio_file = request.files["audio"]
        audio = audio_file.read()

        # Optionally specify the language (this can be dynamic based on the frontend selection)
        language = request.form.get(
            "language", "en"
        )  # Default to English if no language is passed

        file_path = (
            r"C:\Users\asaf2\Documents\Projects\Psych Model\app\backend\audio.wav"
        )
        with open(file_path, "wb") as f:
            # Write the audio data to the temporary file
            f.write(audio)

        # Convert the audio file to WAV format using pydub and resample to 16000 Hz
        # audio_segment = AudioSegment.from_file(file_path)
        # audio_segment = audio_segment.set_frame_rate(16000)
        # audio_segment.export(file_path, format="wav")

        # Transcribe the audio based on the specified language
        if language.lower() == "hebrew":
            print("chose hebrew")
            transcription = transcribe_with_hebrew_model(file_path)
        else:
            print("chose default")
            transcription = transcribe_with_general_model(file_path)
        print("transcription: ", transcription)
        # Return the transcribed text
        return jsonify({"text": transcription})

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)
