import sys
import speech_recognition as sr
import torch
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from gtts import gTTS
import os
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QWidget,
    QPushButton,
)
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
import intel_npu_acceleration_library as npu
from intel_npu_acceleration_library.compiler import CompilerConfig

# Load Hebrew AI Model
model_path = (
    r"C:\Users\asaf2\Documents\Projects\Psych Model\models\hebrew_counseling_model"
)
tokenizer = MT5Tokenizer.from_pretrained(model_path)
model = MT5ForConditionalGeneration.from_pretrained(model_path)

# Optimize model for Intel NPU
compiler_conf = CompilerConfig(dtype=torch.float32, training=True)
model = npu.compile(model, compiler_conf)

# Speech Recognizer
recognizer = sr.Recognizer()
microphone = sr.Microphone()


class ListenThread(QThread):
    recognized = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._running = True

    def run(self):
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
        if not self._running:
            return
        try:
            text = recognizer.recognize_google(audio, language="he-IL")
            self.recognized.emit(text)
        except sr.UnknownValueError:
            self.recognized.emit("Could not understand audio")
        except sr.RequestError as e:
            self.recognized.emit(f"Could not request results; {e}")

    def stop(self):
        self._running = False


class AI_Counselor(QMainWindow):
    def __init__(self):
        super().__init__()

        # UI Setup
        self.setWindowTitle("Hebrew AI Counselor")
        self.setGeometry(100, 100, 400, 200)

        self.label = QLabel("Press the button to start listening", self)
        self.label.setGeometry(50, 50, 300, 50)

        self.button = QPushButton("Start Listening", self)
        self.button.setGeometry(150, 100, 100, 50)
        self.button.clicked.connect(self.toggle_listening)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.button)

        self.container = QWidget()
        self.container.setLayout(self.layout)
        self.setCentralWidget(self.container)

        self.listening = False
        self.listen_thread = ListenThread()
        self.listen_thread.recognized.connect(self.process_recognition)

    def toggle_listening(self):
        if self.listening:
            self.listening = False
            self.button.setText("Start Listening")
            self.listen_thread.stop()
        else:
            self.listening = True
            self.button.setText("Stop Listening")
            self.listen_and_respond()

    def listen_and_respond(self):
        if self.listening:
            self.label.setText("Listening...")
            self.listen_thread = ListenThread()
            self.listen_thread.recognized.connect(self.process_recognition)
            self.listen_thread.start()

    def process_recognition(self, text):
        self.label.setText(f"Recognized: {text}")
        if text.startswith("Could not"):
            self.listening = False
            self.button.setText("Start Listening")
            return

        try:
            # Generate response
            print(text)
            input_ids = tokenizer.encode(text, return_tensors="pt")
            outputs = model.generate(input_ids)
            print(outputs)
            if not outputs:
                raise ValueError("No output from model")
            print(outputs[0])
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(response)
            self.label.setText(f"Response: {response}")

            # Convert response to speech
            tts = gTTS(text=response, lang="he")
            tts.save("response.mp3")
            os.system("start response.mp3")
        except Exception as e:
            self.label.setText(f"Error: {e}")

        self.listening = False
        self.button.setText("Start Listening")


if __name__ == "__main__":
    # app = QApplication(sys.argv)
    # window = AI_Counselor()
    # window.show()
    # sys.exit(app.exec_())
    # Debugging: Check the output of the model
    def debug_model_output(text):
        input_ids = tokenizer.encode(text, return_tensors="pt")
        outputs = model.generate(input_ids)
        print("Input IDs:", input_ids)
        print("Outputs:", outputs)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Response:", response)

    # Test the model with a sample input
    debug_model_output("מה שלומך?")
