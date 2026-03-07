import subprocess
import tempfile
from PyQt6.QtCore import QThreadPool, QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QLineEdit, QProgressBar, QSystemTrayIcon, QMenu
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QIcon
from ollama import chat
from worker import Worker
from piper import PiperVoice
import wave
import os
import soundfile
import sounddevice

class TTS:
    def __init__(self):
        self.voice = PiperVoice.load("assets/voice/en_US-hfc_female-medium.onnx", "assets/voice/en_US-hfc_female-medium.onnx.json")
        self.audio_file_path = None
        self.process = None
    
    def speak(self, text: str):
        with tempfile.NamedTemporaryFile(dir="tmp", suffix=".wav", delete=False) as temp_audio:
            self.audio_file_path = temp_audio.name
        
        try:
            with wave.open(self.audio_file_path, "wb") as wav_file:
                self.voice.synthesize_wav(text, wav_file)

            self.process = subprocess.Popen(["aplay", "-q", self.audio_file_path])
            self.process.wait()
        finally:
            self.remove_audio_file()

    def stop(self):
        if self.process is not None:
            self.process.terminate()
    
    def remove_audio_file(self):
        print('remove_audio_file')
        if self.audio_file_path and os.path.exists(self.audio_file_path):
            try:
                os.remove(self.audio_file_path)
            except OSError:
                pass
            self.audio_file_path = None

# def speak_text(text: str):
#     model_path = "assets/voice/en_US-hfc_female-medium.onnx"
#     config_path = "assets/voice/en_US-hfc_female-medium.onnx.json"

#     voice = PiperVoice.load(model_path, config_path)

#     with tempfile.NamedTemporaryFile(dir="tmp", suffix=".wav", delete=False) as temp_audio:
#         audio_path = temp_audio.name
    
#     with wave.open(audio_path, "wb") as wav_file:
#         voice.synthesize_wav(text, wav_file)

#     data, fs = soundfile.read(audio_path)
#     sounddevice.play(data, fs)
#     sounddevice.wait()
    
#     os.remove(audio_path)

class ModelInterface:
    def __init__(self):
        self.messages = []
        self.model = 'llama3.2'

    def query(self, input: str) -> str:
        self.messages.append({'role': 'user', 'content': input})
        response = chat(
            model=self.model, 
            messages=[{'role': 'user', 'content': input}], 
        )
        self.messages.append({'role': 'assistant', 'content': response.message.content})
        return response.message.content

class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.setWindowFlags(Qt.WindowType.CustomizeWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(QIcon("assets/icon.png"))
        tray_menu = QMenu()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(QApplication.instance().quit)
        tray_menu.addAction(exit_action)
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

        self.counter = 0
        self.model = ModelInterface()
        self.tts = TTS()

        layout = QVBoxLayout()

        self.progress = QProgressBar()
        self.input = QLineEdit()
        button = QPushButton("Send")
        button.pressed.connect(self.send_message)
        button2 = QPushButton("Speak")
        button2.pressed.connect(self.speak_response)
        self.label_response = QLabel("Waiting for input...")

        layout.addWidget(self.progress)
        layout.addWidget(self.input)
        layout.addWidget(button)
        layout.addWidget(button2)
        layout.addWidget(self.label_response)

        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)

        self.threadpool = QThreadPool()

        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.recurring_timer)
        self.timer.start()

        QApplication.instance().aboutToQuit.connect(self.cleanup)

    def cleanup(self):
        print('cleanup')
        self.tts.stop()

    def recurring_timer(self):
        self.counter += 1
        self.progress.setValue(self.counter)
        if self.counter >= 100:
            self.counter = 0

    def handle_response(self, s):
        self.label_response.setText(s)
        self.speak_response(s)

    def handle_error(self, data):
        print(data)

    def handle_finished(self):
        print("Done.")

    def send_message(self):
        worker = Worker(
            self.model.query, self.input.text()
        )
        worker.signals.error.connect(self.handle_error)
        worker.signals.result.connect(self.handle_response)
        worker.signals.finished.connect(self.handle_finished)
        self.threadpool.start(worker)

    def speak_response(self, text: str):
        worker = Worker(
            self.tts.speak, text
        )
        worker.signals.error.connect(self.handle_error)
        worker.signals.finished.connect(self.handle_finished)
        self.threadpool.start(worker)

app = QApplication([])
window = MainWindow()
window.show()
app.exec()