import tempfile
from PyQt6.QtCore import QThreadPool, QTimer, QThread
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QLineEdit, QProgressBar, QSystemTrayIcon, QMenu
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QIcon
from ollama import chat
from worker import Worker
from piper import PiperVoice
import wave
import os
import time
import soundfile as sf
import sounddevice as sd
from faster_whisper import WhisperModel

class STT:
    def __init__(self, model_size="base"):
        """
        Args:
            model_size (str): faster-whisper model.
        """
        print(f"Loading faster-whisper model: {model_size}")
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("Model loaded")
    
    def transcribe(self, audio_file_path: str) -> str:
        if not audio_file_path:
            return ""
        
        segments, info = self.model.transcribe(audio_file_path, language="en")
        full_text = [segment.text for segment in segments]
        return "".join(full_text).strip()

class TTS:
    def __init__(self):
        self.voice = PiperVoice.load("assets/voice/en_US-hfc_female-medium.onnx", "assets/voice/en_US-hfc_female-medium.onnx.json")
    
    def speak(self, text: str):
        with tempfile.NamedTemporaryFile(dir="tmp", suffix=".wav", delete=False) as temp_audio:
            audio_file_path = temp_audio.name
        
        try:
            with wave.open(audio_file_path, "wb") as wav_file:
                self.voice.synthesize_wav(text, wav_file)

            data, fs = sf.read(audio_file_path)
            sd.play(data, fs)
            
            while sd.get_stream() is not None and sd.get_stream().active:
                time.sleep(0.1)
        finally:
            try:
                os.remove(audio_file_path)
            except OSError:
                pass

    def stop(self):
        sd.stop()

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
        self.stt = None

        layout = QVBoxLayout()

        self.progress = QProgressBar()
        self.input = QLineEdit()
        button = QPushButton("Send")
        button.pressed.connect(self.send_message)
        button2 = QPushButton("Speak")
        button2.pressed.connect(self.speak_response)
        button3 = QPushButton("Record")
        button3.pressed.connect(self.record)
        self.label_response = QLabel("Waiting for input...")

        layout.addWidget(self.progress)
        layout.addWidget(self.input)
        layout.addWidget(button)
        layout.addWidget(button2)
        layout.addWidget(button3)
        layout.addWidget(self.label_response)

        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)

        self.threadpool = QThreadPool()

        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.recurring_timer)
        self.timer.start()
        
        self.init_stt()

        QApplication.instance().aboutToQuit.connect(self.cleanup)

    def init_stt(self):
        init_stt_worker = Worker(STT)
        init_stt_worker.signals.error.connect(self.handle_error)
        self.threadpool.start(init_stt_worker)

    def cleanup(self):
        print('cleanup')
        self.tts.stop()
        self.threadpool.waitForDone()

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

    def send_message(self, user_input: str = None):
        if not user_input:
            user_input = self.input.text()
        worker = Worker(
            self.model.query, user_input
        )
        worker.signals.error.connect(self.handle_error)
        worker.signals.result.connect(self.handle_response)
        worker.signals.finished.connect(self.handle_finished)
        self.threadpool.start(worker)

    def speak_response(self, text: str):
        worker = Worker(self.tts.speak, text)
        worker.signals.error.connect(self.handle_error)
        worker.signals.finished.connect(self.handle_finished)
        self.threadpool.start(worker)

    def handle_record_response(self, s):
        self.label_response.setText(s)
        self.send_message(s)
    
    def record(self):
        worker = Worker(self.stt.transcribe, "tmp/test.wav")
        worker.signals.error.connect(self.handle_error)
        worker.signals.result.connect(self.handle_record_response)
        worker.signals.finished.connect(self.handle_finished)
        self.threadpool.start(worker)

app = QApplication([])
window = MainWindow()
window.show()
app.exec()