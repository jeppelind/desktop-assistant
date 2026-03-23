import argparse
from PyQt6.QtCore import QThreadPool, QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QLineEdit, QProgressBar, QSystemTrayIcon, QMenu
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QIcon
from worker import Worker
from text_to_speech import TextToSpeech
from speech_to_text import SpeechToText
from llm_interface import LLMInterface
from enum import Enum

class AppState(Enum):
    INACTIVE = 1
    IDLE = 2
    LISTENING = 3
    WORKING = 4
    RESPONDING = 5

class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        whisper_model = kwargs.pop('whisper_model')

        super().__init__(*args, **kwargs)

        self._state = AppState.INACTIVE
        self.timer_listening = QTimer()
        self.timer_listening.timeout.connect(lambda: self.set_state(AppState.IDLE))
        self.timer_listening.setSingleShot(True)
        
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
        self.llm = LLMInterface()
        self.tts = TextToSpeech()
        self.stt = None

        layout = QVBoxLayout()

        self.progress = QProgressBar()
        self.label_state = QLabel(self._state.name)
        self.input = QLineEdit()
        button = QPushButton("Send")
        button.pressed.connect(lambda: self.send_message(self.input.text()))
        button2 = QPushButton("Listen")
        button2.pressed.connect(self.listen_in_background)
        button3 = QPushButton("Record")
        button3.pressed.connect(self.record)
        self.label_response = QLabel("Waiting for input...")
        button4 = QPushButton("Stop")
        button4.pressed.connect(self.stop_listening)

        layout.addWidget(self.progress)
        layout.addWidget(self.label_state)
        layout.addWidget(self.input)
        layout.addWidget(button)
        layout.addWidget(button2)
        layout.addWidget(button3)
        layout.addWidget(button4)
        layout.addWidget(self.label_response)

        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)

        self.threadpool = QThreadPool()

        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.recurring_timer)
        self.timer.start()
        
        self.init_stt(whisper_model)

        QApplication.instance().aboutToQuit.connect(self.cleanup)

    def init_stt(self, whisper_model: str):
        init_stt_worker = Worker(SpeechToText, whisper_model)
        init_stt_worker.signals.result.connect(self.handle_stt_loaded)
        init_stt_worker.signals.error.connect(self.handle_error)
        self.threadpool.start(init_stt_worker)

    def handle_stt_loaded(self, stt_instance):
        self.stt = stt_instance
        self.stt.signals.new_message.connect(self.handle_record_response)

    def cleanup(self):
        print('cleanup')
        self.tts.stop()
        self.threadpool.waitForDone()
    
    def state(self) -> AppState:
        return self._state
    
    def set_state(self, state: AppState):
        if state == AppState.LISTENING:
            self.stt.toggle_wake_word(False)
            self.timer_listening.start(5000) # Listen for user response for 5 seconds before going IDLE
        elif state == AppState.IDLE:
            self.stt.toggle_wake_word(True)
        elif state == AppState.RESPONDING:
            self.stt.toggle_wake_word(False)
        self._state = state
        self.label_state.setText(state.name)

    def recurring_timer(self):
        self.counter += 1
        self.progress.setValue(self.counter)
        if self.counter >= 100:
            self.counter = 0

    def handle_llm_response(self, s):
        self.label_response.setText(s)
        self.speak_response(s)

    def handle_error(self, data):
        print(data)

    def handle_finished(self):
        print("Done.")

    def send_message(self, user_input: str = None):
        print(f"Sending message: {user_input}")
        self.set_state(AppState.WORKING)
        self.timer_listening.stop()
        worker = Worker(self.llm.query, user_input)
        worker.signals.error.connect(self.handle_error)
        worker.signals.result.connect(self.handle_llm_response)
        worker.signals.finished.connect(self.handle_finished)
        self.threadpool.start(worker)

    def speak_response(self, text: str):
        self.set_state(AppState.RESPONDING)
        worker = Worker(self.tts.speak, text)
        worker.signals.error.connect(self.handle_error)
        worker.signals.finished.connect(lambda: self.set_state(AppState.LISTENING))
        self.threadpool.start(worker)

    def handle_record_response(self, s):
        print(s)
        self.set_state(AppState.LISTENING)
        if not s:
            return
        elif s.lower() == "stop.":
            self.tts.stop()
        else:
            self.label_response.setText(s)
            self.send_message(s)
    
    def record(self):
        worker = Worker(self.stt.record)
        worker.signals.error.connect(self.handle_error)
        worker.signals.result.connect(self.handle_record_response)
        worker.signals.finished.connect(self.handle_finished)
        self.threadpool.start(worker)

    def listen_in_background(self):
        self.set_state(AppState.IDLE)
        worker = Worker(self.stt.listen_in_background)
        worker.signals.error.connect(self.handle_error)
        self.threadpool.start(worker)

    def stop_listening(self):
        worker = Worker(self.stt.stop_listening)
        worker.signals.error.connect(self.handle_error)
        worker.signals.finished.connect(lambda: self.set_state(AppState.INACTIVE))
        self.threadpool.start(worker)

parser = argparse.ArgumentParser()
parser.add_argument('--whisper_model', default='base', help='Whisper model to use')
args = parser.parse_args()

app = QApplication([])
window = MainWindow(whisper_model=args.whisper_model)
window.show()
app.exec()