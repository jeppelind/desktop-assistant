from PyQt6.QtCore import QThreadPool, QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QLineEdit, QProgressBar
from ollama import chat
from worker import Worker

class ModelInterface:
    def __init__(self):
        self.model = 'llama3.2'

    def query(self, input: str) -> str:
        print(input)
        response = chat(
            model=self.model, 
            messages=[{'role': 'user', 'content': input}], 
        )
        return response['message']['content']

class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0
        self.model = ModelInterface()

        layout = QVBoxLayout()

        self.progress = QProgressBar()
        self.input = QLineEdit()
        button = QPushButton("Send")
        button.pressed.connect(self.send_message)
        self.label_response = QLabel("Waiting for input...")


        layout.addWidget(self.progress)
        layout.addWidget(self.input)
        layout.addWidget(button)
        layout.addWidget(self.label_response)

        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)

        self.threadpool = QThreadPool()

        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.recurring_timer)
        self.timer.start()

    def recurring_timer(self):
        self.counter += 1
        self.progress.setValue(self.counter)
        if self.counter >= 100:
            self.counter = 0

    def handle_response(self, s):
        print(s)
        self.label_response.setText(s)

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


app = QApplication([])
window = MainWindow()
window.show()
app.exec()