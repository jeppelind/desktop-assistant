from PyQt6.QtCore import QObject, pyqtSignal
from faster_whisper import WhisperModel
from speech_recognition import Recognizer, Microphone, AudioData
import numpy as np

class SpeechSignals(QObject):
    new_message = pyqtSignal(str)

class SpeechToText:
    def __init__(self, model_size="base.en"):
        """
        Args:
            model_size (str): faster-whisper model.
        """
        print(f"Loading faster-whisper model: {model_size}")
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("Model loaded")
        self.signals = SpeechSignals()
        self._stop_listening_fn = None
    
    def transcribe(self, audio_array: np.ndarray) -> str:
        segments, info = self.model.transcribe(audio_array, language="en")
        full_text = [segment.text for segment in segments]
        return "".join(full_text).strip()

    def record(self):
        recognizer = Recognizer()
        mic = Microphone()
        
        with mic as source:
            print("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("Listening...")
            audio = recognizer.listen(source)
            
        raw_data = audio.get_raw_data(convert_rate=16000, convert_width=2)

        # Convert the raw bytes to a standard float32 numpy array and normalize to -1.0 to 1.0 range
        audio_np = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        return self.transcribe(audio_np)
    
    def listen_in_background(self):
        recognizer = Recognizer()
        mic = Microphone()

        def handle_audio(_, audio: AudioData):
            raw_data = audio.get_raw_data(convert_rate=16000, convert_width=2)

            # Convert the raw bytes to a standard float32 numpy array and normalize to -1.0 to 1.0 range
            audio_np = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            message = self.transcribe(audio_np)
            if message:
                self.signals.new_message.emit(message)
        
        with mic as source:
            print("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("Listening continuously in background...")

        self._stop_listening_fn = recognizer.listen_in_background(mic, handle_audio)

    def stop_listening(self):
        if self._stop_listening_fn:
            self._stop_listening_fn(wait_for_stop=False)
            self._stop_listening_fn = None
            print("Stopped background listening")