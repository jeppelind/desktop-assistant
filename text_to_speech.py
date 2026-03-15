import tempfile
import wave
import os
import time
import soundfile as sf
import sounddevice as sd
from piper import PiperVoice

class TextToSpeech:
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