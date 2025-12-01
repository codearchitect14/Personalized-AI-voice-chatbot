import torch
from TTS.api import TTS
import os

class VoiceCloner:
    def __init__(self, model_name="tts_models/multilingual/multi-dataset/xtts_v2", device=None):
        # Select device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model once
        print("🔄 Loading XTTS model...")
        self.tts = TTS(model_name).to(self.device)
        print("✅ Model loaded successfully.")
        
        self.speaker_wav = None
        self.language = "en"

    def load_voice(self, speaker_wav_path, language="en"):
        """Load and remember the reference speaker voice"""
        if not os.path.exists(speaker_wav_path):
            raise FileNotFoundError(f"Speaker file not found: {speaker_wav_path}")
        
        self.speaker_wav = speaker_wav_path
        self.language = language
        print(f"🎤 Voice reference loaded: {os.path.basename(speaker_wav_path)}")
# transformer,torch,torchaudio
# 4.36.2 2.5.1+cpu 2.5.1+cpu
    def speak(self, text, output_path=None):
        """Generate speech using the remembered voice"""
        if self.speaker_wav is None:
            raise ValueError("❌ No voice loaded. Call load_voice() first.")
        
        if output_path is None:
            output_path = "xtts_output.wav"

        print(f"🗣️ Generating speech for text: \"{text}\"")
        self.tts.tts_to_file(
            text=text,
            speaker_wav=self.speaker_wav,
            language=self.language,
            file_path=output_path
        )
        print(f"💾 Audio saved to: {output_path}")
        return output_path


# ======================
# Example Usage
# ======================

if __name__ == "__main__":
    audio_directory = "/home/qlu/Documents/voice chat/voiceChatTTS"
    speaker_file = os.path.join(audio_directory, "Intro.wav")

    # Create persistent instance
    cloner = VoiceCloner()

    # Load speaker voice once
    cloner.load_voice(speaker_file)

    # Now you can call .speak() multiple times without reloading
    cloner.speak("Hello, this is your cloned voice speaking!", "test1.wav")
    cloner.speak("This is another example using the same cloned voice.", "test2.wav")
