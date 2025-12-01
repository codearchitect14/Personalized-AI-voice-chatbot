# 🎤 Voice Chat Clone AI

**Real-time voice conversations with AI using cloned human voices**

Speak naturally, get intelligent GPT-4 responses, and hear them in a voice that mimics any person you choose!

## ✨ Core Features

- **🎤 Real-time Voice Chat** - Speak and receive voice responses
- **🗣️ Voice Cloning** - Clone any voice from audio samples
- **💬 Text Chat Interface** - Type messages as alternative
- **⚡ Streaming Responses** - Real-time text generation
- **📚 Conversation History** - Save and manage chat sessions
- **🎛️ Multi-Engine TTS** - Chatterbox, XTTS v2, and ElevenLabs support

## 🏗️ Architecture

```
User Voice → Speech-to-Text → GPT-4 Processing → Text-to-Speech (Cloned Voice) → Audio Response
```

## 📦 Dependencies

### Core Requirements
- **Python 3.9+**
- **CUDA 11.8+** (optional, for GPU acceleration)
- **FFmpeg** (audio processing)

### Python Packages
```txt
# AI/ML
torch>=2.0.0
torchaudio>=2.0.0
chatterbox-tts
TTS>=0.21.0
transformers>=4.35.0

# Web Framework
fastapi>=0.104.0
uvicorn>=0.24.0
jinja2>=3.1.2

# Audio Processing
pydub>=0.25.1
soundfile>=0.12.1

# API Clients
openai>=1.3.0
elevenlabs>=0.2.0
python-dotenv>=1.0.0
```

### System Libraries
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg libsndfile1

# macOS
brew install ffmpeg libsndfile

# Windows
choco install ffmpeg
```

### API Keys Required
- **OpenRouter API Key** (for GPT-4 access)
- **ElevenLabs API Key** (optional, for premium voices)
- **OpenAI API Key** (optional, for Whisper STT)

## 🚀 Quick Start

1. **Clone and setup**
```bash
git clone https://github.com/yourusername/voice-chat-clone.git
cd voice-chat-clone
conda create -n voicechat python=3.9
conda activate voicechat
pip install -r requirements.txt
```

2. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. **Add voice sample**
```bash
mkdir -p voice_samples
# Add your WAV/MP3 file to voice_samples/
```

4. **Run the app**
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8502
```

5. **Open browser**
```
http://localhost:8502
```

## 📁 Project Structure
```
voice-chat-clone/
├── main.py                 # FastAPI app
├── cloner.py              # Voice cloning
├── requirements.txt       # Dependencies
├── static/audio/          # Generated audio
├── templates/             # Web interface
└── data/sessions/         # Chat history
```


---

**Give it a ⭐ if you find this project useful!**
