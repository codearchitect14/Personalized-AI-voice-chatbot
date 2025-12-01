import os
import uuid
import json
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, StreamingResponse
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from openai import OpenAI
import tempfile
from datetime import datetime
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import torch
# from TTS.api import TTS
from pydub import AudioSegment
import io
from functools import lru_cache
import torch

load_dotenv()

app = FastAPI(title="Voice Chat")
@lru_cache(maxsize=1)
def get_tts_model():
    audio_directory = "/home/talharashid/voice_chat"
    speaker_file = os.path.join(audio_directory, "Intro.wav")
    model = ChatterboxTTS.from_pretrained(device="cuda")
    return model
model = get_tts_model()
# Create necessary directories
os.makedirs("static/audio", exist_ok=True)
os.makedirs("data/sessions", exist_ok=True)

# Mount static files for serving audio
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize clients
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
openai_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
) if OPENROUTER_API_KEY else None
# Your cloned voice ID - REPLACE WITH YOUR ACTUAL VOICE ID
VOICE_ID = os.getenv("VOICE_ID")  # Replace with the voice ID you got from cloning


audio_directory = "/home/talharashid/voice_chat"
speaker_file = os.path.join(audio_directory, "Intro.wav")

# Store conversations and chat sessions
chat_sessions = {}

def load_sessions():
    """Load chat sessions from disk"""
    global chat_sessions
    try:
        if os.path.exists("data/sessions/sessions.json"):
            with open("data/sessions/sessions.json", "r") as f:
                chat_sessions = json.load(f)
            print(f"Loaded {len(chat_sessions)} sessions from disk")
    except Exception as e:
        print(f"Error loading sessions: {e}")
        chat_sessions = {}

def save_sessions():
    """Save chat sessions to disk"""
    try:
        with open("data/sessions/sessions.json", "w") as f:
            json.dump(chat_sessions, f, indent=2)
        print(f"Saved {len(chat_sessions)} sessions to disk")
    except Exception as e:
        print(f"Error saving sessions: {e}")

def get_or_create_session(session_id, session_name=None):
    """Get or create a chat session"""
    if session_id not in chat_sessions:
        chat_sessions[session_id] = {
            'name': session_name or f"Chat {len(chat_sessions) + 1}",
            'conversations': [],
            'created_at': datetime.now().isoformat(),
            'audio_files': {}  # Store audio file mappings
        }
        save_sessions()
    return chat_sessions[session_id]


def speech_to_text(audio_data: bytes, filename: str = "recording.webm") -> str:
    """Convert speech to text using OpenAI Whisper"""
    try:
        # Save audio data to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            temp_audio.write(audio_data)
            temp_audio_path = temp_audio.name
        
        # Use OpenAI Whisper for speech-to-text
        with open(temp_audio_path, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
                language="en"
            )
        
        # Clean up temporary file
        os.unlink(temp_audio_path)
        
        return transcript.strip()
        
    except Exception as e:
        print(f"Speech-to-text error: {e}")
        return ""

def get_gpt_response(user_input: str, session_id: str) -> str:
    """Get response from GPT-4"""
    try:
        session = get_or_create_session(session_id)
        
        # Build conversation history
        messages = [
            {"role": "system", "content": "You are a helpful and friendly assistant. Keep responses concise and natural for speech."}
        ]
        
        # Add recent conversation history (last 8 exchanges)
        for conv in session['conversations'][-8:]:
            messages.extend([
                {"role": "user", "content": conv['user_input']},
                {"role": "assistant", "content": conv['assistant_response']}
            ])
        
        # Add current user message
        messages.append({"role": "user", "content": user_input})
        
        # Get response from OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )
        
        assistant_response = response.choices[0].message.content
        
        return assistant_response
        
    except Exception as e:
        print(f"GPT error: {e}")
        return "I apologize, but I'm having trouble processing your request right now."
def get_gpt_response_stream(user_input: str, session_id: str):
    """Stream response from GPT-4"""
    try:
        session = get_or_create_session(session_id)
        
        # Build conversation history
        messages = [
            {"role": "system", "content": "You are a helpful and friendly assistant. Keep responses concise and natural for speech."}
        ]
        
        # Add recent conversation history (last 8 exchanges)
        for conv in session['conversations'][-8:]:
            messages.extend([
                {"role": "user", "content": conv['user_input']},
                {"role": "assistant", "content": conv['assistant_response']}
            ])
        
        # Add current user message
        messages.append({"role": "user", "content": user_input})
        
        # Get streaming response from OpenAI
        stream = openai_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=150,
            temperature=0.7,
            stream=True  # Enable streaming
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        print(f"GPT streaming error: {e}")
        yield "I apologize, but I'm having trouble processing your request right now."



def generate_cloned_audio_elenlabs(text, audio_filepath):
    # Generate audio
        audio = elevenlabs_client.text_to_speech.convert(
            text=text,
            voice_id=VOICE_ID,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        
        # Save audio file persistently
        with open(audio_filepath, "wb") as f:
            for chunk in audio:
                if chunk:
                    f.write(chunk)
        
        print("🎙️Using Elevenlabs")
def generate_cloned_audio_chatter_box(text, audio_filepath):
    # audio_directory = "/home/talharashid/voice_chat"
    # speaker_file = os.path.join(audio_directory, "Intro.wav")
    # model = ChatterboxTTS.from_pretrained(device="cuda")
    wav = model.generate(text, audio_prompt_path=speaker_file)
    ta.save(audio_filepath, wav, model.sr)
    print("🎙️Using Chatter-Box")



def text_to_speech(text: str, session_id: str, message_id: str) -> str:
    """Convert text to speech using cloned voice and store persistently"""
    try:
        if not text.strip():
            return None
            
        # Generate unique filename
        audio_filename = f"{message_id}.mp3"
        audio_filepath = f"static/audio/{audio_filename}"
        
        generate_cloned_audio_chatter_box(text, audio_filepath)
        # generate_cloned_audio_tts(text, audio_filepath)
        # generate_cloned_audio_elenlabs(text, audio_filepath)
        
        # Store audio file reference in session
        if session_id in chat_sessions:
            if 'audio_files' not in chat_sessions[session_id]:
                chat_sessions[session_id]['audio_files'] = {}
            chat_sessions[session_id]['audio_files'][message_id] = audio_filename
            save_sessions()
        
        return f"/static/audio/{audio_filename}"
        
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        return None

    

def convert_webm_to_wav(webm_content):
    """Convert WebM audio to WAV format"""
    audio = AudioSegment.from_file(io.BytesIO(webm_content), format="webm")
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    return wav_io.getvalue()

@app.get("/")
async def home(request: Request):
    """Serve the main chat interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/sessions")
async def get_sessions(request: Request):
    """Get all chat sessions for the current user"""
    session_id = str(hash(request.client.host))
    user_sessions = {k: v for k, v in chat_sessions.items() if k.startswith(session_id)}
    return {"sessions": user_sessions}

@app.post("/session/{session_name}")
async def create_session(session_name: str, request: Request):
    """Create a new chat session"""
    session_id = f"{hash(request.client.host)}_{uuid.uuid4().hex[:8]}"
    session = get_or_create_session(session_id, session_name)
    return {
        "success": True,
        "session_id": session_id,
        "session_name": session['name']
    }

@app.get("/session/{session_id}")
async def get_session(session_id: str, request: Request):
    """Get a specific chat session"""
    session = chat_sessions.get(session_id)
    if session:
        return {
            "success": True,
            "session": session
        }
    else:
        return {
            "success": False,
            "error": "Session not found"
        }

@app.post("/chat")
async def chat(request: Request, message: str = Form(...), session_id: str = Form(None)):
    """Handle text chat messages"""
    try:
        if not message.strip():
            return {
                "success": False,
                "error": "Message cannot be empty"
            }
            
        # Use provided session_id or create default one
        if not session_id:
            session_id = f"{hash(request.client.host)}_default"
            
        session = get_or_create_session(session_id)
        gpt_response = get_gpt_response(message, session_id)
        
        # Generate unique message ID for audio
        message_id = uuid.uuid4().hex
        audio_url = text_to_speech(gpt_response, session_id, message_id)
        
        # Store conversation with audio reference
        session['conversations'].append({
            'user_input': message,
            'assistant_response': gpt_response,
            'message_id': message_id,
            'timestamp': datetime.now().isoformat()
        })
        save_sessions()
        
        if audio_url:
            return {
                "success": True,
                "text_response": gpt_response,
                "audio_url": audio_url,
                "session_id": session_id,
                "message_id": message_id
            }
        else:
            return {
                "success": False,
                "error": "Failed to generate audio"
            }
            
    except Exception as e:
        print(f"Chat error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


#*******************************************************************************************************************

@app.post("/voice_chat")
async def voice_chat(request: Request, file: UploadFile = File(...), session_id: str = Form(None)):
    """Handle voice chat - transcribe then get GPT response"""
    try:
        # Read audio data
        content = await file.read()
        print(f"Received audio file: {file.filename}, size: {len(content)} bytes, type: {file.content_type}")
        if len(content) == 0:
            return {
                "success": False,
                "error": "Empty audio file"
            }
        

        if file.filename.endswith('.webm'):
            wav_content = convert_webm_to_wav(content)
            user_input = speech_to_text(wav_content, "converted.wav")
        else:
            user_input = speech_to_text(content, file.filename)
        # # Convert speech to text
        # user_input = speech_to_text(content, file.filename)
        
        if not user_input:
            return {
                "success": False,
                "error": "Could not understand audio. Please speak clearly and try again."
            }
        
        # Use provided session_id or create default one
        if not session_id:
            session_id = f"{hash(request.client.host)}_default"
            
        session = get_or_create_session(session_id)
        gpt_response = get_gpt_response(user_input, session_id)
        
        # Generate unique message ID for audio
        message_id = uuid.uuid4().hex
        audio_url = text_to_speech(gpt_response, session_id, message_id)
        
        # Store conversation with audio reference
        session['conversations'].append({
            'user_input': user_input,
            'assistant_response': gpt_response,
            'message_id': message_id,
            'timestamp': datetime.now().isoformat()
        })
        save_sessions()
        
        if audio_url:
            return {
                "success": True,
                "user_input": user_input,
                "text_response": gpt_response,
                "audio_url": audio_url,
                "session_id": session_id,
                "message_id": message_id
            }
        else:
            return {
                "success": False,
                "user_input": user_input,
                "text_response": gpt_response,
                "error": "Failed to generate audio response"
            }
            
    except Exception as e:
        print(f"Voice chat error: {e}")
        return {
            "success": False,
            "error": f"Voice chat failed: {str(e)}"
        }

@app.delete("/session/{session_id}")
async def delete_session(session_id: str, request: Request):
    """Delete a chat session and its audio files"""
    if session_id in chat_sessions:
        # Delete associated audio files
        session = chat_sessions[session_id]
        if 'audio_files' in session:
            for audio_filename in session['audio_files'].values():
                audio_path = f"static/audio/{audio_filename}"
                try:
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
                        print(f"Deleted audio file: {audio_path}")
                except Exception as e:
                    print(f"Error deleting audio file {audio_path}: {e}")
        
        del chat_sessions[session_id]
        save_sessions()
        return {"success": True}
    else:
        return {"success": False, "error": "Session not found"}

@app.post("/session/{session_id}/rename")
async def rename_session(session_id: str, request: Request, new_name: str = Form(...)):
    """Rename a chat session"""
    if session_id in chat_sessions:
        chat_sessions[session_id]['name'] = new_name
        save_sessions()
        return {"success": True}
    else:
        return {"success": False, "error": "Session not found"}

@app.get("/audio/{session_id}/{message_id}")
async def get_audio(session_id: str, message_id: str):
    """Get audio file for a specific message"""
    try:
        session = chat_sessions.get(session_id)
        if session and 'audio_files' in session and message_id in session['audio_files']:
            audio_filename = session['audio_files'][message_id]
            audio_path = f"static/audio/{audio_filename}"
            if os.path.exists(audio_path):
                return FileResponse(audio_path, media_type="audio/mpeg", filename=audio_filename)
        
        return {"success": False, "error": "Audio not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}
















@app.post("/chat_stream")
async def chat_stream(request: Request, message: str = Form(...), session_id: str = Form(None)):
    """Handle text chat messages with streaming"""
    try:
        if not message.strip():
            return {
                "success": False,
                "error": "Message cannot be empty"
            }
            
        # Use provided session_id or create default one
        if not session_id:
            session_id = f"{hash(request.client.host)}_default"
            
        session = get_or_create_session(session_id)
        
        # Store user message immediately
        user_message_data = {
            'user_input': message,
            'assistant_response': "",  # Will be filled as we stream
            'message_id': uuid.uuid4().hex,
            'timestamp': datetime.now().isoformat(),
            'is_streaming': True
        }
        session['conversations'].append(user_message_data)
        
        def generate():
            full_response = ""
            for chunk in get_gpt_response_stream(message, session_id):
                full_response += chunk
                # Send each chunk as SSE
                yield f"data: {json.dumps({'chunk': chunk, 'session_id': session_id, 'message_id': user_message_data['message_id']})}\n\n"
            
            # Final message with complete response
            user_message_data['assistant_response'] = full_response
            user_message_data['is_streaming'] = False
            
            # Generate audio for the complete response
            audio_url = text_to_speech(full_response, session_id, user_message_data['message_id'])
            
            yield f"data: {json.dumps({'complete': True, 'full_response': full_response, 'audio_url': audio_url, 'session_id': session_id, 'message_id': user_message_data['message_id']})}\n\n"
            
            save_sessions()
        
        return StreamingResponse(generate(), media_type="text/plain")
            
    except Exception as e:
        print(f"Chat stream error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/voice_chat_stream")
async def voice_chat_stream(request: Request, file: UploadFile = File(...), session_id: str = Form(None)):
    """Handle voice chat with streaming response"""
    try:
        # Read audio data
        content = await file.read()
        print(f"Received audio file: {file.filename}, size: {len(content)} bytes, type: {file.content_type}")
        if len(content) == 0:
            return {
                "success": False,
                "error": "Empty audio file"
            }

        # Convert speech to text
        if file.filename.endswith('.webm'):
            wav_content = convert_webm_to_wav(content)
            user_input = speech_to_text(wav_content, "converted.wav")
        else:
            user_input = speech_to_text(content, file.filename)
        
        if not user_input:
            return {
                "success": False,
                "error": "Could not understand audio. Please speak clearly and try again."
            }
        
        # Use provided session_id or create default one
        if not session_id:
            session_id = f"{hash(request.client.host)}_default"
            
        session = get_or_create_session(session_id)
        
        # Store user message immediately
        user_message_data = {
            'user_input': user_input,
            'assistant_response': "",  # Will be filled as we stream
            'message_id': uuid.uuid4().hex,
            'timestamp': datetime.now().isoformat(),
            'is_streaming': True
        }
        session['conversations'].append(user_message_data)
        
        def generate():
            full_response = ""
            for chunk in get_gpt_response_stream(user_input, session_id):
                full_response += chunk
                # Send each chunk as SSE
                yield f"data: {json.dumps({'chunk': chunk, 'session_id': session_id, 'message_id': user_message_data['message_id'], 'user_input': user_input})}\n\n"
            
            # Final message with complete response
            user_message_data['assistant_response'] = full_response
            user_message_data['is_streaming'] = False
            
            # Generate audio for the complete response
            audio_url = text_to_speech(full_response, session_id, user_message_data['message_id'])
            
            yield f"data: {json.dumps({'complete': True, 'full_response': full_response, 'audio_url': audio_url, 'session_id': session_id, 'message_id': user_message_data['message_id'], 'user_input': user_input})}\n\n"
            
            save_sessions()
        
        return StreamingResponse(generate(), media_type="text/plain")
            
    except Exception as e:
        print(f"Voice chat stream error: {e}")
        return {
            "success": False,
            "error": f"Voice chat failed: {str(e)}"
        }
# Load sessions on startup
load_sessions()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8502)