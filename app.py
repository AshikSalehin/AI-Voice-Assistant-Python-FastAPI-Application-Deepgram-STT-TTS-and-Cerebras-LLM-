import os
import base64
import logging
import json
import uuid
import asyncio
import requests
import time
from threading import Event
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from deepgram import LiveTranscriptionEvents, DeepgramClient, LiveOptions

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logging.info("Application started - logging system initialized")

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configuration
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1/"
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_TTS_VOICE = os.getenv("DEEPGRAM_TTS_VOICE")
DEEPGRAM_TTS_CONCURRENCY = int(os.getenv("DEEPGRAM_TTS_CONCURRENCY", "4"))

SYSTEM_PROMPT = f"""
System prompt:
Your role: AI Sales Agent.

Your Name: Alexander.

Communication Rules for you:
1. Always Speak Politely
2. Try to answer to the point.
3. Respond to their queries in a polite manner.
4. Never use a slang
5. Always try to answer in 2 ot 3 sentences.
"""

class ConnectionManager:
    def __init__(self):
        self.active_connections = {}
        self.lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        async with self.lock:
            self.active_connections[session_id] = websocket
        logging.info(f"New connection: {session_id}")

    async def disconnect(self, session_id: str):
        async with self.lock:
            if session_id in self.active_connections:
                del self.active_connections[session_id]

    async def send_message(self, session_id: str, message: dict):
        async with self.lock:
            if session_id in self.active_connections:
                try:
                    await self.active_connections[session_id].send_json(message)
                except Exception as e:
                    logging.error(f"Error sending message: {str(e)}")

manager = ConnectionManager()
session_llms = {}
task_events = {}

class CerebrasLLM:
    def __init__(self, api_key, system_prompt):
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.base_url = CEREBRAS_BASE_URL
        self.conversation_history = []
        self.reset_conversation()

    def reset_conversation(self):
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]

    def get_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def get_available_models(self):
        try:
            response = requests.get(f"{self.base_url}models", 
                                  headers=self.get_headers(),
                                  timeout=10)
            return [model["id"] for model in response.json().get("data", [])]
        except Exception as e:
            logging.error(f"Model fetch error: {str(e)}")
            return []

    def stream_response(self, user_input):
        self.conversation_history.append({"role": "user", "content": user_input})
        try:
            models = self.get_available_models()
            if not models:
                yield "Error: No models available"
                return
            
            selected_model = models[1] if len(models) > 1 else models[0]
            
            response = requests.post(
                f"{self.base_url}chat/completions",
                headers=self.get_headers(),
                json={
                    "model": selected_model,
                    "messages": self.conversation_history,
                    "temperature": 0.7,
                    "max_tokens": 40960,
                    "stream": True
                },
                stream=True,
                timeout=30
            )
            
            full_response = ""
            for chunk in response.iter_lines():
                if chunk:
                    chunk = chunk.decode().strip()
                    if chunk.startswith("data: "):
                        chunk = chunk[6:]
                    if chunk and chunk != "[DONE]":
                        try:
                            data = json.loads(chunk)
                            content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                full_response += content
                                yield content
                        except json.JSONDecodeError:
                            continue
            
            self.conversation_history.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            logging.error(f"LLM Error: {str(e)}")
            yield "I'm having trouble responding. Please try again."

class DeepgramTTS:
    OUTPUT_MIME = "audio/mpeg"

    def __init__(self, api_key, voice_model):
        self.api_key = api_key
        self.voice_model = voice_model or "aura-asteria-en"
        self.speak_url = f"https://api.deepgram.com/v1/speak?model={self.voice_model}"
        # Reuse a single session to keep TCP/TLS connections warm; cap pool size
        from requests.adapters import HTTPAdapter
        self._session = requests.Session()
        adapter = HTTPAdapter(pool_connections=DEEPGRAM_TTS_CONCURRENCY, pool_maxsize=DEEPGRAM_TTS_CONCURRENCY)
        self._session.mount('https://', adapter)
        self._session.mount('http://', adapter)

    def text_to_speech(self, text):
        try:
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json",
                "Accept": self.OUTPUT_MIME
            }
            payload = {"text": text}
            max_attempts = 3
            for attempt in range(max_attempts):
                response = self._session.post(self.speak_url, headers=headers, json=payload, timeout=15)
                if 200 <= response.status_code < 300:
                    return response.content
                if response.status_code == 429:
                    retry_after = response.headers.get('Retry-After')
                    delay = float(retry_after) if retry_after else min(0.5 * (2 ** attempt), 4.0)
                    logging.error(f"TTS 429 rate limited. Backing off for {delay:.2f}s (attempt {attempt+1}/3).")
                    time.sleep(delay)
                    continue
                logging.error(f"TTS Error: HTTP {response.status_code} - {response.text[:200]}")
                break
            return None
        except Exception as e:
            logging.error(f"TTS Error: {str(e)}")
            return None

class DeepgramStreamingSTT:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = DeepgramClient(api_key)
        self.active_connections = {}

    def send_audio(self, session_id, audio_data):
        if session_id in self.active_connections:
            try:
                self.active_connections[session_id].send(audio_data)
                return True
            except Exception as e:
                logging.error(f"Failed to send audio: {str(e)}")
                return False
        return False
    
    def close(self, session_id):
        if session_id in self.active_connections:
            try:
                self.active_connections[session_id].finish()
                del self.active_connections[session_id]
                return True
            except Exception as e:
                logging.error(f"Error closing connection: {str(e)}")
                return False
        return False
    
    def create_streaming_client(self, session_id, callback):
        try:
            options = LiveOptions(
                model="nova-2",
                language="en-US",
                sample_rate=16000,
                smart_format=True,
                interim_results=True,
                punctuate=True,
                endpointing=1000,
                utterance_end_ms=1000
            )
            
            live_transcription = self.client.listen.websocket.v("1")
            
            def on_message(_, result):
                try:
                    if result.is_final:
                        transcript = result.channel.alternatives[0].transcript
                        if transcript.strip():
                            logging.info(f"Transcript: {transcript}")
                            callback(transcript, True)
                    else:
                        transcript = result.channel.alternatives[0].transcript
                        if transcript.strip():
                            callback(transcript, False)
                except Exception as e:
                    logging.error(f"Deepgram error: {str(e)}")
            
            def on_error(_, error):
                logging.error(f"Deepgram error: {error}")
                
            def on_close(_, close_event):
                logging.info(f"Deepgram connection closed: {close_event}")
                if session_id in self.active_connections:
                    del self.active_connections[session_id]

            live_transcription.on(LiveTranscriptionEvents.Transcript, on_message)
            live_transcription.on(LiveTranscriptionEvents.Error, on_error)
            live_transcription.on(LiveTranscriptionEvents.Close, on_close)
            
            live_transcription.start(options)
            self.active_connections[session_id] = live_transcription
            return live_transcription
            
        except Exception as e:
            logging.error(f"Deepgram setup failed: {str(e)}")
            return None

tts = DeepgramTTS(DEEPGRAM_API_KEY, DEEPGRAM_TTS_VOICE)
stt = DeepgramStreamingSTT(DEEPGRAM_API_KEY)

async def process_ai_response(session_id, transcript):
    if session_id in task_events:
        task_events[session_id].set()
    task_events[session_id] = Event()
    
    asyncio.create_task(stream_and_process_response(session_id, transcript, task_events[session_id]))

async def stream_and_process_response(session_id, transcript, cancel_event):
    try:
        chunk_index = 0
        pending_chunks = {}
        last_sent_index = -1
        lock = asyncio.Lock()
        tts_tasks = []
        
        async def process_and_send_audio(text, index):
            nonlocal last_sent_index
            
            if cancel_event.is_set():
                return
                
            # Offload blocking HTTP TTS call to a thread and limit concurrency
            # Create a module-level semaphore after tts is defined
            global tts_semaphore
            if 'tts_semaphore' not in globals() or tts_semaphore is None:
                tts_semaphore = asyncio.Semaphore(DEEPGRAM_TTS_CONCURRENCY)
            async with tts_semaphore:
                audio_data = await asyncio.to_thread(tts.text_to_speech, text)
            if not audio_data:
                return
                
            async with lock:
                pending_chunks[index] = audio_data
                
                while (last_sent_index + 1) in pending_chunks:
                    current_index = last_sent_index + 1
                    audio = pending_chunks.pop(current_index)
                    
                    await manager.send_message(session_id, {
                        "type": "assistant_audio_chunk",
                        "data": {
                            "audio": base64.b64encode(audio).decode('utf-8'),
                            "chunk_index": current_index,
                            "total_chunks": 1,
                            "is_last": False
                        }
                    })
                    last_sent_index = current_index
        
        full_response = []
        sentence_buffer = ""
        sentence_end_marks = ['.', '!', '?']
        
        for content_chunk in session_llms[session_id].stream_response(transcript):
            if cancel_event.is_set():
                break
                
            sentence_buffer += content_chunk
            full_response.append(content_chunk)
            
            while True:
                boundaries = [sentence_buffer.find(mark) for mark in sentence_end_marks]
                last_boundary = max((i for i in boundaries if i != -1), default=-1)
                
                if last_boundary == -1:
                    break
                    
                chunk_to_process = sentence_buffer[:last_boundary+1].strip()
                sentence_buffer = sentence_buffer[last_boundary+1:].strip()
                
                if chunk_to_process:
                    tts_tasks.append(asyncio.create_task(process_and_send_audio(chunk_to_process, chunk_index)))
                    chunk_index += 1
        
        if sentence_buffer.strip() and not cancel_event.is_set():
            tts_tasks.append(asyncio.create_task(process_and_send_audio(sentence_buffer.strip(), chunk_index)))
            chunk_index += 1

        # Wait for all TTS chunks to complete before signaling end
        if tts_tasks:
            await asyncio.gather(*tts_tasks, return_exceptions=True)
            
        await manager.send_message(session_id, {
            "type": "assistant_audio_chunk",
            "data": {
                "audio": "",
                "chunk_index": chunk_index,
                "total_chunks": chunk_index + 1,
                "is_last": True
            }
        })
                
    except Exception as e:
        logging.error(f"Response error: {str(e)}")
        await manager.send_message(session_id, {
            "type": "error",
            "data": {"message": str(e)}
        })

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    session_id = str(uuid.uuid4())
    await manager.connect(websocket, session_id)
    
    try:
        async def receive_messages():
            while True:
                message = await websocket.receive_text()
                data = json.loads(message)
                
                if data.get('type') == 'initial_message':
                    
                    session_llms[session_id] = CerebrasLLM(CEREBRAS_API_KEY, SYSTEM_PROMPT)
                    
                    await manager.send_message(session_id, {
                        "type": "status",
                        "data": {"message": "Initialized"}
                    })

                elif data.get('type') == 'start_stream':
                    loop = asyncio.get_running_loop()
                    
                    def transcription_callback(transcript, is_final):
                        asyncio.run_coroutine_threadsafe(
                            manager.send_message(session_id, {
                                "type": "transcript",
                                "data": {
                                    "text": transcript,
                                    "is_final": is_final
                                }
                            }),
                            loop
                        )
                        if is_final and transcript.strip():
                            asyncio.run_coroutine_threadsafe(
                                process_ai_response(session_id, transcript),
                                loop
                            )

                    connection = stt.create_streaming_client(session_id, transcription_callback)
                    if connection:
                        await manager.send_message(session_id, {
                            "type": "status",
                            "data": {"message": "Streaming started"}
                        })
                        
                        # Send initial greeting
                        audio_data = await asyncio.to_thread(tts.text_to_speech, "Hello there")
                        if audio_data:
                            await manager.send_message(session_id, {
                                "type": "assistant_audio_chunk",
                                "data": {
                                    "audio": base64.b64encode(audio_data).decode('utf-8'),
                                    "chunk_index": 0,
                                    "total_chunks": 1,
                                    "is_last": True
                                }
                            })

                elif data.get('type') == 'audio_data':
                    audio_data = base64.b64decode(data.get('data', ''))
                    stt.send_audio(session_id, audio_data)

                elif data.get('type') == 'stop_stream':
                    stt.close(session_id)
                    await manager.send_message(session_id, {
                        "type": "status",
                        "data": {"message": "Streaming stopped"}
                    })

                elif data.get('type') == 'reset':
                    if session_id in session_llms:
                        session_llms[session_id].reset_conversation()
                    await manager.send_message(session_id, {
                        "type": "status",
                        "data": {"message": "Reset complete"}
                    })

        await receive_messages()

    except WebSocketDisconnect:
        logging.info(f"Client disconnected: {session_id}")
    except Exception as e:
        logging.error(f"WebSocket error: {str(e)}")
    finally:
        stt.close(session_id)
        await manager.disconnect(session_id)
        if session_id in session_llms:
            del session_llms[session_id]

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/reset")
async def handle_reset(request: Request):
    data = await request.json()
    session_id = data.get('session_id')
    if session_id and session_id in session_llms:
        session_llms[session_id].reset_conversation()
        return {"status": "Conversation reset"}
    raise HTTPException(status_code=404, detail="Session not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8116, reload=True)