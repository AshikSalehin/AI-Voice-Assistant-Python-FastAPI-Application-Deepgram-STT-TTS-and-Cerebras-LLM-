# AI-Voice-Assistant-Python-FastAPI-Application-Deepgram-STT-TTS-and-Cerebras-LLM-
This is a AI voice assistant application using Deepgram's STT, TTS and Cerebras LLM using Python FastAPI.


A real‑time voice assistant that listens via the browser, transcribes with Deepgram STT, reasons with Cerebras LLM, and replies with Deepgram TTS. Frontend is a lightweight HTML page that streams mic audio over WebSocket and plays back synthesized audio chunks.
Features
Realtime STT: Deepgram live transcription with interim/final results
LLM: Cerebras chat completions with streaming token handling
TTS: Deepgram Speak (Aura/Aura‑2 voices), mp3 output, chunked playback
Ordering + concurrency: Sentence-level chunking, ordered audio delivery, bounded concurrency and retry/backoff for TTS
Logging: Console + app.log
Architecture
Browser: MediaRecorder → WebSocket /ws → base64 audio → visualizer + audio player
Server (FastAPI):
WebSocket /ws:
audio in → Deepgram STT stream → interim/final transcripts → UI
final transcript → Cerebras LLM stream → sentence chunks
each sentence → Deepgram TTS → mp3 bytes → base64 → UI
HTTP:
GET / UI
POST /api/reset reset conversation
Requirements
Python 3.10+
Deepgram account + API key
Cerebras API key


