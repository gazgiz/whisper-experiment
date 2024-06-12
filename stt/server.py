from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import whisper
import numpy as np
import json
import asyncio
from TTS.api import TTS
import soundfile as sf
import io
import torch

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the Whisper model
model = whisper.load_model("base")

# Load the Coqui TTS model
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)
device = "cuda" if torch.cuda.is_available() else "cpu"
tts.to(device)

# Set default speaker ID and language
default_speaker_id = 'Andrew Chipper'  # Replace with any speaker ID from the list
default_language = 'ko'

# Store connections
send_audio_clients = []
receive_audio_clients = []
send_transcript_clients = []

@app.get('/')
async def get():
    with open('index.html') as f:
        return HTMLResponse(f.read())

@app.websocket("/send_audio")
async def send_audio_endpoint(websocket: WebSocket):
    await websocket.accept()
    send_audio_clients.append(websocket)
    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message:
                data = message["bytes"]
                await process_audio_chunk(data)
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Exception: {e}")
    finally:
        if websocket in send_audio_clients:
            send_audio_clients.remove(websocket)
        try:
            await websocket.close()
        except RuntimeError:
            pass  # WebSocket already closed

@app.websocket("/receive_audio")
async def receive_audio_endpoint(websocket: WebSocket):
    await websocket.accept()
    receive_audio_clients.append(websocket)
    try:
        while True:
            await asyncio.sleep(1)  # Keep connection open
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Exception: {e}")
    finally:
        if websocket in receive_audio_clients:
            receive_audio_clients.remove(websocket)
        try:
            await websocket.close()
        except RuntimeError:
            pass  # WebSocket already closed

@app.websocket("/send_transcript")
async def send_transcript_endpoint(websocket: WebSocket):
    await websocket.accept()
    send_transcript_clients.append(websocket)
    try:
        while True:
            await asyncio.sleep(1)  # Keep connection open
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Exception: {e}")
    finally:
        if websocket in send_transcript_clients:
            send_transcript_clients.remove(websocket)
        try:
            await websocket.close()
        except RuntimeError:
            pass  # WebSocket already closed

async def process_audio_chunk(data):
    try:
        # Convert the received audio data to numpy array
        audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

        # Perform the transcription
        result = model.transcribe(audio_data)
        transcript = result["text"]
        print(f"Transcription: {transcript}")

        # Convert the transcribed text to speech with the default speaker and language
        tts_output = tts.tts(text=transcript, speaker=default_speaker_id, language=default_language)

        # Write the TTS output to a buffer
        with io.BytesIO() as buffer:
            sf.write(buffer, tts_output, samplerate=22050, format='WAV')
            buffer.seek(0)
            audio_bytes = buffer.read()

        # Check the size of the audio bytes before sending
        if len(audio_bytes) > 1048576:
            print("Audio bytes size exceeds 1MB, splitting into smaller chunks.")
            chunks = [audio_bytes[i:i+1048576] for i in range(0, len(audio_bytes), 1048576)]
        else:
            chunks = [audio_bytes]

        # Send the start recording signal
        for client in receive_audio_clients:
            try:
                await client.send_text(json.dumps({"type": "control", "text": "START_RECORDING"}))
            except WebSocketDisconnect:
                receive_audio_clients.remove(client)

        # Send the audio bytes back to all connected clients in chunks
        for chunk in chunks:
            for client in receive_audio_clients:
                try:
                    await client.send_bytes(chunk)
                except WebSocketDisconnect:
                    receive_audio_clients.remove(client)

        # Send the stop recording signal to all connected clients
        for client in receive_audio_clients:
            try:
                await client.send_text(json.dumps({"type": "control", "text": "STOP_RECORDING"}))
            except WebSocketDisconnect:
                receive_audio_clients.remove(client)

        # Send the transcription text to the transcription clients
        for client in send_transcript_clients:
            try:
                await client.send_text(json.dumps({"type": "transcription", "text": transcript}))
            except WebSocketDisconnect:
                send_transcript_clients.remove(client)
    except Exception as e:
        print("Error processing audio chunk:", e)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, ws_max_size=10485760, ws_ping_interval=30, ws_ping_timeout=30, ws_close_timeout=10)  # Increase the max size to 10MB, set ping interval and timeout

