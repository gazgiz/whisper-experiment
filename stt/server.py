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
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

# Store connections
clients = []

@app.get('/')
async def get():
    with open('index.html') as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
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
        if websocket in clients:
            clients.remove(websocket)
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

        # Convert the transcribed text to speech
        tts_output = tts.tts(transcript)

        # Write the TTS output to a buffer
        with io.BytesIO() as buffer:
            sf.write(buffer, tts_output, samplerate=22050, format='WAV')
            buffer.seek(0)
            audio_bytes = buffer.read()

        # Send the audio bytes back to all connected clients
        for client in clients:
            try:
                await client.send_bytes(audio_bytes)
            except WebSocketDisconnect:
                clients.remove(client)
    except Exception as e:
        print("Error processing audio chunk:", e)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

