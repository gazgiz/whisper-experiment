from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import whisper
import numpy as np
import uuid
import json
import asyncio
from scipy.io.wavfile import write as write_wav
import os

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

# In-memory storage for transcription jobs
transcription_jobs = {}

@app.get('/')
async def get():
    with open('index.html') as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive()
            if "text" in message:
                data_json = json.loads(message["text"])
                if data_json["action"] == "start_transcription":
                    job_id = str(uuid.uuid4())
                    transcription_jobs[job_id] = {
                        "status": "IN_PROGRESS",
                        "transcript": []
                    }
                    await websocket.send_json({"job_id": job_id})
            elif "bytes" in message:
                data = message["bytes"]
                await process_audio_chunk(websocket, data)
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Exception: {e}")

async def process_audio_chunk(websocket, data):
    try:
        # Convert the received audio data to numpy array
        audio_data = np.frombuffer(data, dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0
        
        # Save the received audio data to a file for validation
        #sample_rate = 16000  # Assuming the sample rate is 16000 Hz
        #filename = f"received_audio_{uuid.uuid4()}.wav"
        #write_wav(filename, sample_rate, audio_data)
        #print(f"Audio saved to {filename}")

        # Perform the transcription
        result = model.transcribe(audio_data)
        transcript = result["text"]

        # Emit the transcription result back to the client
        await websocket.send_json({"transcript": transcript})
    except Exception as e:
        print("Error processing audio chunk:", e)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

