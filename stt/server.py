from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import numpy as np
import json
import asyncio
from TTS.api import TTS
import soundfile as sf
import io
import torch
from faster_whisper import WhisperModel
from livekit import rtc
import uvicorn
import argparse

# Constants for audio settings
SAMPLE_RATE = 48000  # WebRTC standard sample rate
NUM_CHANNELS = 1

# Function to read the configuration file
def load_config():
    with open('config.json', 'r') as file:
        config = json.load(file)
    return config

app = FastAPI()

config = load_config()
use_gpu = config['use_gpu']
text_only = config['text_only']

# Load the Whisper model
device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
print("Using GPU" if device == "cuda" else "Using CPU")

model_size = "medium"
if device == "cuda":
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
else:
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

# Load the Coqui TTS model
if not text_only:
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
    tts.to(device)

# Set default speaker ID and language
default_speaker_id = 'Andrew Chipper'  # Replace with any speaker ID from the list
default_language = 'ko'

# Store connections
send_transcript_clients = []

@app.get('/')
async def get():
    with open('index.html') as f:
        return HTMLResponse(f.read())

@app.websocket("/receive_audio")
async def receive_audio_endpoint(websocket: WebSocket):
    await websocket.accept()
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
        audio_data = np.frombuffer(data, dtype=np.int16)

        # Ensure the audio data is at the target sample rate
        if len(audio_data) == 0:
            print("Received empty audio data")
            return

        # Perform the transcription
        result, _ = model.transcribe(audio_data.astype(np.float32) / 32768.0, language="ko")
        transcript = " ".join([seg.text.strip() for seg in list(result)])
        print(f"Transcription: {transcript}")

        if not text_only:
            # Convert the transcribed text to speech with the default speaker and language
            tts_output = tts.tts(text=transcript, speaker=default_speaker_id, language=default_language)

            # Write the TTS output to a buffer at the correct sample rate
            with io.BytesIO() as buffer:
                sf.write(buffer, tts_output, samplerate=SAMPLE_RATE, format='WAV')
                buffer.seek(0)
                audio_bytes, sr = sf.read(buffer, dtype='int16')

            # Ensure the audio data is in the correct format
            if sr != SAMPLE_RATE:
                raise ValueError("Sample rate mismatch in audio processing")

            # Send TTS audio to LiveKit room
            await publish_tts_to_livekit(audio_bytes)

        # Send the transcription text to the transcription clients
        for client in send_transcript_clients:
            try:
                await client.send_text(json.dumps({"type": "transcription", "text": transcript}))
            except WebSocketDisconnect:
                send_transcript_clients.remove(client)
    except Exception as e:
        print("Error processing audio chunk:", e)

async def publish_tts_to_livekit(audio_data):
    # Create and configure the audio frame
    audio_frame = rtc.AudioFrame.create(SAMPLE_RATE, NUM_CHANNELS, len(audio_data))
    np.copyto(np.frombuffer(audio_frame.data, dtype=np.int16), audio_data)

    # Send the audio data frame to LiveKit
    await source.capture_frame(audio_frame)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run FastAPI server with LiveKit integration.')
    parser.add_argument('--livekit_url', type=str, required=True, help='LiveKit server URL')
    parser.add_argument('--livekit_token', type=str, required=True, help='LiveKit authentication token')

    args = parser.parse_args()

    # Initialize LiveKit Room
    livekit_room = None
    source = None  # Declare source at the top level to be accessible in the function

    async def initialize_livekit_room():
        global livekit_room
        global source
        livekit_room = rtc.Room()
        await livekit_room.connect(args.livekit_url, args.livekit_token)
        print(f"Connected to LiveKit room: {livekit_room.name}")

        # Create the audio source and track
        source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
        local_audio_track = rtc.LocalAudioTrack.create_audio_track("tts-audio", source)

        # Publish the audio track to the room
        await livekit_room.local_participant.publish_track(local_audio_track)
        print("Published TTS audio track to LiveKit room")

    loop = asyncio.get_event_loop()
    loop.run_until_complete(initialize_livekit_room())

    uvicorn.run(app, host="0.0.0.0", port=8000, ws_max_size=10485760, ws_ping_interval=30, ws_ping_timeout=30)  # Increase the max size to 10MB, set ping interval and timeout

