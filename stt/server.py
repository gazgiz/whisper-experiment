import asyncio
import logging
from signal import SIGINT, SIGTERM
from typing import Union
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import numpy as np
import json
import soundfile as sf
import io
import torch
from faster_whisper import WhisperModel
from TTS.api import TTS
from livekit import api, rtc
import uvicorn
import argparse
from pysilero_vad import SileroVoiceActivityDetector

# Constants for audio settings
SAMPLE_RATE = 22050  # WebRTC standard sample rate
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

# Initialize Silero VAD
vad = SileroVoiceActivityDetector()

# Store connections
send_transcript_clients = []
chat_manager = None  # Declare chat_manager as a global variable

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
        print(f"Exception in receive_audio_endpoint: {e}")
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
        print(f"Exception in send_transcript_endpoint: {e}")
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

        # Check for voice activity
        #if vad(audio_data.astype(np.float32) / 32768.0) < 0.5:
        #    print("Silence detected")
        #    return

        print("Speech detected")

        # Perform the transcription
        result, _ = model.transcribe(audio_data.astype(np.float32) / 32768.0, language="ko")
        transcript = " ".join([seg.text.strip() for seg in list(result)])
        print(f"Transcription: {transcript}")

        # Send the transcription text to the transcription clients
        for client in send_transcript_clients:
            try:
                await client.send_text(json.dumps({"type": "transcription", "text": transcript}))
            except WebSocketDisconnect:
                send_transcript_clients.remove(client)

        # Send the transcription text to LiveKit chat
        global chat_manager
        if chat_manager:
            await chat_manager.send_message(transcript)  # Await the coroutine

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
    except Exception as e:
        print(f"Error processing audio chunk: {e}")

async def publish_tts_to_livekit(audio_data):
    try:
        # Create and configure the audio frame
        audio_frame = rtc.AudioFrame.create(SAMPLE_RATE, NUM_CHANNELS, len(audio_data))
        np.copyto(np.frombuffer(audio_frame.data, dtype=np.int16), audio_data)

        # Send the audio data frame to LiveKit
        await source.capture_frame(audio_frame)
    except Exception as e:
        print(f"Error publishing TTS to LiveKit: {e}")

async def main(room: rtc.Room, livekit_url: str, livekit_token: str) -> None:
    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
        logging.info(
            "participant connected: %s %s", participant.sid, participant.identity
        )

    @room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        logging.info(
            "participant disconnected: %s %s", participant.sid, participant.identity
        )

    @room.on("local_track_published")
    def on_local_track_published(
        publication: rtc.LocalTrackPublication,
        track: Union[rtc.LocalAudioTrack, rtc.LocalVideoTrack],
    ):
        logging.info("local track published: %s", publication.sid)

    @room.on("active_speakers_changed")
    def on_active_speakers_changed(speakers: list[rtc.Participant]):
        logging.info("active speakers changed: %s", speakers)

    @room.on("local_track_unpublished")
    def on_local_track_unpublished(publication: rtc.LocalTrackPublication):
        logging.info("local track unpublished: %s", publication.sid)

    @room.on("track_published")
    def on_track_published(
        publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        logging.info(
            "track published: %s from participant %s (%s)",
            publication.sid,
            participant.sid,
            participant.identity,
        )

    @room.on("track_unpublished")
    def on_track_unpublished(
        publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        logging.info("track unpublished: %s", publication.sid)

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logging.info("track subscribed: %s", publication.sid)
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            print("Subscribed to an Audio Track")
            audio_stream = rtc.AudioStream(track)

            async def process_audio_stream():
                try:
                    async for audio_frame_event in audio_stream:
                        audio_frame = audio_frame_event.frame
                        audio_data = audio_frame.data  # Correctly access data attribute
                        audio_data = np.frombuffer(audio_data, dtype=np.int16)
                        
                        # Use VAD to detect voice activity
                        #if vad(audio_data.astype(np.float32) / 32768.0) >= 0.5:
                        #    print("Speech detected")
                        await process_audio_chunk(audio_data)
                        #else:
                        #    print("Silence detected")
                except Exception as e:
                    print(f"Error processing audio stream: {e}")

            asyncio.create_task(process_audio_stream())

    @room.on("track_unsubscribed")
    def on_track_unsubscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        logging.info("track unsubscribed: %s", publication.sid)

    @room.on("track_muted")
    def on_track_muted(
        publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        logging.info("track muted: %s", publication.sid)

    @room.on("track_unmuted")
    def on_track_unmuted(
        publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        logging.info("track unmuted: %s", publication.sid)

    @room.on("data_received")
    def on_data_received(data: rtc.DataPacket):
        logging.info("received data from %s: %s", data.participant.identity, data.data)

    @room.on("connection_quality_changed")
    def on_connection_quality_changed(
        participant: rtc.Participant, quality: rtc.ConnectionQuality
    ):
        logging.info("connection quality changed for %s", participant.identity)

    @room.on("track_subscription_failed")
    def on_track_subscription_failed(
        participant: rtc.RemoteParticipant, track_sid: str, error: str
    ):
        logging.info("track subscription failed: %s %s", participant.identity, error)

    @room.on("connection_state_changed")
    def on_connection_state_changed(state: rtc.ConnectionState):
        logging.info("connection state changed: %s", state)

    @room.on("connected")
    def on_connected() -> None:
        logging.info("connected")

    @room.on("disconnected")
    def on_disconnected() -> None:
        logging.info("disconnected")

    @room.on("reconnecting")
    def on_reconnecting() -> None:
        logging.info("reconnecting")

    @room.on("reconnected")
    def on_reconnected() -> None:
        logging.info("reconnected")

    await room.connect(livekit_url, livekit_token)
    logging.info("connected to room %s", room.name)
    logging.info("participants: %s", room.participants)

    # Initialize ChatManager
    global chat_manager
    chat_manager = rtc.ChatManager(room)
    if not chat_manager:
        print("Failed to create chat manager")

    # Run the FastAPI server concurrently with LiveKit connection
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, ws_max_size=10485760, ws_ping_interval=30, ws_ping_timeout=30)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run FastAPI server with LiveKit integration.')
    parser.add_argument('--livekit_url', type=str, required=True, help='LiveKit server URL')
    parser.add_argument('--livekit_token', type=str, required=True, help='LiveKit authentication token')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler("basic_room.log"), logging.StreamHandler()],
    )

    loop = asyncio.get_event_loop()
    room = rtc.Room(loop=loop)

    async def cleanup():
        await room.disconnect()
        loop.stop()

    asyncio.ensure_future(main(room, args.livekit_url, args.livekit_token))
    for signal in [SIGINT, SIGTERM]:
        loop.add_signal_handler(signal, lambda: asyncio.ensure_future(cleanup()))

    try:
        loop.run_forever()
    finally:
        loop.close()

