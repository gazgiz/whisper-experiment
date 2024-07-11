import asyncio
import logging
from signal import SIGINT, SIGTERM
import numpy as np
import json
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from TTS.api import TTS
from livekit import api, rtc
import argparse
import librosa
import datetime
import collections
from pysilero_vad import SileroVoiceActivityDetector

# Constants for audio settings
SAMPLE_RATE = 48000  # WebRTC default sample rate
TARGET_SAMPLE_RATE = 16000  # Target sample rate for processing
NUM_CHANNELS = 1
AUDIO_DIR = "audio_segments"

# Initialize VAD
vad = SileroVoiceActivityDetector()
vad_chunk_size = vad.chunk_samples()

def load_config():
    with open('config.json', 'r') as file:
        config = json.load(file)
    return config

config = load_config()
use_gpu = config['use_gpu']
text_only = config['text_only']

# Load Whisper model
device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
print("Using GPU" if device == "cuda" else "Using CPU")
model_size = "medium"
if device == "cuda":
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
else:
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

# Load Coqui TTS model
if not text_only:
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
    tts.to(device)
default_speaker_id = 'Andrew Chipper'  # Replace with any speaker ID from the list
default_language = 'ko'

chat_manager = None  # Declare chat_manager as a global variable

# Ring buffer for accumulating WebRTC audio data
audio_buffer = collections.deque()
resampled_buffer = collections.deque()
clip_buffer = collections.deque()
active_clip = False
silence_counter = 0
clip_silence_trigger_counter = 8

async def process_audio_chunk(data, source="websocket"):
    global audio_buffer, resampled_buffer, clip_buffer, active_clip, silence_counter
    global clip_silence_trigger_counter, chat_manager

    try:
        # Convert the received audio data to numpy array
        audio_data = np.frombuffer(data, dtype=np.int16)

        # Ensure the audio data is at the target sample rate
        if len(audio_data) == 0:
            logging.warning("Received empty audio data")
            return

        resampled_chunk = librosa.resample(audio_data.astype(np.float32), orig_sr=SAMPLE_RATE, target_sr=TARGET_SAMPLE_RATE)
        # Add the resampled audio data to the buffer
        resampled_buffer.extend(resampled_chunk.astype(np.int16))

        # Process the resampled buffer in chunks for VAD
        while len(resampled_buffer) >= vad_chunk_size:
            vad_chunk = [resampled_buffer.popleft() for _ in range(vad_chunk_size)]
            vad_chunk = np.array(vad_chunk)
            # Apply VAD
            if vad.process_chunk(vad_chunk.tobytes()) >= 0.5:
                #logging.info(f"Added chunk of length {len(vad_chunk)} to clip")
                clip_buffer.extend(vad_chunk)
                active_clip = True
                silence_counter = 0
            else:
                silence_counter += 1
                if active_clip and silence_counter > clip_silence_trigger_counter:
                    # Perform the transcription
                    result, _ = model.transcribe(np.array(clip_buffer).astype(np.float32) / 32768.0, language="ko")
                    transcript = " ".join([seg.text.strip() for seg in list(result)])
                    print(f"Transcription: {transcript}")
                    # Send the transcription text to LiveKit chat
                    if chat_manager:
                        await chat_manager.send_message(transcript)  # Await the coroutine
                    clip_buffer.clear()
                    active_clip = False
    except Exception as e:
        logging.error(f"Error processing audio chunk: {e}", exc_info=True)

async def main(room: rtc.Room, livekit_url: str, livekit_token: str) -> None:

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logging.info("track subscribed: %s", publication.sid)
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logging.info("Subscribed to an Audio Track")
            audio_stream = rtc.AudioStream(track)

            async def process_audio_stream():
                try:
                    async for audio_frame_event in audio_stream:
                        audio_frame = audio_frame_event.frame
                        audio_data = audio_frame.data  # Correctly access data attribute
                        audio_data = np.frombuffer(audio_data, dtype=np.int16)
                        
                        # Process the received audio data
                        await process_audio_chunk(audio_data, source="webrtc")
                except Exception as e:
                    logging.error(f"Error processing audio stream: {e}", exc_info=True)

            asyncio.create_task(process_audio_stream())

    await room.connect(livekit_url, livekit_token)
    logging.info("connected to room %s", room.name)
    logging.info("participants: %s", room.participants)

    # Initialize ChatManager
    global chat_manager
    chat_manager = rtc.ChatManager(room)
    if not chat_manager:
        logging.error("Failed to create chat manager")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run FastAPI server with LiveKit integration.')
    parser.add_argument('--livekit_url', type=str, required=True, help='LiveKit server URL')
    parser.add_argument('--livekit_token', type=str, required=True, help='LiveKit authentication token')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler("uam_server.log"), logging.StreamHandler()],
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

