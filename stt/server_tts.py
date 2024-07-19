import asyncio
import json
import io
import logging
import websockets
import soundfile as sf
import numpy as np
import librosa
from TTS.api import TTS
from livekit import rtc
import urllib.parse

# Constants for audio settings
WEBRTC_SAMPLE_RATE = 48000  # WebRTC default sample rate
TTS_SAMPLE_RATE = 22050
NUM_CHANNELS = 1
default_speaker_id = 'Andrew Chipper'  # Replace with any speaker ID from the list

# Load config
def load_config():
    with open('config.json', 'r') as file:
        config = json.load(file)
    return config

config = load_config()
do_tts = config['do_tts']
livekit_url = config['livekit_url']
room_name = config['room_name']
api_key = config['api_key']
api_secret = config['api_secret']
transcript_token = config['transcript_token']
tts_token = config['tts_token']
tts_url = config['tts_url']

# Load Coqui TTS model
if do_tts:
    device_tts = "cpu"  # TTS using CPU
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
    tts.to(device_tts)

async def process_transcript(websocket, path):
    async for message in websocket:
        data = json.loads(message)
        transcript = data.get("transcript")
        if transcript:
            synthesize_and_send_tts(transcript)

async def synthesize_and_send_tts(transcript):
    try:
        tts_output = tts.tts(text=transcript, speaker=default_speaker_id)
        with io.BytesIO(tts_output) as audio_file:
            audio_data, sample_rate = sf.read(audio_file, dtype='int16')
            resampled_audio = librosa.resample(audio_data.astype(np.float32), orig_sr=sample_rate, target_sr=WEBRTC_SAMPLE_RATE)
            send_audio_to_livekit(resampled_audio)
    except Exception as e:
        logging.error(f"Error processing TTS: {e}")

def send_audio_to_livekit(audio_data):
    # Implementation of sending audio data to LiveKit room (depends on LiveKit's API)
    pass

async def main():
    parsed_url = urllib.parse.urlparse(tts_url)
    host = parsed_url.hostname
    port = parsed_url.port
    async with websockets.serve(process_transcript,host , port):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler("server_tts.log"), logging.StreamHandler()],
    )
    asyncio.run(main())
