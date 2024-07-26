import asyncio
import json
import io
import logging
import soundfile as sf
import numpy as np
import librosa
from TTS.api import TTS
from livekit import rtc
import threading
from queue import Queue

# Constants for audio settings
WEBRTC_SAMPLE_RATE = 48000  # WebRTC default sample rate
TTS_SAMPLE_RATE = 22050
NUM_CHANNELS = 1
default_speaker_id = 'Andrew Chipper'  # Replace with any speaker ID from the list

# Load config
def load_config():
    with open('config_tts.json', 'r') as file:
        config = json.load(file)
    return config

config = load_config()
livekit_url = config['livekit_url']
room_name = config['room_name']
api_key = config['api_key']
api_secret = config['api_secret']
tts_token = config['tts_token']
language = config['language']
transcript_identity = config['transcript_identity']

# Blocking queue for chat messages
message_queue = Queue()

async def process_transcript(transcript):
    global tts
    try:
        # Convert the transcribed text to speech with the default speaker and language
        tts_output = tts.tts(text=transcript, speaker=default_speaker_id, language=language)

        # Write the TTS output to a buffer at the correct sample rate
        with io.BytesIO() as buffer:
            sf.write(buffer, tts_output, samplerate=TTS_SAMPLE_RATE, format='WAV')
            buffer.seek(0)
            audio_bytes, sr = sf.read(buffer, dtype='int16')

        # Ensure the audio data is in the correct format
        if sr != TTS_SAMPLE_RATE:
            raise ValueError("Sample rate mismatch in audio processing")

        # Send TTS audio to LiveKit room
        await publish_tts_to_livekit(audio_bytes)
    except Exception as e:
        logging.error(f"Error processing TTS: {e}")

async def publish_tts_to_livekit(audio_data):
    global source
    try:
        # Create and configure the audio frame
        resampled_audio = librosa.resample(audio_data.astype(np.float32), orig_sr=TTS_SAMPLE_RATE, target_sr=WEBRTC_SAMPLE_RATE)
        resampled_audio = resampled_audio.astype(np.int16)
        audio_frame = rtc.AudioFrame.create(WEBRTC_SAMPLE_RATE, NUM_CHANNELS, len(resampled_audio))
        np.copyto(np.frombuffer(audio_frame.data, dtype=np.int16), resampled_audio)

        # Send the audio data frame to LiveKit
        await source.capture_frame(audio_frame)
        logging.info(f"Sent audio frame of length: {len(audio_data)}")
    except Exception as e:
        logging.error(f"Error sending audio to LiveKit: {e}")

def message_processor():
    while True:
        transcript = message_queue.get()
        asyncio.run(process_transcript(transcript))
        message_queue.task_done()

async def main():
    event_loop = asyncio.get_event_loop()
    room = rtc.Room(loop=event_loop)
    await room.connect(livekit_url, tts_token)
    logging.info("Connected to room %s", room.name)
    logging.info("Participants: %s", room.participants)

    global source
    # Create the audio source and track
    source = rtc.AudioSource(WEBRTC_SAMPLE_RATE, NUM_CHANNELS)
    local_audio_track = rtc.LocalAudioTrack.create_audio_track("tts-audio", source)

    # Publish the audio track to the tts room
    await room.local_participant.publish_track(local_audio_track)
    logging.info("Published TTS audio track to room %s", room.name)

    # Initialize ChatManager to receive messages
    chat_manager = rtc.ChatManager(room)

    @chat_manager.on("message_received")
    def on_chat_received(msg: rtc.ChatMessage):
        if not msg.message:
            return
        if msg.participant.identity == transcript_identity:
            logging.info(f"{msg.message}")
            #print(f"message received: {msg.participant.identity}: {msg.message}")
            message_queue.put(msg.message)

    # Load Coqui TTS model
    logging.getLogger('TTS').setLevel(logging.WARNING)  # Turn off Coqui logging
    device_tts = "cpu"  # TTS using CPU
    global tts
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
    tts.to(device_tts)

    # Start the message processor thread
    processor_thread = threading.Thread(target=message_processor, daemon=True)
    processor_thread.start()

    await asyncio.Future()  # Run forever

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler("server_tts.log"), logging.StreamHandler()],
    )
    asyncio.run(main())
