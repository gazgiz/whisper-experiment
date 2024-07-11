import argparse
import asyncio
import numpy as np
import pyaudio
import logging
from livekit import rtc
import threading
import time

logging.basicConfig(level=logging.INFO)

SAMPLE_RATE = 16000  # Microphone standard sample rate
NUM_CHANNELS = 1
CHUNK_SIZE = 1024  # Number of frames per buffer

livekit_room = None
livekit_source = None
chat_manager = None

audio_queue = asyncio.Queue()

async def connect_to_livekit(livekit_url, livekit_token):
    global livekit_room, livekit_source, chat_manager
    livekit_room = rtc.Room()
    await livekit_room.connect(livekit_url, livekit_token)
    logging.info(f"Connected to LiveKit room: {livekit_room.name}")

    # Create the audio source and track
    livekit_source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
    audio_track = rtc.LocalAudioTrack.create_audio_track("audio-track", livekit_source)

    # Publish the audio track to the room
    await livekit_room.local_participant.publish_track(audio_track)
    logging.info("Published audio track to LiveKit room")

    # Initialize ChatManager to receive messages
    chat_manager = rtc.ChatManager(livekit_room)

    def on_message(chat_message):
        logging.info(f"Chat message received from {chat_message.participant.identity}: {chat_message.message}")

    chat_manager.on_message(on_message)

    # Additional logging to verify ChatManager setup
    logging.info(f"ChatManager initialized for room: {livekit_room.name}")

def audio_callback(in_data, frame_count, time_info, status):
    """This function is called for each audio block from the microphone."""
    if status:
        logging.warning(f"Audio callback status: {status}")
    audio_queue.put_nowait(in_data)
    return (in_data, pyaudio.paContinue)

def start_audio_capture():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=NUM_CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE,
                    stream_callback=audio_callback)
    logging.info("Starting audio stream...")
    stream.start_stream()

    while stream.is_active():
        time.sleep(0.1)  # Keep the stream open indefinitely
    stream.stop_stream()
    stream.close()
    p.terminate()

async def send_audio_to_livekit():
    while True:
        audio_data = await audio_queue.get()
        int16_audio_data = np.frombuffer(audio_data, dtype=np.int16)
        audio_frame = rtc.AudioFrame.create(SAMPLE_RATE, NUM_CHANNELS, len(int16_audio_data))
        np.copyto(np.frombuffer(audio_frame.data, dtype=np.int16), int16_audio_data)
        await livekit_source.capture_frame(audio_frame)

async def main(livekit_url, livekit_token):
    await connect_to_livekit(livekit_url, livekit_token)

    logging.info("Recording continuously...")

    send_task = asyncio.create_task(send_audio_to_livekit())
    await send_task

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LiveKit Audio Client.')
    parser.add_argument('--livekit_url', type=str, help='LiveKit server URL')
    parser.add_argument('--livekit_token', type=str, help='LiveKit authentication token')
    args = parser.parse_args()

    loop = asyncio.get_event_loop()

    # Start the audio capturing in a separate thread
    audio_thread = threading.Thread(target=start_audio_capture, daemon=True)
    audio_thread.start()

    # Run the main asyncio event loop
    loop.run_until_complete(main(args.livekit_url, args.livekit_token))
