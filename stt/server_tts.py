import asyncio
import json
import io
import os
import logging
import soundfile as sf
import numpy as np
import torch
from TTS.api import TTS
from livekit import rtc
import threading
from queue import Queue
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

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
use_gpu = config['use_gpu']
gpu_id = config['gpu_id']
livekit_url = config['livekit_url']
room_name = config['room_name']
api_key = config['api_key']
api_secret = config['api_secret']
tts_token = config['tts_token']
language = config['language']
transcript_identity = config['transcript_identity']
system_user_name = config['system_user_name']

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

        # Send TTS audio to GStreamer pipeline
        await publish_tts_to_gst_pipeline(audio_bytes)
    except Exception as e:
        logging.error(f"Error processing TTS: {e}")

async def publish_tts_to_gst_pipeline(audio_data):
    global appsrc
    try:
        # Calculate the duration of the buffer
        duration = len(audio_data) / TTS_SAMPLE_RATE * Gst.SECOND
        buffer = Gst.Buffer.new_wrapped(audio_data.tobytes())

        # Set the timestamp and duration
        buffer.pts = Gst.CLOCK_TIME_NONE  # Use Gst.CLOCK_TIME_NONE to let GStreamer manage timestamps
        buffer.duration = duration

        # Push the buffer to the appsrc element
        appsrc.emit("push-buffer", buffer)
        logging.info(f"Sent audio frame of length: {len(audio_data)}")
    except Exception as e:
        logging.error(f"Error sending audio to GStreamer pipeline: {e}")

def message_processor():
    while True:
        transcript = message_queue.get()
        asyncio.run(process_transcript(transcript))
        message_queue.task_done()

def start_pipeline():
    global pipeline, appsrc
    Gst.init(None)

    pipeline = Gst.parse_launch(
        f"appsrc name=appsrc ! audio/x-raw,format=S16LE,channels=1,rate=22050,layout=interleaved "
        f"! audioconvert ! audioresample ! audio/x-raw,format=S16LE,channels=1,rate={WEBRTC_SAMPLE_RATE},layout=interleaved "
        f"! livekitwebrtcsink name=sink "
        f"signaller::ws-url={livekit_url} "
        f"signaller::api-key={api_key} "
        f"signaller::secret-key={api_secret} "
        f"signaller::room-name={room_name} "
        f"signaller::identity={system_user_name} "
        f"signaller::participant-name={system_user_name}"
    )

    appsrc = pipeline.get_by_name("appsrc")
    appsrc.set_property("format", Gst.Format.TIME)
    appsrc.set_property("block", True)
    appsrc.set_property("caps", Gst.caps_from_string("audio/x-raw,format=S16LE,channels=1,rate=22050,layout=interleaved"))

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message)

    pipeline.set_state(Gst.State.PLAYING)

def restart_pipeline():
    global pipeline
    pipeline.set_state(Gst.State.NULL)
    logging.info("Restarting pipeline...")
    start_pipeline()

def on_message(bus, message):
    if message.type == Gst.MessageType.EOS:
        logging.info("End of stream, restarting pipeline...")
        restart_pipeline()
    elif message.type == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        logging.error(f"Error: {err}, {debug}")
        restart_pipeline()
    elif message.type == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        logging.warning(f"Warning: {err}, {debug}")

def main_gst_loop():
    os.environ["GST_DEBUG"] = "3,livekit:6,webrtcsrc:6"

    start_pipeline()
    glib_loop = GLib.MainLoop()
    try:
        glib_loop.run()
    except KeyboardInterrupt:
        pass

    pipeline.set_state(Gst.State.NULL)

async def main():
    event_loop = asyncio.get_event_loop()
    room = rtc.Room(loop=event_loop)
    await room.connect(livekit_url, tts_token)
    logging.info("connected to room %s", room.name)
    logging.info("remote participants: %s", room.remote_participants)

    global source
    # Create the audio source and track
    source = rtc.AudioSource(WEBRTC_SAMPLE_RATE, NUM_CHANNELS)
    local_audio_track = rtc.LocalAudioTrack.create_audio_track("tts-audio", source)

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

    # setting up device
    logging.getLogger('faster_whisper').setLevel(logging.WARNING)
    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
    if device == "cuda":
        torch.cuda.set_device(gpu_id)
    print("TTS using GPU" if device == "cuda" else "TTS using CPU")

    # Load Coqui TTS model
    global tts
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
    tts.to(device)

    # Start the message processor thread
    processor_thread = threading.Thread(target=message_processor, daemon=True)
    processor_thread.start()

    # Start the GStreamer main loop in a separate thread
    gst_thread = threading.Thread(target=main_gst_loop)
    gst_thread.start()

    await asyncio.Future()  # Run forever

    gst_thread.join()
    processor_thread.join()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler("server_tts.log"), logging.StreamHandler()],
    )
    asyncio.run(main())