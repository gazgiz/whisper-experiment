import threading
import json
import io
import os
import time
import asyncio
import numpy as np
from livekit import api, rtc
import torch
from faster_whisper import WhisperModel
import librosa
import collections
from pysilero_vad import SileroVoiceActivityDetector
import soundfile as sf
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import heapq
from queue import Queue
import logging


# Constants for audio settings
STT_SAMPLE_RATE = 16000  # Target sample rate for processing

# Initialize VAD
vad = SileroVoiceActivityDetector()
vad_chunk_size = vad.chunk_samples()

def load_config():
    with open('config_stt.json', 'r') as file:
        config = json.load(file)
    return config

config = load_config()
use_gpu = config['use_gpu']
language = config['language']
livekit_url = config['livekit_url']
room_name = config['room_name']
api_key = config['api_key']
api_secret = config['api_secret']
transcript_token = config['transcript_token']
peer_user_name = config['peer_user_name']
system_user_name = config['system_user_name']
tts_url = config['tts_url']

print(f"Language set to {language}")

# Load Whisper model
logging.getLogger('faster_whisper').setLevel(logging.WARNING)
device_stt = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
print("STT using GPU" if device_stt == "cuda" else "STT using CPU")
model_size = "medium"
if device_stt == "cuda":
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
else:
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

chat_manager = None  # Declare chat_manager as a global variable
source = None
event_loop = None  # Event loop for asyncio operations

# Ring buffer for accumulating WebRTC audio data
resampled_buffer = collections.deque()
clip_buffer = collections.deque()
active_clip = False
silence_counter = 0
clip_silence_trigger_counter = 5

# Min-heap for storing audio buffers with sequence numbers
sequence_number = 0
min_heap = []
heap_lock = threading.Lock()
buffer_available = threading.Condition(heap_lock)


def process_audio_chunk(data):
    global clip_buffer, active_clip, silence_counter
    global clip_silence_trigger_counter, chat_manager, event_loop
    try:
        #logging.warning(f"Original audio length: {len(data)}")
        # Convert the received audio data to numpy array
        audio_data = np.frombuffer(data, dtype=np.int16)
        #logging.warning(f"np buffer audio length: {len(audio_data)}")

        # Ensure the audio data is at the target sample rate
        if len(audio_data) == 0:
            logging.warning("Received empty audio data")
            return

        # Resample the audio data
        resampled_buffer.extend(audio_data)

        # Process the resampled buffer in chunks for VAD
        while len(resampled_buffer) >= vad_chunk_size:
            vad_chunk = [resampled_buffer.popleft() for _ in range(vad_chunk_size)]
            vad_chunk = np.array(vad_chunk)
            # Apply VAD
            if vad.process_chunk(vad_chunk.tobytes()) >= 0.7:
                clip_buffer.extend(vad_chunk)
                active_clip = True
                silence_counter = 0
            else:
                silence_counter += 1
                if active_clip and silence_counter > clip_silence_trigger_counter:
                    clip_length_seconds = len(clip_buffer) / STT_SAMPLE_RATE
                    if clip_length_seconds >= 1.0:
                        transcribe(list(clip_buffer))  # Process the clip buffer for STT and TTS
                    else:
                        logging.info(f"Discarded clip of length {clip_length_seconds:.2f} seconds")
                    clip_buffer.clear()
                    active_clip = False
    except Exception as e:
        logging.error("Error processing audio chunk", exc_info=True)

def transcribe(clip_buffer):
    global chat_manager, event_loop, tts_queue

    try:
        audio_data = np.array(clip_buffer).astype(np.int16)

        # record to disk
        #wav_file_path = f"/audio_segments/clip_{int(time.time())}.wav"
        #sf.write(wav_file_path, audio_data, STT_SAMPLE_RATE, subtype='PCM_16')

        # Perform the transcription
        result, _ = model.transcribe(audio_data.astype(np.float32) / 32768.0, language=language)
        transcript = " ".join([seg.text.strip() for seg in list(result)])
        # Check if the transcript is empty
        if not transcript.strip():
            print("Transcription is empty, skipping.")
            return
        print(f"{transcript}")

        if chat_manager:
            # Send transcript to chat manager
            event_loop.call_soon_threadsafe(asyncio.create_task, chat_manager.send_message(transcript))
    except Exception as e:
        logging.error(f"Error processing STT: {e}")

def process_audio_from_heap():
    while True:
        with heap_lock:
            while not min_heap:
                buffer_available.wait()
            _, data = heapq.heappop(min_heap)
        process_audio_chunk(data)

def on_message(bus, message, loop):
    if message.type == Gst.MessageType.EOS:
        print("End of stream")
        loop.quit()
    elif message.type == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err}, {debug}")
        loop.quit()
    elif message.type == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        print(f"Warning: {err}, {debug}")

def on_handoff(fakesink, buffer, pad):
    try:
        data = buffer.extract_dup(0, buffer.get_size())
        
        # Extract sequence number from buffer metadata
        seq_num = buffer.pts  # Using PTS (Presentation Time Stamp) as the sequence number
        
        with heap_lock:
            heapq.heappush(min_heap, (seq_num, data))
            buffer_available.notify()
    except Exception as e:
        logging.error("Error in on_handoff", exc_info=True)
    return Gst.FlowReturn.OK

def main_livekit():
    global event_loop
    asyncio.set_event_loop(asyncio.new_event_loop())
    event_loop = asyncio.get_event_loop()
    room = rtc.Room(loop=event_loop)
    event_loop.run_until_complete(room.connect(livekit_url, transcript_token))
    logging.info("connected to room %s", room.name)
    logging.info("remote participants: %s", room.remote_participants)

    global chat_manager
    chat_manager = rtc.ChatManager(room)
    if not chat_manager:
        logging.error("Failed to create chat manager")

    event_loop.run_forever()

def main_gst_loop():
    Gst.init(None)
    pipeline = Gst.parse_launch(
        f"livekitwebrtcsrc name=src "
        f"signaller::ws-url={livekit_url} "
        f"signaller::api-key={api_key} "
        f"signaller::secret-key={api_secret} "
        f"signaller::producer-peer-id={peer_user_name} "
        f"signaller::room-name={room_name} "
        f"signaller::identity={system_user_name} "
        f"signaller::participant-name={system_user_name} "
        f"src. ! queue ! audioconvert ! audio/x-raw,channels=1,rate={STT_SAMPLE_RATE} ! fakesink name=fakesink-1 sync=true signal-handoffs=true"
    )

    fakesink = pipeline.get_by_name("fakesink-1")
    if not fakesink:
        logging.error("Failed to get fakesink element from pipeline")
        return

    fakesink.connect("handoff", on_handoff)

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    glib_loop = GLib.MainLoop()

    bus.connect("message", on_message, glib_loop)

    pipeline.set_state(Gst.State.PLAYING)

    try:
        glib_loop.run()
    except KeyboardInterrupt:
        pass

    pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("uam_server.log"),
            logging.StreamHandler()
        ]
    )
    # Start the LiveKit connection in a separate thread
    livekit_thread = threading.Thread(target=main_livekit)
    livekit_thread.start()

    # Start the GStreamer main loop in a separate thread
    gst_thread = threading.Thread(target=main_gst_loop)
    gst_thread.start()

    # Start a single thread to process audio buffers from the heap
    processor_thread = threading.Thread(target=process_audio_from_heap)
    processor_thread.start()

    # Ensure threads finish
    livekit_thread.join()
    gst_thread.join()
    processor_thread.join()
