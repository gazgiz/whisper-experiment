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
import collections
from pysilero_vad import SileroVoiceActivityDetector
import soundfile as sf
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import heapq
import logging
import langid

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
gpu_id = config['gpu_id']
language = config['language']
livekit_url = config['livekit_url']
room_name = config['room_name']
api_key = config['api_key']
api_secret = config['api_secret']
transcript_token = config['transcript_token']
peer_user_name = config['peer_user_name']
system_user_name = config['system_user_name']
record = config['record']
pause_in_ms = config['pause_in_ms']
initial_prompt = config['initial_prompt']

print(f"Language set to {language}")

# Load Whisper model
logging.getLogger('faster_whisper').setLevel(logging.WARNING)
device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
if device == "cuda":
    torch.cuda.set_device(gpu_id)
print("using CPU" if device == "cpu" else "using GPU")
model_size = "medium.en" if language == 'en' else "medium"
compute_type = "int8" if device == "cpu" else "float16"
model = WhisperModel(model_size, device=device, compute_type=compute_type)
chat_manager = None  # Declare chat_manager as a global variable
source = None
event_loop = None  # Event loop for asyncio operations

# Ring buffer for accumulating WebRTC audio data
resampled_buffer = collections.deque()
clip_buffer = collections.deque()
active_clip = False
silence_counter = 0
# pysilero processes in 30 ms chunks
clip_silence_trigger_counter = pause_in_ms // 30

# Min-heap for storing audio buffers with sequence numbers
sequence_number = 0
min_heap = []
heap_lock = threading.Lock()
buffer_available = threading.Condition(heap_lock)


# Initialize a buffer to accumulate audio data
audio_in = np.array([])

def record_audio_clip(audio_data, sample_rate=STT_SAMPLE_RATE, clip_duration=5):
    """
    Accumulates audio data until a 5-second clip is available and saves it to the 'audio_segments' directory.

    Parameters:
    audio_data (numpy.ndarray): The audio data to record.
    sample_rate (int): The sample rate of the audio data.
    clip_duration (int): The duration of the clip in seconds (default is 5 seconds).
    """
    global audio_in

    # Calculate the number of samples for the given duration
    num_samples = clip_duration * sample_rate

    # Append the new audio data to the buffer
    audio_in = np.append(audio_in, audio_data)

    # Check if the buffer has enough samples for a 5-second clip
    if len(audio_in) >= num_samples:
        # Define the directory path
        directory_path = "audio_segments"

        # Check if the directory exists, and create it if it doesn't
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # Define the file path
        wav_file_path = f"{directory_path}/clip_{int(time.time())}.wav"

        # Write the audio data to the file
        sf.write(wav_file_path, audio_in[:num_samples], sample_rate, subtype='PCM_16')

        # Remove the saved portion from the buffer
        audio_in = audio_in[num_samples:]


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

        if record:
            record_audio_clip(audio_data / 32768.0)

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
                    if clip_length_seconds >= 0.2:
                        transcribe(list(clip_buffer))
                    else:
                        logging.info(f"Discarded clip of length {clip_length_seconds:.2f} seconds")
                    clip_buffer.clear()
                    active_clip = False
    except Exception as e:
        logging.error("Error processing audio chunk", exc_info=True)

def detect_language(text):
    lang, confidence = langid.classify(text)
    return lang, confidence

def transcribe(clip_buffer):
    global chat_manager, event_loop

    try:
        audio_data = np.array(clip_buffer).astype(np.int16)
        result, _ = model.transcribe(audio_data.astype(np.float32) / 32768.0, language=language, initial_prompt=initial_prompt)
        transcript = " ".join([seg.text.strip() for seg in list(result)])

        # Check if the transcript is empty
        if not transcript.strip():
            print("Transcription is empty, skipping.")
            return
        print(f"{transcript}")

        # disable
        #lang, confidence = detect_language(transcript)
        #logging.info(f"Language {lang}, confidence {confidence}")

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


def start_pipeline():
    global pipeline
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

def main_gst_loop():
     # 0 = none, 1 = ERROR, 2 = WARNING, 3 = INFO, 4 = DEBUG, 5 = LOG
    #os.environ["GST_DEBUG"] = "livekit:6,webrtcsrc:6"
    os.environ["GST_DEBUG"] = "0"

    Gst.init(None)
    start_pipeline()
    glib_loop = GLib.MainLoop()
    try:
        glib_loop.run()
    except KeyboardInterrupt:
        pass

    pipeline.set_state(Gst.State.NULL)

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
