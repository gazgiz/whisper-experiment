import threading
import logging
import json
import io
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor
from signal import SIGINT, SIGTERM
import numpy as np
from livekit import api, rtc
import torch
from faster_whisper import WhisperModel
from TTS.api import TTS
import librosa
import collections
from pysilero_vad import SileroVoiceActivityDetector
import soundfile as sf
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Constants for audio settings
WEBRTC_SAMPLE_RATE = 48000  # WebRTC default sample rate
STT_SAMPLE_RATE = 16000  # Target sample rate for processing
TTS_SAMPLE_RATE = 22050
NUM_CHANNELS = 1

# Initialize VAD
vad = SileroVoiceActivityDetector()
vad_chunk_size = vad.chunk_samples()

def load_config():
    with open('config.json', 'r') as file:
        config = json.load(file)
    return config

config = load_config()
use_gpu = config['use_gpu']
do_tts = config['do_tts']
language = config['language']
livekit_url = config['livekit_url']
room_name = config['peer_user_name']
peer_user_name = config['peer_user_name']
system_user_name = config['system_user_name']
api_key = config['api_key']
api_secret = config['api_secret']
system_token = config['system_token']

print(f"Language set to {language}")

# Load Whisper model
device_stt = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
print("STT using GPU" if device_stt == "cuda" else "STT using CPU")
model_size = "medium"
if device_stt == "cuda":
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
else:
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

# Load Coqui TTS model
if do_tts:
    device_tts = "cpu"
    print("TTS using GPU" if device_tts == "cuda" else "TTS using CPU")
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
    tts.to(device_tts)
default_speaker_id = 'Andrew Chipper'  # Replace with any speaker ID from the list

chat_manager = None  # Declare chat_manager as a global variable
source = None
event_loop = None  # Event loop for asyncio operations

# Thread-safe queue for audio buffers
audio_buffer_queue = queue.Queue()

# Ring buffer for accumulating WebRTC audio data
resampled_buffer = collections.deque()
clip_buffer = collections.deque()
active_clip = False
silence_counter = 0
clip_silence_trigger_counter = 10

def process_audio_chunk(data):
    global resampled_buffer, clip_buffer, active_clip, silence_counter
    global clip_silence_trigger_counter, chat_manager, event_loop

    try:
        # Convert the received audio data to numpy array
  #      logging.info("Converting audio data to numpy array")
        audio_data = np.frombuffer(data, dtype=np.int16)
  #      logging.info(f"Audio data length: {len(audio_data)}")

        # Ensure the audio data is at the target sample rate
        if len(audio_data) == 0:
            logging.warning("Received empty audio data")
            return

 #       logging.info("Resampling audio data")
        resampled_chunk = librosa.resample(audio_data.astype(np.float32), orig_sr=WEBRTC_SAMPLE_RATE, target_sr=STT_SAMPLE_RATE)
 #       logging.info(f"Resampled chunk length: {len(resampled_chunk)}")
        # Add the resampled audio data to the buffer
        resampled_buffer.extend(resampled_chunk.astype(np.int16))
 #       logging.info(f"Resampled buffer length: {len(resampled_buffer)}")
        # Process the resampled buffer in chunks for VAD
        while len(resampled_buffer) >= vad_chunk_size:
            vad_chunk = [resampled_buffer.popleft() for _ in range(vad_chunk_size)]
            vad_chunk = np.array(vad_chunk)
            # Apply VAD
            if vad.process_chunk(vad_chunk.tobytes()) >= 0.7:
 #               logging.info(f"Added chunk of length {len(vad_chunk)} to clip")
                clip_buffer.extend(vad_chunk)
                active_clip = True
                silence_counter = 0
            else:
                silence_counter += 1
                if active_clip and silence_counter > clip_silence_trigger_counter:
                    clip_length_seconds = len(clip_buffer) / STT_SAMPLE_RATE
                    if clip_length_seconds >= 1.0:
 #                       logging.info(f"Processing clip buffer of length {clip_length_seconds} seconds")
                        transcribe_and_tts_clip(list(clip_buffer))  # Process the clip buffer for STT and TTS
                    else:
                        logging.info(f"Discarded clip of length {clip_length_seconds:.2f} seconds")
                    clip_buffer.clear()
                    active_clip = False
    except Exception as e:
        logging.error("Error processing audio chunk", exc_info=True)

def transcribe_and_tts_clip(clip_buffer):
    global chat_manager, event_loop

    try:
        # Perform the transcription
        result, _ = model.transcribe(np.array(clip_buffer).astype(np.float32) / 32768.0, language=language)

        transcript = " ".join([seg.text.strip() for seg in list(result)])
        # Check if the transcript is empty
        if not transcript.strip():
            logging.info("Transcription is empty, skipping.")
            return
        logging.info(f"\nTranscription: {transcript}\n")
        if chat_manager:
            # Use event loop to run the send_message coroutine
            event_loop.call_soon_threadsafe(asyncio.create_task, chat_manager.send_message(transcript))
        if do_tts:
            synthesize_tts(transcript)
    except Exception as e:
        logging.error(f"Error processing STT: {e}")

def synthesize_tts(transcript):
    try:
        tts_output = tts.tts(text=transcript, speaker=default_speaker_id, language=language)

        with io.BytesIO() as buffer:
            sf.write(buffer, tts_output, samplerate=TTS_SAMPLE_RATE, format='WAV')
            buffer.seek(0)
            audio_bytes, sr = sf.read(buffer, dtype='int16')

        global source
        # Create and configure the audio frame
        resampled_chunk = librosa.resample(audio_bytes.astype(np.float32), orig_sr=TTS_SAMPLE_RATE, target_sr=WEBRTC_SAMPLE_RATE)
        resampled_buffer.extend(resampled_chunk.astype(np.int16))
        audio_frame = rtc.AudioFrame.create(WEBRTC_SAMPLE_RATE, NUM_CHANNELS, len(resampled_buffer))
        np.copyto(np.frombuffer(audio_frame.data, dtype=np.int16), resampled_buffer)

        # Send the audio data frame to LiveKit
        source.capture_frame(audio_frame)

    except Exception as e:
        logging.error(f"Error processing TTS: {e}")

def process_audio_from_queue():
    while True:
        data = audio_buffer_queue.get()
        process_audio_chunk(data)
        audio_buffer_queue.task_done()

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
#        logging.info("Handoff received")
        data = buffer.extract_dup(0, buffer.get_size())
#        logging.info(f"Buffer extracted of size: {len(data)}")
        audio_buffer_queue.put(data)  # Put the data into the thread-safe queue
    except Exception as e:
        logging.error("Error in on_handoff", exc_info=True)
    return Gst.FlowReturn.OK

def main_livekit():
    global event_loop
    asyncio.set_event_loop(asyncio.new_event_loop())
    event_loop = asyncio.get_event_loop()
    room = rtc.Room(loop=event_loop)
    event_loop.run_until_complete(room.connect(livekit_url, system_token))
    logging.info("connected to room %s", room.name)
    logging.info("participants: %s", room.participants)

    global chat_manager
    chat_manager = rtc.ChatManager(room)
    if not chat_manager:
        logging.error("Failed to create chat manager")

    if do_tts:
        global source
        # Create the audio source and track
        source = rtc.AudioSource(WEBRTC_SAMPLE_RATE, NUM_CHANNELS)
        local_audio_track = rtc.LocalAudioTrack.create_audio_track("tts-audio", source)

        # Publish the audio track to the room
        event_loop.run_until_complete(room.local_participant.publish_track(local_audio_track))
        print("Published TTS audio track to LiveKit tts room")

    event_loop.run_forever()

def main_gst_loop():
    Gst.init(None)
    pipeline = Gst.parse_launch(
        f"livekitwebrtcsrc name=src "
        f"signaller::ws-url={livekit_url} "
        f"signaller::api-key={api_key} "
        f"signaller::secret-key={api_secret} "
        f"signaller::room-name={room_name} "
        f"signaller::producer-peer-id={peer_user_name} "
        f"signaller::identity={system_user_name} "
        f"signaller::participant-name={system_user_name} "
        "src. ! queue ! audioconvert ! fakesink name=fakesink-1 sync=true signal-handoffs=true"
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
        handlers=[logging.FileHandler("uam_server.log"), logging.StreamHandler()],
    )

    # Start the LiveKit connection in a separate thread
    livekit_thread = threading.Thread(target=main_livekit)
    livekit_thread.start()

    # Start the GStreamer main loop in a separate thread
    gst_thread = threading.Thread(target=main_gst_loop)
    gst_thread.start()

    # Start a thread pool to process audio buffers from the queue
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.submit(process_audio_from_queue)

    # Ensure threads finish
    livekit_thread.join()
    gst_thread.join()

