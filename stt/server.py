import asyncio
import logging
from signal import SIGINT, SIGTERM
import numpy as np
import json
import io
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from TTS.api import TTS
from livekit import api, rtc
import argparse
import librosa
import collections
from pysilero_vad import SileroVoiceActivityDetector

# Constants for audio settings
WEBRTC_SAMPLE_RATE = 48000  # WebRTC default sample rate
STT_SAMPLE_RATE = 16000  # Target sample rate for processing
TTS_SAMPLE_RATE = 22050
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
do_tts = config['do_tts']
language = config['language']

print(f"Language set to {language}")

# Load Whisper model
# Note: force to CPU if do_tts is true
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

# Ring buffer for accumulating WebRTC audio data
audio_buffer = collections.deque()
resampled_buffer = collections.deque()
clip_buffer = collections.deque()
active_clip = False
silence_counter = 0
clip_silence_trigger_counter = 10

# Queue for TTS and STT transcripts
transcript_queue = asyncio.Queue()
stt_queue = asyncio.Queue()

async def process_tts_queue():
    while True:
        transcript = await transcript_queue.get()
        await process_tts(transcript)
        transcript_queue.task_done()

async def process_tts(transcript):
    try:
        if not transcript:
            raise ValueError("Transcript is empty. Define `text` for synthesis.")

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
        await source.capture_frame(audio_frame)

    except Exception as e:
        logging.error(f"Error processing TTS: {e}", exc_info=True)

async def process_stt_queue():
    while True:
        clip_buffer = await stt_queue.get()
        await process_stt(clip_buffer)
        stt_queue.task_done()

async def process_stt(clip_buffer):
    try:
        # Perform the transcription
        result, _ = model.transcribe(np.array(clip_buffer).astype(np.float32) / 32768.0, language=language)

        transcript = " ".join([seg.text.strip() for seg in list(result)])
        # Check if the transcript is empty
        if not transcript.strip():
            print("Transcription is empty, skipping.")
            return
        print(f"\nTranscription: {transcript}\n")
        if chat_manager:
            await chat_manager.send_message(transcript)
        if do_tts:
            await transcript_queue.put(transcript)  # Enqueue the transcript for TTS processing
    except Exception as e:
        logging.error(f"Error processing STT: {e}", exc_info=True)

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

        resampled_chunk = librosa.resample(audio_data.astype(np.float32), orig_sr=WEBRTC_SAMPLE_RATE, target_sr=STT_SAMPLE_RATE)
        # Add the resampled audio data to the buffer
        resampled_buffer.extend(resampled_chunk.astype(np.int16))

        # Process the resampled buffer in chunks for VAD
        while len(resampled_buffer) >= vad_chunk_size:
            vad_chunk = [resampled_buffer.popleft() for _ in range(vad_chunk_size)]
            vad_chunk = np.array(vad_chunk)
            # Apply VAD
            if vad.process_chunk(vad_chunk.tobytes()) >= 0.7:
                #logging.info(f"Added chunk of length {len(vad_chunk)} to clip")
                clip_buffer.extend(vad_chunk)
                active_clip = True
                silence_counter = 0
            else:
                silence_counter += 1
                if active_clip and silence_counter > clip_silence_trigger_counter:
                    clip_length_seconds = len(clip_buffer) / STT_SAMPLE_RATE
                    if clip_length_seconds >= 1.0:
                        await stt_queue.put(list(clip_buffer))  # Enqueue the clip buffer for STT processing
                    else:
                        logging.info(f"Discarded clip of length {clip_length_seconds:.2f} seconds")
                    clip_buffer.clear()
                    active_clip = False
    except Exception as e:
        logging.error(f"Error processing audio chunk: {e}", exc_info=True)

async def main(livekit_url: str, room_stt: rtc.Room, livekit_token_stt: str, room_tts: rtc.Room, livekit_token_tts: str) -> None:

    @room_stt.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logging.info("track subscribed: %s", publication.sid)
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logging.info("Subscribed to STT Audio Track")
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
                    logging.error(f"Error processing STT audio stream: {e}", exc_info=True)

            asyncio.create_task(process_audio_stream())

    await room_stt.connect(livekit_url, livekit_token_stt)
    logging.info("connected to stt room %s", room_stt.name)
    logging.info("participants: %s", room_stt.participants)

    global chat_manager
    chat_manager = rtc.ChatManager(room_stt)
    if not chat_manager:
        logging.error("Failed to create chat manager")

    if do_tts:
        await room_tts.connect(livekit_url, livekit_token_tts)
        logging.info("connected to tts room %s", room_tts.name)
        logging.info("participants: %s", room_tts.participants)

        global source
        # Create the audio source and track
        source = rtc.AudioSource(WEBRTC_SAMPLE_RATE, NUM_CHANNELS)
        local_audio_track = rtc.LocalAudioTrack.create_audio_track("tts-audio", source)

        # Publish the audio track to the tts room
        await room_tts.local_participant.publish_track(local_audio_track)
        print("Published TTS audio track to LiveKit tts room")

    # Start the STT and TTS processing tasks
    asyncio.create_task(process_stt_queue())
    asyncio.create_task(process_tts_queue())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run FastAPI server with LiveKit integration.')
    parser.add_argument('--livekit_url', type=str, required=True, help='LiveKit server URL')
    parser.add_argument('--livekit_token_stt', type=str, required=True, help='LiveKit authentication token for STT audio')
    parser.add_argument('--livekit_token_tts', type=str, required=True, help='LiveKit authentication token for TTS audio')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler("uam_server.log"), logging.StreamHandler()],
    )

    loop = asyncio.get_event_loop()
    room_stt = rtc.Room(loop=loop)
    room_tts = None
    if do_tts:
        room_tts = rtc.Room(loop=loop)

    async def cleanup():
        await room_stt.disconnect()
        if room_tts:
            await room_tts.disconnect()
        loop.stop()

    asyncio.ensure_future(main(args.livekit_url, room_stt, args.livekit_token_stt, room_tts, args.livekit_token_tts))
    for signal in [SIGINT, SIGTERM]:
        loop.add_signal_handler(signal, lambda: asyncio.ensure_future(cleanup()))

    try:
        loop.run_forever()
    finally:
        loop.close()
