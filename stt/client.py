import argparse
import asyncio
import numpy as np
import sounddevice as sd
import webrtcvad
import collections
import json
import logging
import wave
import os
from livekit import rtc

logging.basicConfig(level=logging.INFO)

SAMPLE_RATE = 16000  # microphone standard sample rate
NUM_CHANNELS = 1

is_recording = False
livekit_room = None
livekit_source = None
chat_manager = None

async def receive_transcription_from_server(uri):
    import websockets
    while True:
        try:
            async with websockets.connect(uri, ping_interval=60, ping_timeout=60, close_timeout=20) as websocket:
                logging.info("Listening for transcription from server...")
                while True:
                    message = await websocket.recv()
                    if isinstance(message, str):
                        if message.strip():  # Only process non-empty messages
                            message_data = json.loads(message)
                            if message_data["type"] == "transcription":
                                logging.info(f"Transcription received: {message_data['text']}")
        except websockets.ConnectionClosedError as e:
            logging.error(f"Connection closed with error: {e}")
            await asyncio.sleep(5)  # Wait before reconnecting
        except Exception as e:
            logging.error(f"Exception: {e}")
            await asyncio.sleep(5)  # Wait before reconnecting

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

async def record_audio(sample_rate=16000, frame_duration_ms=30, padding_duration_ms=300, vad=None):
    global is_recording
    num_padding_frames = padding_duration_ms // frame_duration_ms
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []

    stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16")
    stream.start()

    logging.info("Recording... Press Ctrl+C to stop.")

    try:
        while True:
            frame = stream.read(int(sample_rate * frame_duration_ms / 1000))[0]
            is_speech = vad.is_speech(frame.tobytes(), sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True
                    voiced_frames.extend([f for f, s in ring_buffer])
                    ring_buffer.clear()
            else:
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    triggered = False
                    stream.stop()
                    break
    except KeyboardInterrupt:
        pass
    finally:
        stream.close()

    logging.info("Recording complete")
    is_recording = False
    return b"".join([f.tobytes() for f in voiced_frames])

async def send_audio_to_server(audio_data, sample_rate=16000):
    import websockets
    uri = "ws://localhost:8000/receive_audio"
    retries = 5

    while retries > 0:
        try:
            async with websockets.connect(uri, ping_interval=60, ping_timeout=60, close_timeout=20) as websocket:
                # Send audio data
                await websocket.send(audio_data)
                logging.info("Audio sent to server")
                return
        except (websockets.ConnectionClosedError, asyncio.TimeoutError) as e:
            logging.error(f"Connection error: {e}, retrying...")
            retries -= 1
            await asyncio.sleep(2)
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            await asyncio.sleep(2)
    logging.error("Failed to send audio to server after multiple attempts.")

async def main(livekit_url, livekit_token):
    if livekit_url and livekit_token:
        await connect_to_livekit(livekit_url, livekit_token)

    vad = webrtcvad.Vad()
    vad.set_mode(1)  # 0: least aggressive, 3: most aggressive

    if not (livekit_url and livekit_token):
        receive_transcript_uri = "ws://localhost:8000/send_transcript"
        receive_transcript_task = asyncio.create_task(receive_transcription_from_server(receive_transcript_uri))

    logging.info("Recording continuously...")

    while True:
        audio_data = await record_audio(sample_rate=SAMPLE_RATE, vad=vad)
        
        if livekit_url and livekit_token:
            # Convert bytes back to int16 array for LiveKit
            int16_audio_data = np.frombuffer(audio_data, dtype=np.int16)
            send_audio_task = asyncio.create_task(send_audio_to_livekit(int16_audio_data))
        else:
            send_audio_task = asyncio.create_task(send_audio_to_server(audio_data))
        
        await send_audio_task  # Wait for the send task to complete before starting a new recording

async def send_audio_to_livekit(audio_data):
    audio_frame = rtc.AudioFrame.create(SAMPLE_RATE, NUM_CHANNELS, len(audio_data))
    np.copyto(np.frombuffer(audio_frame.data, dtype=np.int16), audio_data)
    await livekit_source.capture_frame(audio_frame)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LiveKit Audio Client.')
    parser.add_argument('--livekit_url', type=str, help='LiveKit server URL')
    parser.add_argument('--livekit_token', type=str, help='LiveKit authentication token')
    args = parser.parse_args()

    asyncio.run(main(args.livekit_url, args.livekit_token))

