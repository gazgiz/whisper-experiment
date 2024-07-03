import asyncio
import websockets
import sounddevice as sd
import soundfile as sf
import numpy as np
import webrtcvad
import collections
import io
import json

is_recording = False
recorded_audio = []
audio_clip_number = 0

async def receive_transcription_from_server(uri):
    while True:
        try:
            async with websockets.connect(uri, ping_interval=60, ping_timeout=60, close_timeout=20) as websocket:
                print("Listening for transcription from server...")
                while True:
                    message = await websocket.recv()
                    if isinstance(message, str):
                        if message.strip():  # Only process non-empty messages
                            message_data = json.loads(message)
                            if message_data["type"] == "transcription":
                                print(f"Transcription received: {message_data['text']}")
        except websockets.ConnectionClosedError as e:
            print(f"Connection closed with error: {e}")
            await asyncio.sleep(5)  # Wait before reconnecting
        except Exception as e:
            print(f"Exception: {e}")
            await asyncio.sleep(5)  # Wait before reconnecting

async def record_audio(sample_rate=16000, frame_duration_ms=30, padding_duration_ms=300, vad=None):
    global is_recording
    num_padding_frames = padding_duration_ms // frame_duration_ms
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []

    stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16")
    stream.start()

    print("Recording... Press Ctrl+C to stop.")

    try:
        while True:
            frame = stream.read(int(sample_rate * frame_duration_ms / 1000))[0].tobytes()
            is_speech = vad.is_speech(frame, sample_rate)

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

    print("Recording complete")
    is_recording = False
    return b"".join(voiced_frames)

async def send_audio_to_server(audio_data, sample_rate=16000):
    uri = "ws://localhost:8000/receive_audio"
    retries = 5

    while retries > 0:
        try:
            async with websockets.connect(uri, ping_interval=60, ping_timeout=60, close_timeout=20) as websocket:
                # Convert audio data to bytes
                audio_bytes = audio_data

                # Send audio data
                await websocket.send(audio_bytes)
                print("Audio sent to server")
                return
        except (websockets.ConnectionClosedError, asyncio.TimeoutError) as e:
            print(f"Connection error: {e}, retrying...")
            retries -= 1
            await asyncio.sleep(2)
        except Exception as e:
            print(f"Unexpected error: {e}")
            await asyncio.sleep(2)
    print("Failed to send audio to server after multiple attempts.")

async def main():
    global is_recording
    sample_rate = 16000  # Sample rate in Hz
    vad = webrtcvad.Vad()
    vad.set_mode(1)  # 0: least aggressive, 3: most aggressive

    receive_transcript_uri = "ws://localhost:8000/send_transcript"

    # Start the transcription listening coroutine
    receive_transcript_task = asyncio.create_task(receive_transcription_from_server(receive_transcript_uri))

    print("Recording continuously...")

    while True:
        audio_data = await record_audio(sample_rate=sample_rate, vad=vad)
        send_audio_task = asyncio.create_task(send_audio_to_server(audio_data, sample_rate))
        await send_audio_task  # Wait for the send task to complete before starting a new recording

if __name__ == "__main__":
    asyncio.run(main())

