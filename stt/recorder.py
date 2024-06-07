import asyncio
import websockets
import sounddevice as sd
import numpy as np
import webrtcvad
import collections

is_recording = False

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
    uri = "ws://localhost:8000/ws"

    async with websockets.connect(uri) as websocket:
        # Convert audio data to bytes
        audio_bytes = audio_data

        # Send audio data
        await websocket.send(audio_bytes)
        print("Audio sent to server")

        # Receive and print the transcription
        transcription = await websocket.recv()
        print(f"Transcription received: {transcription}")

async def main():
    global is_recording
    sample_rate = 16000  # Sample rate in Hz
    vad = webrtcvad.Vad()
    vad.set_mode(1)  # 0: least aggressive, 3: most aggressive

    print("Press 'r' to start recording...")

    loop = asyncio.get_event_loop()

    while True:
        # Wait for 'r' key press to start recording
        while not is_recording:
            user_input = await loop.run_in_executor(None, input)
            if user_input.strip().lower() == 'r':
                is_recording = True
                break

        if is_recording:
            audio_data = await record_audio(sample_rate=sample_rate, vad=vad)
            await send_audio_to_server(audio_data, sample_rate)
            print("Press 'r' to start recording again...")

if __name__ == "__main__":
    asyncio.run(main())

