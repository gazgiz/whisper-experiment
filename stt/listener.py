import asyncio
import websockets
import sounddevice as sd
import soundfile as sf
import io
import numpy as np

is_recording = False
recorded_audio = []
audio_clip_number = 0

async def play_and_record_audio(audio_bytes):
    global recorded_audio
    # Play the audio
    with io.BytesIO(audio_bytes) as buffer:
        audio, sample_rate = sf.read(buffer)
        sd.play(audio, samplerate=sample_rate)
        sd.wait()

    if is_recording:
        recorded_audio.extend(audio)

async def receive_audio_from_server():
    uri = "ws://localhost:8000/ws"
    global audio_clip_number

    while True:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=10, close_timeout=10) as websocket:
                print("Listening for audio from server...")
                while True:
                    message = await websocket.recv()
                    if isinstance(message, bytes):
                        await play_and_record_audio(message)
                    elif isinstance(message, str):
                        if message == "START_RECORDING":
                            print("Start recording signal received.")
                            global is_recording, recorded_audio
                            is_recording = True
                            recorded_audio = []
                        elif message == "STOP_RECORDING":
                            print("Stop recording signal received.")
                            is_recording = False
                            # Save the recorded audio to a file
                            filename = f"recorded_output_{audio_clip_number}.wav"
                            with io.BytesIO() as buffer:
                                sf.write(buffer, np.array(recorded_audio), samplerate=22050, format='WAV')
                                buffer.seek(0)
                                with open(filename, "wb") as f:
                                    f.write(buffer.read())
                            print(f"Recorded audio saved to '{filename}'")
                            audio_clip_number += 1
        except websockets.ConnectionClosedError as e:
            print(f"Connection closed with error: {e}")
            await asyncio.sleep(5)  # Wait before reconnecting
        except Exception as e:
            print(f"Exception: {e}")
            await asyncio.sleep(5)  # Wait before reconnecting

async def main():
    await receive_audio_from_server()

if __name__ == "__main__":
    asyncio.run(main())

