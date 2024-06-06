import asyncio
import websockets
import sounddevice as sd
import soundfile as sf
import io
import numpy as np

is_recording = False
recorded_audio = []

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

    async with websockets.connect(uri) as websocket:
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
                    with io.BytesIO() as buffer:
                        sf.write(buffer, np.array(recorded_audio), samplerate=22050, format='WAV')
                        buffer.seek(0)
                        with open("recorded_output.wav", "wb") as f:
                            f.write(buffer.read())
                    print("Recorded audio saved to 'recorded_output.wav'")
                    break

async def main():
    await receive_audio_from_server()

if __name__ == "__main__":
    asyncio.run(main())

