import asyncio
import websockets
import sounddevice as sd
import soundfile as sf
import io
import numpy as np

async def play_and_record_audio(audio_bytes):
    # Play the audio
    with io.BytesIO(audio_bytes) as buffer:
        audio, sample_rate = sf.read(buffer)
        sd.play(audio, samplerate=sample_rate)
        sd.wait()

    # Record the audio being played
    print("Recording the received audio...")
    recorded_audio = sd.rec(int(len(audio) * sample_rate / len(audio)), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    print("Recording complete")

    # Save the recorded audio to a file
    with io.BytesIO() as buffer:
        sf.write(buffer, recorded_audio, samplerate=sample_rate, format='WAV')
        buffer.seek(0)
        with open("recorded_output.wav", "wb") as f:
            f.write(buffer.read())

    print("Recorded audio saved to 'recorded_output.wav'")

async def receive_audio_from_server():
    uri = "ws://localhost:8000/ws"

    async with websockets.connect(uri) as websocket:
        print("Listening for audio from server...")
        while True:
            audio_bytes = await websocket.recv()
            await play_and_record_audio(audio_bytes)

async def main():
    await receive_audio_from_server()

if __name__ == "__main__":
    asyncio.run(main())

