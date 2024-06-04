import asyncio
import websockets
import sounddevice as sd
import numpy as np
import json

async def record_audio(duration=5, sample_rate=16000):
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    print("Recording complete")
    return audio_data

async def send_audio_to_server(audio_data, sample_rate=16000):
    uri = "ws://localhost:8000/ws"

    async with websockets.connect(uri) as websocket:
        # Start transcription
        await websocket.send(json.dumps({"action": "start_transcription"}))
        response = await websocket.recv()
        response_data = json.loads(response)
        job_id = response_data.get("job_id")
        print(f"Job ID: {job_id}")

        # Convert audio data to bytes
        audio_bytes = audio_data.tobytes()

        # Send audio data
        await websocket.send(audio_bytes)

        # Wait for transcription result
        transcription = await websocket.recv()
        transcription_data = json.loads(transcription)
        print(f"Transcription: {transcription_data.get('transcript')}")

async def main():
    duration = 5  # Duration of recording in seconds
    sample_rate = 16000  # Sample rate in Hz

    audio_data = await record_audio(duration, sample_rate)
    await send_audio_to_server(audio_data, sample_rate)

if __name__ == "__main__":
    asyncio.run(main())

