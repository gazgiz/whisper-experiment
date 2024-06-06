import asyncio
import websockets
import sounddevice as sd

is_recording = False

async def record_audio(duration=5, sample_rate=16000):
    global is_recording
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    print("Recording complete")
    is_recording = False
    return audio_data

async def send_audio_to_server(audio_data, sample_rate=16000):
    uri = "ws://localhost:8000/ws"

    async with websockets.connect(uri) as websocket:
        # Convert audio data to bytes
        audio_bytes = audio_data.tobytes()

        # Send audio data
        await websocket.send(audio_bytes)
        print("Audio sent to server")

async def main():
    global is_recording
    duration = 10  # Duration of recording in seconds
    sample_rate = 16000  # Sample rate in Hz

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
            audio_data = await record_audio(duration, sample_rate)
            await send_audio_to_server(audio_data, sample_rate)
            print("Press 'r' to start recording again...")

if __name__ == "__main__":
    asyncio.run(main())

