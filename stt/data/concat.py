from pydub import AudioSegment
import os

def get_next_segment(audio, start_ms, duration_ms):
    """Get the next segment of audio from the specified start point."""
    end_ms = start_ms + duration_ms
    return audio[start_ms:end_ms], end_ms

def main():
    # Input MP3 file paths and durations
    mp3_one = input("Enter the path for the first MP3 file: ")
    duration_one = int(input("Enter the duration (in seconds) for the first MP3 segment: ")) * 1000  # convert to milliseconds
    mp3_two = input("Enter the path for the second MP3 file: ")
    duration_two = int(input("Enter the duration (in seconds) for the second MP3 segment: ")) * 1000  # convert to milliseconds

    if not os.path.exists(mp3_one) or not os.path.exists(mp3_two):
        print("One or both of the MP3 files do not exist. Please check the file paths and try again.")
        return

    # Load the audio files
    audio_one = AudioSegment.from_file(mp3_one)
    audio_two = AudioSegment.from_file(mp3_two)
    pause_segment = AudioSegment.silent(duration=500)  # 0.5 second pause

    # Create the final mixed audio
    final_audio = AudioSegment.empty()

    # Define the starting points
    start_one = 0
    start_two = 0

    while start_one < len(audio_one) and start_two < len(audio_two):
        # Get the next segments from each audio file
        segment_one, start_one = get_next_segment(audio_one, start_one, duration_one)
        segment_two, start_two = get_next_segment(audio_two, start_two, duration_two)

        # Add the segments and pauses to the final audio
        final_audio += segment_one
        final_audio += pause_segment
        final_audio += segment_two
        final_audio += pause_segment

    # Export the final mixed audio
    final_output_path = "mixed_audio.mp3"
    final_audio.export(final_output_path, format="mp3")
    print(f"Final mixed audio has been exported to {final_output_path}")

if __name__ == "__main__":
    main()

