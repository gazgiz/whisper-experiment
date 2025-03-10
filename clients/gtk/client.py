import os
import json
import sys
import time
import gi
import asyncio
import threading
import logging
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Gdk, GLib
from livekit import rtc
import numpy as np
import pyaudio
import soundfile as sf


SAMPLE_RATE_MIC = 16000  # Microphone standard sample rate
SAMPLE_RATE_WEBRTC = 48000  # WebRTC sample rate
NUM_CHANNELS = 1
CHUNK_SIZE = 1024  # Number of frames per buffer
CONFIG_FILE = os.path.expanduser("~/.cache/livekit_config.json")

# Configure logging
logging.basicConfig(level=logging.INFO)

class LiveKitApp(Gtk.Application):

    def __init__(self):
        super().__init__(application_id="com.example.LiveKitApp")
        self.audio_in = np.array([])
        self.connect("activate", self.on_activate)
        self.livekit_room = None
        self.local_audio_track = None
        self.chat_manager = None
        self.livekit_source = None
        self.loop = asyncio.new_event_loop()
        self.stop_event = asyncio.Event()
        self.tasks = []
        self.loop_thread = threading.Thread(target=self.start_loop, args=(self.loop,), daemon=True)
        self.loop_thread.start()
        self.scroll_idle_id = None

    def start_loop(self, loop):
        asyncio.set_event_loop(loop)
        logging.info("Starting asyncio event loop...")
        try:
            loop.run_forever()
        except Exception as e:
            logging.error(f"Exception in event loop: {e}")
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception as e:
                logging.error(f"Exception during loop shutdown: {e}")
            loop.close()
            logging.info("Event loop closed.")

    def on_activate(self, app):
        logging.info("Activating application...")
        self.window = Gtk.ApplicationWindow(application=app)
        self.window.set_title("LiveKit WebRTC Session")
        self.window.set_default_size(800, 600)
        self.window.connect("destroy", self.on_window_destroy)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.window.set_child(box)

        # URL Entry
        url_label = Gtk.Label(label="LiveKit URL:")
        box.append(url_label)
        self.url_entry = Gtk.Entry()
        self.url_entry.set_placeholder_text("Enter LiveKit URL")
        box.append(self.url_entry)

        # Token Entry
        token_label = Gtk.Label(label="Enter LiveKit Token:")
        box.append(token_label)
        self.token_entry = Gtk.TextView()
        self.token_entry.set_wrap_mode(Gtk.WrapMode.WORD)
        token_scroll = Gtk.ScrolledWindow()
        token_scroll.set_min_content_height(100)
        token_scroll.set_child(self.token_entry)
        box.append(token_scroll)

        # Room Name Entry
        room_label = Gtk.Label(label="Room Name:")
        box.append(room_label)
        self.room_entry = Gtk.Entry()
        self.room_entry.set_placeholder_text("Enter room name")
        box.append(self.room_entry)

        # Connect Button
        self.connect_button = Gtk.Button(label="Connect")
        self.connect_button.connect("clicked", self.on_connect_clicked)
        box.append(self.connect_button)

        # Disconnect Button
        self.disconnect_button = Gtk.Button(label="Disconnect")
        self.disconnect_button.connect("clicked", self.on_disconnect_clicked)
        box.append(self.disconnect_button)

        # Status Label
        self.status_label = Gtk.Label(label="Disconnected")
        box.append(self.status_label)

        # Chat Messages View
        chat_label = Gtk.Label(label="Chat Messages:")
        box.append(chat_label)
        self.chat_view = Gtk.TextView()
        self.chat_view.set_editable(False)
        self.chat_view.set_cursor_visible(False)
        self.chat_view.set_wrap_mode(Gtk.WrapMode.WORD)
        self.chat_buffer = self.chat_view.get_buffer()
        chat_scroll = Gtk.ScrolledWindow()
        chat_scroll.set_min_content_height(200)
        chat_scroll.set_child(self.chat_view)
        box.append(chat_scroll)

        # Load the last configuration if it exists
        self.load_last_config()

        # Set the connect button as the default widget
        self.window.set_default_widget(self.connect_button)

        self.window.show()

    def scroll_to_end(self):
        adj = self.chat_view.get_vadjustment()
        adj.set_value(adj.get_upper() - adj.get_page_size())
        self.scroll_idle_id = None
        return False  # Returning False removes the idle source after execution

    def show_chat_message(self, message):
        def insert_message():
            buffer = self.chat_view.get_buffer()
            buffer.insert(buffer.get_end_iter(), f"{message}\n")
            # Schedule scroll_to_end to run on the main UI thread
            if self.scroll_idle_id is None:
                self.scroll_idle_id = GLib.idle_add(self.scroll_to_end)
            return False  # Returning False to remove the idle source

        # Ensure the message is inserted from the main thread
        GLib.idle_add(insert_message)

    def load_last_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as file:
                config = json.load(file)
                self.url_entry.set_text(config.get("url", ""))
                buffer = self.token_entry.get_buffer()
                buffer.set_text(config.get("token", ""))
                self.room_entry.set_text(config.get("room_name", ""))

    def save_last_config(self, url, token, room_name):
        config = {
            "url": url,
            "token": token,
            "room_name": room_name
        }
        with open(CONFIG_FILE, "w") as file:
            json.dump(config, file)

    def on_connect_clicked(self, button):
        url = self.url_entry.get_text()
        token_buffer = self.token_entry.get_buffer()
        token = token_buffer.get_text(token_buffer.get_start_iter(), token_buffer.get_end_iter(), False)
        room_name = self.room_entry.get_text()
        logging.info(f"Connecting to {room_name} at {url}...")

        # Save the configuration
        self.save_last_config(url, token, room_name)

        task = asyncio.run_coroutine_threadsafe(
            self.join_room(url, token, room_name), self.loop
        )
        self.tasks.append(task)



    async def play_audio(self, audio_stream):
        p = pyaudio.PyAudio()

        # Open stream for playback
        stream = p.open(format=pyaudio.paInt16,
                        channels=NUM_CHANNELS,
                        rate=SAMPLE_RATE_WEBRTC,
                        output=True,
                        frames_per_buffer=480)

        logging.info("Audio playback stream opened")

        try:
            async for frame_event in audio_stream:
                frame = frame_event.frame
                audio_data = np.frombuffer(frame.data, dtype=np.int16)
                #logging.info(f"Received audio frame with {len(audio_data)} samples")
                # Optional: Log the first few samples for debugging
                #logging.debug(f"Audio data: {audio_data[:10]}")
                #self.record_audio_clip(audio_data / 32768.0)
                stream.write(audio_data.tobytes())
                #logging.info("Audio data written to playback stream")


        except asyncio.CancelledError:
            logging.info("play_audio cancelled")
        except Exception as e:
            logging.error(f"Exception in play_audio: {e}")
        finally:
            logging.info("Stopping play_audio")
            try:
                stream.stop_stream()
                stream.close()
                p.terminate()
            except Exception as e:
                logging.error(f"Exception during stream close: {e}")

    def record_audio_clip(self, record_data, sample_rate=SAMPLE_RATE_MIC, clip_duration=5):
        """
        Accumulates audio data until a 5-second clip is available and saves it to the 'client_audio_segments' directory.

        Parameters:
        audio_data (numpy.ndarray): The audio data to record.
        sample_rate (int): The sample rate of the audio data.
        clip_duration (int): The duration of the clip in seconds (default is 5 seconds).
        """
        # Calculate the number of samples needed for the desired clip duration
        num_samples = clip_duration * sample_rate

        # Append the new audio data to the buffer
        self.audio_in = np.append(self.audio_in, record_data)

        # Check if the buffer has enough samples for a full clip
        if len(self.audio_in) >= num_samples:
            # Define the directory path
            directory_path = "client_audio_segments"
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            # Define the file path using the current timestamp to ensure uniqueness
            wav_file_path = f"{directory_path}/clip_{int(time.time())}.wav"

            # Write the audio data to the file, ensuring proper format
            sf.write(wav_file_path, self.audio_in[:num_samples], sample_rate, subtype='PCM_16')

            # Remove the saved portion from the buffer
            self.audio_in = self.audio_in[num_samples:]

            logging.info(f"Audio clip saved: {wav_file_path}")


    async def join_room(self, url, token, room_name):
        self.livekit_room = rtc.Room()

        @self.livekit_room.on("participant_connected")
        def on_participant_connected(participant: rtc.RemoteParticipant):
            logging.info("Participant connected: %s %s", participant.sid, participant.identity)

        @self.livekit_room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            logging.info("Track subscribed: %s", publication.sid)
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                logging.info("Audio track received: %s", track.sid)
                _audio_stream = rtc.AudioStream(track)
                task = asyncio.run_coroutine_threadsafe(
                    self.play_audio(_audio_stream), self.loop
                )
                self.tasks.append(task)

        try:
            await self.livekit_room.connect(url, token)
            logging.info(f"Connected to LiveKit room: {self.livekit_room.name}")
            logging.info("remote participants: %s", self.livekit_room.remote_participants)
            self.update_status("Connected", "red")

            # Set the disconnect button as the default widget
            self.window.set_default_widget(self.disconnect_button)

            # Create the audio source and track
            self.livekit_source = rtc.AudioSource(SAMPLE_RATE_MIC, NUM_CHANNELS)
            self.local_audio_track = rtc.LocalAudioTrack.create_audio_track("audio-track", self.livekit_source)

            # Publish the audio track to the room
            await self.livekit_room.local_participant.publish_track(self.local_audio_track)
            logging.info("Published audio track to LiveKit room")

            # Initialize ChatManager to receive messages
            self.chat_manager = rtc.ChatManager(self.livekit_room)

            @self.chat_manager.on("message_received")
            def on_chat_received(msg: rtc.ChatMessage):
                if not msg.message:
                    return
                print(f"{msg.message}")
                self.show_chat_message(msg.message)

            # Additional logging to verify ChatManager setup
            logging.info(f"ChatManager initialized for room: {self.livekit_room.name}")

            # Start publishing frames
            publish_task = asyncio.create_task(self.publish_frames(self.livekit_source))
            self.tasks.append(publish_task)

        except Exception as e:
            logging.error(f"Exception occurred: {e}")

    async def publish_frames(self, source):
        p = pyaudio.PyAudio()

        # Open stream
        stream = p.open(format=pyaudio.paInt16,
                        channels=NUM_CHANNELS,
                        rate=SAMPLE_RATE_MIC,
                        input=True,
                        frames_per_buffer=CHUNK_SIZE)

        logging.info("Microphone stream opened for publishing frames")

        audio_frame = rtc.AudioFrame.create(SAMPLE_RATE_MIC, NUM_CHANNELS, CHUNK_SIZE)
        audio_data = np.frombuffer(audio_frame.data, dtype=np.int16)

        try:
            while not self.stop_event.is_set():
                # Read data from microphone
                mic_data = stream.read(CHUNK_SIZE)
                #logging.info(f"Read {len(mic_data)} bytes from microphone")
                np.copyto(audio_data, np.frombuffer(mic_data, dtype=np.int16))
                #self.record_audio_clip(audio_data / 32768.0)
                await source.capture_frame(audio_frame)
                logging.debug("Captured audio frame from microphone")
        except asyncio.CancelledError:
            logging.info("publish_frames cancelled")
        except Exception as e:
            logging.error(f"Exception in publish_frames: {e}")
        finally:
            logging.info("Stopping publish_frames")
            try:
                stream.stop_stream()
                stream.close()
                p.terminate()
            except Exception as e:
                logging.error(f"Exception during stream close: {e}")

    def on_disconnect_clicked(self, button):
        self.stop_event.set()
        for task in self.tasks:
            if isinstance(task, asyncio.Task):
                task.cancel()
        disconnect_task = asyncio.run_coroutine_threadsafe(self.disconnect(), self.loop)
        disconnect_task.result()  # Wait for disconnect to complete

    def on_window_destroy(self, window):
        logging.info("Window destroy event triggered.")
        if self.livekit_room:
            self.on_disconnect_clicked(None)  # Call the disconnect handler
        else:
            self.stop_event.set()  # Ensure the event is set even if not connected
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.loop_thread.join()  # Ensure the loop thread has finished
        except Exception as e:
            logging.error(f"Exception during window destroy: {e}")
        Gtk.main_quit()  # Exit the GTK main loop

    async def disconnect(self):
        logging.info("Starting disconnect process...")
        await self.stop_tasks()
        logging.info("Tasks stopped.")
        if self.local_audio_track:
            try:
                publication_sid = self.local_audio_track.sid
                await self.livekit_room.local_participant.unpublish_track(publication_sid)
                self.local_audio_track = None
                logging.info("Local audio track unpublished")
            except Exception as e:
                logging.error(f"Exception during unpublish track: {e}")
        if self.livekit_room:
            try:
                await self.livekit_room.disconnect()
                self.livekit_room = None
                logging.info("Room disconnected.")
            except Exception as e:
                logging.error(f"Exception during room disconnect: {e}")
        if self.chat_manager:
            self.chat_manager = None
            logging.info("Chat manager released.")
        if self.livekit_source:
            self.livekit_source = None
            logging.info("Audio source released.")
        self.update_status("Disconnected", "black")

        # Set the connect button as the default widget
        self.window.set_default_widget(self.connect_button)

        self.stop_event.clear()
        logging.info("Disconnect process complete.")

    async def stop_tasks(self):
        logging.info("Stopping tasks...")
        try:
            await asyncio.gather(*[task for task in self.tasks if isinstance(task, asyncio.Task)], return_exceptions=True)
        except Exception as e:
            logging.error(f"Exception during stopping tasks: {e}")
        logging.info("All tasks cancelled")
        self.tasks.clear()

    def update_status(self, status, color):
        self.status_label.set_text(status)
        context = self.status_label.get_style_context()
        context.remove_class("connected")
        context.remove_class("disconnected")
        if color == "red":
            context.add_class("connected")
        else:
            context.add_class("disconnected")

if __name__ == "__main__":
    try:
        app = LiveKitApp()

        # Add CSS for connected and disconnected classes
        css = """
        .connected {
            color: red;
        }
        .disconnected {
            color: black;
        }
        """
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(css.encode())
        display = Gdk.Display.get_default()
        Gtk.StyleContext.add_provider_for_display(
            display, css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        app.run(None)
    except Exception as e:
        logging.error(f"Exception in main: {e}")
        sys.exit(1)
    finally:
        logging.info("Application shutdown.")
