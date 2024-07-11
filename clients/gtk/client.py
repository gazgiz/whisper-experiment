import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Gdk
import asyncio
import threading
import logging
from livekit import rtc
import numpy as np
import pyaudio
import os
import json

SAMPLE_RATE = 16000  # Microphone standard sample rate
NUM_CHANNELS = 1
CHUNK_SIZE = 1024  # Number of frames per buffer
CONFIG_FILE = os.path.expanduser("~/.cache/livekit_config.json")

# Configure logging
logging.basicConfig(level=logging.INFO)

class LiveKitApp(Gtk.Application):

    def __init__(self):
        super().__init__(application_id="com.example.LiveKitApp")
        self.connect("activate", self.on_activate)
        self.room = None
        self.local_audio_track = None
        self.chat_manager = None
        self.loop = asyncio.new_event_loop()
        self.stop_event = asyncio.Event()
        self.tasks = []
        threading.Thread(target=self.start_loop, args=(self.loop,), daemon=True).start()

    def start_loop(self, loop):
        asyncio.set_event_loop(loop)
        print("Starting asyncio event loop...")
        loop.run_forever()

    def on_activate(self, app):
        print("Activating application...")
        window = Gtk.ApplicationWindow(application=app)
        window.set_title("LiveKit WebRTC Session")
        window.set_default_size(800, 600)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        window.set_child(box)

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
        chat_scroll = Gtk.ScrolledWindow()
        chat_scroll.set_min_content_height(200)
        chat_scroll.set_child(self.chat_view)
        box.append(chat_scroll)

        # Load the last configuration if it exists
        self.load_last_config()

        window.show()

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
        print(f"Connecting to {room_name} at {url}...")

        # Save the configuration
        self.save_last_config(url, token, room_name)

        task = asyncio.run_coroutine_threadsafe(
            self.join_room(url, token, room_name), self.loop
        )
        self.tasks.append(task)

    async def join_room(self, url, token, room_name):
        global livekit_room, livekit_source, chat_manager

        livekit_room = rtc.Room()

        @livekit_room.on("participant_connected")
        def on_participant_connected(participant: rtc.RemoteParticipant):
            logging.info(
                "Participant connected: %s %s", participant.sid, participant.identity
            )

        @livekit_room.on("track_subscribed")
        async def on_track_subscribed(
            track: rtc.AudioTrack,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            logging.info("Audio track subscribed: %s", publication.sid)
            if track.kind == "audio":
                logging.info("Audio track received: %s", track.sid)
                task = asyncio.create_task(self.play_audio(track))
                self.tasks.append(task)

        try:
            await livekit_room.connect(url, token)
            logging.info(f"Connected to LiveKit room: {livekit_room.name}")
            self.update_status("Connected", "red")

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
                self.show_chat_message(chat_message)

            chat_manager.on_message(on_message)

            # Additional logging to verify ChatManager setup
            logging.info(f"ChatManager initialized for room: {livekit_room.name}")

            # Start publishing frames
            publish_task = asyncio.create_task(self.publish_frames(livekit_source))
            self.tasks.append(publish_task)

        except Exception as e:
            logging.error(f"Exception occurred: {e}")

    async def publish_frames(self, source):
        p = pyaudio.PyAudio()

        # Open stream
        stream = p.open(format=pyaudio.paInt16,
                        channels=NUM_CHANNELS,
                        rate=SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=CHUNK_SIZE)

        audio_frame = rtc.AudioFrame.create(SAMPLE_RATE, NUM_CHANNELS, CHUNK_SIZE)
        audio_data = np.frombuffer(audio_frame.data, dtype=np.int16)

        try:
            while not self.stop_event.is_set():
                # Read data from microphone
                mic_data = stream.read(CHUNK_SIZE)
                np.copyto(audio_data, np.frombuffer(mic_data, dtype=np.int16))
                await source.capture_frame(audio_frame)
        except asyncio.CancelledError:
            print("publish_frames cancelled")
        finally:
            print("Stopping publish_frames")
            # Close stream
            stream.stop_stream()
            stream.close()
            p.terminate()

    async def play_audio(self, track):
        p = pyaudio.PyAudio()

        # Open stream for playback
        stream = p.open(format=pyaudio.paInt16,
                        channels=NUM_CHANNELS,
                        rate=SAMPLE_RATE,
                        output=True,
                        frames_per_buffer=480)

        def on_audio_frame(frame):
            audio_data = np.frombuffer(frame.data, dtype=np.int16)
            stream.write(audio_data)

        track.on("audio_frame", on_audio_frame)

        try:
            while not self.stop_event.is_set():
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            print("play_audio cancelled")
        finally:
            print("Stopping play_audio")
            # Close stream
            stream.stop_stream()
            stream.close()
            p.terminate()

    def show_chat_message(self, chat_message):
        buffer = self.chat_view.get_buffer()
        buffer.insert(buffer.get_end_iter(), f"From {chat_message.participant.identity}:\n{chat_message.message}\n")

    def on_disconnect_clicked(self, button):
        self.stop_event.set()
        for task in self.tasks:
            task.cancel()
        asyncio.run_coroutine_threadsafe(self.disconnect(), self.loop)

    async def disconnect(self):
        await self.stop_tasks()
        if self.local_audio_track:
            await self.local_audio_track.stop()
            self.local_audio_track = None
            logging.info("Local audio track stopped")
        if self.room:
            await self.room.disconnect()
            self.room = None
            logging.info("Room disconnected.")
        self.update_status("Disconnected", "black")
        self.stop_event.clear()

    async def stop_tasks(self):
        await asyncio.gather(*self.tasks, return_exceptions=True)
        print("All tasks cancelled")
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
