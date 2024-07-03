import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GObject, Gdk, Pango
import asyncio
import threading
import logging
from livekit import rtc
import numpy as np
import pyaudio

SAMPLE_RATE = 48000
NUM_CHANNELS = 1

# Configure logging
logging.basicConfig(level=logging.DEBUG)


class LiveKitApp(Gtk.Application):

    def __init__(self):
        super().__init__(application_id="com.example.LiveKitApp")
        self.connect("activate", self.on_activate)
        self.room = None
        self.local_audio_track = None
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
        self.url_entry.set_text("wss://jupiter7-rdxxacqa.livekit.cloud")
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
        self.room_entry.set_text("SKT")
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

        window.show()

    def on_connect_clicked(self, button):
        print("Connect button clicked...")
        url = self.url_entry.get_text()
        token_buffer = self.token_entry.get_buffer()
        token = token_buffer.get_text(
            token_buffer.get_start_iter(), token_buffer.get_end_iter(), False
        )
        room_name = self.room_entry.get_text()
        print(f"URL: {url}")
        # Print only the first 30 characters of the token for privacy
        print(f"Token: {token[:30]}...")
        print(f"Room name: {room_name}")
        task = asyncio.run_coroutine_threadsafe(
            self.join_room(url, token, room_name), self.loop
        )
        self.tasks.append(task)

    async def publish_frames(self, source: rtc.AudioSource):
        p = pyaudio.PyAudio()

        # Open stream
        stream = p.open(format=pyaudio.paInt16,
                        channels=NUM_CHANNELS,
                        rate=SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=480)

        audio_frame = rtc.AudioFrame.create(SAMPLE_RATE, NUM_CHANNELS, 480)
        audio_data = np.frombuffer(audio_frame.data, dtype=np.int16)

        try:
            while not self.stop_event.is_set():
                # Read data from microphone
                mic_data = stream.read(480)
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

    async def join_room(self, url, token, room_name):
        print("Attempting to join room...")
        self.room = rtc.Room()

        @self.room.on("participant_connected")
        def on_participant_connected(participant: rtc.RemoteParticipant):
            logging.info(
                "Participant connected: %s %s", participant.sid, participant.identity
            )

        @self.room.on("track_subscribed")
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
            await self.room.connect(url, token)
            logging.info("Connected to room %s", self.room.name)
            self.update_status("Connected", "red")

            # Create and publish local audio track
            source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
            self.local_audio_track = rtc.LocalAudioTrack.create_audio_track(
                "mic-audio", source
            )
            options = rtc.TrackPublishOptions()
            options.source = rtc.TrackSource.SOURCE_MICROPHONE
            await self.room.local_participant.publish_track(
                self.local_audio_track, options
            )
            logging.info("Local audio track published")
            task = asyncio.create_task(self.publish_frames(source))
            self.tasks.append(task)
        except Exception as e:
            logging.error(f"Exception occurred: {e}")

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

    def on_disconnect_clicked(self, button):
        print("Disconnect button clicked...")
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
    """
    css_provider = Gtk.CssProvider()
    css_provider.load_from_data(css.encode())
    display = Gdk.Display.get_default()
    Gtk.StyleContext.add_provider_for_display(
        display, css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
    )

    app.run(None)

