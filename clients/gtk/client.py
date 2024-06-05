import gi
import asyncio
import threading
import websockets
import json
from gi.repository import Gtk, GObject
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRecorder

# Ensure GTK 4 is used
gi.require_version('Gtk', '4.0')

class LiveKitApp(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="com.example.LiveKitApp")
        self.connect("activate", self.on_activate)

    def on_activate(self, app):
        print("Activating application...")
        window = Gtk.ApplicationWindow(application=app)
        window.set_title("LiveKit WebRTC Session")
        window.set_default_size(800, 600)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        window.set_child(box)

        self.url_entry = Gtk.Entry()
        self.url_entry.set_text("wss://jupiter7-rdxxacqa.livekit.cloud")
        self.url_entry.set_placeholder_text("Enter LiveKit URL")
        box.append(self.url_entry)

        self.token_label = Gtk.Label(label="Enter LiveKit Token:")
        box.append(self.token_label)
        
        self.token_entry = Gtk.TextView()
        self.token_entry.set_wrap_mode(Gtk.WrapMode.WORD)
        token_buffer = self.token_entry.get_buffer()
        token_buffer.set_text(
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3MTcxODYyNTMsImlzcyI6IkFQSTQ0Q0F1TlNyODdyOCIsIm5iZiI6MTcxNzA5NjI1Mywic3ViIjoiU3dpZnQiLCJ2aWRlbyI6eyJjYW5QdWJsaXNoIjp0cnVlLCJjYW5QdWJsaXNoRGF0YSI6dHJ1ZSwiY2FuU3Vic2NyaWJlIjp0cnVlLCJyb29tIjoiVUFUTSIsInJvb21Kb2luIjp0cnVlfX0.yyDNHVHA3oztzTqhc9RCEtEovZefngrwoI2Pn1pst10",
            -1
        )
        token_scroll = Gtk.ScrolledWindow()
        token_scroll.set_min_content_height(100)
        token_scroll.set_child(self.token_entry)
        box.append(token_scroll)

        self.room_entry = Gtk.Entry()
        self.room_entry.set_text("SKT")
        self.room_entry.set_placeholder_text("Enter room name")
        box.append(self.room_entry)

        self.button = Gtk.Button(label="Connect")
        self.button.connect("clicked", self.on_connect_clicked)
        box.append(self.button)

        self.video_area = Gtk.DrawingArea()
        box.append(self.video_area)

        self.peer_connection = None
        self.recorder = None

        window.show()

    def on_connect_clicked(self, button):
        print("Connect button clicked...")
        url = self.url_entry.get_text()
        token_buffer = self.token_entry.get_buffer()
        token = token_buffer.get_text(token_buffer.get_start_iter(), token_buffer.get_end_iter(), False)
        room_name = self.room_entry.get_text()
        print(f"URL: {url}")
        print(f"Token: {token[:30]}...")  # Print only the first 30 characters of the token for privacy
        print(f"Room name: {room_name}")
        loop = asyncio.get_event_loop()
        if loop.is_running():
            print("Event loop is running, scheduling join_room...")
            asyncio.run_coroutine_threadsafe(self.join_room(url, token, room_name), loop)
        else:
            print("Event loop is not running. Starting new event loop...")
            asyncio.run(self.join_room(url, token, room_name))

    async def join_room(self, url, token, room_name):
        print("Attempting to join room...")
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        websocket_url = url.replace("https", "wss") + "/rtc"

        try:
            print(f"Connecting to WebSocket at {websocket_url} with headers: {headers}")
            async with websockets.connect(websocket_url, extra_headers=headers) as websocket:
                join_data = {
                    'method': 'join',
                    'room': room_name
                }
                await websocket.send(json.dumps(join_data))
                
                response = await websocket.recv()
                try:
                    response_text = response.decode('utf-8')
                except (UnicodeDecodeError, AttributeError):
                    response_text = response  # Handle cases where response is not a string
                
                try:
                    join_response = json.loads(response_text)
                    print(f"Join response: {join_response}")
                except json.JSONDecodeError:
                    print(f"Failed to decode JSON response: {response_text}")
                    return

                if 'sdp' in join_response and 'type' in join_response:
                    self.peer_connection = RTCPeerConnection()
                    
                    # Set up local audio capture
                    player = MediaPlayer(None)  # Use default microphone
                    self.peer_connection.addTrack(player.audio)

                    # Handle incoming media
                    @self.peer_connection.on("track")
                    async def on_track(track):
                        if track.kind == "audio":
                            print("Received audio track")
                            if self.recorder is None:
                                self.recorder = MediaRecorder(None)  # Use default speakers
                                await self.recorder.start()
                            self.recorder.addTrack(track)

                    offer = RTCSessionDescription(sdp=join_response['sdp'], type=join_response['type'])
                    print("Setting remote description...")
                    await self.peer_connection.setRemoteDescription(offer)
                    print("Remote description set.")
                    answer = await self.peer_connection.createAnswer()
                    print("Creating answer...")
                    await self.peer_connection.setLocalDescription(answer)
                    print("Local description set.")
                    print(f"Joined room {room_name} successfully!")
                else:
                    print("Join response did not contain SDP information.")
        except Exception as e:
            print(f"Exception occurred: {e}")

def start_loop(loop):
    asyncio.set_event_loop(loop)
    print("Starting asyncio event loop...")
    loop.run_forever()

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    threading.Thread(target=start_loop, args=(loop,), daemon=True).start()

    app = LiveKitApp()
    app.run(None)

