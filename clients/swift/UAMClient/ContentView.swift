
import SwiftUI
import LiveKit
struct ContentView: View {
@StateObject private var liveKitManager = LiveKitManager()
var body: some View {
VStack {
    Text("Hello, World!")
        .padding()

    Button(action: {
        Task {
            await liveKitManager.connect()
        }
    }) {
        Text("Connect to LiveKit")
    }

    if let room = liveKitManager.room {
        Text("Connected to the room: \(room.name ?? "Unknown")")
    }

    if let localVideoTrack = liveKitManager.localVideoTrack {
      //  VideoView(track: localVideoTrack)
       //     .frame(width: 200, height: 200)
       //     .border(Color.black, width: 1)
    }

    if let remoteVideoTrack = liveKitManager.remoteVideoTrack {
      //  VideoView(track: remoteVideoTrack)
      //      .frame(width: 200, height: 200)
      //      .border(Color.black, width: 1)
    }
}
.onAppear {
    // Optionally connect when the view appears
    // Task { await liveKitManager.connect() }
}
}
}
