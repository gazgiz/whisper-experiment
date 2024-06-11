import Foundation
import LiveKit
import Combine
class LiveKitManager: NSObject, ObservableObject {
    @Published var room: Room?
    @Published var localVideoTrack: VideoTrack?
    @Published var remoteVideoTrack: VideoTrack?

    override init() {
        super.init()
        room = Room(delegate: self)
    }
func connect() async {
    let url = "wss://your-livekit-server-url"
    let token = "your-access-token"

    do {
        try await room?.connect(url: url, token: token)
        DispatchQueue.main.async {
            // Update room state or other UI-related tasks here
            print("Connected to the room: \(self.room?.name ?? "Unknown")")
        }

        // Publishing camera & mic...
        guard let localParticipant = room?.localParticipant else {
            print("Failed to get local participant")
            return
        }

        try await localParticipant.setCamera(enabled: true)
        try await localParticipant.setMicrophone(enabled: true)
    } catch {
        print("Failed to connect to the room: \(error)")
    }
}
}
extension LiveKitManager: RoomDelegate {
    func room(_ room: Room, participant: LocalParticipant, didPublishTrack publication: LocalTrackPublication) {
        guard let track = publication.track as? VideoTrack else { return }
        DispatchQueue.main.async {
            self.localVideoTrack = track
        }
    }
func room(_ room: Room, participant: RemoteParticipant, didSubscribeTrack publication: RemoteTrackPublication) {
    guard let track = publication.track as? VideoTrack else { return }
    DispatchQueue.main.async {
        self.remoteVideoTrack = track
    }
}
}
