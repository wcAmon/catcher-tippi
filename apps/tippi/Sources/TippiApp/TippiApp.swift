import SwiftUI
import TippiCore

@main
struct TippiApp: App {
    @State private var controller: TranscriptionController

    init() {
        let modelStore = ModelStore()
        let audio = AudioRecorder()
        _controller = State(
            initialValue: TranscriptionController(
                modelInstaller: modelStore,
                audio: audio,
                catcherFactory: { modelURL in
                    try CatcherClient(modelDirectory: modelURL, language: "auto", lookahead: 3)
                }
            )
        )
    }

    var body: some Scene {
        WindowGroup("Tippi") {
            ContentView(controller: controller)
                .task {
                    guard controller.state == .modelMissing else { return }
                    await controller.prepare()
                }
        }
        .defaultSize(width: 720, height: 560)
        .windowResizability(.contentMinSize)
    }
}
