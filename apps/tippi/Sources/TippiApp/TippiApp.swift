import SwiftUI
import TippiCore

@main
struct TippiApp: App {
    @State private var controller: TranscriptionController

    init() {
        let bundleInstaller = ModelBundleInstaller(
            asr: ModelStore(),
            asrTotalBytes: [ModelFile].catcherRelease.totalByteCount,
            diar: ModelStore(
                baseURL: ModelStore.diarizationRepositoryURL,
                files: .diarizationRelease,
                directoryName: "catcher-diar-mlx-int8"
            ),
            diarTotalBytes: [ModelFile].diarizationRelease.totalByteCount
        )
        let audio = AudioRecorder()
        let injector = CGEventTextInjector()
        let coordinator = TextInjectionCoordinator(
            injector: injector,
            targetProvider: FrontmostApplicationProvider(),
            ownBundleIdentifier: "com.wcamon.tippi"
        )
        let keywordInstaller = KeywordModelInstaller()
        let modelMigrator = ModelDirectoryMigrator()
        _controller = State(
            initialValue: TranscriptionController(
                modelInstaller: bundleInstaller,
                audio: audio,
                catcherFactory: { bundle in
                    try CatcherClient(
                        modelDirectory: bundle.asr,
                        diarModelDirectory: bundle.diar,
                        language: "auto",
                        lookahead: 3
                    )
                },
                modelMigrator: modelMigrator,
                keywordInstaller: keywordInstaller,
                keywordFactory: { directory in
                    try KeywordSpotterClient(modelDirectory: directory)
                },
                injectionCoordinator: coordinator
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
