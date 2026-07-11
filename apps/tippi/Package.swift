// swift-tools-version: 6.2

import PackageDescription
import Foundation

let packageDirectory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
let repositoryRoot = packageDirectory
    .deletingLastPathComponent()
    .deletingLastPathComponent()
let catcherLibraryPath = repositoryRoot.appending(path: "target/release").path

let package = Package(
    name: "Tippi",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "TippiCore", targets: ["TippiCore"]),
        .executable(name: "Tippi", targets: ["TippiApp"]),
    ],
    targets: [
        .systemLibrary(name: "CCatcher", path: "Sources/CCatcher"),
        .target(
            name: "TippiCore",
            dependencies: ["CCatcher"],
            linkerSettings: [
                .unsafeFlags([
                    "-L\(catcherLibraryPath)",
                    "-lcatcher_ffi",
                    "-Xlinker", "-rpath",
                    "-Xlinker", catcherLibraryPath,
                    "-Xlinker", "-rpath",
                    "-Xlinker", "@executable_path/../Frameworks",
                ]),
            ]
        ),
        .executableTarget(name: "TippiApp", dependencies: ["TippiCore"]),
        .testTarget(name: "TippiCoreTests", dependencies: ["TippiCore"]),
    ]
)
