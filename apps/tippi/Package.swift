// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "Tippi",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "TippiCore", targets: ["TippiCore"]),
    ],
    targets: [
        .systemLibrary(name: "CCatcher", path: "Sources/CCatcher"),
        .target(name: "TippiCore"),
        .testTarget(name: "TippiCoreTests", dependencies: ["TippiCore"]),
    ]
)
