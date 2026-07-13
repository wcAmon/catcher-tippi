import Foundation
import Testing
@testable import TippiCore

@Suite("ModelDirectoryMigratorTests")
struct ModelDirectoryMigratorTests {
    private enum TestFailure: Error {
        case forcedMoveFailure
    }

    @Test
    func defaultPathsUseLegacyContainerAndUnsandboxedApplicationSupport() {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let expectedLegacy = home.appending(
            path: "Library/Containers/com.wcamon.tippi/Data/Library/Application Support/Tippi/Models",
            directoryHint: .isDirectory
        )
        let expectedCurrent = home.appending(
            path: "Library/Application Support/Tippi/Models",
            directoryHint: .isDirectory
        )

        #expect(ModelDirectoryMigrator.defaultLegacyRoot().standardizedFileURL == expectedLegacy.standardizedFileURL)
        #expect(ModelStore.defaultRootDirectory().standardizedFileURL == expectedCurrent.standardizedFileURL)
    }

    @Test
    func movesLegacyModelsWhenDestinationIsAbsent() async throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        try writeModelTree(at: fixture.source)

        let migrator = ModelDirectoryMigrator(
            source: fixture.source,
            destination: fixture.destination
        )

        try await migrator.migrateIfNeeded()

        #expect(!FileManager.default.fileExists(atPath: fixture.source.path))
        #expect(try Data(contentsOf: fixture.destination.appending(path: "asr/model.bin"))
            == Data("asr-model".utf8))
        #expect(try Data(contentsOf: fixture.destination.appending(path: "diar/model.bin"))
            == Data("diar-model".utf8))
    }

    @Test
    func nonEmptyDestinationIsNeverOverwritten() async throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        try writeModelTree(at: fixture.source)
        try FileManager.default.createDirectory(
            at: fixture.destination,
            withIntermediateDirectories: true
        )
        let sentinel = fixture.destination.appending(path: "keep-me.txt")
        try Data("existing".utf8).write(to: sentinel)

        let migrator = ModelDirectoryMigrator(
            source: fixture.source,
            destination: fixture.destination
        )

        try await migrator.migrateIfNeeded()

        #expect(FileManager.default.fileExists(atPath: fixture.source.appending(path: "asr/model.bin").path))
        #expect(try Data(contentsOf: sentinel) == Data("existing".utf8))
        #expect(!FileManager.default.fileExists(
            atPath: fixture.destination.appending(path: "asr/model.bin").path
        ))
    }

    @Test
    func moveFailureCopiesThroughStagingAndRemovesSourceOnlyAfterPromotion() async throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        try writeModelTree(at: fixture.source)
        let staging = fixture.destination.deletingLastPathComponent()
            .appending(path: ".Models.migration.partial", directoryHint: .isDirectory)

        let migrator = ModelDirectoryMigrator(
            source: fixture.source,
            destination: fixture.destination,
            moveItem: { _, _ in throw TestFailure.forcedMoveFailure }
        )

        try await migrator.migrateIfNeeded()

        #expect(!FileManager.default.fileExists(atPath: fixture.source.path))
        #expect(!FileManager.default.fileExists(atPath: staging.path))
        #expect(try Data(contentsOf: fixture.destination.appending(path: "asr/model.bin"))
            == Data("asr-model".utf8))
        #expect(try Data(contentsOf: fixture.destination.appending(path: "diar/model.bin"))
            == Data("diar-model".utf8))
    }

    @Test
    func copyFailureKeepsTheOnlyLegacyModelCopy() async throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        try writeModelTree(at: fixture.source)
        let unreadable = fixture.source.appending(path: "asr/model.bin")
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o000],
            ofItemAtPath: unreadable.path
        )
        defer {
            try? FileManager.default.setAttributes(
                [.posixPermissions: 0o600],
                ofItemAtPath: unreadable.path
            )
        }
        let staging = fixture.destination.deletingLastPathComponent()
            .appending(path: ".Models.migration.partial", directoryHint: .isDirectory)

        let migrator = ModelDirectoryMigrator(
            source: fixture.source,
            destination: fixture.destination,
            moveItem: { _, _ in throw TestFailure.forcedMoveFailure }
        )

        await #expect(throws: (any Error).self) {
            try await migrator.migrateIfNeeded()
        }

        #expect(FileManager.default.fileExists(atPath: fixture.source.path))
        #expect(FileManager.default.fileExists(atPath: unreadable.path))
        #expect(!FileManager.default.fileExists(atPath: fixture.destination.path))
        #expect(!FileManager.default.fileExists(atPath: staging.path))
    }

    private func makeFixture() throws -> (root: URL, source: URL, destination: URL) {
        let root = FileManager.default.temporaryDirectory
            .appending(path: UUID().uuidString, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return (
            root,
            root.appending(path: "legacy/Models", directoryHint: .isDirectory),
            root.appending(path: "current/Models", directoryHint: .isDirectory)
        )
    }

    private func writeModelTree(at root: URL) throws {
        let asr = root.appending(path: "asr", directoryHint: .isDirectory)
        let diar = root.appending(path: "diar", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: asr, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: diar, withIntermediateDirectories: true)
        try Data("asr-model".utf8).write(to: asr.appending(path: "model.bin"))
        try Data("diar-model".utf8).write(to: diar.appending(path: "model.bin"))
    }
}
