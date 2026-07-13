import Foundation

public protocol ModelDownloading: Sendable {
    func download(
        from url: URL,
        to destination: URL,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws
}

public protocol ModelInstalling: Sendable {
    func installIfNeeded(
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> URL
}

public enum ModelStoreError: Error, LocalizedError {
    case invalidChecksum(file: String)
    case invalidHTTPStatus(Int)
    case incompleteModel

    public var errorDescription: String? {
        switch self {
        case let .invalidChecksum(file):
            "Downloaded model file failed SHA-256 verification: \(file)"
        case let .invalidHTTPStatus(status):
            "Model download returned HTTP status \(status)"
        case .incompleteModel:
            "The installed Catcher model is incomplete"
        }
    }
}

public actor ModelStore {
    public static let repositoryURL = URL(
        string: "https://huggingface.co/wcamon/catcher-asr-mlx-int8/resolve/main/"
    )!

    public static let diarizationRepositoryURL = URL(
        string: "https://huggingface.co/wcamon/catcher-diar-mlx-int8/resolve/main/"
    )!

    private let rootDirectory: URL
    private let baseURL: URL
    private let files: [ModelFile]
    private let downloader: any ModelDownloading
    private let fileManager: FileManager
    private let modelDirectoryName: String

    public init(
        rootDirectory: URL? = nil,
        baseURL: URL = ModelStore.repositoryURL,
        files: [ModelFile] = .catcherRelease,
        directoryName: String = "catcher-asr-mlx-int8",
        downloader: any ModelDownloading = URLSessionModelDownloader(),
        fileManager: FileManager = .default
    ) {
        self.rootDirectory = rootDirectory ?? Self.defaultRootDirectory(fileManager: fileManager)
        self.baseURL = baseURL
        self.files = files
        self.modelDirectoryName = directoryName
        self.downloader = downloader
        self.fileManager = fileManager
    }

    public func installIfNeeded(
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> URL {
        let destination = rootDirectory.appending(path: modelDirectoryName, directoryHint: .isDirectory)
        if try verifyModel(at: destination) {
            progress(1.0)
            return destination
        }

        let staging = rootDirectory.appending(
            path: ".\(modelDirectoryName).partial",
            directoryHint: .isDirectory
        )
        try fileManager.createDirectory(at: rootDirectory, withIntermediateDirectories: true)
        try? fileManager.removeItem(at: staging)
        try fileManager.createDirectory(at: staging, withIntermediateDirectories: true)

        do {
            let totalBytes = Double(files.reduce(0) { $0 + $1.byteCount })
            var completedBytes: Int64 = 0
            for file in files {
                let source = baseURL.appending(path: file.name)
                let target = staging.appending(path: file.name)
                let completedBefore = completedBytes
                try await downloader.download(from: source, to: target) { localProgress in
                    let downloaded = Double(completedBefore) + localProgress * Double(file.byteCount)
                    progress(downloaded / totalBytes)
                }
                guard try ModelChecksum.sha256(of: target) == file.sha256 else {
                    throw ModelStoreError.invalidChecksum(file: file.name)
                }
                completedBytes += file.byteCount
            }
            guard try verifyModel(at: staging) else {
                throw ModelStoreError.incompleteModel
            }
            try? fileManager.removeItem(at: destination)
            try fileManager.moveItem(at: staging, to: destination)
            progress(1.0)
            return destination
        } catch {
            try? fileManager.removeItem(at: staging)
            throw error
        }
    }

    private func verifyModel(at directory: URL) throws -> Bool {
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: directory.path, isDirectory: &isDirectory),
              isDirectory.boolValue
        else {
            return false
        }
        for file in files where file.required {
            let url = directory.appending(path: file.name)
            guard fileManager.fileExists(atPath: url.path),
                  try ModelChecksum.sha256(of: url) == file.sha256
            else {
                return false
            }
        }
        return true
    }

    public static func defaultRootDirectory(fileManager: FileManager) -> URL {
        let applicationSupport = fileManager.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        )[0]
        return applicationSupport
            .appending(path: "Tippi", directoryHint: .isDirectory)
            .appending(path: "Models", directoryHint: .isDirectory)
    }
}

extension ModelStore: ModelInstalling {}

public struct URLSessionModelDownloader: ModelDownloading {
    public init() {}

    public func download(
        from url: URL,
        to destination: URL,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws {
        let operation = DownloadOperation(destination: destination, progress: progress)
        try await operation.run(url: url)
    }
}

private final class DownloadOperation: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let destination: URL
    private let progress: @Sendable (Double) -> Void
    private var continuation: CheckedContinuation<Void, any Error>?
    private var session: URLSession?
    private var moveError: (any Error)?

    init(destination: URL, progress: @escaping @Sendable (Double) -> Void) {
        self.destination = destination
        self.progress = progress
    }

    func run(url: URL) async throws {
        try await withCheckedThrowingContinuation { continuation in
            self.continuation = continuation
            let queue = OperationQueue()
            queue.maxConcurrentOperationCount = 1
            let session = URLSession(configuration: .default, delegate: self, delegateQueue: queue)
            self.session = session
            session.downloadTask(with: url).resume()
        }
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        guard totalBytesExpectedToWrite > 0 else { return }
        progress(min(1.0, Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)))
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        do {
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.moveItem(at: location, to: destination)
        } catch {
            moveError = error
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: (any Error)?
    ) {
        defer {
            continuation = nil
            session.finishTasksAndInvalidate()
            self.session = nil
        }
        if let error {
            continuation?.resume(throwing: error)
        } else if let moveError {
            continuation?.resume(throwing: moveError)
        } else if let response = task.response as? HTTPURLResponse,
                  !(200 ... 299).contains(response.statusCode)
        {
            continuation?.resume(throwing: ModelStoreError.invalidHTTPStatus(response.statusCode))
        } else {
            progress(1.0)
            continuation?.resume()
        }
    }
}
