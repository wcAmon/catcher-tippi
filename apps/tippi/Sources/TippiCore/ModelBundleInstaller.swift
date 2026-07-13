import Foundation

/// The two on-disk model directories Tippi needs: Catcher ASR plus the
/// Sortformer diarization artifact.
public struct ModelBundle: Equatable, Sendable {
    public let asr: URL
    public let diar: URL

    public init(asr: URL, diar: URL) {
        self.asr = asr
        self.diar = diar
    }
}

public protocol ModelBundleInstalling: Sendable {
    func installIfNeeded(
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> ModelBundle
}

/// Installs the ASR and diarization artifacts sequentially, reporting one
/// merged progress value weighted by each artifact's total byte count.
public actor ModelBundleInstaller: ModelBundleInstalling {
    private let asr: any ModelInstalling
    private let diar: any ModelInstalling
    private let asrWeight: Double
    private let diarWeight: Double

    public init(
        asr: any ModelInstalling,
        asrTotalBytes: Int64,
        diar: any ModelInstalling,
        diarTotalBytes: Int64
    ) {
        self.asr = asr
        self.diar = diar
        let total = Double(asrTotalBytes + diarTotalBytes)
        asrWeight = Double(asrTotalBytes) / total
        diarWeight = Double(diarTotalBytes) / total
    }

    public func installIfNeeded(
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> ModelBundle {
        let asrWeight = asrWeight
        let diarWeight = diarWeight
        let asrURL = try await asr.installIfNeeded { value in
            progress(value * asrWeight)
        }
        let diarURL = try await diar.installIfNeeded { value in
            progress(asrWeight + value * diarWeight)
        }
        progress(1.0)
        return ModelBundle(asr: asrURL, diar: diarURL)
    }
}
