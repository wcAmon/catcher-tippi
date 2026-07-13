import Foundation

public struct ModelFile: Equatable, Sendable {
    public let name: String
    public let sha256: String
    public let required: Bool
    public let byteCount: Int64

    public init(name: String, sha256: String, required: Bool, byteCount: Int64 = 1) {
        self.name = name
        self.sha256 = sha256
        self.required = required
        self.byteCount = byteCount
    }
}

public extension Array where Element == ModelFile {
    static let catcherRelease: [ModelFile] = [
        ModelFile(name: "weights.safetensors", sha256: "157d0efe1a3fff7a04a4709e365755a76b4c7c972bc8b1f8d58ef33d5f93acee", required: true, byteCount: 658_663_198),
        ModelFile(name: "manifest.json", sha256: "40f7eacf4ff4929049e951d943fe15f33a01f7d238ec622737582731335dc16b", required: true, byteCount: 100_746),
        ModelFile(name: "config.json", sha256: "62d186fd91f518e00e7867500f1f5819225e8ee95ea3e21b546514bf2048e845", required: true, byteCount: 1_376),
        ModelFile(name: "generation_config.json", sha256: "993e5d4cb74a6fe9d6e7084a76b3313c1446740679be4676570c23b664fdc07e", required: true, byteCount: 193),
        ModelFile(name: "processor_config.json", sha256: "ec47870f1091ea4f25539208387b45b902c92d0e3f997a30061ef88f73437ab0", required: true, byteCount: 2_519),
        ModelFile(name: "tokenizer.json", sha256: "3f3d481deb073b64c2082e8c7860d487a3a62774bf4e9e4faac83007e181f246", required: true, byteCount: 752_051),
        ModelFile(name: "tokenizer_config.json", sha256: "5c641c5b3f50702a60082690d27c1ce7fcb5a92c4a624793bcae0f21eda3d6e0", required: true, byteCount: 881),
        ModelFile(name: "LICENSE", sha256: "dd6b7d50e7d7f8ce3fb28487011c6a324d812e0315ed7c6f34f2a9048932b3bf", required: true, byteCount: 2_619),
        ModelFile(name: "NOTICE.md", sha256: "5a220b42f4625219699656bc053188fe8b69b6c273c030512eaefdc1a28c7aaa", required: true, byteCount: 726),
        ModelFile(name: "NVIDIA_MODEL_CARD.md", sha256: "e8335332bed2d69e790f61d7619098dc464f3075985f68856e53213c7aeddccb", required: true, byteCount: 53_753),
    ]

    static let diarizationRelease: [ModelFile] = [
        ModelFile(name: "weights.safetensors", sha256: "a02b1a83ceb6c1f9cf048ab3420c86c84421b0f4e64c433da75b506411445987", required: true, byteCount: 127_218_628),
        ModelFile(name: "manifest.json", sha256: "b777fce8ee72fa7ec90a54637709b3831df30e988006e2bc28ff1d8a1ec7403d", required: true, byteCount: 139_278),
        ModelFile(name: "config.json", sha256: "6c0418a4b7e5e3256abe9ed6c077995118ac8d3be9082615ec89b60b6dba6470", required: true, byteCount: 4_469),
        ModelFile(name: "LICENSE", sha256: "aae70c7d06968fee034365d5b18bcef9ac0f54d58c2c60e0c9dbe6e1e1e6093e", required: true, byteCount: 10_321),
        ModelFile(name: "NOTICE.md", sha256: "5ce32ffbe2c279712d9e456820851ebf00d4f7970cfb182b092bf1831781b6ef", required: true, byteCount: 458),
        ModelFile(name: "NVIDIA_MODEL_CARD.md", sha256: "86d9ff0886b098dac53fccbab660c231b0bbfee4c54068d57ab16c5fdb9776d6", required: true, byteCount: 27_357),
        ModelFile(name: "README.md", sha256: "0e2e2491c3ddb719a256ce4e051710fd0ea7e1fecf1f04a086079bfff0186399", required: true, byteCount: 642),
    ]

    var totalByteCount: Int64 {
        reduce(0) { $0 + $1.byteCount }
    }
}
