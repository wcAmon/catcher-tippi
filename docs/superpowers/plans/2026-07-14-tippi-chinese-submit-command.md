# Tippi Chinese Submit Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unreliable English `Tippi Go` submit command with the single Mandarin command 「幫我送出」 while preserving safe cross-App text injection and upgrading existing keyword-model installations without another download.

**Architecture:** Add one Swift command contract that owns the user-facing phrase, sherpa keyword tokens, KWS parameters, and `SUBMIT_ZH` event identifier. The existing two-model stream remains unchanged: Catcher supplies stable text from a 16 kHz sample clock and sherpa-onnx independently emits the submit event; the installer repairs only generated keyword metadata when all pinned runtime files are intact.

**Tech Stack:** Swift 6.2, SwiftUI, Swift Testing, Rust 2024, sherpa-onnx 1.13.4, Catcher C ABI, macOS `say`/`afconvert`, Cargo, ad-hoc signed macOS app bundle.

## Global Constraints

- The only submit phrase is `幫我送出`; `Tippi Go` is not an alias.
- The event identifier is exactly `SUBMIT_ZH` in Swift, generated `keywords.txt`, and Rust.
- The initial keyword line is exactly `b āng w ǒ s òng ch ū :1.5 #0.25 @SUBMIT_ZH\n`.
- The command stays local, offline, and independent of the main ASR transcript; do not add a third model, fuzzy text matching, or command cleanup by Backspace.
- All audio remains mono Float32 at 16 kHz. Stable text uses `floor(receivedSampleCount * 1000 / 16000) - 2000 ms`; KWS timestamps remain diagnostic-only.
- An empty safe prefix never sends Return. An accepted non-empty command sends Return exactly once, then resets ASR, KWS, coordinator, duplicate guard, and the sample clock.
- Stop, stream failure, Tippi-frontmost, an unavailable target, and an untrusted Accessibility state remain fail-closed.
- Keep the pinned sherpa archive URL, directory name, archive SHA-256, four runtime filenames, and four runtime SHA-256 values unchanged.
- Repair generated files in place only when the installed directory contains exactly the four pinned runtime files plus `keywords.txt` and `THIRD_PARTY_NOTICES.md`, and all four runtime hashes match.
- Do not commit the sherpa archive or ONNX model files. The four small PCM16 WAV command fixtures are allowed.
- Active runtime source, SwiftUI copy, and `README.md` must not contain `Tippi Go` or `TIPPI_GO`; the old fixture and its Rust negative regression may retain the old phrase.

---

## File Map

- Create `apps/tippi/Sources/TippiCore/VoiceSubmitCommand.swift`: single Swift source of truth for phrase, identifier, pinyin tokens, boost, threshold, and generated keyword line.
- Modify `apps/tippi/Sources/TippiCore/KeywordModelManifest.swift`: generate `keywords.txt` from `VoiceSubmitCommand` while preserving the pinned archive and runtime hashes.
- Modify `apps/tippi/Sources/TippiCore/KeywordModelInstaller.swift`: separate runtime verification from generated-file verification, repair stale generated files atomically, and use Chinese keyword-model errors.
- Modify `apps/tippi/Sources/TippiCore/KeywordSpotterClient.swift`: keep the ABI unchanged and replace English user-visible detector errors.
- Modify `apps/tippi/Sources/TippiCore/VoiceInputTiming.swift`: increase the sample-clock holdback to 2,000 ms.
- Modify `apps/tippi/Sources/TippiCore/TranscriptionController.swift`: accept only `SUBMIT_ZH` through the shared command definition and use the Chinese retry message.
- Modify `apps/tippi/Sources/TippiApp/VoiceInputTabView.swift`: display only 「幫我送出」 and the 2-second delay.
- Modify `crates/catcher-ffi/src/kws.rs`: change the accepted sherpa event identifier without changing the C ABI.
- Modify `crates/catcher-ffi/tests/kws_ffi.rs`: add two Mandarin positives, partial-phrase negatives, the old-English negative, reset coverage, and the existing timestamp diagnostic.
- Modify `apps/tippi/Tests/TippiCoreTests/KeywordModelInstallerTests.swift`: cover the command contract, generated-only repair, full-install fallback, and exact error copy.
- Modify `apps/tippi/Tests/TippiCoreTests/KeywordSpotterClientTests.swift`: carry `SUBMIT_ZH` through the Swift protocol.
- Modify `apps/tippi/Tests/TippiCoreTests/TranscriptionControllerTests.swift`: cover the 2-second sample-clock cutoff, identifier filtering, exact-once Return, reset, stop, failure, and frontmost behavior.
- Create `tests/fixtures/bang-wo-song-chu-zh-cn.wav`: Tingting deterministic positive command.
- Create `tests/fixtures/bang-wo-song-chu-zh-tw.wav`: Meijia deterministic positive command.
- Create `tests/fixtures/bang-wo-zh-tw.wav`: Meijia partial-command negative.
- Create `tests/fixtures/song-chu-zh-tw.wav`: Meijia partial-command negative.
- Modify `tests/fixtures/README.md`: document exact fixture origins and reconstruction commands; reclassify `tippi-go.wav` as a negative regression.
- Modify `README.md`: document the Chinese workflow, 2-second holdback, and safety behavior.

---

### Task 1: Define and prove the Chinese KWS command contract

**Files:**
- Create: `apps/tippi/Sources/TippiCore/VoiceSubmitCommand.swift`
- Create: `tests/fixtures/bang-wo-song-chu-zh-cn.wav`
- Create: `tests/fixtures/bang-wo-song-chu-zh-tw.wav`
- Create: `tests/fixtures/bang-wo-zh-tw.wav`
- Create: `tests/fixtures/song-chu-zh-tw.wav`
- Modify: `apps/tippi/Sources/TippiCore/KeywordModelManifest.swift`
- Modify: `apps/tippi/Sources/TippiCore/KeywordSpotterClient.swift`
- Modify: `apps/tippi/Tests/TippiCoreTests/KeywordSpotterClientTests.swift`
- Modify: `apps/tippi/Tests/TippiCoreTests/KeywordModelInstallerTests.swift`
- Modify: `crates/catcher-ffi/src/kws.rs`
- Modify: `crates/catcher-ffi/tests/kws_ffi.rs`
- Modify: `tests/fixtures/README.md`

**Interfaces:**
- Consumes: the existing `KeywordDetection`, `KeywordSpotting`, `KeywordModelManifest.release`, `catcher_kws_*` C ABI, pinned sherpa runtime, and the installed pinyin tokens `b/āng/w/ǒ/s/òng/ch/ū`.
- Produces: `VoiceSubmitCommand.displayPhrase: String`, `eventIdentifier: String`, `tokenSequence: String`, `keywordBoost: Double`, `triggerThreshold: Double`, and `keywordDefinition: String`; Rust `EXPECTED_KEYWORD = "SUBMIT_ZH"`; four deterministic command fixtures.

- [ ] **Step 1: Generate the deterministic Mandarin fixtures**

Run from the repository root:

```bash
/usr/bin/say -v Tingting -r 170 -o /tmp/bang-wo-song-chu-zh-cn.aiff "幫我送出"
/usr/bin/say -v Meijia -r 170 -o /tmp/bang-wo-song-chu-zh-tw.aiff "幫我送出"
/usr/bin/say -v Meijia -r 170 -o /tmp/bang-wo-zh-tw.aiff "幫我"
/usr/bin/say -v Meijia -r 170 -o /tmp/song-chu-zh-tw.aiff "送出"
/usr/bin/afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/bang-wo-song-chu-zh-cn.aiff tests/fixtures/bang-wo-song-chu-zh-cn.wav
/usr/bin/afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/bang-wo-song-chu-zh-tw.aiff tests/fixtures/bang-wo-song-chu-zh-tw.wav
/usr/bin/afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/bang-wo-zh-tw.aiff tests/fixtures/bang-wo-zh-tw.wav
/usr/bin/afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/song-chu-zh-tw.aiff tests/fixtures/song-chu-zh-tw.wav
file tests/fixtures/bang-wo-song-chu-zh-cn.wav tests/fixtures/bang-wo-song-chu-zh-tw.wav tests/fixtures/bang-wo-zh-tw.wav tests/fixtures/song-chu-zh-tw.wav
```

Expected: every `file` result reports Microsoft PCM, 16 bit, mono, 16000 Hz.

- [ ] **Step 2: Write the failing Swift command-contract tests**

Add this test to `KeywordSpotterClientTests.swift`, and change the fake detection and existing expectation from the old literal to `VoiceSubmitCommand.eventIdentifier`:

```swift
@Test
func chineseSubmitCommandContractIsExact() {
    #expect(VoiceSubmitCommand.displayPhrase == "幫我送出")
    #expect(VoiceSubmitCommand.eventIdentifier == "SUBMIT_ZH")
    #expect(VoiceSubmitCommand.tokenSequence == "b āng w ǒ s òng ch ū")
    #expect(VoiceSubmitCommand.keywordBoost == 1.5)
    #expect(VoiceSubmitCommand.triggerThreshold == 0.25)
    #expect(VoiceSubmitCommand.keywordDefinition
        == "b āng w ǒ s òng ch ū :1.5 #0.25 @SUBMIT_ZH\n")
}

private actor FakeKeywordSpotter: KeywordSpotting {
    func start() async throws {}

    func push(_ samples: [Float]) async throws -> KeywordDetection? {
        KeywordDetection(
            keyword: VoiceSubmitCommand.eventIdentifier,
            startMs: UInt64(samples.count)
        )
    }

    func reset() async throws {}
}

@Test
func keywordDetectionCarriesCommandAndCutoff() async throws {
    let service: any KeywordSpotting = FakeKeywordSpotter()
    let detection = try await service.push([0, 0, 0])
    #expect(detection == KeywordDetection(keyword: "SUBMIT_ZH", startMs: 3))
}
```

Change the release-manifest assertion in `KeywordModelInstallerTests.swift` to:

```swift
#expect(KeywordModelManifest.keywords == VoiceSubmitCommand.keywordDefinition)
#expect(KeywordModelManifest.keywords
    == "b āng w ǒ s òng ch ū :1.5 #0.25 @SUBMIT_ZH\n")
```

In `installsOnlyVerifiedRuntimeFilesAndGeneratedMetadata`, replace the old
generated-keyword expectation with:

```swift
#expect(try Data(contentsOf: installed.appending(path: "keywords.txt"))
    == Data(VoiceSubmitCommand.keywordDefinition.utf8))
```

- [ ] **Step 3: Write the failing real-model Rust tests**

In `kws_ffi.rs`, add `PathBuf` to the imports and replace the English-positive helpers/tests with the following exact fixture helper and matrix:

```rust
use std::path::{Path, PathBuf};

fn padded_fixture(name: &str) -> Vec<f32> {
    let path: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name);
    let spoken = read_wav(path);
    let mut samples = vec![0.0; 16_000];
    samples.extend(spoken);
    samples.extend(vec![0.0; 16_000]);
    samples
}

#[test]
#[ignore = "requires SHERPA_KWS_MODEL"]
fn detects_both_mandarin_submit_fixtures_and_resets() {
    let model = CString::new(std::env::var("SHERPA_KWS_MODEL").unwrap()).unwrap();
    let handle = unsafe { catcher_kws_create(model.as_ptr()) };
    assert!(!handle.is_null(), "KWS model should load");

    unsafe {
        for fixture in [
            "bang-wo-song-chu-zh-cn.wav",
            "bang-wo-song-chu-zh-tw.wav",
        ] {
            let samples = padded_fixture(fixture);
            assert_eq!(catcher_kws_start(handle), CATCHER_OK);
            assert!(feed_until_detected(handle, &samples), "missed {fixture}");
            assert_eq!(
                CStr::from_ptr(catcher_kws_keyword(handle)).to_str().unwrap(),
                "SUBMIT_ZH"
            );
            let start_ms = catcher_kws_start_ms(handle);
            assert!((500..=3_000).contains(&start_ms), "{fixture}: {start_ms}ms");

            assert_eq!(catcher_kws_start(handle), CATCHER_OK);
            assert_eq!(CStr::from_ptr(catcher_kws_keyword(handle)).to_bytes(), b"");
            assert_eq!(catcher_kws_start_ms(handle), 0);
            assert!(
                feed_until_detected(handle, &samples),
                "missed {fixture} after reset"
            );
        }
        catcher_kws_destroy(handle);
    }
}

#[test]
#[ignore = "requires SHERPA_KWS_MODEL"]
fn partial_old_and_unrelated_audio_do_not_trigger_submit() {
    let model = CString::new(std::env::var("SHERPA_KWS_MODEL").unwrap()).unwrap();
    let handle = unsafe { catcher_kws_create(model.as_ptr()) };
    assert!(!handle.is_null(), "KWS model should load");

    unsafe {
        for fixture in [
            "bang-wo-zh-tw.wav",
            "song-chu-zh-tw.wav",
            "tippi-go.wav",
            "hello-streaming.wav",
            "conversation.wav",
        ] {
            assert_eq!(catcher_kws_start(handle), CATCHER_OK);
            assert_no_detection(handle, &padded_fixture(fixture));
        }
        catcher_kws_destroy(handle);
    }
}
```

Update `reported_timestamp_is_not_an_absolute_long_stream_offset` to load `bang-wo-song-chu-zh-tw.wav` instead of `tippi-go.wav`. Keep its `[1, 5, 10, 20]` second prefixes and diagnostic-only assertions unchanged.

- [ ] **Step 4: Run the tests to prove the new contract is absent**

Prepare a disposable real-model directory with the new generated keyword file, without changing the user's installed model:

```bash
KWS_SOURCE="$HOME/Library/Application Support/Tippi/Models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
KWS_TEST=/tmp/tippi-kws-submit-zh
rm -rf "$KWS_TEST"
cp -R "$KWS_SOURCE" "$KWS_TEST"
printf '%s\n' 'b āng w ǒ s òng ch ū :1.5 #0.25 @SUBMIT_ZH' > "$KWS_TEST/keywords.txt"
cargo build -p catcher-ffi --release
swift test --package-path apps/tippi --filter chineseSubmitCommandContractIsExact
SHERPA_KWS_MODEL="$KWS_TEST" cargo test -p catcher-ffi --test kws_ffi detects_both_mandarin_submit_fixtures_and_resets -- --ignored --nocapture
```

Expected: Swift fails because `VoiceSubmitCommand` does not exist. The ignored Rust test fails to detect because Rust still rejects identifiers other than `TIPPI_GO`.

- [ ] **Step 5: Implement the shared Swift contract and Rust identifier**

Create `VoiceSubmitCommand.swift` with:

```swift
public enum VoiceSubmitCommand {
    public static let displayPhrase = "幫我送出"
    public static let eventIdentifier = "SUBMIT_ZH"
    public static let tokenSequence = "b āng w ǒ s òng ch ū"
    public static let keywordBoost = 1.5
    public static let triggerThreshold = 0.25

    public static var keywordDefinition: String {
        "\(tokenSequence) :\(keywordBoost) #\(triggerThreshold) @\(eventIdentifier)\n"
    }
}
```

In `KeywordModelManifest.swift`, replace the literal with:

```swift
public static let keywords = VoiceSubmitCommand.keywordDefinition
```

In `KeywordSpotterClientError.errorDescription`, use:

```swift
case let .creationFailed(message):
    "無法載入「\(VoiceSubmitCommand.displayPhrase)」口令模型：\(message)"
case let .operationFailed(message):
    "「\(VoiceSubmitCommand.displayPhrase)」口令偵測失敗：\(message)"
```

In `crates/catcher-ffi/src/kws.rs`, make the only runtime change:

```rust
pub const EXPECTED_KEYWORD: &str = "SUBMIT_ZH";
```

Do not change the KWS model filenames, config, stream lifecycle, timestamp fallback, FFI status codes, or C headers.

- [ ] **Step 6: Document fixture reconstruction and the old-command negative**

Replace the first fixture section in `tests/fixtures/README.md` with:

````markdown
## Mandarin submit-command fixtures

`bang-wo-song-chu-zh-cn.wav` and `bang-wo-song-chu-zh-tw.wav` are deterministic
positive fixtures for 「幫我送出」, spoken by the macOS Tingting (`zh_CN`) and
Meijia (`zh_TW`) voices. `bang-wo-zh-tw.wav` and `song-chu-zh-tw.wav` are
partial-command negatives. All four files are 16 kHz mono PCM16.

Reconstruction commands:

```sh
/usr/bin/say -v Tingting -r 170 -o /tmp/bang-wo-song-chu-zh-cn.aiff "幫我送出"
/usr/bin/say -v Meijia -r 170 -o /tmp/bang-wo-song-chu-zh-tw.aiff "幫我送出"
/usr/bin/say -v Meijia -r 170 -o /tmp/bang-wo-zh-tw.aiff "幫我"
/usr/bin/say -v Meijia -r 170 -o /tmp/song-chu-zh-tw.aiff "送出"
/usr/bin/afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/bang-wo-song-chu-zh-cn.aiff tests/fixtures/bang-wo-song-chu-zh-cn.wav
/usr/bin/afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/bang-wo-song-chu-zh-tw.aiff tests/fixtures/bang-wo-song-chu-zh-tw.wav
/usr/bin/afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/bang-wo-zh-tw.aiff tests/fixtures/bang-wo-zh-tw.wav
/usr/bin/afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/song-chu-zh-tw.aiff tests/fixtures/song-chu-zh-tw.wav
```

## tippi-go.wav

Deterministic negative regression for the removed English command. It remains
checked in to prove that `Tippi Go` no longer submits.

Keep the existing Samantha reconstruction commands immediately under the `tippi-go.wav` paragraph.
````

- [ ] **Step 7: Run the command-contract and real-model matrix**

```bash
cargo fmt --check
cargo build -p catcher-ffi --release
swift test --package-path apps/tippi --filter chineseSubmitCommandContractIsExact
swift test --package-path apps/tippi --filter keywordDetectionCarriesCommandAndCutoff
swift test --package-path apps/tippi --filter keywordReleaseManifestPinsOfficialChunk16Artifacts
SHERPA_KWS_MODEL=/tmp/tippi-kws-submit-zh cargo test -p catcher-ffi --test kws_ffi -- --ignored --nocapture
```

Expected: both Mandarin positives detect `SUBMIT_ZH` before and after reset; 「幫我」, 「送出」, old English, hello, and conversation fixtures never detect; the long-prefix timestamp diagnostic passes; all Swift contract tests pass. If the exact `1.5/0.25` line misses either positive or triggers any negative, stop before committing and report the failed fixture and observed result because changing KWS parameters requires a separately recorded full-matrix calibration under the approved spec.

- [ ] **Step 8: Commit the command contract and fixtures**

```bash
git add apps/tippi/Sources/TippiCore/VoiceSubmitCommand.swift apps/tippi/Sources/TippiCore/KeywordModelManifest.swift apps/tippi/Sources/TippiCore/KeywordSpotterClient.swift apps/tippi/Tests/TippiCoreTests/KeywordSpotterClientTests.swift apps/tippi/Tests/TippiCoreTests/KeywordModelInstallerTests.swift crates/catcher-ffi/src/kws.rs crates/catcher-ffi/tests/kws_ffi.rs tests/fixtures/README.md tests/fixtures/bang-wo-song-chu-zh-cn.wav tests/fixtures/bang-wo-song-chu-zh-tw.wav tests/fixtures/bang-wo-zh-tw.wav tests/fixtures/song-chu-zh-tw.wav
git commit -m "feat: switch voice submit command to Chinese"
```

---

### Task 2: Repair generated keyword files without downloading the model

**Files:**
- Modify: `apps/tippi/Sources/TippiCore/KeywordModelInstaller.swift`
- Modify: `apps/tippi/Tests/TippiCoreTests/KeywordModelInstallerTests.swift`

**Interfaces:**
- Consumes: `KeywordModelArchive.files`, `VoiceSubmitCommand.keywordDefinition`, `KeywordModelInstaller.thirdPartyNotice`, `ModelChecksum.sha256`, and the existing downloader/extractor/promoter protocols.
- Produces: `KeywordModelInstallerError.generatedFileRepairFailed`, separated inventory/runtime/generated verification, `writeGeneratedFiles(to:)`, and the no-download repair path.

- [ ] **Step 1: Write failing generated-repair and fallback tests**

Add this damage type and helper to `KeywordModelInstallerTests.swift`:

```swift
private enum InstalledRuntimeDamage {
    case missing
    case corrupt
}

private func assertInvalidRuntimeForcesFullInstall(
    _ damage: InstalledRuntimeDamage
) async throws {
    let fixture = keywordInstallerFixture()
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    let destination = fixture.root.appending(
        path: fixture.manifest.directoryName,
        directoryHint: .isDirectory
    )
    try writeVerifiedInstall(at: destination, payloads: fixture.payloads)
    let damaged = destination.appending(path: keywordRuntimeNames[0])
    switch damage {
    case .missing:
        try FileManager.default.removeItem(at: damaged)
    case .corrupt:
        try Data("corrupt installed runtime".utf8).write(to: damaged)
    }
    let downloader = KeywordArchiveDownloader(payload: fixture.archive)
    let extractor = KeywordArchiveExtractor(
        modelDirectoryName: fixture.manifest.directoryName,
        payloads: fixture.payloads
    )
    let installer = KeywordModelInstaller(
        rootDirectory: fixture.root,
        manifest: fixture.manifest,
        downloader: downloader,
        extractor: extractor
    )

    let installed = try await installer.installIfNeeded { _ in }

    #expect(installed == destination)
    #expect(await downloader.callCount() == 1)
    #expect(await extractor.calls() == 1)
    #expect(try Data(contentsOf: installed.appending(path: keywordRuntimeNames[0]))
        == fixture.payloads[keywordRuntimeNames[0]]!)
}
```

Add these exact tests:

```swift
@Test
func staleGeneratedFilesAreRepairedWithoutDownloadExtractionOrPromotion() async throws {
    let fixture = keywordInstallerFixture()
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    let destination = fixture.root.appending(
        path: fixture.manifest.directoryName,
        directoryHint: .isDirectory
    )
    try writeVerifiedInstall(at: destination, payloads: fixture.payloads)
    try Data("T IH1 P IY0 G OW1 :1.5 #0.25 @TIPPI_GO\n".utf8).write(
        to: destination.appending(path: "keywords.txt")
    )
    try Data("stale notice".utf8).write(
        to: destination.appending(path: "THIRD_PARTY_NOTICES.md")
    )
    let downloader = KeywordArchiveDownloader(payload: fixture.archive)
    let extractor = KeywordArchiveExtractor(
        modelDirectoryName: fixture.manifest.directoryName,
        payloads: fixture.payloads
    )
    let installer = KeywordModelInstaller(
        rootDirectory: fixture.root,
        manifest: fixture.manifest,
        downloader: downloader,
        extractor: extractor,
        promoter: FailingKeywordModelPromoter()
    )

    let installed = try await installer.installIfNeeded { _ in }

    #expect(installed == destination)
    #expect(await downloader.callCount() == 0)
    #expect(await extractor.calls() == 0)
    for (name, payload) in fixture.payloads {
        #expect(try Data(contentsOf: installed.appending(path: name)) == payload)
    }
    #expect(try Data(contentsOf: installed.appending(path: "keywords.txt"))
        == Data(VoiceSubmitCommand.keywordDefinition.utf8))
    #expect(try Data(contentsOf: installed.appending(path: "THIRD_PARTY_NOTICES.md"))
        == Data(expectedThirdPartyNotice.utf8))
}

@Test
func missingInstalledRuntimeForcesFullInstall() async throws {
    try await assertInvalidRuntimeForcesFullInstall(.missing)
}

@Test
func corruptInstalledRuntimeForcesFullInstall() async throws {
    try await assertInvalidRuntimeForcesFullInstall(.corrupt)
}

@Test
func generatedRepairFailureHasGenericChineseError() {
    #expect(KeywordModelInstallerError.generatedFileRepairFailed.errorDescription
        == "無法更新口令模型設定")
}
```

- [ ] **Step 2: Run the installer tests to verify the repair path is missing**

```bash
cargo build -p catcher-ffi --release
swift test --package-path apps/tippi --filter staleGeneratedFilesAreRepairedWithoutDownloadExtractionOrPromotion
swift test --package-path apps/tippi --filter generatedRepairFailureHasGenericChineseError
```

Expected: the first test calls the downloader/promoter path instead of succeeding locally, and the second test does not compile because `generatedFileRepairFailed` is absent.

- [ ] **Step 3: Add the repair error and Chinese installer errors**

Change `KeywordModelInstallerError` to include the new case and these exact descriptions:

```swift
public enum KeywordModelInstallerError: Error, LocalizedError {
    case invalidArchiveChecksum
    case missingRuntimeFile(String)
    case invalidRuntimeFileChecksum(String)
    case incompleteInstallation
    case extractionFailed(String)
    case generatedFileRepairFailed

    public var errorDescription: String? {
        switch self {
        case .invalidArchiveChecksum:
            "口令模型封存檔未通過 SHA-256 驗證"
        case let .missingRuntimeFile(file):
            "口令模型封存檔缺少 \(file)"
        case let .invalidRuntimeFileChecksum(file):
            "口令模型檔案未通過 SHA-256 驗證：\(file)"
        case .incompleteInstallation:
            "已安裝的口令模型不完整"
        case let .extractionFailed(message):
            "無法解壓縮口令模型：\(message)"
        case .generatedFileRepairFailed:
            "無法更新口令模型設定"
        }
    }
}
```

- [ ] **Step 4: Split verification and implement atomic generated-file repair**

After removing stale partial paths and before entering the download block in `installIfNeeded`, use:

```swift
if try verifyInstallation(at: destination) {
    progress(1.0)
    return destination
}
if try hasExpectedInventory(at: destination), try verifyRuntimeFiles(at: destination) {
    do {
        try writeGeneratedFiles(to: destination)
        guard try verifyInstallation(at: destination) else {
            throw KeywordModelInstallerError.incompleteInstallation
        }
    } catch {
        throw KeywordModelInstallerError.generatedFileRepairFailed
    }
    progress(1.0)
    return destination
}
```

Replace the duplicated writes in the full installation path with:

```swift
try writeGeneratedFiles(to: partials.install)
```

Replace the current `verifyInstallation` implementation and add these helpers:

```swift
private var expectedInstalledNames: Set<String> {
    Set(manifest.files.map(\.name) + ["keywords.txt", "THIRD_PARTY_NOTICES.md"])
}

private func hasExpectedInventory(at directory: URL) throws -> Bool {
    var isDirectory: ObjCBool = false
    guard fileManager.fileExists(atPath: directory.path, isDirectory: &isDirectory),
          isDirectory.boolValue
    else {
        return false
    }
    return Set(try fileManager.contentsOfDirectory(atPath: directory.path))
        == expectedInstalledNames
}

private func verifyRuntimeFiles(at directory: URL) throws -> Bool {
    for file in manifest.files {
        let url = directory.appending(path: file.name)
        guard fileManager.fileExists(atPath: url.path),
              try ModelChecksum.sha256(of: url) == file.sha256
        else {
            return false
        }
    }
    return true
}

private func verifyGeneratedFiles(at directory: URL) -> Bool {
    guard let keywords = try? Data(contentsOf: directory.appending(path: "keywords.txt")),
          let notice = try? Data(
              contentsOf: directory.appending(path: "THIRD_PARTY_NOTICES.md")
          )
    else {
        return false
    }
    return keywords == Data(VoiceSubmitCommand.keywordDefinition.utf8)
        && notice == Data(Self.thirdPartyNotice.utf8)
}

private func verifyInstallation(at directory: URL) throws -> Bool {
    guard try hasExpectedInventory(at: directory),
          try verifyRuntimeFiles(at: directory)
    else {
        return false
    }
    return verifyGeneratedFiles(at: directory)
}

private func writeGeneratedFiles(to directory: URL) throws {
    try Data(VoiceSubmitCommand.keywordDefinition.utf8).write(
        to: directory.appending(path: "keywords.txt"),
        options: .atomic
    )
    try Data(Self.thirdPartyNotice.utf8).write(
        to: directory.appending(path: "THIRD_PARTY_NOTICES.md"),
        options: .atomic
    )
}
```

This exact-inventory gate deliberately sends a missing generated file, extra file, missing runtime, or corrupt runtime through the existing verified full-install path. Only stale contents in the two present generated files qualify for repair.

- [ ] **Step 5: Run focused and full installer tests**

```bash
cargo build -p catcher-ffi --release
swift test --package-path apps/tippi
```

Expected: every installer test passes; stale generated data causes zero downloader/extractor calls and cannot invoke the failing promoter; missing/corrupt runtime causes one downloader and one extractor call; unsafe inventory still uses atomic promotion.

- [ ] **Step 6: Commit the installer upgrade path**

```bash
git add apps/tippi/Sources/TippiCore/KeywordModelInstaller.swift apps/tippi/Tests/TippiCoreTests/KeywordModelInstallerTests.swift
git commit -m "fix: repair installed Chinese keyword metadata"
```

---

### Task 3: Apply the 2-second safety window and Chinese event in the controller

**Files:**
- Modify: `apps/tippi/Sources/TippiCore/VoiceInputTiming.swift`
- Modify: `apps/tippi/Sources/TippiCore/TranscriptionController.swift`
- Modify: `apps/tippi/Tests/TippiCoreTests/TranscriptionControllerTests.swift`

**Interfaces:**
- Consumes: `VoiceSubmitCommand.eventIdentifier`, `VoiceSubmitCommand.displayPhrase`, `CatcherServing.text(before:)`, `CatcherServing.finish(before:)`, `KeywordDetection.startMs` as diagnostics only, and `TextInjectionCoordinator` exact-once behavior.
- Produces: `VoiceInputTiming.holdbackMs == 2_000`, `SUBMIT_ZH`-only command handling, Chinese target-retry copy, and unchanged fail-closed/reset semantics.

- [ ] **Step 1: Change tests to the new timing boundary and event identifier**

Replace the timing table with:

```swift
@Test(arguments: zip(
    [UInt64(0), 31_999, 32_000, 48_000],
    [UInt64(0), 0, 0, 1_000]
))
func stableCutoffUsesSixteenKilohertzSampleClock(
    sampleCount: UInt64,
    expectedMs: UInt64
) {
    #expect(VoiceInputTiming.stableCutoffMs(receivedSampleCount: sampleCount) == expectedMs)
}
```

Replace every command detection literal in this test file with:

```swift
KeywordDetection(keyword: VoiceSubmitCommand.eventIdentifier, startMs: startMs)
```

For fixed timestamps such as `320` and `960`, retain the number and replace only the keyword expression. In the tests that assert a 1,000 ms cutoff, change the emitted command chunk from 40,000 samples to 48,000 samples. This applies to:

- `commandCutoffUsesSampleClockInsteadOfKwsTimestamp`
- the first turn in `commandResetStartsTheNextTurnAtZeroMilliseconds`
- the first recording in `stopAndRestartResetTheSampleClock`

Replace `heldTextBecomesInjectableAsSilenceAdvancesTheClock` with:

```swift
@MainActor
@Test
func heldTextBecomesInjectableAsSilenceAdvancesTheClock() async throws {
    let log = EventLog()
    let fixture = makeFixture(log: log)
    await fixture.catcher.script(
        pushes: [TranscriptUpdate(text: "你好", segments: [], warning: nil), nil, nil],
        stableTexts: ["", "", "你好"]
    )
    await fixture.keywordSpotter.script([nil, nil, nil])
    await prepareVoiceFixture(fixture)
    log.removeAll()

    await fixture.controller.toggleRecording(mode: .voiceInput)
    await fixture.audio.emit([Float](repeating: 0, count: 16_000))
    try await waitUntil { log.snapshot().filter { $0 == "asr.textBefore:0" }.count == 1 }
    #expect(fixture.injector.injected.isEmpty)

    await fixture.audio.emit([Float](repeating: 0, count: 16_000))
    try await waitUntil { log.snapshot().filter { $0 == "asr.textBefore:0" }.count == 2 }
    #expect(fixture.injector.injected.isEmpty)

    await fixture.audio.emit([Float](repeating: 0, count: 16_000))
    try await waitUntil { log.snapshot().contains("asr.textBefore:1000") }
    #expect(fixture.injector.injected == ["你好"])
    await fixture.controller.toggleRecording(mode: .voiceInput)
}
```

- [ ] **Step 2: Add identifier-filter and preparation-isolation regressions**

Add this setter to `FakeKeywordInstaller` so preparation failures can be
scripted without exposing actor state:

```swift
func fail(with error: TestFailure?) {
    self.error = error
}
```

Add:

```swift
@MainActor
@Test
func nonSubmitKeywordNeverSendsReturn() async throws {
    let log = EventLog()
    let fixture = makeFixture(log: log)
    await fixture.catcher.script(stableTexts: ["安全內容"])
    await fixture.keywordSpotter.script([
        KeywordDetection(keyword: "OTHER_COMMAND", startMs: 320)
    ])
    await prepareVoiceFixture(fixture)
    log.removeAll()

    await fixture.controller.toggleRecording(mode: .voiceInput)
    await fixture.audio.emit([Float](repeating: 0, count: 48_000))
    try await waitUntil { log.snapshot().contains("asr.textBefore:1000") }

    #expect(fixture.injector.injected == ["安全內容"])
    #expect(fixture.injector.submitCount == 0)
    #expect(!log.snapshot().contains(where: { $0.hasPrefix("asr.finishBefore:") }))
    await fixture.controller.toggleRecording(mode: .voiceInput)
}

@MainActor
@Test
func keywordPreparationFailureLeavesTranscriptionUsable() async {
    let fixture = makeFixture()
    await fixture.controller.prepare()
    await fixture.keywordInstaller.fail(with: .installation)

    await fixture.controller.prepareVoiceInput()

    #expect(fixture.controller.state == .ready)
    #expect(fixture.controller.voiceInputPreparation == .failed("installation failed"))
    #expect(fixture.controller.canToggle(.transcription))
    #expect(!fixture.controller.canToggle(.voiceInput))
}
```

Change the frontmost assertion to:

```swift
try await waitUntil {
    fixture.controller.voiceInputMessage == "請切到目標輸入框後重說「幫我送出」"
}
```

- [ ] **Step 3: Run the focused tests to verify the old controller fails**

```bash
cargo build -p catcher-ffi --release
swift test --package-path apps/tippi --filter stableCutoffUsesSixteenKilohertzSampleClock
swift test --package-path apps/tippi --filter nonSubmitKeywordNeverSendsReturn
swift test --package-path apps/tippi --filter keywordPreparationFailureLeavesTranscriptionUsable
swift test --package-path apps/tippi --filter tippiFrontmostFailsSafeAndRequiresCommandToBeRepeated
```

Expected: the focused group fails on the 2,000 ms timing boundary and Chinese
frontmost copy. The identifier-filter and preparation-isolation tests pass as
regressions for behavior that must remain fail-closed.

- [ ] **Step 4: Implement the timing and identifier changes**

In `VoiceInputTiming.swift` change only:

```swift
public static let holdbackMs: UInt64 = 2_000
```

In `TranscriptionController.processVoiceInput`, replace the command condition and target retry with:

```swift
if let detection, detection.keyword == VoiceSubmitCommand.eventIdentifier {
    if suppressImmediateDuplicateCommand {
        try await catcher.start()
        try await keywordSpotter.reset()
        resetVoiceTurn()
        return
    }

    let final = try await catcher.finish(before: cutoffMs)
    let event = try injectionCoordinator.submit(final.text)
    applyInjectionEvent(event)
    if event == .waitingForTarget {
        voiceInputMessage = "請切到目標輸入框後重說「\(VoiceSubmitCommand.displayPhrase)」"
    }
    try await catcher.start()
    try await keywordSpotter.reset()
    resetVoiceTurn()
    suppressImmediateDuplicateCommand = true
    return
}
```

Leave `receivedSampleCount` advancement before both model calls, the `catcher.text(before:)` non-command path, empty-submit behavior, duplicate suppression, stop cleanup, and all reset sites unchanged.

- [ ] **Step 5: Run every controller and injection regression**

```bash
cargo build -p catcher-ffi --release
swift test --package-path apps/tippi
```

Expected: all tests pass, including timestamp independence for `startMs = 0/320/960`, 2-second held text, empty-turn suppression, one Return for a duplicate command, sample-clock reset after command/stop/failure, and Tippi-frontmost safety.

- [ ] **Step 6: Commit the controller safety-window change**

```bash
git add apps/tippi/Sources/TippiCore/VoiceInputTiming.swift apps/tippi/Sources/TippiCore/TranscriptionController.swift apps/tippi/Tests/TippiCoreTests/TranscriptionControllerTests.swift
git commit -m "feat: use Chinese submit event with two-second holdback"
```

---

### Task 4: Replace command-facing UI and user documentation

**Files:**
- Modify: `apps/tippi/Sources/TippiApp/VoiceInputTabView.swift`
- Modify: `README.md`

**Interfaces:**
- Consumes: `VoiceSubmitCommand.displayPhrase`, `VoiceInputTiming.holdbackMs`, controller preparation/recording state, and the approved user workflow.
- Produces: Chinese-only command copy, an explicit 2-second delay, and documentation that matches runtime behavior.

- [ ] **Step 1: Replace the Voice Input header and badge**

Use the shared phrase in `VoiceInputTabView.header`:

```swift
private var header: some View {
    HStack(alignment: .firstTextBaseline) {
        VStack(alignment: .leading, spacing: 5) {
            Text("語音輸入")
                .font(.system(size: 32, weight: .semibold, design: .rounded))
            Text("內容說完後短暫停頓，再說「\(VoiceSubmitCommand.displayPhrase)」。")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
        Spacer()
        Text("口令：\(VoiceSubmitCommand.displayPhrase)")
            .font(.callout.weight(.semibold))
            .padding(.horizontal, 12)
            .padding(.vertical, 7)
            .background(.indigo.opacity(0.12), in: Capsule())
    }
}
```

- [ ] **Step 2: Replace status, retry, and recording-hint copy**

Use these exact strings:

```swift
case .notPrepared:
    Label("正在準備口令模型", systemImage: "clock")
        .foregroundStyle(.secondary)
// downloading and loading branches stay structurally unchanged
case .ready:
    Label("「\(VoiceSubmitCommand.displayPhrase)」口令模型已就緒", systemImage: "checkmark.circle.fill")
        .foregroundStyle(.green)
```

In the target-attention message while recording, use:

```swift
Text("請切到目標輸入框；Tippi 不會把文字輸入到自己。切換後請重說「\(VoiceSubmitCommand.displayPhrase)」。")
```

In `recordingHint`, use:

```swift
if controller.isRecording(.voiceInput) {
    return "文字約延遲 2 秒；短暫停頓後說「\(VoiceSubmitCommand.displayPhrase)」。"
}
```

- [ ] **Step 3: Rewrite active README command behavior**

Make the introduction end with:

```markdown
record speaker-attributed transcripts, or stream recognized text into the
frontmost app and say `幫我送出` to press Return.
```

Replace the Voice Input usage and timing block with:

```markdown
Voice Input sends the same 16 kHz microphone stream to two local models. The
main Catcher ASR model produces the text to inject. A second, smaller offline
[`sherpa-onnx`](https://github.com/k2-fsa/sherpa-onnx) keyword-spotting model
detects the fixed `幫我送出` command without sending audio to a cloud service.
Tippi downloads and verifies the pinned keyword model when this tab is first
prepared. Existing valid installations update the generated command files in
place without downloading the model archive again.

To use it:

1. 打開「語音輸入」分頁，等待語音辨識與「幫我送出」口令模型就緒。
2. 授予 Tippi「系統設定 → 隱私權與安全性 → 輔助使用」權限。
3. 按「開始語音輸入」，切到目標 App 並點進輸入框。
4. 說完內容後短暫停頓（約 0.5 秒），再說「幫我送出」。
5. 口令不會進入輸入框；停止按鈕不會自動送出未完成內容。

- Live text is intentionally held for about 2 seconds before injection.
- After finishing the content, pause briefly (about 0.5 seconds), then say `幫我送出`.
- The pause keeps the final content outside the command safety window.
- If content runs directly into the command, Tippi prefers dropping a very short tail over leaking command words.
- sherpa-onnx timestamps are diagnostic-only; cutoff follows the shared 16 kHz sample clock.
```

Change the frontmost retry sentence to `Switch back to the target input field and say 「幫我送出」 again.` Change the limitations bullet to say the fixed `幫我送出` command. Do not rewrite historical files under `docs/superpowers/specs` or `docs/superpowers/plans`.

- [ ] **Step 4: Run source and Swift checks**

```bash
if rg -n 'Tippi Go|TIPPI_GO' apps/tippi/Sources crates/catcher-ffi/src README.md; then
  exit 1
fi
cargo build -p catcher-ffi --release
swift test --package-path apps/tippi
swift build --package-path apps/tippi
```

Expected: the source scan prints nothing; all Swift tests pass; both library and app targets build.

- [ ] **Step 5: Commit UI and documentation**

```bash
git add apps/tippi/Sources/TippiApp/VoiceInputTabView.swift README.md
git commit -m "docs: teach the Chinese voice submit command"
```

---

### Task 5: Run full regression, package the app, and perform real cross-App acceptance

**Files:**
- Verify: all files changed in Tasks 1-4
- Build output: `apps/tippi/build/Tippi.app` (not committed)

**Interfaces:**
- Consumes: the completed Rust/Swift command, installer, controller, UI, fixture, and documentation changes; installed local Catcher/sherpa models; current Microphone and Accessibility permissions.
- Produces: evidence that unit, real-model, packaging, generated-file upgrade, and human cross-App behavior satisfy the approved spec.

- [ ] **Step 1: Run the complete automated regression**

```bash
cargo fmt --check
cargo test --workspace
cargo build -p catcher-ffi --release
swift test --package-path apps/tippi
```

Expected: every non-ignored Rust workspace test and every Swift test passes. No existing transcription, diarization, injection, or installer test regresses.

- [ ] **Step 2: Run the complete real KWS matrix from a disposable model copy**

```bash
KWS_SOURCE="$HOME/Library/Application Support/Tippi/Models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
KWS_TEST=/tmp/tippi-kws-submit-zh
rm -rf "$KWS_TEST"
cp -R "$KWS_SOURCE" "$KWS_TEST"
printf '%s\n' 'b āng w ǒ s òng ch ū :1.5 #0.25 @SUBMIT_ZH' > "$KWS_TEST/keywords.txt"
SHERPA_KWS_MODEL="$KWS_TEST" cargo test -p catcher-ffi --test kws_ffi -- --ignored --nocapture
```

Expected: both Mandarin positives pass twice with reset; all five negative fixtures produce no detection; the long-prefix timestamp diagnostic remains green.

- [ ] **Step 3: Build and verify the signed app bundle**

```bash
apps/tippi/scripts/build-app.sh
apps/tippi/scripts/verify-app.sh
```

Expected: `apps/tippi/build/Tippi.app` is arm64, contains and links `libcatcher_ffi.dylib` through `@rpath`, has no worktree library path, is ad-hoc signed with microphone entitlement/usage text, and passes deep strict code-sign verification.

- [ ] **Step 4: Verify the installed generated-file upgrade**

Before launching, confirm the existing installed runtime hashes still match the
pinned manifest:

```bash
KWS_INSTALLED="$HOME/Library/Application Support/Tippi/Models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
test "$(shasum -a 256 "$KWS_INSTALLED/encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx" | cut -d ' ' -f 1)" = 408bbd740838c42d5bf6d1c5b80b3c88b616c7860b92d980328b5b068c76ae48
test "$(shasum -a 256 "$KWS_INSTALLED/decoder-epoch-13-avg-2-chunk-16-left-64.onnx" | cut -d ' ' -f 1)" = 63a22dd60f40fff082ac3e09afa507f6787da36df76ded2fbe145fa233e22c21
test "$(shasum -a 256 "$KWS_INSTALLED/joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx" | cut -d ' ' -f 1)" = 190d4067b4cc20b72a42a1916e69d92052000fb7051a427ebb1bc72a69207dc1
test "$(shasum -a 256 "$KWS_INSTALLED/tokens.txt" | cut -d ' ' -f 1)" = 2d3f32311f9b692b964da3c90e830258d3e78e013cb0c992dbfb15cd5a1a71b0
open -n apps/tippi/build/Tippi.app
```

Open 語音輸入 once and wait for the KWS ready state, then run:

```bash
KWS_INSTALLED="$HOME/Library/Application Support/Tippi/Models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
test "$(cat "$KWS_INSTALLED/keywords.txt")" = 'b āng w ǒ s òng ch ū :1.5 #0.25 @SUBMIT_ZH'
test "$(find "$KWS_INSTALLED" -maxdepth 1 -type f | wc -l | tr -d ' ')" = 6
```

Expected: both checks pass. The installer unit test from Task 2 is the proof that this path does not call the downloader, extractor, or promoter; the live check proves the user's old generated file was migrated.

- [ ] **Step 5: Perform three positive TextEdit turns**

With TextEdit frontmost and its insertion point active, start Voice Input and perform three separate turns. In each turn, speak ordinary Mandarin content, pause about 0.5 seconds, then say 「幫我送出」 at a normal pace.

Expected for all three turns:

- The ordinary content appears after the intentional delay.
- Tippi detects the command and presses Return exactly once.
- Neither 「幫我送出」 nor an ASR approximation of it appears in TextEdit.
- Turns two and three work without restarting recording, proving KWS, duplicate guard, coordinator, ASR, and sample clock reset.

- [ ] **Step 6: Perform negative and fail-closed acceptance**

While recording into TextEdit, say each of these as a separate turn: 「送出」, 「幫我」, and `Tippi Go`.

Expected: none sends Return. Then verify:

1. Speak unsent content and press Stop: no held tail and no Return is emitted.
2. Put Tippi frontmost and say 「幫我送出」: no injection and no Return occurs; the UI asks for the target and a repeated command.
3. Focus a non-text target and say 「幫我送出」: injection and Return remain fail-closed.
4. Return to TextEdit and repeat 「幫我送出」 after content: one Return occurs, proving the discarded command was not deferred.

- [ ] **Step 7: Confirm the repository handoff is clean**

```bash
git status --short
git log -5 --oneline
```

Expected: `git status --short` prints nothing. The recent history contains the four implementation commits from Tasks 1-4; Task 5 creates only the ignored app bundle and validation evidence reported in the final handoff.
