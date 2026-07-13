# Sortformer 階段三 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tippi 端到端 who-said-what 體驗——雙模型合併下載、分說話者訊息列表、說話者命名、複製/匯出——並完成階段二遺留的三個 Rust 旗標。

**Architecture:** Rust 側先修三旗標(FFI diar 可恢復、fusion 邊界標點、rust-toolchain.toml),Swift 側走方案 A:兩個 `ModelStore` 實例 + `ModelBundleInstaller` 合併進度;segments JSON 在 `CatcherClient` 解成型別;`TranscriptionController` 組 `Message` 列表;`ContentView` 呈現訊息、改名 popover、警告橫幅、複製/匯出。

**Tech Stack:** Rust(catcher-ffi、nemotron-mlx fusion)、Swift 6 / SwiftUI(apps/tippi,swift-testing)、無新第三方依賴。

**Spec:** `docs/superpowers/specs/2026-07-13-sortformer-phase3-design.md`

## Global Constraints

- 不新增任何第三方依賴(Rust 與 Swift 皆是)。
- JSON 欄位名(Rust serde 決定,Swift 必須對齊):`speaker`、`start_ms`、`end_ms`、`text`、`final`。
- 繁中 UI 文案精確字串:未命名說話者顯示 `說話者 N`(N = speaker 索引 + 1);警告橫幅 `說話者分離已暫停,文字繼續轉寫`;按鈕 `複製全部`、`匯出…`、context menu `複製此則`。
- 複製/匯出行格式:`[mm:ss] 顯示名：內文`(冒號是全形 `:`,mm 至少兩位數、超過 99 分鐘自然進位如 `[102:07]`),多行以 `\n` join;複製與匯出共用同一個格式化函式。
- rust-toolchain.toml 鎖 `channel = "1.95.0"`。
- diar artifact hash/位元組數必須逐字使用 Task 4 表列值(來源:main commit `e556258` body)。
- 所有會驅動 MLX 的 Rust gated 測試(`#[ignore = "requires …"]`)必須先取 `serialize_mlx()` guard;環境變數 `NEMOTRON_MLX_ARTIFACT`(/tmp/catcher-asr-mlx-int8)、`SORTFORMER_MLX_ARTIFACT`(/tmp/catcher-diar-mlx-int8)。
- Rust 驗證:`cargo test -p <crate>`(gated 加 `-- --ignored`)、`cargo fmt --check`。Swift 驗證:`cd apps/tippi && swift build && swift test`。
- 錯誤處理慣例:FFI 沿用 `with_handle_mut`/`catch_unwind`/`safe_c_string`;Swift 沿用 `CatcherClientError`。
- diar 執行期降級為非致命:錄音繼續、warning 供查詢;`catcher_start` 重建失敗同樣非致命(回 `CATCHER_OK` 並設 warning)。

---

## File Structure

| 檔案 | 動作 | 職責 |
|---|---|---|
| `rust-toolchain.toml` | Create | 鎖定 toolchain 1.95.0 |
| `crates/nemotron-mlx/src/fusion.rs` | Modify | `segments()` 尾端加邊界標點後處理 |
| `crates/nemotron-mlx/tests/fusion.rs` | Modify | 標點後處理單元測試 |
| `crates/catcher-ffi/src/lib.rs` | Modify | handle 保留 diar 路徑、start 重建 diarizer、test hook |
| `crates/catcher-ffi/tests/ffi_lifecycle.rs` | Modify | degrade→重建 gated 測試 |
| `crates/catcher-ffi/include/catcher.h`、`apps/tippi/Sources/CCatcher/include/catcher.h` | Modify | `catcher_start` 契約註解 |
| `apps/tippi/Sources/TippiCore/ModelStore.swift` | Modify | `directoryName` 建構參數、diar repo URL |
| `apps/tippi/Sources/TippiCore/ModelManifest.swift` | Modify | `.diarizationRelease`、`totalByteCount` |
| `apps/tippi/Sources/TippiCore/ModelBundleInstaller.swift` | Create | `ModelBundle`/`ModelBundleInstalling`/`ModelBundleInstaller` |
| `apps/tippi/Sources/TippiCore/Transcript.swift` | Create | `SpeakerSegment`/`TranscriptUpdate`/`Message`/`TranscriptFormatter` |
| `apps/tippi/Sources/TippiCore/CatcherClient.swift` | Modify | CatcherServing v2、JSON 解碼、diar 必填 |
| `apps/tippi/Sources/TippiCore/TranscriptionController.swift` | Modify | `messages`/`speakerNames`/`warningMessage` |
| `apps/tippi/Sources/TippiApp/TippiApp.swift` | Modify | bundle installer + 雙目錄 factory 佈線 |
| `apps/tippi/Sources/TippiApp/ContentView.swift` | Modify | 訊息列表、橫幅、複製/匯出 |
| `apps/tippi/Sources/TippiApp/MessageRow.swift` | Create | 單則訊息 view + 改名 popover |
| `apps/tippi/Resources/Tippi.entitlements` | Modify | 加 `files.user-selected.read-write` |
| `apps/tippi/Tests/TippiCoreTests/*` | Modify/Create | 對應單元測試 |
| `README.md` | Modify | Tippi 雙模型/訊息 UI 說明 |

---

### Task 1: rust-toolchain.toml

**Files:**
- Create: `rust-toolchain.toml`(workspace 根)

**Interfaces:**
- Consumes: 無
- Produces: 固定 toolchain;後續所有 `cargo` 命令走 1.95.0

- [ ] **Step 1: 建立檔案**

```toml
[toolchain]
channel = "1.95.0"
components = ["rustfmt", "clippy"]
```

- [ ] **Step 2: 驗證 toolchain 生效且 fmt 乾淨**

Run: `cd /Users/wake/Desktop/catcher-tippi && rustc --version && cargo fmt --check`
Expected: `rustc 1.95.0 (…)`;fmt 無輸出、exit 0。若 rustup 缺該版會自動下載。

- [ ] **Step 3: 快速編譯煙霧測試**

Run: `cargo check -p nemotron-mlx -p sortformer-mlx -p catcher-ffi`
Expected: 全部通過。

- [ ] **Step 4: Commit**

```bash
git add rust-toolchain.toml
git commit -m "chore: pin Rust toolchain to 1.95.0"
```

---

### Task 2: fusion 邊界標點後處理

**Files:**
- Modify: `crates/nemotron-mlx/src/fusion.rs`(`segments()` 約 line 119-162;檔尾加私有函式)
- Test: `crates/nemotron-mlx/tests/fusion.rs`

**Interfaces:**
- Consumes: `Fusion::segments(detokenize)`、`SpeakerSegment`(既有)
- Produces: `segments()` 回傳值保證「非首段落的文字不以邊界標點開頭」;全標點段落被併入前段後移除。`SpeakerSegment` 欄位不變。

**語義(spec §1b):** `segments()` 組完結果後,對第 1..n 段:若文字以邊界標點(`,。?!、:;…` 八個字元)開頭,把整串前導標點移到前一段文字尾端;搬完若該段文字為空,移除該段。首段不動。時間戳(start_ms/end_ms)不調整——這是純文字外觀修正。tentative tail 一併處理(每次 `segments()` 全量重算,跨次呼叫自洽)。

- [ ] **Step 1: 寫失敗測試**

在 `crates/nemotron-mlx/tests/fusion.rs` 追加(沿用該檔既有的 TimedToken/Fusion 建構慣例;`TimedToken { id, frame }` 來自 `nemotron_mlx::model::TimedToken`):

```rust
/// 45 diar frames:0..=17 為 speaker0 高機率,18..=44 為 speaker1。
fn two_speaker_diar_frames() -> Vec<[f32; 4]> {
    (0..45)
        .map(|frame| {
            if frame <= 17 {
                [0.9, 0.05, 0.0, 0.0]
            } else {
                [0.05, 0.9, 0.0, 0.0]
            }
        })
        .collect()
}

#[test]
fn leading_punctuation_moves_to_previous_segment() {
    let mut fusion = Fusion::new(FusionConfig::default());
    fusion.push_tokens(&[
        TimedToken { id: 1, frame: 2 },
        TimedToken { id: 2, frame: 10 },
        TimedToken { id: 3, frame: 25 },
        TimedToken { id: 4, frame: 35 },
    ]);
    fusion.push_diar_frames(&two_speaker_diar_frames());
    fusion.flush();

    let segments = fusion.segments(|ids| {
        if ids.contains(&3) {
            ",第二段".to_string()
        } else {
            "第一段".to_string()
        }
    });

    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].text, "第一段,");
    assert_eq!(segments[1].text, "第二段");
    assert_eq!(segments[1].speaker, 1);
}

#[test]
fn all_punctuation_segment_is_absorbed_and_dropped() {
    let mut fusion = Fusion::new(FusionConfig::default());
    fusion.push_tokens(&[
        TimedToken { id: 1, frame: 2 },
        TimedToken { id: 2, frame: 10 },
        TimedToken { id: 3, frame: 25 },
        TimedToken { id: 4, frame: 35 },
    ]);
    fusion.push_diar_frames(&two_speaker_diar_frames());
    fusion.flush();

    let segments = fusion.segments(|ids| {
        if ids.contains(&3) {
            "?!".to_string()
        } else {
            "第一段".to_string()
        }
    });

    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].text, "第一段?!");
}

#[test]
fn first_segment_leading_punctuation_is_untouched() {
    let mut fusion = Fusion::new(FusionConfig::default());
    fusion.push_tokens(&[TimedToken { id: 1, frame: 2 }, TimedToken { id: 2, frame: 10 }]);
    fusion.push_diar_frames(&two_speaker_diar_frames());
    fusion.flush();

    let segments = fusion.segments(|_| ",開頭".to_string());

    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].text, ",開頭");
}
```

(token 佈局說明:speaker0 tokens 在 frame 2、10 → 段長 720ms,speaker1 在 25、35 → 880ms,均大於 `min_turn_ms` 500,避免被 anti-flicker 合併;smooth_frames 3 的窗口不跨越 frame 17/18 邊界。)

- [ ] **Step 2: 確認失敗**

Run: `cargo test -p nemotron-mlx --test fusion`
Expected: 前兩個新測試 FAIL(標點仍留在第二段);第三個可能已過。

- [ ] **Step 3: 實作**

`fusion.rs` `segments()` 的 `result` 建完(`build_tail` 追加之後、`return` 之前)呼叫後處理:

```rust
        shift_boundary_punctuation(&mut result);

        result
    }
}
```

檔尾(`attribute_speaker` 等私有函式區)加:

```rust
/// Boundary punctuation that reads as "closing" the previous utterance.
/// A segment beginning with these characters looks wrong attributed to the
/// *next* speaker, so the leading run migrates to the previous segment's
/// tail (cosmetic only: timestamps are untouched).
const BOUNDARY_PUNCTUATION: [char; 8] = [',', '。', '?', '!', '、', ':', ';', '…'];

fn shift_boundary_punctuation(segments: &mut Vec<SpeakerSegment>) {
    let mut index = 1;
    while index < segments.len() {
        let text = &segments[index].text;
        let prefix_end = text
            .char_indices()
            .find(|&(_, c)| !BOUNDARY_PUNCTUATION.contains(&c))
            .map(|(byte, _)| byte)
            .unwrap_or(text.len());
        if prefix_end > 0 {
            let prefix = segments[index].text[..prefix_end].to_string();
            segments[index].text.replace_range(..prefix_end, "");
            segments[index - 1].text.push_str(&prefix);
        }
        if segments[index].text.is_empty() {
            segments.remove(index);
        } else {
            index += 1;
        }
    }
}
```

- [ ] **Step 4: 確認通過(含既有測試)**

Run: `cargo test -p nemotron-mlx --test fusion`
Expected: 全 PASS(既有 fusion 測試不得回歸;若某個既有 fixture 恰有前導標點被搬移,以新語義為準更新該測試的期望值並在 commit message 註明)。

- [ ] **Step 5: Commit**

```bash
git add crates/nemotron-mlx/src/fusion.rs crates/nemotron-mlx/tests/fusion.rs
git commit -m "fix: migrate leading boundary punctuation to the previous fusion segment"
```

---

### Task 3: FFI diar 降級可恢復

**Files:**
- Modify: `crates/catcher-ffi/src/lib.rs`
- Modify: `crates/catcher-ffi/include/catcher.h`(`catcher_start` 註解)
- Modify: `apps/tippi/Sources/CCatcher/include/catcher.h`(同步註解)
- Test: `crates/catcher-ffi/tests/ffi_lifecycle.rs`

**Interfaces:**
- Consumes: `StreamingDiarizer::from_artifact_dir(&str) -> Result<_, ModelError>`(既有)
- Produces: `CatcherHandle` 欄位 `diar_model_path: Option<String>` 取代 `diarization_requested: bool`;`catcher_start` 新契約:diarizer 若已降級則就地重建,重建失敗回 `CATCHER_OK` 並設 warning;`#[doc(hidden)] pub unsafe fn test_degrade_diarizer(handle: *mut CatcherHandle)` 供整合測試注入降級。C ABI 簽名零改動。

- [ ] **Step 1: 寫失敗測試(gated)**

在 `crates/catcher-ffi/tests/ffi_lifecycle.rs` 追加。沿用檔內既有慣例:`serialize_mlx()` guard、雙模型測試(line ~172)的環境變數與音訊載入 helper(推 `tests/fixtures/conversation.wav` 的樣本;直接重用該測試已有的讀檔程式;若它是行內邏輯,抽成 `fn conversation_samples() -> Vec<f32>` 讓兩個測試共用)。

```rust
#[test]
#[ignore = "requires NEMOTRON_MLX_ARTIFACT and SORTFORMER_MLX_ARTIFACT"]
fn degraded_diarizer_is_rebuilt_on_next_start() {
    let _guard = serialize_mlx();
    let model = CString::new(std::env::var("NEMOTRON_MLX_ARTIFACT").unwrap()).unwrap();
    let diar_model = CString::new(std::env::var("SORTFORMER_MLX_ARTIFACT").unwrap()).unwrap();
    let language = CString::new("auto").unwrap();
    let handle =
        unsafe { catcher_create(model.as_ptr(), diar_model.as_ptr(), language.as_ptr(), 3) };
    assert!(!handle.is_null());
    let samples = conversation_samples();

    unsafe {
        assert_eq!(catcher_start(handle), CATCHER_OK);
        catcher_push_audio(handle, samples.as_ptr(), 16_000);
        assert!(catcher_warning(handle).is_null());

        // 注入執行期降級:warning 出現、後續 push 仍可運作(純 ASR)。
        catcher_ffi::test_degrade_diarizer(handle);
        assert!(!catcher_warning(handle).is_null());
        let status = catcher_push_audio(handle, samples[16_000..].as_ptr(), 16_000);
        assert!(status == CATCHER_OK || status == CATCHER_NO_UPDATE);
        assert!(!catcher_warning(handle).is_null());
        catcher_finish(handle);

        // 下一次 start 就地重建:warning 清空、diarization 恢復產出 segments。
        assert_eq!(catcher_start(handle), CATCHER_OK);
        assert!(catcher_warning(handle).is_null());
        let mut offset = 0usize;
        let chunk = 16_000;
        let mut recovered = false;
        while offset + chunk <= samples.len() {
            catcher_push_audio(handle, samples[offset..].as_ptr(), chunk);
            offset += chunk;
            let segments = unsafe { CStr::from_ptr(catcher_segments(handle)) };
            if segments.to_bytes() != b"[]" {
                recovered = true;
                break;
            }
        }
        assert!(recovered, "diarization produced no segments after rebuild");
        catcher_destroy(handle);
    }
}

#[test]
#[ignore = "requires NEMOTRON_MLX_ARTIFACT and SORTFORMER_MLX_ARTIFACT"]
fn failed_rebuild_keeps_warning_and_start_succeeds() {
    let _guard = serialize_mlx();
    // 把 diar artifact 複製到暫存目錄,degrade 後刪除,迫使重建失敗。
    let source = std::path::PathBuf::from(std::env::var("SORTFORMER_MLX_ARTIFACT").unwrap());
    let staging = std::env::temp_dir().join(format!("catcher-ffi-rebuild-{}", std::process::id()));
    std::fs::create_dir_all(&staging).unwrap();
    for entry in std::fs::read_dir(&source).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), staging.join(entry.file_name())).unwrap();
    }

    let model = CString::new(std::env::var("NEMOTRON_MLX_ARTIFACT").unwrap()).unwrap();
    let diar_model = CString::new(staging.to_str().unwrap()).unwrap();
    let language = CString::new("auto").unwrap();
    let handle =
        unsafe { catcher_create(model.as_ptr(), diar_model.as_ptr(), language.as_ptr(), 3) };
    assert!(!handle.is_null());

    unsafe {
        assert_eq!(catcher_start(handle), CATCHER_OK);
        catcher_ffi::test_degrade_diarizer(handle);
        std::fs::remove_dir_all(&staging).unwrap();

        // 重建失敗必須非致命:start 回 OK、warning 保留、純 ASR 繼續可用。
        assert_eq!(catcher_start(handle), CATCHER_OK);
        assert!(!catcher_warning(handle).is_null());
        let samples = conversation_samples();
        let status = catcher_push_audio(handle, samples.as_ptr(), 16_000);
        assert!(status == CATCHER_OK || status == CATCHER_NO_UPDATE);
        catcher_destroy(handle);
    }
}
```

- [ ] **Step 2: 確認編譯失敗**

Run: `cargo test -p catcher-ffi --test ffi_lifecycle --no-run`
Expected: FAIL——`test_degrade_diarizer` 不存在。

- [ ] **Step 3: 實作**

`crates/catcher-ffi/src/lib.rs`:

1. `CatcherHandle`:`diarization_requested: bool` 換成路徑欄位(doc comment 一併改寫):

```rust
    /// The `diar_model_path` given to `catcher_create`, kept so
    /// `catcher_start` can rebuild a diarizer that degraded to `None` after
    /// a runtime error. `None` means diarization was never requested, in
    /// which case `catcher_segments` stays `[]` forever.
    diar_model_path: Option<String>,
```

2. `catcher_create`:刪掉 `let diarization_requested = …`,建構 handle 時 `diar_model_path: diar_model_path.clone()`(在 `match diar_model_path` 之前先 clone,或改寫成先建 diarizer 再存路徑——以編譯通過的最小改寫為準)。
3. `rebuild_strings_and_report_segment_change` 內 `if handle.diarization_requested` → `if handle.diar_model_path.is_some()`。
4. `catcher_start` 重建邏輯(取代現有 `if let Some(diarizer) = handle.diarizer.as_mut() { diarizer.reset(); }`):

```rust
            handle.warning = None;
            match handle.diarizer.as_mut() {
                Some(diarizer) => diarizer.reset(),
                None => {
                    if let Some(path) = handle.diar_model_path.clone() {
                        match StreamingDiarizer::from_artifact_dir(&path) {
                            Ok(diarizer) => handle.diarizer = Some(diarizer),
                            Err(error) => {
                                handle.warning = Some(diarizer_rebuild_warning(&error));
                            }
                        }
                    }
                }
            }
```

(原本無條件的 `handle.warning = None;` 行移除,以上面這段開頭的清除為準。)

5. 檔尾加:

```rust
/// Formats the warning stored when `catcher_start` fails to rebuild a
/// previously degraded diarizer. Distinct wording from
/// [`diarizer_disabled_warning`] so logs distinguish "died mid-session"
/// from "could not come back".
fn diarizer_rebuild_warning(error: &sortformer_mlx::model::ModelError) -> CString {
    safe_c_string(&format!(
        "diarization unavailable: failed to reload the model: {error}"
    ))
}

/// Test-only hook: simulates the runtime degradation path (diarizer dropped,
/// warning set) without needing a real mid-session model failure. Not part
/// of the C ABI (no `#[unsafe(no_mangle)]`), only reachable from Rust
/// integration tests.
///
/// # Safety
///
/// `handle` must be a live pointer returned by `catcher_create`.
#[doc(hidden)]
pub unsafe fn test_degrade_diarizer(handle: *mut CatcherHandle) {
    let handle = unsafe { &mut *handle };
    handle.warning = Some(safe_c_string(
        "diarization disabled after a runtime error: injected by test",
    ));
    handle.diarizer = None;
}
```

6. 兩份 `catcher.h`(`crates/catcher-ffi/include/catcher.h` 完整版與 `apps/tippi/Sources/CCatcher/include/catcher.h` 精簡版)在 `catcher_start` 的註解補上契約:start 會嘗試重建先前降級的 diarizer;重建失敗仍回 `CATCHER_OK`,以 `catcher_warning` 回報,轉寫以純 ASR 繼續。

- [ ] **Step 4: 非 gated 測試通過**

Run: `cargo test -p catcher-ffi`
Expected: 全 PASS(gated 不跑)。

- [ ] **Step 5: gated 測試通過**

Run: `NEMOTRON_MLX_ARTIFACT=/tmp/catcher-asr-mlx-int8 SORTFORMER_MLX_ARTIFACT=/tmp/catcher-diar-mlx-int8 cargo test -p catcher-ffi --test ffi_lifecycle -- --ignored`
Expected: 全 PASS(含既有 4 個 gated 測試與新增 2 個)。

- [ ] **Step 6: Commit**

```bash
git add crates/catcher-ffi apps/tippi/Sources/CCatcher/include/catcher.h
git commit -m "feat: rebuild a degraded diarizer on catcher_start"
```

---

### Task 4: ModelStore directoryName 參數 + diar manifest

**Files:**
- Modify: `apps/tippi/Sources/TippiCore/ModelStore.swift`
- Modify: `apps/tippi/Sources/TippiCore/ModelManifest.swift`
- Test: `apps/tippi/Tests/TippiCoreTests/ModelStoreTests.swift`

**Interfaces:**
- Consumes: 既有 `ModelStore`/`ModelFile`
- Produces:`ModelStore.init(rootDirectory:baseURL:files:directoryName:downloader:fileManager:)`(`directoryName` 預設 `"catcher-asr-mlx-int8"`);`ModelStore.diarizationRepositoryURL`;`[ModelFile].diarizationRelease`;`[ModelFile].totalByteCount: Int64`。

- [ ] **Step 1: 寫失敗測試**

`ModelStoreTests.swift` 追加(沿用檔內 `FakeDownloader`/`sha256`):

```swift
@Test
func customDirectoryNameInstallsIntoThatDirectory() async throws {
    let root = FileManager.default.temporaryDirectory
        .appending(path: UUID().uuidString, directoryHint: .isDirectory)
    defer { try? FileManager.default.removeItem(at: root) }
    let weights = Data("diar-weights".utf8)
    let downloader = FakeDownloader(payloads: ["weights.safetensors": weights])
    let store = ModelStore(
        rootDirectory: root,
        baseURL: URL(string: "https://example.test/diar/")!,
        files: [ModelFile(name: "weights.safetensors", sha256: sha256(weights), required: true)],
        directoryName: "catcher-diar-mlx-int8",
        downloader: downloader
    )

    let installed = try await store.installIfNeeded { _ in }

    #expect(installed.lastPathComponent == "catcher-diar-mlx-int8")
    #expect(FileManager.default.fileExists(atPath: installed.appending(path: "weights.safetensors").path))
    #expect(!FileManager.default.fileExists(atPath: root.appending(path: ".catcher-diar-mlx-int8.partial").path))
}

@Test
func diarizationManifestPinsSevenFilesAndTotalBytes() {
    let files = [ModelFile].diarizationRelease
    #expect(files.count == 7)
    #expect(files.allSatisfy(\.required))
    #expect(files.totalByteCount == 127_401_153)
    #expect(files.first { $0.name == "weights.safetensors" }?.sha256
        == "a02b1a83ceb6c1f9cf048ab3420c86c84421b0f4e64c433da75b506411445987")
}
```

- [ ] **Step 2: 確認失敗**

Run: `cd apps/tippi && swift test --filter TippiCoreTests`
Expected: 編譯 FAIL(`directoryName` 參數與 `diarizationRelease` 不存在)。

- [ ] **Step 3: 實作**

`ModelStore.swift`:

- `private let modelDirectoryName = "catcher-asr-mlx-int8"` 刪除,改為 `private let modelDirectoryName: String`。
- `init` 在 `files:` 後加 `directoryName: String = "catcher-asr-mlx-int8",`,並 `self.modelDirectoryName = directoryName`。
- `repositoryURL` 旁加:

```swift
    public static let diarizationRepositoryURL = URL(
        string: "https://huggingface.co/wcamon/catcher-diar-mlx-int8/resolve/main/"
    )!
```

`ModelManifest.swift` 追加(hash/bytes 逐字來自 main commit `e556258` body):

```swift
public extension Array where Element == ModelFile {
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
```

- [ ] **Step 4: 確認通過**

Run: `cd apps/tippi && swift test --filter TippiCoreTests`
Expected: 全 PASS。

- [ ] **Step 5: Commit**

```bash
git add apps/tippi/Sources/TippiCore/ModelStore.swift apps/tippi/Sources/TippiCore/ModelManifest.swift apps/tippi/Tests/TippiCoreTests/ModelStoreTests.swift
git commit -m "feat: parameterize ModelStore directory and pin the diarization manifest"
```

---

### Task 5: ModelBundleInstaller

**Files:**
- Create: `apps/tippi/Sources/TippiCore/ModelBundleInstaller.swift`
- Test: `apps/tippi/Tests/TippiCoreTests/ModelBundleInstallerTests.swift`(Create)

**Interfaces:**
- Consumes: `ModelInstalling`(既有)
- Produces:

```swift
public struct ModelBundle: Equatable, Sendable { public let asr: URL; public let diar: URL }
public protocol ModelBundleInstalling: Sendable {
    func installIfNeeded(progress: @escaping @Sendable (Double) -> Void) async throws -> ModelBundle
}
public actor ModelBundleInstaller: ModelBundleInstalling
// init(asr: any ModelInstalling, asrTotalBytes: Int64, diar: any ModelInstalling, diarTotalBytes: Int64)
```

- [ ] **Step 1: 寫失敗測試**

`ModelBundleInstallerTests.swift`:

```swift
import Foundation
import Testing
@testable import TippiCore

private actor ScriptedInstaller: ModelInstalling {
    let url: URL
    let steps: [Double]
    let error: (any Error)?
    init(url: URL, steps: [Double], error: (any Error)? = nil) {
        self.url = url
        self.steps = steps
        self.error = error
    }
    func installIfNeeded(progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        for step in steps { progress(step) }
        if let error { throw error }
        return url
    }
}

private actor ProgressRecorder {
    private(set) var values: [Double] = []
    func append(_ value: Double) { values.append(value) }
    func snapshot() -> [Double] { values }
}

private enum TestFailure: Error { case download }

@Test
func mergesProgressWeightedByBytes() async throws {
    // ASR 600、diar 200 bytes → 權重 0.75 / 0.25。
    let installer = ModelBundleInstaller(
        asr: ScriptedInstaller(url: URL(fileURLWithPath: "/tmp/asr"), steps: [0.5, 1.0]),
        asrTotalBytes: 600,
        diar: ScriptedInstaller(url: URL(fileURLWithPath: "/tmp/diar"), steps: [0.5, 1.0]),
        diarTotalBytes: 200
    )
    let recorder = ProgressRecorder()

    let bundle = try await installer.installIfNeeded { value in
        Task { await recorder.append(value) }
    }
    try await Task.sleep(for: .milliseconds(20))

    #expect(bundle == ModelBundle(
        asr: URL(fileURLWithPath: "/tmp/asr"),
        diar: URL(fileURLWithPath: "/tmp/diar")
    ))
    let values = await recorder.snapshot()
    #expect(values == values.sorted())
    #expect(values.contains(0.375))   // ASR 一半:0.5 × 0.75
    #expect(values.contains(0.75))    // ASR 完成
    #expect(values.contains(0.875))   // diar 一半:0.75 + 0.5 × 0.25
    #expect(values.last == 1.0)
}

@Test
func alreadyInstalledAsrJumpsStraightToItsShare() async throws {
    let installer = ModelBundleInstaller(
        asr: ScriptedInstaller(url: URL(fileURLWithPath: "/tmp/asr"), steps: [1.0]),
        asrTotalBytes: 600,
        diar: ScriptedInstaller(url: URL(fileURLWithPath: "/tmp/diar"), steps: [1.0]),
        diarTotalBytes: 200
    )
    let recorder = ProgressRecorder()

    _ = try await installer.installIfNeeded { value in
        Task { await recorder.append(value) }
    }
    try await Task.sleep(for: .milliseconds(20))

    let values = await recorder.snapshot()
    #expect(values.first == 0.75)
    #expect(values.last == 1.0)
}

@Test
func diarFailurePropagates() async {
    let installer = ModelBundleInstaller(
        asr: ScriptedInstaller(url: URL(fileURLWithPath: "/tmp/asr"), steps: [1.0]),
        asrTotalBytes: 600,
        diar: ScriptedInstaller(
            url: URL(fileURLWithPath: "/tmp/diar"),
            steps: [0.5],
            error: TestFailure.download
        ),
        diarTotalBytes: 200
    )

    await #expect(throws: TestFailure.self) {
        _ = try await installer.installIfNeeded { _ in }
    }
}
```

- [ ] **Step 2: 確認失敗**

Run: `cd apps/tippi && swift test --filter TippiCoreTests`
Expected: 編譯 FAIL(型別不存在)。

- [ ] **Step 3: 實作**

`ModelBundleInstaller.swift`:

```swift
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
```

- [ ] **Step 4: 確認通過**

Run: `cd apps/tippi && swift test --filter TippiCoreTests`
Expected: 全 PASS。

- [ ] **Step 5: Commit**

```bash
git add apps/tippi/Sources/TippiCore/ModelBundleInstaller.swift apps/tippi/Tests/TippiCoreTests/ModelBundleInstallerTests.swift
git commit -m "feat: add ModelBundleInstaller with byte-weighted merged progress"
```

---

### Task 6: Transcript 型別與 TranscriptFormatter

**Files:**
- Create: `apps/tippi/Sources/TippiCore/Transcript.swift`
- Test: `apps/tippi/Tests/TippiCoreTests/TranscriptTests.swift`(Create)

**Interfaces:**
- Consumes: 無(純型別)
- Produces:

```swift
public struct SpeakerSegment: Codable, Equatable, Sendable
// speaker: Int, startMs: UInt64("start_ms"), endMs: UInt64("end_ms"), text: String, isFinal: Bool("final")
// static func decodeArray(from json: String) throws -> [SpeakerSegment]
public struct TranscriptUpdate: Equatable, Sendable  // segments: [SpeakerSegment], warning: String?
public struct Message: Identifiable, Equatable, Sendable
// id: Int, speaker: Int, startMs: UInt64, text: String, isFinal: Bool
// init(id: Int, segment: SpeakerSegment)
public enum TranscriptFormatter
// displayName(for: Int, names: [Int: String]) -> String
// timestamp(forMs: UInt64) -> String
// line(for: Message, names: [Int: String]) -> String
// transcript(messages: [Message], names: [Int: String]) -> String
```

- [ ] **Step 1: 寫失敗測試**

`TranscriptTests.swift`:

```swift
import Foundation
import Testing
@testable import TippiCore

@Test
func decodesRustSegmentJSON() throws {
    let json = """
    [{"speaker":0,"start_ms":400,"end_ms":2000,"text":"今天先討論這個。","final":true},
     {"speaker":1,"start_ms":2080,"end_ms":2400,"text":"好。","final":false}]
    """
    let segments = try SpeakerSegment.decodeArray(from: json)
    #expect(segments == [
        SpeakerSegment(speaker: 0, startMs: 400, endMs: 2000, text: "今天先討論這個。", isFinal: true),
        SpeakerSegment(speaker: 1, startMs: 2080, endMs: 2400, text: "好。", isFinal: false),
    ])
}

@Test
func decodeFailureThrows() {
    #expect(throws: (any Error).self) {
        _ = try SpeakerSegment.decodeArray(from: "not json")
    }
}

@Test
func formatsLinesWithNamesAndDefaults() {
    let named = Message(id: 0, speaker: 0, startMs: 204_000, text: "今天先討論這個。", isFinal: true)
    let unnamed = Message(id: 1, speaker: 1, startMs: 6_132_000, text: "好。", isFinal: true)
    let names = [0: "小明"]

    #expect(TranscriptFormatter.line(for: named, names: names) == "[03:24] 小明：今天先討論這個。")
    #expect(TranscriptFormatter.line(for: unnamed, names: names) == "[102:12] 說話者 2：好。")
    #expect(TranscriptFormatter.transcript(messages: [named, unnamed], names: names)
        == "[03:24] 小明：今天先討論這個。\n[102:12] 說話者 2：好。")
}

@Test
func messageIsBuiltFromSegment() {
    let segment = SpeakerSegment(speaker: 2, startMs: 80, endMs: 160, text: "喂?", isFinal: false)
    let message = Message(id: 5, segment: segment)
    #expect(message == Message(id: 5, speaker: 2, startMs: 80, text: "喂?", isFinal: false))
}
```

- [ ] **Step 2: 確認失敗**

Run: `cd apps/tippi && swift test --filter TippiCoreTests`
Expected: 編譯 FAIL。

- [ ] **Step 3: 實作**

`Transcript.swift`:

```swift
import Foundation

/// One speaker-attributed run of transcript text, decoded from the
/// `catcher_segments` JSON produced by the Rust fusion module. Field names
/// mirror the Rust `SpeakerSegment` serde output exactly.
public struct SpeakerSegment: Codable, Equatable, Sendable {
    public let speaker: Int
    public let startMs: UInt64
    public let endMs: UInt64
    public let text: String
    public let isFinal: Bool

    enum CodingKeys: String, CodingKey {
        case speaker
        case startMs = "start_ms"
        case endMs = "end_ms"
        case text
        case isFinal = "final"
    }

    public init(speaker: Int, startMs: UInt64, endMs: UInt64, text: String, isFinal: Bool) {
        self.speaker = speaker
        self.startMs = startMs
        self.endMs = endMs
        self.text = text
        self.isFinal = isFinal
    }

    public static func decodeArray(from json: String) throws -> [SpeakerSegment] {
        try JSONDecoder().decode([SpeakerSegment].self, from: Data(json.utf8))
    }
}

/// What one successful push/finish call reports back to the UI layer.
public struct TranscriptUpdate: Equatable, Sendable {
    public let segments: [SpeakerSegment]
    public let warning: String?

    public init(segments: [SpeakerSegment], warning: String?) {
        self.segments = segments
        self.warning = warning
    }
}

/// One row in Tippi's message list. `id` is the row's index; the whole list
/// is rebuilt from segments on every update.
public struct Message: Identifiable, Equatable, Sendable {
    public let id: Int
    public let speaker: Int
    public let startMs: UInt64
    public let text: String
    public let isFinal: Bool

    public init(id: Int, speaker: Int, startMs: UInt64, text: String, isFinal: Bool) {
        self.id = id
        self.speaker = speaker
        self.startMs = startMs
        self.text = text
        self.isFinal = isFinal
    }

    public init(id: Int, segment: SpeakerSegment) {
        self.init(
            id: id,
            speaker: segment.speaker,
            startMs: segment.startMs,
            text: segment.text,
            isFinal: segment.isFinal
        )
    }
}

/// Shared line formatting for on-screen copy actions and file export, so the
/// two never drift apart. Line shape: `[mm:ss] 顯示名：內文` (fullwidth
/// colon; minutes grow past two digits naturally).
public enum TranscriptFormatter {
    public static func displayName(for speaker: Int, names: [Int: String]) -> String {
        names[speaker] ?? "說話者 \(speaker + 1)"
    }

    public static func timestamp(forMs milliseconds: UInt64) -> String {
        let totalSeconds = milliseconds / 1000
        return String(format: "[%02d:%02d]", totalSeconds / 60, totalSeconds % 60)
    }

    public static func line(for message: Message, names: [Int: String]) -> String {
        "\(timestamp(forMs: message.startMs)) \(displayName(for: message.speaker, names: names))：\(message.text)"
    }

    public static func transcript(messages: [Message], names: [Int: String]) -> String {
        messages.map { line(for: $0, names: names) }.joined(separator: "\n")
    }
}
```

- [ ] **Step 4: 確認通過**

Run: `cd apps/tippi && swift test --filter TippiCoreTests`
Expected: 全 PASS。

- [ ] **Step 5: Commit**

```bash
git add apps/tippi/Sources/TippiCore/Transcript.swift apps/tippi/Tests/TippiCoreTests/TranscriptTests.swift
git commit -m "feat: add transcript domain types and shared line formatter"
```

---

### Task 7: CatcherServing v2 + Controller messages + 佈線

**Files:**
- Modify: `apps/tippi/Sources/TippiCore/CatcherClient.swift`
- Modify: `apps/tippi/Sources/TippiCore/TranscriptionController.swift`
- Modify: `apps/tippi/Sources/TippiApp/TippiApp.swift`
- Modify: `apps/tippi/Sources/TippiApp/ContentView.swift`(最小編譯修正,訊息 UI 留給 Task 8)
- Test: `apps/tippi/Tests/TippiCoreTests/TranscriptionControllerTests.swift`

**Interfaces:**
- Consumes: Task 5 `ModelBundle`/`ModelBundleInstalling`、Task 6 `SpeakerSegment`/`TranscriptUpdate`/`Message`/`TranscriptFormatter`
- Produces:

```swift
public protocol CatcherServing: Sendable {
    func start() async throws
    func push(_ samples: [Float]) async throws -> TranscriptUpdate?  // CATCHER_NO_UPDATE → nil
    func finish() async throws -> TranscriptUpdate
}
public typealias CatcherFactory = @Sendable (ModelBundle) async throws -> any CatcherServing
// CatcherClient.init(modelDirectory: URL, diarModelDirectory: URL, language: String = "auto", lookahead: UInt32 = 3) throws
// TranscriptionController: messages: [Message], speakerNames: [Int: String](可寫), warningMessage: String?
// TranscriptionController.init(modelInstaller: any ModelBundleInstalling, audio:, catcherFactory:)
```

- [ ] **Step 1: 改寫 controller 測試(失敗先行)**

`TranscriptionControllerTests.swift` 全檔改寫:

```swift
import Foundation
import Testing
@testable import TippiCore

private actor FakeInstaller: ModelBundleInstalling {
    let bundle: ModelBundle
    init(bundle: ModelBundle) { self.bundle = bundle }
    func installIfNeeded(
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> ModelBundle {
        progress(0.5)
        progress(1.0)
        return bundle
    }
}

private actor FakeCatcher: CatcherServing {
    private(set) var events: [String] = []
    private var pushUpdates: [TranscriptUpdate?] = []
    private var finishUpdate = TranscriptUpdate(segments: [], warning: nil)

    func start() async throws { events.append("start") }

    func push(_ samples: [Float]) async throws -> TranscriptUpdate? {
        events.append("push:\(samples.count)")
        return pushUpdates.isEmpty ? nil : pushUpdates.removeFirst()
    }

    func finish() async throws -> TranscriptUpdate {
        events.append("finish")
        return finishUpdate
    }

    func script(pushes: [TranscriptUpdate?], finish: TranscriptUpdate) {
        pushUpdates = pushes
        finishUpdate = finish
    }

    func snapshot() -> [String] { events }
}

private actor FakeAudio: AudioRecording {
    private var sink: (@Sendable ([Float]) -> Void)?
    private(set) var events: [String] = []
    var startError: (any Error)?

    func start(onSamples: @escaping @Sendable ([Float]) -> Void) async throws {
        if let startError { throw startError }
        sink = onSamples
        events.append("start")
    }

    func stop() async {
        events.append("stop")
        sink = nil
    }

    func emit(_ samples: [Float]) { sink?(samples) }
    func snapshot() -> [String] { events }
    func setStartError(_ error: any Error) { startError = error }
}

private enum TestFailure: Error { case microphone }

private let testBundle = ModelBundle(
    asr: URL(fileURLWithPath: "/tmp/asr"),
    diar: URL(fileURLWithPath: "/tmp/diar")
)

private func segment(
    _ speaker: Int, _ startMs: UInt64, _ text: String, final isFinal: Bool
) -> SpeakerSegment {
    SpeakerSegment(speaker: speaker, startMs: startMs, endMs: startMs + 80, text: text, isFinal: isFinal)
}

@MainActor
@Test
func recordingPublishesMessagesThenFinalizesOnStop() async throws {
    let catcher = FakeCatcher()
    await catcher.script(
        pushes: [TranscriptUpdate(
            segments: [segment(0, 400, "今天先討論這個。", final: false)],
            warning: nil
        )],
        finish: TranscriptUpdate(
            segments: [
                segment(0, 400, "今天先討論這個。", final: true),
                segment(1, 2080, "好。", final: true),
            ],
            warning: nil
        )
    )
    let audio = FakeAudio()
    let controller = TranscriptionController(
        modelInstaller: FakeInstaller(bundle: testBundle),
        audio: audio,
        catcherFactory: { _ in catcher }
    )
    await controller.prepare()
    #expect(controller.state == .ready)

    await controller.toggleRecording()
    #expect(controller.state == .recording)
    await audio.emit([0.1, 0.2, 0.3])
    try await Task.sleep(for: .milliseconds(20))
    #expect(controller.messages == [
        Message(id: 0, speaker: 0, startMs: 400, text: "今天先討論這個。", isFinal: false)
    ])

    await controller.toggleRecording()
    #expect(controller.state == .ready)
    #expect(controller.messages.count == 2)
    #expect(controller.messages.allSatisfy(\.isFinal))
    #expect(controller.messages[1].speaker == 1)
    #expect(await catcher.snapshot() == ["start", "push:3", "finish"])
}

@MainActor
@Test
func warningIsPublishedAndClearedOnNextRecording() async throws {
    let catcher = FakeCatcher()
    await catcher.script(
        pushes: [TranscriptUpdate(
            segments: [segment(0, 0, "喂?", final: false)],
            warning: "diarization disabled after a runtime error: injected"
        )],
        finish: TranscriptUpdate(segments: [segment(0, 0, "喂?", final: true)], warning: nil)
    )
    let audio = FakeAudio()
    let controller = TranscriptionController(
        modelInstaller: FakeInstaller(bundle: testBundle),
        audio: audio,
        catcherFactory: { _ in catcher }
    )
    await controller.prepare()

    await controller.toggleRecording()
    await audio.emit([0.1])
    try await Task.sleep(for: .milliseconds(20))
    #expect(controller.warningMessage == "diarization disabled after a runtime error: injected")
    await controller.toggleRecording()
    #expect(controller.warningMessage == nil)

    // 新錄音清空訊息、命名與警告。
    controller.speakerNames[0] = "小明"
    await controller.toggleRecording()
    #expect(controller.messages.isEmpty)
    #expect(controller.speakerNames.isEmpty)
    #expect(controller.warningMessage == nil)
    await controller.toggleRecording()
}

@MainActor
@Test
func microphoneFailureLeavesRecordingOffAndCanBeRetried() async {
    let catcher = FakeCatcher()
    let audio = FakeAudio()
    await audio.setStartError(TestFailure.microphone)
    let controller = TranscriptionController(
        modelInstaller: FakeInstaller(bundle: testBundle),
        audio: audio,
        catcherFactory: { _ in catcher }
    )
    await controller.prepare()

    await controller.toggleRecording()

    guard case .failed = controller.state else {
        Issue.record("expected failed state")
        return
    }
    #expect(!controller.isRecording)
}
```

(注意第二個測試:finish 的 warning 為 nil → `warningMessage` 直接吸收為 nil,驗證「warning 每次更新都整體覆寫」的語義。)

- [ ] **Step 2: 確認失敗**

Run: `cd apps/tippi && swift test --filter TippiCoreTests`
Expected: 編譯 FAIL。

- [ ] **Step 3: 實作 CatcherClient v2**

`CatcherClient.swift` 改寫(保留 `CatcherHandleOwner`/`CatcherClientError`/`check`/`currentError`/`globalError` 原樣):

```swift
public protocol CatcherServing: Sendable {
    func start() async throws
    func push(_ samples: [Float]) async throws -> TranscriptUpdate?
    func finish() async throws -> TranscriptUpdate
}
```

`CatcherClient`:

```swift
public actor CatcherClient: CatcherServing {
    private let owner: CatcherHandleOwner

    public init(
        modelDirectory: URL,
        diarModelDirectory: URL,
        language: String = "auto",
        lookahead: UInt32 = 3
    ) throws {
        let pointer = modelDirectory.path.withCString { modelPath in
            diarModelDirectory.path.withCString { diarPath in
                language.withCString { languageCode in
                    catcher_create(modelPath, diarPath, languageCode, lookahead)
                }
            }
        }
        guard let pointer else {
            throw CatcherClientError.creationFailed(Self.globalError())
        }
        owner = CatcherHandleOwner(pointer: pointer)
    }

    public func start() async throws {
        try check(catcher_start(owner.pointer), allowNoUpdate: false)
    }

    public func push(_ samples: [Float]) async throws -> TranscriptUpdate? {
        let status = samples.withUnsafeBufferPointer { buffer in
            catcher_push_audio(owner.pointer, buffer.baseAddress, buffer.count)
        }
        if status == CATCHER_NO_UPDATE { return nil }
        try check(status, allowNoUpdate: false)
        return try currentUpdate()
    }

    public func finish() async throws -> TranscriptUpdate {
        try check(catcher_finish(owner.pointer), allowNoUpdate: true)
        return try currentUpdate()
    }

    private func currentUpdate() throws -> TranscriptUpdate {
        let json = catcher_segments(owner.pointer).map { String(cString: $0) } ?? "[]"
        let segments: [SpeakerSegment]
        do {
            segments = try SpeakerSegment.decodeArray(from: json)
        } catch {
            throw CatcherClientError.operationFailed("segments JSON decode failed: \(error)")
        }
        let warning = catcher_warning(owner.pointer).map { String(cString: $0) }
        return TranscriptUpdate(segments: segments, warning: warning)
    }
    // check / currentError / globalError 原樣保留;currentText 刪除。
}
```

- [ ] **Step 4: 實作 TranscriptionController**

`TranscriptionController.swift`:

- `public typealias CatcherFactory = @Sendable (ModelBundle) async throws -> any CatcherServing`
- 屬性:`text` 刪除,改 `public private(set) var messages: [Message] = []`、`public var speakerNames: [Int: String] = [:]`、`public private(set) var warningMessage: String?`。
- `modelInstaller` 型別改 `any ModelBundleInstalling`(init 參數同步)。
- `prepare()`:`installIfNeeded` 回 `ModelBundle`,`catcher = try await catcherFactory(bundle)`。
- `startRecording()`:`text = ""` 換成:

```swift
            messages = []
            speakerNames = [:]
            warningMessage = nil
```

- push 迴圈:

```swift
                        if let update = try await catcher.push(samples) {
                            self?.apply(update)
                        }
```

- `stopRecording()`:`text = try await catcher.finish()` 換成 `apply(try await catcher.finish())`。
- 新私有方法:

```swift
    private func apply(_ update: TranscriptUpdate) {
        messages = update.segments.enumerated().map { index, segment in
            Message(id: index, segment: segment)
        }
        warningMessage = update.warning
    }
```

- [ ] **Step 5: 佈線 TippiApp 與 ContentView 最小修正**

`TippiApp.swift` `init()`:

```swift
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
                }
            )
        )
```

`ContentView.swift` 最小編譯修正(訊息 UI 是 Task 8):transcript 區塊的 `controller.text` 兩處改為

```swift
TranscriptFormatter.transcript(messages: controller.messages, names: controller.speakerNames)
```

(以 local `let transcriptText` 存起來判空與顯示。)

- [ ] **Step 6: 確認通過**

Run: `cd apps/tippi && swift build && swift test --filter TippiCoreTests`
Expected: build 成功、測試全 PASS。

- [ ] **Step 7: Commit**

```bash
git add apps/tippi/Sources apps/tippi/Tests
git commit -m "feat: drive Tippi from speaker segments via CatcherServing v2"
```

---

### Task 8: ContentView 訊息列表、改名、警告橫幅

**Files:**
- Create: `apps/tippi/Sources/TippiApp/MessageRow.swift`
- Modify: `apps/tippi/Sources/TippiApp/ContentView.swift`

**Interfaces:**
- Consumes: `controller.messages`/`speakerNames`/`warningMessage`、`TranscriptFormatter.displayName/timestamp`
- Produces: 純 UI;`MessageRow(message:name:accent:onRename:)` view。

- [ ] **Step 1: MessageRow**

`MessageRow.swift`:

```swift
import SwiftUI
import TippiCore

struct MessageRow: View {
    let message: Message
    let name: String
    let accent: Color
    let onRename: (String) -> Void

    @State private var isRenaming = false
    @State private var draftName = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Button {
                    draftName = name
                    isRenaming = true
                } label: {
                    Text(name)
                        .font(.callout.weight(.semibold))
                        .foregroundStyle(accent)
                }
                .buttonStyle(.plain)
                .popover(isPresented: $isRenaming, arrowEdge: .bottom) {
                    TextField("說話者名稱", text: $draftName)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 180)
                        .padding(12)
                        .onSubmit {
                            onRename(draftName.trimmingCharacters(in: .whitespacesAndNewlines))
                            isRenaming = false
                        }
                }
                Text(TranscriptFormatter.timestamp(forMs: message.startMs))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Text(message.text)
                .font(.system(size: 19, weight: .regular, design: .rounded))
                .foregroundStyle(message.isFinal ? AnyShapeStyle(.primary) : AnyShapeStyle(.secondary))
                .textSelection(.enabled)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
```

- [ ] **Step 2: ContentView 訊息列表 + 橫幅**

`ContentView.swift` 的 `transcript` 改為:

```swift
    private static let accents: [Color] = [.blue, .green, .orange, .purple]

    private var transcript: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("TRANSCRIPT")
                .font(.caption.weight(.semibold))
                .tracking(1.2)
                .foregroundStyle(.secondary)
            if controller.warningMessage != nil {
                Label("說話者分離已暫停,文字繼續轉寫", systemImage: "person.2.slash")
                    .font(.callout)
                    .foregroundStyle(.orange)
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(.orange.opacity(0.12), in: RoundedRectangle(cornerRadius: 10))
            }
            if controller.messages.isEmpty {
                Text(placeholder)
                    .font(.system(size: 23, weight: .regular, design: .rounded))
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            } else {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 16) {
                            ForEach(controller.messages) { message in
                                MessageRow(
                                    message: message,
                                    name: TranscriptFormatter.displayName(
                                        for: message.speaker,
                                        names: controller.speakerNames
                                    ),
                                    accent: Self.accents[message.speaker % Self.accents.count],
                                    onRename: { newName in
                                        rename(speaker: message.speaker, to: newName)
                                    }
                                )
                                .id(message.id)
                            }
                        }
                        .padding(.vertical, 4)
                    }
                    .onChange(of: controller.messages.last?.text) {
                        if let lastID = controller.messages.last?.id {
                            proxy.scrollTo(lastID, anchor: .bottom)
                        }
                    }
                }
            }
        }
        .padding(22)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(.background.opacity(0.72), in: RoundedRectangle(cornerRadius: 20))
        .overlay(RoundedRectangle(cornerRadius: 20).stroke(.separator.opacity(0.45)))
    }

    private func rename(speaker: Int, to newName: String) {
        if newName.isEmpty {
            controller.speakerNames.removeValue(forKey: speaker)
        } else {
            controller.speakerNames[speaker] = newName
        }
    }
```

(Task 7 加的 `TranscriptFormatter.transcript` 暫用文字顯示移除。改名寫入 `speakerNames` 後,`displayName` 是每次 render 重查,整個列表即時換名——不需額外重繪邏輯。)

- [ ] **Step 3: Build 驗證**

Run: `cd apps/tippi && swift build && swift test`
Expected: build 成功、全部測試 PASS。

- [ ] **Step 4: Commit**

```bash
git add apps/tippi/Sources/TippiApp
git commit -m "feat: render speaker message list with rename popover and warning banner"
```

---

### Task 9: 複製全部、匯出、複製此則

**Files:**
- Modify: `apps/tippi/Sources/TippiApp/ContentView.swift`
- Modify: `apps/tippi/Sources/TippiApp/MessageRow.swift`(context menu)
- Modify: `apps/tippi/Resources/Tippi.entitlements`

**Interfaces:**
- Consumes: `TranscriptFormatter.transcript/line`(Task 6)
- Produces: 純 UI 動作;entitlements 增 `com.apple.security.files.user-selected.read-write`。

- [ ] **Step 1: entitlements**

`Tippi.entitlements` 的 `<dict>` 內加:

```xml
    <key>com.apple.security.files.user-selected.read-write</key>
    <true/>
```

- [ ] **Step 2: ContentView 動作**

`footer` 的 default case HStack,在錄音按鈕前加:

```swift
                Button("複製全部") { copyAll() }
                    .disabled(controller.messages.isEmpty)
                Button("匯出…") { exportTranscript() }
                    .disabled(controller.messages.isEmpty)
```

`ContentView` 加方法:

```swift
    private var fullTranscript: String {
        TranscriptFormatter.transcript(
            messages: controller.messages,
            names: controller.speakerNames
        )
    }

    private func copyAll() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(fullTranscript, forType: .string)
    }

    private func exportTranscript() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.plainText]
        panel.nameFieldStringValue = "Tippi 逐字稿.txt"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try Data(fullTranscript.utf8).write(to: url)
        } catch {
            NSSound.beep()
        }
    }
```

(頂部 `import UniformTypeIdentifiers`。)

- [ ] **Step 3: MessageRow context menu**

`MessageRow` 增加屬性 `let lineText: String`,`body` 外層 `VStack` 加:

```swift
        .contextMenu {
            Button("複製此則") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(lineText, forType: .string)
            }
        }
```

`ContentView` 建 `MessageRow` 處補 `lineText: TranscriptFormatter.line(for: message, names: controller.speakerNames),`。

- [ ] **Step 4: Build 驗證**

Run: `cd apps/tippi && swift build && swift test`
Expected: 成功、全 PASS。

- [ ] **Step 5: Commit**

```bash
git add apps/tippi/Sources/TippiApp apps/tippi/Resources/Tippi.entitlements
git commit -m "feat: add copy-all, per-message copy, and .txt export"
```

---

### Task 10: README 與全面驗證

**Files:**
- Modify: `README.md`(Tippi 章節)

**Interfaces:**
- Consumes: 全部前置任務
- Produces: 文件與綠色全套測試

- [ ] **Step 1: README 更新**

Tippi 章節補:首次啟動下載兩個模型(ASR `wcamon/catcher-asr-mlx-int8` ≈628MB + diarization `wcamon/catcher-diar-mlx-int8` ≈121MB,合併進度條);錄音顯示分說話者訊息列表(未命名顯示 說話者 N,點名字改名);「複製全部」「匯出…」與每則「複製此則」,行格式 `[03:24] 小明：今天先討論這個。`;diarization 執行期錯誤顯示非阻斷橫幅、下次錄音自動恢復。C ABI 章節若提及 `catcher_start`,補一句重建語義。

- [ ] **Step 2: Rust 全套驗證**

Run: `cargo fmt --check && cargo test --workspace`
Expected: 全 PASS。

- [ ] **Step 3: Rust gated 驗證**

Run: `NEMOTRON_MLX_ARTIFACT=/tmp/catcher-asr-mlx-int8 SORTFORMER_MLX_ARTIFACT=/tmp/catcher-diar-mlx-int8 cargo test -p catcher-ffi -p nemotron-mlx -p sortformer-mlx -- --ignored`(逐 package 執行,不可合併 `--test` 旗標)
Expected: 全 PASS。

- [ ] **Step 4: Swift 全套驗證**

Run: `cd apps/tippi && swift build && swift test`
Expected: 全 PASS。

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: document Tippi dual-model download and speaker message UI"
```

---

## 驗收(合併前,人工)

1. 刪除 `~/Library/Application Support/Tippi/Models`,啟動 app:單一進度條跑完雙模型下載。
2. 錄一段兩人對話:訊息按說話者分列、最後一則即時長大、說話者變更開新則。
3. 點說話者名改名:全列表即時換名;停止後再錄,命名與訊息重設。
4. 「複製全部」與「匯出…」產出相同的 `[mm:ss] 名字:文字` 逐行內容;匯出檔為 UTF-8 .txt。
