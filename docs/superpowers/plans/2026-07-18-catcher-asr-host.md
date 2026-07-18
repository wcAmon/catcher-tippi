# catcher-asr-host(mac Engine Host)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 catcher 的 MLX ASR runtime 抽成獨立 console engine host,講 tomato-ears 的 stdin/stdout JSON-lines 協定 v1,並產出可發布的 tar.gz + SHA-256。

**Architecture:** 新 workspace crate `crates/catcher-asr-host`。協定層(serde 型別)、會話狀態機(session)、引擎抽象(trait + FakeEngine 供無 Metal/無模型測試 + MlxEngine 實體)三層分離;stdio 迴圈只做編解碼與轉發。協定合約文件是本計畫與 Plan 2(win host)、Plan 3(Deno 配方)的共同介面。

**Tech Stack:** Rust 1.85 / edition 2024(workspace 繼承)、serde + serde_json、base64、clap、nemotron-mlx(path 依賴)。

**相關 spec:** `docs/superpowers/specs/2026-07-18-tomato-ears-design.md`、`2026-07-18-mini-app-store-design.md`

## Global Constraints

- 平台:arm64 Apple Silicon macOS 15+;Rust 1.85+(workspace `rust-version = "1.85"`、`edition = "2024"`)
- 音訊:mono 16 kHz PCM16(little-endian),每 chunk ≈100 ms(1600 samples)
- 協定:asr-host-v1(見 Task 1,訊息格式逐字不可改)
- 輸出文字:繁體中文經 `nemotron_mlx::opencc::to_traditional`(s2twp)
- 模型:`wcamon/catcher-asr-mlx-int8`;tokenizer 參數 `Tokenizer::from_json(dir.join("tokenizer.json"), 0, 13_087)`
- language 預設 `auto`;lookahead 預設 `3`(合法值 0/3/6/13)
- 所有 cargo 指令在 repo 根目錄(`/Users/wake/Desktop/catcher-tippi`)執行

---

### Task 1: 協定合約文件 asr-host-v1

**Files:**
- Create: `docs/protocol/asr-host-v1.md`

**Interfaces:**
- Produces: 協定 v1 語義,Plan 2(nemotron-asr-host)與 Plan 3(Deno `engine.ts`)逐字依賴本文件。

- [ ] **Step 1: 寫協定文件**

````markdown
# asr-host protocol v1

傳輸:stdin/stdout,JSON-lines(每行一則 JSON,UTF-8,`\n` 結尾)。
stderr 僅供人類閱讀的日誌,消費端必須忽略。

## Host 生命週期
1. 行程啟動 → 載入模型 → 輸出 `ready`(之前不得輸出任何 stdout 行)。
2. 之後接受指令。stdin EOF = 結束行程(exit code 0)。
3. 模型載入失敗:輸出 `error` 後以 exit code 1 結束。

## 指令(stdin →)
| 指令 | 格式 | 語義 |
|---|---|---|
| start | `{"cmd":"start","lang":"auto","sample_rate":16000}` | 開新會話。`lang`:`auto`/`en-US`/`zh-CN`/`zh-TW` 等 checkpoint locale。`sample_rate` 僅接受 16000。會話進行中再收 start → `error`(會話不中斷)。 |
| audio | `{"cmd":"audio","pcm16_b64":"<base64>"}` | mono 16 kHz PCM16-LE。建議每 chunk 1600 samples(100 ms)。無會話時收到 → `error`。 |
| stop | `{"cmd":"stop"}` | 沖洗解碼器,輸出 `final`,會話結束。無會話時收到 → `error`。 |

## 事件(stdout ←)
| 事件 | 格式 | 語義 |
|---|---|---|
| ready | `{"event":"ready","backend":"mlx"}` | 模型載入完成(win host 為 `"dml"` 或 `"cpu"`)。 |
| partial | `{"event":"partial","text":"..."}` | 會話累積轉錄的最新全文(非增量)。僅在有新 token 時輸出。 |
| final | `{"event":"final","text":"..."}` | stop 之後的定稿全文,一個會話恰好一次。 |
| error | `{"event":"error","message":"..."}` | 可恢復錯誤(格式錯、狀態錯)。行程不退出;致命錯誤才退出(exit 1)。 |

## 錯誤處理原則
- 無法解析的行 → `error`,繼續讀下一行。
- `pcm16_b64` 非法 base64 或位元組數為奇數 → `error`,會話保留。
- 文字一律 host 端已轉繁體(opencc s2twp);消費端不再轉換。
````

- [ ] **Step 2: Commit**

```bash
git add docs/protocol/asr-host-v1.md
git commit -m "docs: define asr-host stdin/stdout protocol v1"
```

---

### Task 2: crate 骨架 + 協定型別

**Files:**
- Create: `crates/catcher-asr-host/Cargo.toml`
- Create: `crates/catcher-asr-host/src/main.rs`(暫時只掛模組)
- Create: `crates/catcher-asr-host/src/protocol.rs`
- Modify: `Cargo.toml:2`(workspace members 加入 `"crates/catcher-asr-host"`)

**Interfaces:**
- Produces: `protocol::Command`(enum:`Start { lang: String, sample_rate: u32 }` / `Audio { pcm16_b64: String }` / `Stop`)、`protocol::Event`(enum:`Ready { backend: &'static str }` / `Partial { text: String }` / `Final { text: String }` / `Error { message: String }`)、`protocol::parse_command(line: &str) -> Result<Command, String>`、`protocol::emit(event: &Event) -> String`(單行 JSON,不含換行)。

- [ ] **Step 1: 建 crate 與依賴**

`crates/catcher-asr-host/Cargo.toml`:

```toml
[package]
name = "catcher-asr-host"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true

[dependencies]
nemotron-mlx = { path = "../nemotron-mlx" }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
base64 = "0.22"
clap = { version = "4", features = ["derive"] }
```

root `Cargo.toml` members 加 `"crates/catcher-asr-host"`。`src/main.rs` 先放:

```rust
mod protocol;

fn main() {}
```

- [ ] **Step 2: 寫 failing 測試(protocol.rs 內嵌 #[cfg(test)])**

```rust
// crates/catcher-asr-host/src/protocol.rs 尾端
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_start_command() {
        let cmd = parse_command(r#"{"cmd":"start","lang":"auto","sample_rate":16000}"#).unwrap();
        assert!(matches!(cmd, Command::Start { ref lang, sample_rate: 16000 } if lang == "auto"));
    }

    #[test]
    fn parses_audio_and_stop() {
        assert!(matches!(
            parse_command(r#"{"cmd":"audio","pcm16_b64":"AAA="}"#).unwrap(),
            Command::Audio { .. }
        ));
        assert!(matches!(parse_command(r#"{"cmd":"stop"}"#).unwrap(), Command::Stop));
    }

    #[test]
    fn rejects_unknown_and_malformed() {
        assert!(parse_command(r#"{"cmd":"dance"}"#).is_err());
        assert!(parse_command("not json").is_err());
    }

    #[test]
    fn emits_events_as_single_json_lines() {
        assert_eq!(emit(&Event::Ready { backend: "mlx" }), r#"{"event":"ready","backend":"mlx"}"#);
        assert_eq!(
            emit(&Event::Partial { text: "你好".into() }),
            r#"{"event":"partial","text":"你好"}"#
        );
        assert_eq!(
            emit(&Event::Error { message: "x".into() }),
            r#"{"event":"error","message":"x"}"#
        );
    }
}
```

- [ ] **Step 3: 跑測試確認失敗**

Run: `cargo test -p catcher-asr-host`
Expected: 編譯錯誤(`Command`/`Event`/`parse_command`/`emit` 未定義)——TDD 的紅燈。

- [ ] **Step 4: 實作 protocol.rs**

```rust
//! asr-host protocol v1 的訊息型別與編解碼。
//! 格式定義以 docs/protocol/asr-host-v1.md 為準,逐字不可改。

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
#[serde(tag = "cmd", rename_all = "lowercase", deny_unknown_fields)]
pub enum Command {
    Start { lang: String, sample_rate: u32 },
    Audio { pcm16_b64: String },
    Stop,
}

#[derive(Debug, Serialize)]
#[serde(tag = "event", rename_all = "lowercase")]
pub enum Event {
    Ready { backend: &'static str },
    Partial { text: String },
    Final { text: String },
    Error { message: String },
}

pub fn parse_command(line: &str) -> Result<Command, String> {
    serde_json::from_str(line).map_err(|error| format!("無法解析指令:{error}"))
}

pub fn emit(event: &Event) -> String {
    serde_json::to_string(event).expect("protocol events serialize infallibly")
}
```

- [ ] **Step 5: 跑測試確認通過**

Run: `cargo test -p catcher-asr-host`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml crates/catcher-asr-host
git commit -m "feat: catcher-asr-host crate with protocol v1 types"
```

---

### Task 3: 引擎抽象 + FakeEngine + 會話狀態機

**Files:**
- Create: `crates/catcher-asr-host/src/engine.rs`
- Create: `crates/catcher-asr-host/src/session.rs`
- Modify: `crates/catcher-asr-host/src/main.rs`(掛 `mod engine; mod session;`)

**Interfaces:**
- Consumes: `protocol::Event`
- Produces:
  - `engine::AsrEngine` trait:`fn push(&mut self, samples: &[f32]) -> Result<Vec<u32>, String>` / `fn finish(&mut self) -> Result<Vec<u32>, String>` / `fn decode(&self, ids: &[u32]) -> Result<String, String>` / `fn backend(&self) -> &'static str`
  - `engine::FakeEngine::new()`(每 1600 samples 產一個遞增 token id;decode 把 id 序列組成 `"字0字1…"`;backend 回 `"fake"`)
  - `session::Session::new(engine: Box<dyn AsrEngine>)`、`fn handle(&mut self, cmd: protocol::Command) -> Vec<protocol::Event>`

- [ ] **Step 1: 寫 failing 測試(session.rs 內嵌)**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::FakeEngine;
    use crate::protocol::{Command, Event};
    use base64::Engine as _;

    fn b64_pcm(samples: usize) -> String {
        base64::engine::general_purpose::STANDARD.encode(vec![0u8; samples * 2])
    }

    fn start() -> Command {
        Command::Start { lang: "auto".into(), sample_rate: 16000 }
    }

    #[test]
    fn happy_path_start_audio_stop() {
        let mut session = Session::new(Box::new(FakeEngine::new()));
        assert!(session.handle(start()).is_empty());
        let events = session.handle(Command::Audio { pcm16_b64: b64_pcm(1600) });
        assert!(matches!(&events[..], [Event::Partial { text }] if text == "字0"));
        let events = session.handle(Command::Stop);
        assert!(matches!(&events[..], [Event::Final { text }] if text == "字0"));
    }

    #[test]
    fn partial_only_when_new_tokens() {
        let mut session = Session::new(Box::new(FakeEngine::new()));
        session.handle(start());
        // 不足 1600 samples → FakeEngine 不吐 token → 不得有 partial
        let events = session.handle(Command::Audio { pcm16_b64: b64_pcm(100) });
        assert!(events.is_empty());
    }

    #[test]
    fn state_errors_do_not_kill_session() {
        let mut session = Session::new(Box::new(FakeEngine::new()));
        assert!(matches!(&session.handle(Command::Stop)[..], [Event::Error { .. }]));
        assert!(matches!(
            &session.handle(Command::Audio { pcm16_b64: b64_pcm(1600) })[..],
            [Event::Error { .. }]
        ));
        session.handle(start());
        assert!(matches!(&session.handle(start())[..], [Event::Error { .. }]));
        // 原會話仍活著
        let events = session.handle(Command::Audio { pcm16_b64: b64_pcm(1600) });
        assert!(matches!(&events[..], [Event::Partial { .. }]));
    }

    #[test]
    fn rejects_bad_audio_payload() {
        let mut session = Session::new(Box::new(FakeEngine::new()));
        session.handle(start());
        // 非法 base64
        assert!(matches!(
            &session.handle(Command::Audio { pcm16_b64: "!!!".into() })[..],
            [Event::Error { .. }]
        ));
        // 奇數位元組
        let odd = base64::engine::general_purpose::STANDARD.encode([0u8; 3]);
        assert!(matches!(
            &session.handle(Command::Audio { pcm16_b64: odd })[..],
            [Event::Error { .. }]
        ));
    }

    #[test]
    fn rejects_wrong_sample_rate() {
        let mut session = Session::new(Box::new(FakeEngine::new()));
        let events = session.handle(Command::Start { lang: "auto".into(), sample_rate: 44100 });
        assert!(matches!(&events[..], [Event::Error { .. }]));
    }
}
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `cargo test -p catcher-asr-host`
Expected: 編譯錯誤(engine/session 未定義)

- [ ] **Step 3: 實作 engine.rs**

```rust
//! 推論引擎抽象。Session 只認識這個 trait,
//! 讓狀態機能在沒有 Metal 與模型檔的環境(CI、驗收測試)用 FakeEngine 驗證。

pub trait AsrEngine {
    /// 餵入 16 kHz mono f32 samples,回傳新產生的 token ids。
    fn push(&mut self, samples: &[f32]) -> Result<Vec<u32>, String>;
    /// 沖洗解碼器,回傳最後一批 token ids。
    fn finish(&mut self) -> Result<Vec<u32>, String>;
    /// 把「會話累積的全部 ids」解成文字(host 端已含繁化)。
    fn decode(&self, ids: &[u32]) -> Result<String, String>;
    fn backend(&self) -> &'static str;
}

/// 決定性假引擎:每滿 1600 samples 產出一個遞增 id;
/// decode 把每個 id 映成 "字N",讓測試能精確斷言 partial/final 內容。
pub struct FakeEngine {
    buffered: usize,
    next_id: u32,
}

impl FakeEngine {
    pub fn new() -> Self {
        Self { buffered: 0, next_id: 0 }
    }
}

impl AsrEngine for FakeEngine {
    fn push(&mut self, samples: &[f32]) -> Result<Vec<u32>, String> {
        self.buffered += samples.len();
        let mut ids = Vec::new();
        while self.buffered >= 1600 {
            self.buffered -= 1600;
            ids.push(self.next_id);
            self.next_id += 1;
        }
        Ok(ids)
    }

    fn finish(&mut self) -> Result<Vec<u32>, String> {
        self.buffered = 0;
        Ok(Vec::new())
    }

    fn decode(&self, ids: &[u32]) -> Result<String, String> {
        Ok(ids.iter().map(|id| format!("字{id}")).collect())
    }

    fn backend(&self) -> &'static str {
        "fake"
    }
}
```

- [ ] **Step 4: 實作 session.rs**

```rust
//! 會話狀態機:協定指令 → 引擎呼叫 → 協定事件。
//! 不做 I/O;stdio 迴圈(main.rs)負責讀寫,這裡保持純邏輯以便單元測試。

use base64::Engine as _;

use crate::engine::AsrEngine;
use crate::protocol::{Command, Event};

pub struct Session {
    engine: Box<dyn AsrEngine>,
    /// None = 無進行中會話;Some = 會話累積的 token ids。
    accumulated: Option<Vec<u32>>,
}

impl Session {
    pub fn new(engine: Box<dyn AsrEngine>) -> Self {
        Self { engine, accumulated: None }
    }

    pub fn backend(&self) -> &'static str {
        self.engine.backend()
    }

    pub fn handle(&mut self, cmd: Command) -> Vec<Event> {
        match cmd {
            Command::Start { sample_rate, .. } if sample_rate != 16000 => {
                error(format!("sample_rate 僅支援 16000,收到 {sample_rate}"))
            }
            Command::Start { .. } if self.accumulated.is_some() => {
                error("會話進行中,請先 stop".into())
            }
            Command::Start { .. } => {
                self.accumulated = Some(Vec::new());
                Vec::new()
            }
            Command::Audio { pcm16_b64 } => {
                let Some(ids) = self.accumulated.as_mut() else {
                    return error("尚未 start".into());
                };
                let samples = match decode_pcm16(&pcm16_b64) {
                    Ok(samples) => samples,
                    Err(message) => return error(message),
                };
                match self.engine.push(&samples) {
                    Ok(new_ids) if new_ids.is_empty() => Vec::new(),
                    Ok(new_ids) => {
                        ids.extend(new_ids);
                        let snapshot = ids.clone();
                        self.partial(&snapshot)
                    }
                    Err(message) => error(message),
                }
            }
            Command::Stop => {
                let Some(mut ids) = self.accumulated.take() else {
                    return error("尚未 start".into());
                };
                match self.engine.finish() {
                    Ok(tail) => ids.extend(tail),
                    Err(message) => return error(message),
                }
                match self.engine.decode(&ids) {
                    Ok(text) => vec![Event::Final { text }],
                    Err(message) => error(message),
                }
            }
        }
    }

    fn partial(&self, ids: &[u32]) -> Vec<Event> {
        match self.engine.decode(ids) {
            Ok(text) => vec![Event::Partial { text }],
            Err(message) => error(message),
        }
    }
}

fn error(message: String) -> Vec<Event> {
    vec![Event::Error { message }]
}

/// PCM16-LE bytes → f32 samples(±1.0)。奇數位元組視為格式錯誤。
fn decode_pcm16(b64: &str) -> Result<Vec<f32>, String> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|error| format!("pcm16_b64 非法 base64:{error}"))?;
    if bytes.len() % 2 != 0 {
        return Err("pcm16_b64 位元組數必須為偶數".into());
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|pair| i16::from_le_bytes([pair[0], pair[1]]) as f32 / 32768.0)
        .collect())
}
```

main.rs 掛模組:

```rust
mod engine;
mod protocol;
mod session;

fn main() {}
```

- [ ] **Step 5: 跑測試確認通過**

Run: `cargo test -p catcher-asr-host`
Expected: 9 passed(protocol 4 + session 5)

- [ ] **Step 6: Commit**

```bash
git add crates/catcher-asr-host/src
git commit -m "feat: asr-host engine trait, fake engine, and session state machine"
```

---

### Task 4: stdio 迴圈 + 二進位整合測試

**Files:**
- Modify: `crates/catcher-asr-host/src/main.rs`
- Create: `crates/catcher-asr-host/tests/stdio.rs`

**Interfaces:**
- Consumes: `Session::new` / `Session::handle` / `Session::backend`、`protocol::parse_command` / `protocol::emit`、`engine::FakeEngine`
- Produces: 可執行檔 `catcher-asr-host`,CLI:`--model <dir>`(真引擎,Task 5 接上)、`--language <s>`(預設 auto)、`--lookahead <n>`(預設 3)、`--fake-engine`(隱藏旗標,測試/驗收用,不需 `--model`)

- [ ] **Step 1: 寫 failing 整合測試**

```rust
// crates/catcher-asr-host/tests/stdio.rs
//! 對編譯出的二進位做黑箱協定測試(cargo 提供 CARGO_BIN_EXE_ 路徑)。

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdout, Command, Stdio};

fn spawn_fake_host() -> (Child, BufReader<ChildStdout>) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_catcher-asr-host"))
        .arg("--fake-engine")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn catcher-asr-host");
    let reader = BufReader::new(child.stdout.take().expect("stdout"));
    (child, reader)
}

fn read_line(reader: &mut BufReader<ChildStdout>) -> serde_json::Value {
    let mut line = String::new();
    reader.read_line(&mut line).expect("read stdout line");
    serde_json::from_str(line.trim()).expect("stdout line is JSON")
}

#[test]
fn full_protocol_roundtrip_with_fake_engine() {
    let (mut child, mut reader) = spawn_fake_host();

    let ready = read_line(&mut reader);
    assert_eq!(ready["event"], "ready");
    assert_eq!(ready["backend"], "fake");

    let stdin = child.stdin.as_mut().expect("stdin");
    use base64::Engine as _;
    let chunk = base64::engine::general_purpose::STANDARD.encode(vec![0u8; 1600 * 2]);
    writeln!(stdin, r#"{{"cmd":"start","lang":"auto","sample_rate":16000}}"#).unwrap();
    writeln!(stdin, r#"{{"cmd":"audio","pcm16_b64":"{chunk}"}}"#).unwrap();
    writeln!(stdin, r#"{{"cmd":"stop"}}"#).unwrap();

    let partial = read_line(&mut reader);
    assert_eq!(partial["event"], "partial");
    assert_eq!(partial["text"], "字0");

    let final_event = read_line(&mut reader);
    assert_eq!(final_event["event"], "final");
    assert_eq!(final_event["text"], "字0");

    drop(child.stdin.take()); // EOF → 行程正常結束
    let status = child.wait().expect("wait");
    assert!(status.success());
}

#[test]
fn malformed_line_yields_error_and_keeps_running() {
    let (mut child, mut reader) = spawn_fake_host();
    assert_eq!(read_line(&mut reader)["event"], "ready");

    let stdin = child.stdin.as_mut().expect("stdin");
    writeln!(stdin, "garbage").unwrap();
    assert_eq!(read_line(&mut reader)["event"], "error");

    // 行程還活著:正常會話仍可走完
    writeln!(stdin, r#"{{"cmd":"start","lang":"auto","sample_rate":16000}}"#).unwrap();
    writeln!(stdin, r#"{{"cmd":"stop"}}"#).unwrap();
    assert_eq!(read_line(&mut reader)["event"], "final");

    drop(child.stdin.take());
    assert!(child.wait().expect("wait").success());
}
```

`Cargo.toml` 補 dev-dependency(整合測試要用 serde_json/base64,已在 dependencies,無需新增;若編譯器要求,將兩者同列 `[dev-dependencies]`)。

- [ ] **Step 2: 跑測試確認失敗**

Run: `cargo test -p catcher-asr-host --test stdio`
Expected: FAIL(main 尚未輸出 ready、未讀 stdin)

- [ ] **Step 3: 實作 main.rs**

```rust
//! catcher-asr-host:tomato-ears 的 macOS MLX engine host。
//! 協定見 docs/protocol/asr-host-v1.md;本檔只做 stdio 編解碼與轉發,
//! 邏輯在 session.rs,引擎在 engine.rs。

mod engine;
mod protocol;
mod session;

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

use engine::{AsrEngine, FakeEngine};
use protocol::Event;
use session::Session;

#[derive(Debug, Parser)]
#[command(name = "catcher-asr-host", version, about = "Nemotron ASR engine host (MLX)")]
struct Arguments {
    /// MLX INT8 artifact 目錄(含 tokenizer.json)。
    #[arg(long, required_unless_present = "fake_engine")]
    model: Option<PathBuf>,
    /// Checkpoint prompt locale,例如 en-US、zh-TW、auto。
    #[arg(long, default_value = "auto")]
    language: String,
    /// Encoder 右側 attention context:0、3、6、13。
    #[arg(long, default_value_t = 3)]
    lookahead: usize,
    /// 測試用假引擎,不載入模型。
    #[arg(long, hide = true)]
    fake_engine: bool,
}

fn main() -> ExitCode {
    let arguments = Arguments::parse();
    let engine: Box<dyn AsrEngine> = if arguments.fake_engine {
        Box::new(FakeEngine::new())
    } else {
        // Task 5 接上 MlxEngine;在那之前真引擎路徑回報未實作。
        emit_line(&Event::Error { message: "MLX engine 尚未接上,請用 --fake-engine".into() });
        return ExitCode::FAILURE;
    };

    let mut session = Session::new(engine);
    emit_line(&Event::Ready { backend: session.backend() });

    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let events = match protocol::parse_command(&line) {
            Ok(cmd) => session.handle(cmd),
            Err(message) => vec![Event::Error { message }],
        };
        for event in &events {
            emit_line(event);
        }
    }
    ExitCode::SUCCESS
}

fn emit_line(event: &Event) {
    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "{}", protocol::emit(event)).expect("write stdout");
    stdout.flush().expect("flush stdout");
}
```

- [ ] **Step 4: 跑全部測試確認通過**

Run: `cargo test -p catcher-asr-host`
Expected: 11 passed(單元 9 + 整合 2)

- [ ] **Step 5: Commit**

```bash
git add crates/catcher-asr-host
git commit -m "feat: asr-host stdio loop with black-box protocol tests"
```

---

### Task 5: MlxEngine(真引擎)

**Files:**
- Modify: `crates/catcher-asr-host/src/engine.rs`(新增 `MlxEngine`)
- Modify: `crates/catcher-asr-host/src/main.rs`(接上真引擎分支)
- Create: `crates/catcher-asr-host/tests/real_model.rs`(gated,`#[ignore]`)

**Interfaces:**
- Consumes: `nemotron_mlx::{weights::Artifact, model::StreamingTranscriber, tokenizer::Tokenizer, opencc}`(簽名與 `crates/nemotron-cli/src/main.rs:95-145` 相同)
- Produces: `engine::MlxEngine::load(model: &Path, language: &str, lookahead: usize) -> Result<MlxEngine, String>`

- [ ] **Step 1: 寫 gated failing 測試**

```rust
// crates/catcher-asr-host/tests/real_model.rs
//! 真模型整合測試。需要 Metal 與已下載的 artifact,預設 #[ignore]。
//! 執行:
//!   CATCHER_ASR_MODEL_DIR=~/path/to/catcher-asr-mlx-int8 \
//!   CATCHER_ASR_FIXTURE_WAV=~/path/to/fixture-zh.wav \
//!   CATCHER_ASR_FIXTURE_TEXT="預期的轉錄內容" \
//!   cargo test -p catcher-asr-host --test real_model -- --ignored

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

#[test]
#[ignore = "needs Metal + downloaded model artifact"]
fn transcribes_fixture_wav_end_to_end() {
    let model = std::env::var("CATCHER_ASR_MODEL_DIR").expect("CATCHER_ASR_MODEL_DIR");
    let wav_path = std::env::var("CATCHER_ASR_FIXTURE_WAV").expect("CATCHER_ASR_FIXTURE_WAV");
    let expected = std::env::var("CATCHER_ASR_FIXTURE_TEXT").expect("CATCHER_ASR_FIXTURE_TEXT");

    let samples = read_wav_pcm16(&wav_path);
    let mut child = Command::new(env!("CARGO_BIN_EXE_catcher-asr-host"))
        .args(["--model", &model])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn");
    let mut reader = BufReader::new(child.stdout.take().unwrap());

    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let ready: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
    assert_eq!(ready["event"], "ready");
    assert_eq!(ready["backend"], "mlx");

    let stdin = child.stdin.as_mut().unwrap();
    writeln!(stdin, r#"{{"cmd":"start","lang":"auto","sample_rate":16000}}"#).unwrap();
    use base64::Engine as _;
    for chunk in samples.chunks(1600 * 2) {
        let b64 = base64::engine::general_purpose::STANDARD.encode(chunk);
        writeln!(stdin, r#"{{"cmd":"audio","pcm16_b64":"{b64}"}}"#).unwrap();
    }
    writeln!(stdin, r#"{{"cmd":"stop"}}"#).unwrap();

    // 讀到 final 為止(中途的 partial 全部略過)
    let final_text = loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap() == 0 {
            panic!("host 在 final 之前結束");
        }
        let value: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
        if value["event"] == "final" {
            break value["text"].as_str().unwrap().to_string();
        }
    };

    // 串流解碼容許小差異:斷言預期字元的覆蓋率 ≥ 60%,而非逐字相等。
    let hits = expected.chars().filter(|c| final_text.contains(*c)).count();
    let coverage = hits as f64 / expected.chars().count().max(1) as f64;
    assert!(
        coverage >= 0.6,
        "coverage {coverage:.2} too low\nexpected: {expected}\ngot: {final_text}"
    );
    drop(child.stdin.take());
    child.wait().unwrap();
}

/// 讀 mono 16 kHz PCM16 WAV 的 data chunk(僅測試用的最小 parser)。
fn read_wav_pcm16(path: &str) -> Vec<u8> {
    let bytes = std::fs::read(path).expect("read wav");
    let data_pos = bytes
        .windows(4)
        .position(|window| window == b"data")
        .expect("wav data chunk");
    bytes[data_pos + 8..].to_vec()
}
```

- [ ] **Step 2: 確認測試紅燈(真引擎未接)**

Run: `CATCHER_ASR_MODEL_DIR=… CATCHER_ASR_FIXTURE_WAV=… CATCHER_ASR_FIXTURE_TEXT=… cargo test -p catcher-asr-host --test real_model -- --ignored`
Expected: FAIL(host 輸出 error「MLX engine 尚未接上」後退出)

- [ ] **Step 3: 實作 MlxEngine 並接上 main**

`engine.rs` 追加:

```rust
use std::path::Path;

use nemotron_mlx::{
    model::StreamingTranscriber, opencc, tokenizer::Tokenizer, weights::Artifact,
};

/// 真引擎:與 nemotron-cli 的 Transcribe 子命令同一條推論路徑
/// (Artifact → StreamingTranscriber → Tokenizer → opencc 繁化)。
pub struct MlxEngine {
    transcriber: StreamingTranscriber,
    tokenizer: Tokenizer,
}

impl MlxEngine {
    pub fn load(model: &Path, language: &str, lookahead: usize) -> Result<Self, String> {
        let artifact = Artifact::load(model).map_err(|error| format!("載入模型失敗:{error}"))?;
        let transcriber = StreamingTranscriber::new(&artifact, language, lookahead)
            .map_err(|error| format!("初始化 transcriber 失敗:{error}"))?;
        // 0 與 13_087 是 Nemotron tokenizer 的 id 邊界,與 nemotron-cli 一致。
        let tokenizer = Tokenizer::from_json(model.join("tokenizer.json"), 0, 13_087)
            .map_err(|error| format!("載入 tokenizer 失敗:{error}"))?;
        Ok(Self { transcriber, tokenizer })
    }
}

impl AsrEngine for MlxEngine {
    fn push(&mut self, samples: &[f32]) -> Result<Vec<u32>, String> {
        let tokens = self.transcriber.push_samples(samples).map_err(|e| e.to_string())?;
        Ok(tokens.iter().map(|token| token.id).collect())
    }

    fn finish(&mut self) -> Result<Vec<u32>, String> {
        let tokens = self.transcriber.finish().map_err(|e| e.to_string())?;
        Ok(tokens.iter().map(|token| token.id).collect())
    }

    fn decode(&self, ids: &[u32]) -> Result<String, String> {
        let text = self.tokenizer.decode(ids, true).map_err(|e| e.to_string())?;
        Ok(opencc::to_traditional(&text))
    }

    fn backend(&self) -> &'static str {
        "mlx"
    }
}
```

main.rs 的引擎分支改為:

```rust
    let engine: Box<dyn AsrEngine> = if arguments.fake_engine {
        Box::new(FakeEngine::new())
    } else {
        let model = arguments.model.as_deref().expect("clap enforces --model");
        match engine::MlxEngine::load(model, &arguments.language, arguments.lookahead) {
            Ok(engine) => Box::new(engine),
            Err(message) => {
                emit_line(&Event::Error { message });
                return ExitCode::FAILURE;
            }
        }
    };
```

(若 `StreamingTranscriber::push_samples` 回傳型別的欄位名與 `token.id` 不符,以
`crates/nemotron-mlx/src/model/` 中實際定義為準修正 map;`transcribe_samples`
在 `nemotron-cli/src/main.rs:143-144` 的用法是 `tokens.iter().map(|token| token.id)`。)

- [ ] **Step 4: 跑全部測試**

Run: `cargo test -p catcher-asr-host`(11 passed,gated 測試仍 ignored)
Run: 帶環境變數跑 `--test real_model -- --ignored`
Expected: PASS(final 覆蓋率 ≥ 0.6)

- [ ] **Step 5: Commit**

```bash
git add crates/catcher-asr-host
git commit -m "feat: wire MLX engine into asr-host with gated real-model test"
```

---

### Task 6: Release 打包腳本 + SHA-256

**Files:**
- Create: `scripts/build-asr-host.sh`

**Interfaces:**
- Produces: `dist/catcher-asr-host-v<version>-macos-arm64.tar.gz` + `.sha256`,供 tomato-ears 配方 manifest 的 `dependencies` 欄位引用。

- [ ] **Step 1: 寫打包腳本**

```bash
#!/bin/zsh
# 打包 catcher-asr-host 為可發布的 tar.gz 並產生 SHA-256。
# 產物路徑:dist/catcher-asr-host-v<version>-macos-arm64.tar.gz(.sha256)
set -euo pipefail
cd "$(dirname "$0")/.."

cargo build --release -p catcher-asr-host

version=$(cargo pkgid -p catcher-asr-host | sed 's/.*[@#]//')
stage=$(mktemp -d)
name="catcher-asr-host-v${version}-macos-arm64"
mkdir -p "dist" "${stage}/${name}"

cp target/release/catcher-asr-host "${stage}/${name}/"
cp docs/protocol/asr-host-v1.md "${stage}/${name}/PROTOCOL.md"
tar -czf "dist/${name}.tar.gz" -C "${stage}" "${name}"
shasum -a 256 "dist/${name}.tar.gz" | tee "dist/${name}.tar.gz.sha256"
rm -rf "${stage}"
echo "done: dist/${name}.tar.gz"
```

- [ ] **Step 2: 執行並驗證**

Run: `chmod +x scripts/build-asr-host.sh && scripts/build-asr-host.sh`
Expected: `dist/catcher-asr-host-v0.1.0-macos-arm64.tar.gz` 與 `.sha256` 存在

- [ ] **Step 3: Release binary 冒煙測試**

Run:
```bash
printf '%s\n%s\n' '{"cmd":"start","lang":"auto","sample_rate":16000}' '{"cmd":"stop"}' \
  | target/release/catcher-asr-host --fake-engine
```
Expected 輸出兩行:`ready`(backend fake)、`final`(text 空字串)——注意無 audio 時 final 為空是合法的。

- [ ] **Step 4: Commit**

```bash
git add scripts/build-asr-host.sh
git commit -m "build: catcher-asr-host release packaging with pinned sha256"
```

- [ ] **Step 5: 發布(需 wake 同意後執行)**

```bash
gh release create asr-host-v0.1.0 \
  dist/catcher-asr-host-v0.1.0-macos-arm64.tar.gz \
  dist/catcher-asr-host-v0.1.0-macos-arm64.tar.gz.sha256 \
  --repo wcAmon/catcher-tippi \
  --title "catcher-asr-host v0.1.0 (macOS arm64)" \
  --notes "tomato-ears mac engine host,協定 asr-host-v1。"
```

發布是對外動作:執行前向 wake 確認 tag 名與 repo 可見性。

---

## 後續計畫(本檔不含)

- **Plan 2** `nemotron-asr-host`(Windows):從 `codex/windows-auto-backend` 抽出
  `NemotronEngine.cs` / `InferenceBackend.cs` / `ModelInstaller.cs` 成 .NET console host,
  講同一份 asr-host-v1 協定;在 Windows 機器驗證 DirectML 探測。
- **Plan 3** tomato-ears Deno 配方包(reference/、verify/、SPEC/PLAN/SECURITY/manifest)。
- **Plan 4** tmuh.ai mini-app store 實作文件(交伺服器端 Claude Code)。
