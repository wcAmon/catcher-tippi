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
