//! 真模型整合測試。需要 Metal 與已下載的 artifact,預設 #[ignore]。
//! 執行:
//!   CATCHER_ASR_MODEL_DIR=~/path/to/catcher-asr-mlx-int8 \
//!   CATCHER_ASR_FIXTURE_WAV=~/path/to/fixture-zh.wav \
//!   CATCHER_ASR_FIXTURE_TEXT="預期的轉錄內容" \
//!   cargo test -p catcher-asr-host --test real_model -- --ignored
//!
//! 第二組(中文)執行範例——AISHELL-3 真人語音 14 句串接
//! (ground truth 見 tests/fixtures/README.md;簡體原文以
//! nemotron_mlx::opencc::to_traditional 轉繁,與 host 輸出同一轉換路徑):
//!   CATCHER_ASR_MODEL_DIR=~/path/to/catcher-asr-mlx-int8 \
//!   CATCHER_ASR_FIXTURE_WAV=tests/fixtures/conversation.wav \
//!   CATCHER_ASR_FIXTURE_TEXT="伺候老婆是老公的責任領導幹部廉潔從政自查阿爾泰的生物有什麼河宕村民委員會計劃生育服務室文成縣的學校有什麼基層醫院的醫生缺乏不斷學習和提高水平的動力但有人說我非常耀眼下輩子不做女人微軟推出免費增值策略三百六十五山海經地名有什麼當你孤單你會想起誰我們稱她為母夜叉搜狐娛樂訊據香港媒體報道" \
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

    // 正規化編輯距離:0.0 = 完全相同。比 presence-based 覆蓋率嚴格——
    // 亂碼即使字元集重疊也會因插入/替換代價而距離飆高。
    let distance = normalized_levenshtein(&expected, &final_text);
    assert!(
        distance <= 0.25,
        "normalized edit distance {distance:.3} > 0.25\nexpected: {expected}\ngot: {final_text}"
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

/// 字元級 Levenshtein 距離除以較長字串的字元數(0.0 = 相同,1.0 = 完全不同)。
fn normalized_levenshtein(a: &str, b: &str) -> f64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (prev[j] + cost).min(prev[j + 1] + 1).min(curr[j] + 1);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()] as f64 / a.len().max(b.len()) as f64
}
