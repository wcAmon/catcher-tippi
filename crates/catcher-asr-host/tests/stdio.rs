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
