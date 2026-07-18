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
