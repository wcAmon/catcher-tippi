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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::FakeEngine;
    use crate::protocol::{Command, Event};

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
