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
        let model = arguments.model.as_deref().expect("clap enforces --model");
        match engine::MlxEngine::load(model, &arguments.language, arguments.lookahead) {
            Ok(engine) => Box::new(engine),
            Err(message) => {
                emit_line(&Event::Error { message });
                return ExitCode::FAILURE;
            }
        }
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
