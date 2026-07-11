use std::{
    path::{Path, PathBuf},
    process::ExitCode,
};

use clap::{Parser, Subcommand};
use nemotron_mlx::{model::StreamingTranscriber, tokenizer::Tokenizer, weights::Artifact};

#[derive(Debug, Parser)]
#[command(name = "catcher", version, about = "Nemotron 3.5 ASR on Apple MLX")]
struct Arguments {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Transcribe a mono 16 kHz WAV with cache-aware streaming inference.
    Transcribe {
        /// Converted MLX INT8 artifact directory.
        #[arg(long)]
        model: PathBuf,
        /// Mono 16 kHz PCM or float WAV file.
        #[arg(long)]
        audio: PathBuf,
        /// Checkpoint prompt locale such as en-US, zh-CN, zh-TW, or auto.
        #[arg(long, default_value = "auto")]
        language: String,
        /// Right attention context in encoder frames: 0, 3, 6, or 13.
        #[arg(long, default_value_t = 3)]
        lookahead: usize,
        /// tokenizer.json; defaults to MODEL/tokenizer.json.
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        /// Emit a JSON object instead of plain text.
        #[arg(long)]
        json: bool,
    },
}

fn main() -> ExitCode {
    match run(Arguments::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("catcher: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(arguments: Arguments) -> Result<(), Box<dyn std::error::Error>> {
    match arguments.command {
        Command::Transcribe {
            model,
            audio,
            language,
            lookahead,
            tokenizer,
            json,
        } => {
            let samples = read_wav(&audio)?;
            let artifact = Artifact::load(&model)?;
            let mut transcriber = StreamingTranscriber::new(&artifact, &language, lookahead)?;
            let token_ids = transcriber.transcribe_samples(&samples)?;
            let tokenizer_path = tokenizer.unwrap_or_else(|| model.join("tokenizer.json"));
            let tokenizer = Tokenizer::from_json(tokenizer_path, 0, 13_087)?;
            let text = tokenizer.decode(&token_ids, true)?;
            if json {
                println!(
                    "{}",
                    serde_json::json!({
                        "text": text,
                        "token_ids": token_ids,
                        "language": language,
                        "lookahead": lookahead,
                    })
                );
            } else {
                println!("{text}");
            }
        }
    }
    Ok(())
}

fn read_wav(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate != 16_000 {
        return Err(format!(
            "WAV must be mono 16 kHz, found {} channel(s) at {} Hz",
            spec.channels, spec.sample_rate
        )
        .into());
    }
    match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into),
        hound::SampleFormat::Int if spec.bits_per_sample <= 16 => reader
            .samples::<i16>()
            .map(|sample| sample.map(|value| value as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into),
        hound::SampleFormat::Int => {
            let scale = 2_f32.powi(spec.bits_per_sample as i32 - 1);
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / scale))
                .collect::<Result<Vec<_>, _>>()
                .map_err(Into::into)
        }
    }
}
