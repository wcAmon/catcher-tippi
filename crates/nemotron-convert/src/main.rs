use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use nemotron_mlx::weights::convert_model_with_group_size;

/// Convert the published Nemotron 3.5 F32 safetensors checkpoint to an MLX INT8 artifact.
#[derive(Debug, Parser)]
#[command(name = "nemotron-convert", version, about)]
struct Arguments {
    /// Source Hugging Face model.safetensors file.
    #[arg(long)]
    source: PathBuf,

    /// New output directory for weights.safetensors and manifest.json.
    #[arg(long)]
    output: PathBuf,

    /// Affine INT8 values per scale/bias group.
    #[arg(long, default_value_t = 128)]
    group_size: usize,
}

fn main() -> ExitCode {
    let arguments = Arguments::parse();
    match convert_model_with_group_size(&arguments.source, &arguments.output, arguments.group_size)
    {
        Ok(()) => {
            println!("wrote MLX INT8 artifact to {}", arguments.output.display());
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("conversion failed: {error}");
            ExitCode::FAILURE
        }
    }
}
