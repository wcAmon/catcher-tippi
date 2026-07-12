use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use sortformer_mlx::weights::convert_model;

/// Convert exported Sortformer v2.1 F32 safetensors to an MLX INT8 artifact.
#[derive(Debug, Parser)]
#[command(name = "sortformer-convert", version, about)]
struct Arguments {
    /// Exported model.safetensors from tools/export_sortformer_weights.py.
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
    match convert_model(&arguments.source, &arguments.output, arguments.group_size) {
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
