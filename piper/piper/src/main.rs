use clap::Parser;
use log::{debug, error, info};
use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// ONNX model path
    #[arg(short, long, value_name = "ONNX_FILE")]
    onnx: PathBuf,

    /// Model config
    #[arg(short, long, value_name = "CONFIG_FILE")]
    config: PathBuf,

}


fn main() {
    enable_logging();
    let cli = Cli::parse();
    info!("Using ONNX model: `{}`", cli.onnx.display());
    info!("Using model config: `{}`", cli.config.display());
}

fn enable_logging() {
    env_logger::Builder::from_env(env_logger::Env::default().filter_or("PIPER_LOG", "info")).init();
}