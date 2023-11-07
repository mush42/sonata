use clap::Parser;
use libtashkeel_base::{create_inference_engine, do_tashkeel, DynamicInferenceEngine, CHAR_LIMIT};
use std::fs::File;
use std::io::{self, prelude::*};
use std::path::PathBuf;

const TASKEEN_REJECTION_THRESHOLD: &str = "0.95";

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Input file (default `stdin`)
    #[arg(short = 'f', long, value_name = "INPUT_FILE")]
    input_file: Option<PathBuf>,
    /// Output file (default `stdout`)
    #[arg(short, long, value_name = "OUTPUT_FILE")]
    output_file: Option<PathBuf>,
    /// Use interactive mode (useful for testing)
    #[arg(short, long)]
    interactive: bool,
    /// Use sukoon for case-ending diacritic if the model is uncertain
    #[arg(short, long)]
    taskeen: bool,
    /// Taskeen threshold probability
    #[arg(long, short, default_value = TASKEEN_REJECTION_THRESHOLD, required = false)]
    prob: Option<f32>,
    /// ONNX model (default: use bundled model if available)
    #[arg(short = 'x', long, value_name = "ONNX_MODEL")]
    onnx: Option<PathBuf>,
}

fn write_to_stdout(text: &str) -> anyhow::Result<()> {
    let mut stdout = io::stdout().lock();
    stdout.write_all(text.as_bytes())?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    Ok(())
}

fn get_input_text(args: &Cli) -> anyhow::Result<String> {
    let mut input_buffer = String::new();
    if let Some(ref input_filename) = args.input_file {
        let mut file = File::open(input_filename)?;
        file.read_to_string(&mut input_buffer)?;
    } else {
        let stdin = io::stdin();
        stdin.read_line(&mut input_buffer)?;
    }

    Ok(input_buffer)
}

fn tashkeel_main(
    model: &DynamicInferenceEngine,
    args: &Cli,
    input_text: String,
) -> anyhow::Result<()> {
    let taskeen_threshold = if args.taskeen { args.prob } else { None };
    let mut diacritized_text: String = String::new();
    if args.input_file.is_none() {
        let input = String::from_iter(input_text.chars().take(CHAR_LIMIT));
        diacritized_text.push_str(&do_tashkeel(model, &input, taskeen_threshold)?);
    } else {
        let mut diacritized_lines = String::new();
        for input_line in input_text.lines() {
            let input = String::from_iter(input_line.chars().take(CHAR_LIMIT));
            let diacritized_line = do_tashkeel(model, &input, taskeen_threshold)?;
            if args.output_file.is_none() {
                write_to_stdout(&diacritized_line)?;
            } else {
                diacritized_lines.push_str(&diacritized_line);
                diacritized_lines.push('\n');
            }
        }
        diacritized_text.push_str(&diacritized_lines);
    }

    if let Some(ref output_filename) = args.output_file {
        let mut file = File::create(output_filename)?;
        file.write_all(diacritized_text.as_bytes())?;
        log::info!("Wrote output to file `{}`", output_filename.display());
    } else {
        write_to_stdout(&diacritized_text)?
    }

    Ok(())
}

fn main() -> anyhow::Result<()> {
    setup_logging();

    let mut args = Cli::parse();

    if args.input_file.is_some() || args.output_file.is_some() {
        if args.interactive {
            anyhow::bail!(
                "Interactive mode is not available when `--input-file` or `--output-file` is passed"
            )
        }
    } else {
        args.interactive = true;
    }

    let model = create_inference_engine(args.onnx.take())?;

    let mut input_text = get_input_text(&args)?;
    if args.interactive {
        loop {
            if !input_text.trim().is_empty() {
                tashkeel_main(&model, &args, std::mem::take(&mut input_text))?;
            }
            input_text = get_input_text(&args)?;
        }
    } else {
        tashkeel_main(&model, &args, input_text)?;
    }

    Ok(())
}

fn setup_logging() {
    env_logger::Builder::from_env(env_logger::Env::default().filter_or("TASHKEEL_LOG", "info"))
        .init();
}
