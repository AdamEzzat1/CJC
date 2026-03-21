//! `cjc forge` — Content-addressable pipeline runner.
//!
//! Executes a CJC program and captures its output as a content-addressable
//! artifact identified by a SHA-256 hash. Supports:
//! - Running .cjc files with deterministic seed
//! - Capturing stdout as an artifact
//! - Computing SHA-256 hash of output
//! - Storing artifacts in a local cache directory (.cjc-forge/)
//! - Verifying cached artifacts match expected hashes
//! - Listing cached artifacts
//!
//! Content-addressable storage ensures identical inputs always produce
//! identical outputs with identical hashes — the foundation of
//! reproducible pipelines.

use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use crate::output::{self, OutputMode};

const FORGE_DIR: &str = ".cjc-forge";

/// Parsed arguments for `cjc forge`.
pub struct ForgeArgs {
    pub action: ForgeAction,
    pub seed: u64,
    pub output: OutputMode,
    pub cache_dir: PathBuf,
    pub verbose: bool,
}

#[derive(Debug, Clone)]
pub enum ForgeAction {
    /// Run a .cjc file and store its output as a content-addressed artifact.
    Run { file: PathBuf },
    /// Verify that running a file produces the expected hash.
    Verify { file: PathBuf, expected_hash: String },
    /// List all cached artifacts.
    List,
    /// Show details of a specific artifact by hash prefix.
    Show { hash_prefix: String },
    /// Clean the forge cache.
    Clean,
}

impl Default for ForgeArgs {
    fn default() -> Self {
        Self {
            action: ForgeAction::List,
            seed: 42,
            output: OutputMode::Color,
            cache_dir: PathBuf::from(FORGE_DIR),
            verbose: false,
        }
    }
}

pub fn parse_args(args: &[String]) -> ForgeArgs {
    let mut fa = ForgeArgs::default();
    let mut positionals = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                if i < args.len() {
                    fa.seed = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("error: --seed requires a numeric argument");
                        process::exit(1);
                    });
                }
            }
            "--cache-dir" => {
                i += 1;
                if i < args.len() {
                    fa.cache_dir = PathBuf::from(&args[i]);
                }
            }
            "-v" | "--verbose" => fa.verbose = true,
            "--plain" => fa.output = OutputMode::Plain,
            "--json" => fa.output = OutputMode::Json,
            "--color" => fa.output = OutputMode::Color,
            "run" if positionals.is_empty() => positionals.push("run".to_string()),
            "verify" if positionals.is_empty() => positionals.push("verify".to_string()),
            "list" if positionals.is_empty() => positionals.push("list".to_string()),
            "show" if positionals.is_empty() => positionals.push("show".to_string()),
            "clean" if positionals.is_empty() => positionals.push("clean".to_string()),
            other if !other.starts_with('-') => positionals.push(other.to_string()),
            other => {
                eprintln!("error: unknown flag `{}` for `cjc forge`", other);
                process::exit(1);
            }
        }
        i += 1;
    }

    if positionals.is_empty() {
        fa.action = ForgeAction::List;
        return fa;
    }

    match positionals[0].as_str() {
        "run" => {
            if positionals.len() < 2 {
                eprintln!("error: `cjc forge run` requires a .cjc file argument");
                process::exit(1);
            }
            fa.action = ForgeAction::Run { file: PathBuf::from(&positionals[1]) };
        }
        "verify" => {
            if positionals.len() < 3 {
                eprintln!("error: `cjc forge verify` requires <file> <expected_hash>");
                process::exit(1);
            }
            fa.action = ForgeAction::Verify {
                file: PathBuf::from(&positionals[1]),
                expected_hash: positionals[2].clone(),
            };
        }
        "list" => fa.action = ForgeAction::List,
        "show" => {
            if positionals.len() < 2 {
                eprintln!("error: `cjc forge show` requires a hash prefix");
                process::exit(1);
            }
            fa.action = ForgeAction::Show { hash_prefix: positionals[1].clone() };
        }
        "clean" => fa.action = ForgeAction::Clean,
        other => {
            // Treat as a file to run
            fa.action = ForgeAction::Run { file: PathBuf::from(other) };
        }
    }

    fa
}

/// A stored forge artifact.
struct Artifact {
    hash: String,
    source_file: String,
    seed: u64,
    output_lines: usize,
    size: u64,
}

/// Entry point for `cjc forge`.
pub fn run(args: &[String]) {
    let fa = parse_args(args);

    match &fa.action {
        ForgeAction::Run { file } => forge_run(&fa, file),
        ForgeAction::Verify { file, expected_hash } => forge_verify(&fa, file, expected_hash),
        ForgeAction::List => forge_list(&fa),
        ForgeAction::Show { hash_prefix } => forge_show(&fa, hash_prefix),
        ForgeAction::Clean => forge_clean(&fa),
    }
}

fn forge_run(fa: &ForgeArgs, file: &Path) {
    let source = match fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: could not read `{}`: {}", file.display(), e);
            process::exit(1);
        }
    };

    let filename = file.display().to_string();

    // Parse and execute
    let (program, diags) = cjc_parser::parse_source(&source);
    if diags.has_errors() {
        let rendered = diags.render_all_color(&source, &filename, fa.output.use_color());
        eprint!("{}", rendered);
        process::exit(1);
    }

    let mut interpreter = cjc_eval::Interpreter::new(fa.seed);
    if let Err(e) = interpreter.exec(&program) {
        eprintln!("error: execution failed: {}", e);
        process::exit(1);
    }

    // Collect output
    let output_text = interpreter.output.join("\n");
    let output_bytes = output_text.as_bytes();

    // Compute content hash
    let hash_bytes = cjc_snap::hash::sha256(output_bytes);
    let hash_hex = hash_bytes.iter().map(|b| format!("{:02x}", b)).collect::<String>();

    // Ensure cache directory exists
    let _ = fs::create_dir_all(&fa.cache_dir);

    // Write artifact: <cache>/<hash>.artifact
    let artifact_path = fa.cache_dir.join(format!("{}.artifact", &hash_hex));
    let meta = format!(
        "source: {}\nseed: {}\nlines: {}\nhash: {}\n---\n",
        filename.replace('\\', "/"),
        fa.seed,
        interpreter.output.len(),
        hash_hex,
    );

    let mut content = meta;
    content.push_str(&output_text);

    if let Err(e) = fs::write(&artifact_path, &content) {
        eprintln!("error: could not write artifact: {}", e);
        process::exit(1);
    }

    // Report
    match fa.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"action\": \"run\",");
            println!("  \"source\": \"{}\",", filename.replace('\\', "/"));
            println!("  \"seed\": {},", fa.seed);
            println!("  \"hash\": \"{}\",", hash_hex);
            println!("  \"output_lines\": {},", interpreter.output.len());
            println!("  \"output_bytes\": {}", output_bytes.len());
            println!("}}");
        }
        _ => {
            let mut t = crate::table::Table::new(vec!["Property", "Value"]);
            t.add_row_owned(vec!["Source".into(), filename.replace('\\', "/")]);
            t.add_row_owned(vec!["Seed".into(), format!("{}", fa.seed)]);
            t.add_row_owned(vec!["Output lines".into(), format!("{}", interpreter.output.len())]);
            t.add_row_owned(vec!["Output bytes".into(), format!("{}", output_bytes.len())]);
            t.add_row_owned(vec!["SHA-256".into(), hash_hex.clone()]);
            t.add_row_owned(vec!["Artifact".into(), artifact_path.display().to_string().replace('\\', "/")]);
            eprint!("{}", t.render());
            eprintln!("{}", output::colorize(fa.output, output::BOLD_GREEN, "forged"));
        }
    }
}

fn forge_verify(fa: &ForgeArgs, file: &Path, expected_hash: &str) {
    let source = match fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: could not read `{}`: {}", file.display(), e);
            process::exit(1);
        }
    };

    let filename = file.display().to_string();

    let (program, diags) = cjc_parser::parse_source(&source);
    if diags.has_errors() {
        let rendered = diags.render_all_color(&source, &filename, fa.output.use_color());
        eprint!("{}", rendered);
        process::exit(1);
    }

    let mut interpreter = cjc_eval::Interpreter::new(fa.seed);
    if let Err(e) = interpreter.exec(&program) {
        eprintln!("error: execution failed: {}", e);
        process::exit(1);
    }

    let output_text = interpreter.output.join("\n");
    let hash_bytes = cjc_snap::hash::sha256(output_text.as_bytes());
    let hash_hex = hash_bytes.iter().map(|b| format!("{:02x}", b)).collect::<String>();

    let matches = hash_hex.starts_with(expected_hash) || expected_hash.starts_with(&hash_hex);

    match fa.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"action\": \"verify\",");
            println!("  \"source\": \"{}\",", filename.replace('\\', "/"));
            println!("  \"seed\": {},", fa.seed);
            println!("  \"expected\": \"{}\",", expected_hash);
            println!("  \"actual\": \"{}\",", hash_hex);
            println!("  \"match\": {}", matches);
            println!("}}");
        }
        _ => {
            eprintln!("Verifying `{}`...", filename);
            eprintln!("  Expected: {}", expected_hash);
            eprintln!("  Actual:   {}", hash_hex);
            if matches {
                eprintln!("  {}", output::colorize(fa.output, output::BOLD_GREEN, "MATCH"));
            } else {
                eprintln!("  {}", output::colorize(fa.output, output::BOLD_RED, "MISMATCH"));
                process::exit(1);
            }
        }
    }
}

fn forge_list(fa: &ForgeArgs) {
    let read_dir = match fs::read_dir(&fa.cache_dir) {
        Ok(rd) => rd,
        Err(_) => {
            if fa.output == OutputMode::Json {
                println!("[]");
            } else {
                eprintln!("No forge cache found ({})", fa.cache_dir.display());
            }
            return;
        }
    };

    let mut artifacts: Vec<Artifact> = Vec::new();

    let mut entries: Vec<_> = read_dir.filter_map(|e| e.ok()).collect();
    entries.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

    for entry in entries {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("artifact") {
            continue;
        }

        if let Some(artifact) = parse_artifact_meta(&path) {
            artifacts.push(artifact);
        }
    }

    match fa.output {
        OutputMode::Json => {
            println!("[");
            for (i, a) in artifacts.iter().enumerate() {
                print!("  {{\"hash\": \"{}\", \"source\": \"{}\", \"seed\": {}, \"lines\": {}, \"size\": {}}}",
                    a.hash, a.source_file, a.seed, a.output_lines, a.size);
                if i + 1 < artifacts.len() { print!(","); }
                println!();
            }
            println!("]");
        }
        _ => {
            if artifacts.is_empty() {
                eprintln!("No artifacts in forge cache.");
                return;
            }
            let mut t = crate::table::Table::new(vec!["Hash (short)", "Source", "Seed", "Lines", "Size"]);
            for a in &artifacts {
                t.add_row_owned(vec![
                    a.hash.chars().take(16).collect(),
                    a.source_file.clone(),
                    format!("{}", a.seed),
                    format!("{}", a.output_lines),
                    output::format_size(a.size),
                ]);
            }
            eprint!("{}", t.render());
            eprintln!("{} artifacts", artifacts.len());
        }
    }
}

fn forge_show(fa: &ForgeArgs, hash_prefix: &str) {
    let read_dir = match fs::read_dir(&fa.cache_dir) {
        Ok(rd) => rd,
        Err(_) => {
            eprintln!("error: forge cache not found ({})", fa.cache_dir.display());
            process::exit(1);
        }
    };

    let mut found: Option<PathBuf> = None;
    for entry in read_dir.filter_map(|e| e.ok()) {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with(hash_prefix) && name.ends_with(".artifact") {
            found = Some(entry.path());
            break;
        }
    }

    let artifact_path = match found {
        Some(p) => p,
        None => {
            eprintln!("error: no artifact matching prefix `{}`", hash_prefix);
            process::exit(1);
        }
    };

    let content = match fs::read_to_string(&artifact_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: could not read artifact: {}", e);
            process::exit(1);
        }
    };

    // Print the full artifact content
    println!("{}", content);
}

fn forge_clean(fa: &ForgeArgs) {
    let read_dir = match fs::read_dir(&fa.cache_dir) {
        Ok(rd) => rd,
        Err(_) => {
            eprintln!("Forge cache already clean.");
            return;
        }
    };

    let mut count = 0u64;
    for entry in read_dir.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("artifact") {
            if fs::remove_file(&path).is_ok() {
                count += 1;
            }
        }
    }

    match fa.output {
        OutputMode::Json => {
            println!("{{\"action\": \"clean\", \"removed\": {}}}", count);
        }
        _ => {
            eprintln!("Removed {} artifacts from forge cache.", count);
        }
    }
}

/// Parse artifact metadata from the header section (before ---).
fn parse_artifact_meta(path: &Path) -> Option<Artifact> {
    let content = fs::read_to_string(path).ok()?;
    let size = content.len() as u64;

    let mut source_file = String::new();
    let mut seed = 0u64;
    let mut lines = 0usize;
    let mut hash = String::new();

    for line in content.lines() {
        if line.starts_with("---") { break; }
        if let Some(val) = line.strip_prefix("source: ") {
            source_file = val.to_string();
        } else if let Some(val) = line.strip_prefix("seed: ") {
            seed = val.parse().unwrap_or(0);
        } else if let Some(val) = line.strip_prefix("lines: ") {
            lines = val.parse().unwrap_or(0);
        } else if let Some(val) = line.strip_prefix("hash: ") {
            hash = val.to_string();
        }
    }

    Some(Artifact {
        hash,
        source_file,
        seed,
        output_lines: lines,
        size,
    })
}

pub fn print_help() {
    eprintln!("cjc forge — Content-addressable pipeline runner");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  cjc forge run <file.cjc>              Run and store output artifact");
    eprintln!("  cjc forge verify <file.cjc> <hash>    Verify output matches expected hash");
    eprintln!("  cjc forge list                        List cached artifacts");
    eprintln!("  cjc forge show <hash-prefix>          Show artifact details");
    eprintln!("  cjc forge clean                       Remove all cached artifacts");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  --seed <N>            RNG seed (default: 42)");
    eprintln!("  --cache-dir <path>    Cache directory (default: .cjc-forge/)");
    eprintln!("  -v, --verbose         Show execution details");
    eprintln!("  --plain               Plain text output");
    eprintln!("  --json                JSON output");
    eprintln!("  --color               Color output (default)");
}
