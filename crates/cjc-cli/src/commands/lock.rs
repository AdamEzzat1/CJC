//! `cjc lock` — Deterministic lockfile generator & verifier.
//!
//! Generates a lockfile recording source hash, seed, executor version,
//! platform, and expected output hash. With `--verify`, checks that the
//! current run matches the recorded lockfile.

use std::fs;
use std::path::PathBuf;
use std::process;
use crate::output::{self, OutputMode};

/// Parsed arguments for `cjc lock`.
pub struct LockArgs {
    pub file: PathBuf,
    pub seed: u64,
    pub verify: bool,
    pub output: OutputMode,
}

impl Default for LockArgs {
    fn default() -> Self {
        Self {
            file: PathBuf::new(),
            seed: 42,
            verify: false,
            output: OutputMode::Color,
        }
    }
}

pub fn parse_args(args: &[String]) -> LockArgs {
    let mut la = LockArgs::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                if i < args.len() {
                    la.seed = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("error: --seed requires a numeric argument");
                        process::exit(1);
                    });
                }
            }
            "--verify" => la.verify = true,
            "--plain" => la.output = OutputMode::Plain,
            "--json" => la.output = OutputMode::Json,
            "--color" => la.output = OutputMode::Color,
            other if !other.starts_with('-') => la.file = PathBuf::from(other),
            other => {
                eprintln!("error: unknown flag `{}` for `cjc lock`", other);
                process::exit(1);
            }
        }
        i += 1;
    }
    if la.file.as_os_str().is_empty() {
        eprintln!("error: `cjc lock` requires a .cjc file argument");
        process::exit(1);
    }
    la
}

/// Compute SHA-256 hex string of the given bytes.
fn sha256_hex(data: &[u8]) -> String {
    let hash = cjc_snap::hash::sha256(data);
    hash.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Parse a lockfile into its key-value pairs.
fn parse_lockfile(content: &str) -> std::collections::BTreeMap<String, String> {
    let mut map = std::collections::BTreeMap::new();
    for line in content.lines() {
        if let Some((key, val)) = line.split_once(": ") {
            map.insert(key.trim().to_string(), val.trim().to_string());
        }
    }
    map
}

/// Entry point for `cjc lock`.
pub fn run(args: &[String]) {
    let la = parse_args(args);

    let source = match fs::read_to_string(&la.file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: could not read `{}`: {}", la.file.display(), e);
            process::exit(1);
        }
    };

    let filename = la.file.display().to_string();
    let source_hash = sha256_hex(source.as_bytes());

    // Parse the source
    let (program, diags) = cjc_parser::parse_source(&source);
    if diags.has_errors() {
        eprintln!("error: parse errors in `{}`", filename);
        let rendered = diags.render_all_color(&source, &filename, la.output.use_color());
        eprint!("{}", rendered);
        process::exit(1);
    }

    // Execute with eval interpreter
    let mut interpreter = cjc_eval::Interpreter::new(la.seed);
    let exec_ok = match interpreter.exec(&program) {
        Ok(_) => true,
        Err(e) => {
            eprintln!("error: execution failed: {}", e);
            process::exit(1);
        }
    };
    let _ = exec_ok;

    let output_text = interpreter.output.join("\n");
    let output_hash = sha256_hex(output_text.as_bytes());

    let platform = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);
    let executor_str = "eval v0.1.0";

    if la.verify {
        run_verify(&la, &filename, &source_hash, &output_hash);
    } else {
        run_generate(&la, &filename, &source_hash, &output_hash, &platform, executor_str);
    }
}

fn run_generate(
    la: &LockArgs,
    filename: &str,
    source_hash: &str,
    output_hash: &str,
    platform: &str,
    executor: &str,
) {
    let lockfile_path = format!("{}.lock", la.file.display());

    let lockfile_content = format!(
        "source_sha256: {}\nseed: {}\nexecutor: {}\nplatform: {}\noutput_sha256: {}\n",
        source_hash, la.seed, executor, platform, output_hash
    );

    match fs::write(&lockfile_path, &lockfile_content) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("error: could not write lockfile `{}`: {}", lockfile_path, e);
            process::exit(1);
        }
    }

    match la.output {
        OutputMode::Json => {
            println!("{}", output::json_object(&[
                ("file", &filename.replace('\\', "/")),
                ("lockfile", &lockfile_path.replace('\\', "/")),
                ("source_sha256", source_hash),
                ("seed", &la.seed.to_string()),
                ("executor", executor),
                ("platform", platform),
                ("output_sha256", output_hash),
                ("status", "generated"),
            ]));
        }
        _ => {
            let label = output::colorize(la.output, output::BOLD_CYAN, "[lock]");
            eprintln!("{} Generated lockfile: {}", label, lockfile_path);

            let mut t = crate::table::Table::new(vec!["Field", "Value"]);
            t.add_row_owned(vec!["source_sha256".to_string(), source_hash.to_string()]);
            t.add_row_owned(vec!["seed".to_string(), la.seed.to_string()]);
            t.add_row_owned(vec!["executor".to_string(), executor.to_string()]);
            t.add_row_owned(vec!["platform".to_string(), platform.to_string()]);
            t.add_row_owned(vec!["output_sha256".to_string(), output_hash.to_string()]);
            eprint!("{}", t.render());
        }
    }
}

fn run_verify(
    la: &LockArgs,
    filename: &str,
    current_source_hash: &str,
    current_output_hash: &str,
) {
    let lockfile_path = format!("{}.lock", la.file.display());

    let lockfile_content = match fs::read_to_string(&lockfile_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: could not read lockfile `{}`: {}", lockfile_path, e);
            process::exit(1);
        }
    };

    let fields = parse_lockfile(&lockfile_content);

    let expected_source = fields.get("source_sha256").cloned().unwrap_or_default();
    let expected_output = fields.get("output_sha256").cloned().unwrap_or_default();
    let expected_seed = fields.get("seed").cloned().unwrap_or_default();

    let source_match = current_source_hash == expected_source;
    let output_match = current_output_hash == expected_output;
    let seed_match = la.seed.to_string() == expected_seed;
    let all_pass = source_match && output_match && seed_match;

    match la.output {
        OutputMode::Json => {
            println!("{}", output::json_object(&[
                ("file", &filename.replace('\\', "/")),
                ("lockfile", &lockfile_path.replace('\\', "/")),
                ("source_match", if source_match { "true" } else { "false" }),
                ("output_match", if output_match { "true" } else { "false" }),
                ("seed_match", if seed_match { "true" } else { "false" }),
                ("verdict", if all_pass { "MATCH" } else { "MISMATCH" }),
            ]));
        }
        _ => {
            let label = output::colorize(la.output, output::BOLD_CYAN, "[lock]");
            eprintln!("{} Verifying against {}", label, lockfile_path);
            eprintln!();

            let mut t = crate::table::Table::new(vec!["Check", "Expected", "Actual", "Status"]);
            t.add_row_owned(vec![
                "source_sha256".to_string(),
                short_hash(&expected_source),
                short_hash(current_source_hash),
                verdict_str(la.output, source_match),
            ]);
            t.add_row_owned(vec![
                "seed".to_string(),
                expected_seed.clone(),
                la.seed.to_string(),
                verdict_str(la.output, seed_match),
            ]);
            t.add_row_owned(vec![
                "output_sha256".to_string(),
                short_hash(&expected_output),
                short_hash(current_output_hash),
                verdict_str(la.output, output_match),
            ]);
            eprint!("{}", t.render());

            let verdict = if all_pass {
                output::colorize(la.output, output::BOLD_GREEN, "MATCH")
            } else {
                output::colorize(la.output, output::BOLD_RED, "MISMATCH")
            };
            eprintln!("\nVerdict: {}", verdict);
        }
    }

    if !all_pass {
        process::exit(1);
    }
}

/// Show first 16 hex chars of a hash for display.
fn short_hash(h: &str) -> String {
    if h.len() > 16 {
        format!("{}...", &h[..16])
    } else {
        h.to_string()
    }
}

fn verdict_str(mode: OutputMode, pass: bool) -> String {
    if pass {
        output::colorize(mode, output::BOLD_GREEN, "PASS")
    } else {
        output::colorize(mode, output::BOLD_RED, "FAIL")
    }
}

pub fn print_help() {
    eprintln!("cjc lock — Deterministic lockfile generator & verifier");
    eprintln!();
    eprintln!("Usage: cjc lock <file.cjc> [flags]");
    eprintln!();
    eprintln!("Generates a lockfile recording source hash, seed, executor version,");
    eprintln!("platform, and expected output hash. Use --verify to check against it.");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  --seed <N>          RNG seed (default: 42)");
    eprintln!("  --verify            Verify current run against existing lockfile");
    eprintln!("  --plain             Plain text output");
    eprintln!("  --json              JSON output");
    eprintln!("  --color             Color output (default)");
}
