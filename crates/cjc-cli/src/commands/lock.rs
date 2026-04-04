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
    pub update: bool,
    pub show: bool,
    pub diff: bool,
    pub executor: LockExecutor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockExecutor {
    Eval,
    Mir,
}

impl Default for LockArgs {
    fn default() -> Self {
        Self {
            file: PathBuf::new(),
            seed: 42,
            verify: false,
            output: OutputMode::Color,
            update: false,
            show: false,
            diff: false,
            executor: LockExecutor::Eval,
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
            "--update" => la.update = true,
            "--show" => la.show = true,
            "--diff" => la.diff = true,
            "--executor" => {
                i += 1;
                if i < args.len() {
                    match args[i].as_str() {
                        "eval" => la.executor = LockExecutor::Eval,
                        "mir" => la.executor = LockExecutor::Mir,
                        other => {
                            eprintln!("error: --executor expects `eval` or `mir`, got `{}`", other);
                            process::exit(1);
                        }
                    }
                } else {
                    eprintln!("error: --executor requires an argument (eval or mir)");
                    process::exit(1);
                }
            }
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

/// Execute the program with the selected executor and return (output_text, executor_label).
fn execute_program(program: &cjc_ast::Program, seed: u64, executor: LockExecutor) -> (String, &'static str) {
    match executor {
        LockExecutor::Eval => {
            let mut interpreter = cjc_eval::Interpreter::new(seed);
            match interpreter.exec(program) {
                Ok(_) => {}
                Err(e) => {
                    eprintln!("error: execution failed: {}", e);
                    process::exit(1);
                }
            }
            let output_text = interpreter.output.join("\n");
            (output_text, "eval v0.1.0")
        }
        LockExecutor::Mir => {
            match cjc_mir_exec::run_program_with_executor(program, seed) {
                Ok((_val, exec)) => {
                    let output_text = exec.output.join("\n");
                    (output_text, "mir-exec v0.1.0")
                }
                Err(e) => {
                    eprintln!("error: MIR execution failed: {}", e);
                    process::exit(1);
                }
            }
        }
    }
}

/// Entry point for `cjc lock`.
pub fn run(args: &[String]) {
    let la = parse_args(args);

    // --show: display lockfile contents without executing
    if la.show {
        run_show(&la);
        return;
    }

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

    // Execute with selected executor
    let (output_text, executor_str) = execute_program(&program, la.seed, la.executor);
    let output_hash = sha256_hex(output_text.as_bytes());

    let platform = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);

    if la.diff {
        run_diff(&la, &filename, &source_hash, &output_hash, &platform, executor_str);
    } else if la.verify {
        run_verify(&la, &filename, &source_hash, &output_hash);
    } else {
        // --update or default generate
        run_generate(&la, &filename, &source_hash, &output_hash, &platform, executor_str);
    }
}

fn run_show(la: &LockArgs) {
    let lockfile_path = format!("{}.lock", la.file.display());

    let lockfile_content = match fs::read_to_string(&lockfile_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: could not read lockfile `{}`: {}", lockfile_path, e);
            process::exit(1);
        }
    };

    let fields = parse_lockfile(&lockfile_content);

    match la.output {
        OutputMode::Json => {
            println!("{{");
            let entries: Vec<_> = fields.iter().collect();
            for (i, (k, v)) in entries.iter().enumerate() {
                print!("  \"{}\": \"{}\"", k, v);
                if i + 1 < entries.len() { print!(","); }
                println!();
            }
            println!("}}");
        }
        _ => {
            let label = output::colorize(la.output, output::BOLD_CYAN, "[lock]");
            eprintln!("{} Contents of {}", label, lockfile_path);
            eprintln!();

            let mut t = crate::table::Table::new(vec!["Field", "Value"]);
            for (k, v) in &fields {
                t.add_row_owned(vec![k.clone(), v.clone()]);
            }
            eprint!("{}", t.render());
        }
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

    // If --update is not set and lockfile already exists, generate anyway (original behavior).
    // --update explicitly means "overwrite existing".

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
                ("status", if la.update { "updated" } else { "generated" }),
            ]));
        }
        _ => {
            let label = output::colorize(la.output, output::BOLD_CYAN, "[lock]");
            let action = if la.update { "Updated" } else { "Generated" };
            eprintln!("{} {} lockfile: {}", label, action, lockfile_path);

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

fn run_diff(
    la: &LockArgs,
    filename: &str,
    current_source_hash: &str,
    current_output_hash: &str,
    current_platform: &str,
    current_executor: &str,
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

    let locked_source = fields.get("source_sha256").cloned().unwrap_or_default();
    let locked_output = fields.get("output_sha256").cloned().unwrap_or_default();
    let locked_seed = fields.get("seed").cloned().unwrap_or_default();
    let locked_executor = fields.get("executor").cloned().unwrap_or_default();
    let locked_platform = fields.get("platform").cloned().unwrap_or_default();

    match la.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"file\": \"{}\",", filename.replace('\\', "/"));
            println!("  \"lockfile\": \"{}\",", lockfile_path.replace('\\', "/"));
            println!("  \"diffs\": {{");
            let mut diffs = Vec::new();
            if locked_source != current_source_hash {
                diffs.push(format!("    \"source_sha256\": {{\"locked\": \"{}\", \"current\": \"{}\"}}", locked_source, current_source_hash));
            }
            if locked_seed != la.seed.to_string() {
                diffs.push(format!("    \"seed\": {{\"locked\": \"{}\", \"current\": \"{}\"}}", locked_seed, la.seed));
            }
            if locked_executor != current_executor {
                diffs.push(format!("    \"executor\": {{\"locked\": \"{}\", \"current\": \"{}\"}}", locked_executor, current_executor));
            }
            if locked_platform != current_platform {
                diffs.push(format!("    \"platform\": {{\"locked\": \"{}\", \"current\": \"{}\"}}", locked_platform, current_platform));
            }
            if locked_output != current_output_hash {
                diffs.push(format!("    \"output_sha256\": {{\"locked\": \"{}\", \"current\": \"{}\"}}", locked_output, current_output_hash));
            }
            for (i, d) in diffs.iter().enumerate() {
                print!("{}", d);
                if i + 1 < diffs.len() { print!(","); }
                println!();
            }
            println!("  }}");
            println!("}}");
        }
        _ => {
            let label = output::colorize(la.output, output::BOLD_CYAN, "[lock]");
            eprintln!("{} Diff against {}", label, lockfile_path);
            eprintln!();

            let mut t = crate::table::Table::new(vec!["Field", "Locked", "Current", "Status"]);

            let seed_str = la.seed.to_string();
            let fields_to_check: Vec<(&str, &str, &str)> = vec![
                ("source_sha256", &locked_source, current_source_hash),
                ("seed", &locked_seed, &seed_str),
                ("executor", &locked_executor, current_executor),
                ("platform", &locked_platform, current_platform),
                ("output_sha256", &locked_output, current_output_hash),
            ];

            let mut any_diff = false;
            for (name, locked, current) in &fields_to_check {
                let is_hash = name.contains("sha256");
                let locked_display = if is_hash { short_hash(locked) } else { locked.to_string() };
                let current_display = if is_hash { short_hash(current) } else { current.to_string() };
                let matches = *locked == *current;
                if !matches { any_diff = true; }
                t.add_row_owned(vec![
                    name.to_string(),
                    locked_display,
                    current_display,
                    if matches {
                        output::colorize(la.output, output::BOLD_GREEN, "same")
                    } else {
                        output::colorize(la.output, output::BOLD_YELLOW, "CHANGED")
                    },
                ]);
            }
            eprint!("{}", t.render());

            if !any_diff {
                eprintln!("\nNo differences found.");
            }
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
    eprintln!("  --seed <N>            RNG seed (default: 42)");
    eprintln!("  --verify              Verify current run against existing lockfile");
    eprintln!("  --update              Regenerate the lockfile (overwrite existing)");
    eprintln!("  --show                Display the contents of an existing lockfile");
    eprintln!("  --diff                Show differences between current run and lockfile");
    eprintln!("  --executor eval|mir   Specify which executor to use (default: eval)");
    eprintln!("  --plain               Plain text output");
    eprintln!("  --json                JSON output");
    eprintln!("  --color               Color output (default)");
}
