//! `cjcl seek` — Deterministic file and data discovery.
//!
//! Recursively searches for files matching patterns with stable, sorted output.
//! Supports:
//! - Glob patterns (*.cjcl, **/*.snap)
//! - Content search within files (--contains)
//! - Type filtering (--type cjc, snap, csv)
//! - Size filtering (--min-size, --max-size)
//! - Exclude patterns (--exclude)
//! - Build artifact skipping (--ignore-build-artifacts)
//! - Result limiting (--first N)
//! - Sort modes (--sort size|name|modified)
//! - SHA-256 hashing (--hash)
//! - Manifest output (--manifest)
//! - Deterministic output ordering (always sorted by path)

use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::time::SystemTime;
use crate::output::{self, OutputMode};

/// Sort mode for seek results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortMode {
    Name,
    Size,
    Modified,
}

pub struct SeekArgs {
    pub root: PathBuf,
    pub pattern: Option<String>,
    pub file_type: Option<String>,
    pub contains: Option<String>,
    pub min_size: Option<u64>,
    pub max_size: Option<u64>,
    pub output: OutputMode,
    pub count_only: bool,
    pub max_depth: Option<usize>,
    pub exclude: Vec<String>,
    pub ignore_build_artifacts: bool,
    pub first: Option<usize>,
    pub sort: SortMode,
    pub hash: bool,
    pub manifest: bool,
}

impl Default for SeekArgs {
    fn default() -> Self {
        Self {
            root: PathBuf::from("."),
            pattern: None,
            file_type: None,
            contains: None,
            min_size: None,
            max_size: None,
            output: OutputMode::Color,
            count_only: false,
            max_depth: None,
            exclude: Vec::new(),
            ignore_build_artifacts: false,
            first: None,
            sort: SortMode::Name,
            hash: false,
            manifest: false,
        }
    }
}

pub fn parse_args(args: &[String]) -> SeekArgs {
    let mut sa = SeekArgs::default();
    let mut positionals = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--type" | "-t" => {
                i += 1;
                if i < args.len() { sa.file_type = Some(args[i].clone()); }
            }
            "--contains" | "--grep" => {
                i += 1;
                if i < args.len() { sa.contains = Some(args[i].clone()); }
            }
            "--min-size" => {
                i += 1;
                if i < args.len() { sa.min_size = Some(parse_size(&args[i])); }
            }
            "--max-size" => {
                i += 1;
                if i < args.len() { sa.max_size = Some(parse_size(&args[i])); }
            }
            "--max-depth" => {
                i += 1;
                if i < args.len() { sa.max_depth = args[i].parse().ok(); }
            }
            "--exclude" => {
                i += 1;
                if i < args.len() { sa.exclude.push(args[i].clone()); }
            }
            "--ignore-build-artifacts" => sa.ignore_build_artifacts = true,
            "--first" => {
                i += 1;
                if i < args.len() { sa.first = args[i].parse().ok(); }
            }
            "--sort" => {
                i += 1;
                if i < args.len() {
                    sa.sort = match args[i].as_str() {
                        "size" => SortMode::Size,
                        "modified" => SortMode::Modified,
                        "name" => SortMode::Name,
                        other => {
                            eprintln!("error: unknown sort mode `{}` (expected: name, size, modified)", other);
                            process::exit(1);
                        }
                    };
                }
            }
            "--hash" => sa.hash = true,
            "--manifest" => { sa.manifest = true; sa.hash = true; }
            "--count" => sa.count_only = true,
            "--plain" => sa.output = OutputMode::Plain,
            "--json" => sa.output = OutputMode::Json,
            "--color" => sa.output = OutputMode::Color,
            other if !other.starts_with('-') => positionals.push(other.to_string()),
            other => {
                eprintln!("error: unknown flag `{}` for `cjcl seek`", other);
                process::exit(1);
            }
        }
        i += 1;
    }
    // First positional is pattern or root
    if positionals.len() == 1 {
        // If it looks like a glob pattern, treat as pattern
        if positionals[0].contains('*') || positionals[0].contains('?') {
            sa.pattern = Some(positionals[0].clone());
        } else {
            sa.root = PathBuf::from(&positionals[0]);
        }
    } else if positionals.len() >= 2 {
        sa.root = PathBuf::from(&positionals[0]);
        sa.pattern = Some(positionals[1].clone());
    }
    sa
}

fn parse_size(s: &str) -> u64 {
    let s = s.trim().to_lowercase();
    if s.ends_with("kb") || s.ends_with("k") {
        let num: f64 = s.trim_end_matches(|c: char| c.is_alphabetic()).parse().unwrap_or(0.0);
        (num * 1024.0) as u64
    } else if s.ends_with("mb") || s.ends_with("m") {
        let num: f64 = s.trim_end_matches(|c: char| c.is_alphabetic()).parse().unwrap_or(0.0);
        (num * 1024.0 * 1024.0) as u64
    } else {
        s.parse().unwrap_or(0)
    }
}

/// Directories to skip when --ignore-build-artifacts is set.
const BUILD_ARTIFACT_DIRS: &[&str] = &["target", "node_modules", "__pycache__", ".git", "build"];

struct SeekResult {
    path: String,
    size: u64,
    modified: u64, // seconds since UNIX_EPOCH, 0 if unavailable
    match_line: Option<(usize, String)>, // line number + content for --contains
    hash: Option<String>, // SHA-256 hex, computed when --hash is set
}

pub fn run(args: &[String]) {
    let sa = parse_args(args);
    let mut results = Vec::new();

    seek_recursive(&sa.root, &sa, 0, &mut results);

    // Sort according to --sort mode (deterministic: secondary sort by path)
    match sa.sort {
        SortMode::Name => {
            results.sort_by(|a, b| a.path.cmp(&b.path));
        }
        SortMode::Size => {
            results.sort_by(|a, b| a.size.cmp(&b.size).then_with(|| a.path.cmp(&b.path)));
        }
        SortMode::Modified => {
            results.sort_by(|a, b| a.modified.cmp(&b.modified).then_with(|| a.path.cmp(&b.path)));
        }
    }

    // Apply --first limit BEFORE computing hashes (performance: only hash displayed files)
    if let Some(n) = sa.first {
        results.truncate(n);
    }

    // Compute hashes only for files that will be displayed
    if sa.hash {
        for r in results.iter_mut() {
            let file_path = r.path.replace('/', std::path::MAIN_SEPARATOR_STR);
            r.hash = fs::read(&file_path).ok().map(|data| {
                let h = cjc_snap::hash::sha256(&data);
                h.iter().map(|b| format!("{:02x}", b)).collect::<String>()
            });
        }
    }

    if sa.count_only {
        println!("{}", results.len());
        return;
    }

    // --manifest mode: <hash> <size> <path> per line
    if sa.manifest {
        for r in &results {
            let h = r.hash.as_deref().unwrap_or("????????????????????????????????????????????????????????????????");
            println!("{} {} {}", h, r.size, r.path);
        }
        return;
    }

    match sa.output {
        OutputMode::Json => {
            println!("[");
            for (i, r) in results.iter().enumerate() {
                print!("  {{\"path\": \"{}\"", r.path);
                print!(", \"size\": {}", r.size);
                if let Some(ref h) = r.hash {
                    print!(", \"hash\": \"{}\"", h);
                }
                if let Some((line, content)) = &r.match_line {
                    print!(", \"match_line\": {}, \"match_content\": \"{}\"",
                        line, content.replace('\\', "\\\\").replace('"', "\\\""));
                }
                print!("}}");
                if i + 1 < results.len() { print!(","); }
                println!();
            }
            println!("]");
        }
        _ => {
            for r in &results {
                let display = if sa.output.use_color() {
                    let ext = Path::new(&r.path).extension().and_then(|e| e.to_str()).unwrap_or("");
                    let color = match ext {
                        "cjcl" => output::GREEN,
                        "snap" => output::CYAN,
                        "csv" | "tsv" | "json" => output::YELLOW,
                        _ => "",
                    };
                    output::colorize(sa.output, color, &r.path)
                } else {
                    r.path.clone()
                };

                if let Some(ref h) = r.hash {
                    if let Some((line, content)) = &r.match_line {
                        println!("{} {}:{}:{}", &h[..16], display, line, content.trim());
                    } else {
                        println!("{} {}", &h[..16], display);
                    }
                } else if let Some((line, content)) = &r.match_line {
                    println!("{}:{}:{}", display, line, content.trim());
                } else {
                    println!("{}", display);
                }
            }
            eprintln!("{} files found", results.len());
        }
    }
}

fn seek_recursive(dir: &Path, sa: &SeekArgs, depth: usize, results: &mut Vec<SeekResult>) {
    if let Some(max) = sa.max_depth {
        if depth > max { return; }
    }

    let read_dir = match fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(_) => return,
    };

    let mut entries: Vec<_> = read_dir.filter_map(|e| e.ok()).collect();
    entries.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            let dir_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

            // --ignore-build-artifacts: skip known build directories
            if sa.ignore_build_artifacts && BUILD_ARTIFACT_DIRS.contains(&dir_name) {
                continue;
            }

            // --exclude: skip directories matching exclude patterns
            if sa.exclude.iter().any(|pat| glob_match(pat, dir_name)) {
                continue;
            }

            seek_recursive(&path, sa, depth + 1, results);
            continue;
        }

        let meta = entry.metadata().ok();
        let size = meta.as_ref().map(|m| m.len()).unwrap_or(0);

        let modified = meta.as_ref()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        // --exclude: skip files matching exclude patterns
        if sa.exclude.iter().any(|pat| glob_match(pat, file_name)) {
            continue;
        }

        // Filter by type
        if let Some(ref ft) = sa.file_type {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if ext != ft { continue; }
        }

        // Filter by pattern (simple glob)
        if let Some(ref pattern) = sa.pattern {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !glob_match(pattern, name) { continue; }
        }

        // Filter by size
        if let Some(min) = sa.min_size {
            if size < min { continue; }
        }
        if let Some(max) = sa.max_size {
            if size > max { continue; }
        }

        let display_path = path.to_string_lossy().to_string().replace('\\', "/");

        // Content search
        let match_line = if let Some(ref needle) = sa.contains {
            find_in_file(&path, needle)
        } else {
            None
        };

        // If --contains is specified, only include files with matches
        if sa.contains.is_some() && match_line.is_none() {
            continue;
        }

        // Hash is computed lazily after --first limit is applied (see run())
        results.push(SeekResult { path: display_path, size, modified, match_line, hash: None });
    }
}

/// Simple glob matching: supports * (any chars) and ? (single char).
fn glob_match(pattern: &str, name: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let n: Vec<char> = name.chars().collect();
    glob_match_inner(&p, &n, 0, 0)
}

fn glob_match_inner(p: &[char], n: &[char], pi: usize, ni: usize) -> bool {
    if pi == p.len() && ni == n.len() { return true; }
    if pi == p.len() { return false; }

    if p[pi] == '*' {
        // Try matching zero or more characters
        for skip in 0..=(n.len() - ni) {
            if glob_match_inner(p, n, pi + 1, ni + skip) {
                return true;
            }
        }
        false
    } else if ni < n.len() && (p[pi] == '?' || p[pi] == n[ni]) {
        glob_match_inner(p, n, pi + 1, ni + 1)
    } else {
        false
    }
}

/// Find the first occurrence of `needle` in a file, return (line_number, line_content).
fn find_in_file(path: &Path, needle: &str) -> Option<(usize, String)> {
    let content = fs::read_to_string(path).ok()?;
    for (i, line) in content.lines().enumerate() {
        if line.contains(needle) {
            return Some((i + 1, line.to_string()));
        }
    }
    None
}

pub fn print_help() {
    eprintln!("cjcl seek — Deterministic file and data discovery");
    eprintln!();
    eprintln!("Usage: cjcl seek [path] [pattern] [flags]");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  -t, --type <ext>       Filter by extension (cjc, snap, csv, etc.)");
    eprintln!("  --contains <text>      Search file contents");
    eprintln!("  --min-size <size>      Minimum file size (e.g., 1kb, 10mb)");
    eprintln!("  --max-size <size>      Maximum file size");
    eprintln!("  --max-depth <N>        Maximum recursion depth");
    eprintln!("  --exclude <glob>       Exclude files/dirs matching glob (repeatable)");
    eprintln!("  --ignore-build-artifacts  Skip target/, node_modules/, __pycache__/, .git/, build/");
    eprintln!("  --first <N>            Only return the first N results");
    eprintln!("  --sort <mode>          Sort by: name (default), size, modified");
    eprintln!("  --hash                 Compute and display SHA-256 hash per file");
    eprintln!("  --manifest             Output manifest format: <hash> <size> <path>");
    eprintln!("  --count                Only print match count");
    eprintln!("  --plain                Plain text output");
    eprintln!("  --json                 JSON output");
    eprintln!("  --color                Color output (default)");
}
