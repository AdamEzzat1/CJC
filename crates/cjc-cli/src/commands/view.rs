//! `cjcl view` — Deterministic, effect-aware directory listing.
//!
//! Lists files with stable ordering, color-coded by type, and annotates:
//! - `.cjcl` files with effect flags (nogc, pure, etc.)
//! - `.snap` files with content hashes
//! - Directories, executables, and data files
//!
//! Output is always sorted lexicographically for determinism.

use std::fs;
use std::path::{Path, PathBuf};
use crate::output::{self, OutputMode};

/// Parsed arguments for `cjcl view`.
pub struct ViewArgs {
    pub path: PathBuf,
    pub recursive: bool,
    pub no_header: bool,
    pub output: OutputMode,
    pub show_hash: bool,
    pub show_effects: bool,
    pub show_size: bool,
}

impl Default for ViewArgs {
    fn default() -> Self {
        Self {
            path: PathBuf::from("."),
            recursive: false,
            no_header: false,
            output: OutputMode::Color,
            show_hash: false,
            show_effects: true,
            show_size: true,
        }
    }
}

pub fn parse_args(args: &[String]) -> ViewArgs {
    let mut va = ViewArgs::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-r" | "--recursive" => va.recursive = true,
            "--no-header" => va.no_header = true,
            "--hash" => va.show_hash = true,
            "--no-hash" => va.show_hash = false,
            "--no-effects" => va.show_effects = false,
            "--no-size" => va.show_size = false,
            "--plain" => va.output = OutputMode::Plain,
            "--json" => va.output = OutputMode::Json,
            "--table" => va.output = OutputMode::Table,
            "--output" => {
                i += 1;
                if i < args.len() {
                    va.output = OutputMode::from_str(&args[i]).unwrap_or_else(|e| {
                        eprintln!("error: {}", e);
                        std::process::exit(1);
                    });
                }
            }
            "--color" => va.output = OutputMode::Color,
            other if !other.starts_with('-') => va.path = PathBuf::from(other),
            other => {
                eprintln!("error: unknown flag `{}` for `cjcl view`", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }
    va
}

/// Entry point for `cjcl view`.
pub fn run(args: &[String]) {
    let va = parse_args(args);
    let entries = collect_entries(&va.path, va.recursive);

    if entries.is_empty() {
        if va.output == OutputMode::Json {
            println!("[]");
        } else {
            eprintln!("No files found in `{}`", va.path.display());
        }
        return;
    }

    match va.output {
        OutputMode::Json => print_json(&entries, &va),
        OutputMode::Table | OutputMode::Plain | OutputMode::Color => print_table(&entries, &va),
    }
}

/// File entry with metadata extracted deterministically.
struct FileEntry {
    path: String,
    kind: FileKind,
    size: u64,
    effects: Vec<String>,
    snap_hash: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileKind {
    Directory,
    CjcSource,
    SnapFile,
    DataFile,
    Other,
}

impl FileKind {
    fn label(&self) -> &'static str {
        match self {
            FileKind::Directory => "dir",
            FileKind::CjcSource => "cjcl",
            FileKind::SnapFile => "snap",
            FileKind::DataFile => "data",
            FileKind::Other => "file",
        }
    }

    fn color(&self) -> &'static str {
        match self {
            FileKind::Directory => output::BOLD_BLUE,
            FileKind::CjcSource => output::BOLD_GREEN,
            FileKind::SnapFile => output::BOLD_CYAN,
            FileKind::DataFile => output::BOLD_YELLOW,
            FileKind::Other => "",
        }
    }
}

fn classify(path: &Path) -> FileKind {
    if path.is_dir() {
        return FileKind::Directory;
    }
    match path.extension().and_then(|e| e.to_str()) {
        Some("cjcl") => FileKind::CjcSource,
        Some("snap") => FileKind::SnapFile,
        Some("csv") | Some("tsv") | Some("json") => FileKind::DataFile,
        _ => FileKind::Other,
    }
}

/// Collect file entries with deterministic (sorted) ordering.
fn collect_entries(root: &Path, recursive: bool) -> Vec<FileEntry> {
    let mut entries = Vec::new();
    collect_recursive(root, recursive, &mut entries);
    // Deterministic sort: by path, lexicographic
    entries.sort_by(|a, b| a.path.cmp(&b.path));
    entries
}

fn collect_recursive(dir: &Path, recursive: bool, entries: &mut Vec<FileEntry>) {
    let read_dir = match fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) => {
            eprintln!("warning: cannot read `{}`: {}", dir.display(), e);
            return;
        }
    };

    // Collect and sort directory entries for determinism
    let mut dir_entries: Vec<_> = read_dir.filter_map(|e| e.ok()).collect();
    dir_entries.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

    for entry in dir_entries {
        let path = entry.path();
        let kind = classify(&path);
        let meta = entry.metadata().ok();
        let size = meta.as_ref().map(|m| m.len()).unwrap_or(0);

        let display_path = path.to_string_lossy().to_string()
            .replace('\\', "/"); // Normalize path separators

        let effects = if kind == FileKind::CjcSource {
            extract_effects(&path)
        } else {
            Vec::new()
        };

        let snap_hash = if kind == FileKind::SnapFile {
            extract_snap_hash(&path)
        } else {
            None
        };

        entries.push(FileEntry {
            path: display_path,
            kind,
            size,
            effects,
            snap_hash,
        });

        if recursive && kind == FileKind::Directory {
            collect_recursive(&path, true, entries);
        }
    }
}

/// Extract effect annotations from a .cjcl file by scanning for fn declarations.
fn extract_effects(path: &Path) -> Vec<String> {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };

    let mut effects = std::collections::BTreeSet::new();

    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("@nogc") || trimmed.contains("/nogc") {
            effects.insert("nogc".to_string());
        }
        if trimmed.starts_with("@pure") || trimmed.contains("/pure") {
            effects.insert("pure".to_string());
        }
        if trimmed.starts_with("@memoize") {
            effects.insert("memoize".to_string());
        }
        if trimmed.starts_with("@timed") {
            effects.insert("timed".to_string());
        }
        if trimmed.starts_with("@trace") || trimmed.starts_with("@log") {
            effects.insert("traced".to_string());
        }
    }

    effects.into_iter().collect()
}

/// Extract content hash from a .snap file header.
fn extract_snap_hash(path: &Path) -> Option<String> {
    let data = fs::read(path).ok()?;
    if data.len() < 32 {
        return None;
    }
    // Snap files store a 32-byte SHA-256 hash at the start
    Some(output::format_hash_short(&data[..32]))
}

fn print_table(entries: &[FileEntry], va: &ViewArgs) {
    let mode = va.output;

    if !va.no_header {
        let header = format!("  {} ({} entries)",
            output::colorize(mode, output::BOLD, &va.path.display().to_string()),
            entries.len()
        );
        eprintln!("{}", header);
        eprintln!();
    }

    // Build table
    let mut headers = vec!["Name", "Kind"];
    if va.show_size { headers.push("Size"); }
    if va.show_effects { headers.push("Effects"); }
    if va.show_hash { headers.push("Hash"); }

    let mut t = crate::table::Table::new(headers);

    for entry in entries {
        let name = if mode.use_color() {
            output::colorize(mode, entry.kind.color(), &entry.path)
        } else {
            entry.path.clone()
        };

        let mut row = vec![
            name,
            entry.kind.label().to_string(),
        ];

        if va.show_size {
            row.push(if entry.kind == FileKind::Directory {
                "-".to_string()
            } else {
                output::format_size(entry.size)
            });
        }

        if va.show_effects {
            row.push(if entry.effects.is_empty() {
                "-".to_string()
            } else {
                entry.effects.join(", ")
            });
        }

        if va.show_hash {
            row.push(entry.snap_hash.clone().unwrap_or_else(|| "-".to_string()));
        }

        t.add_row_owned(row);
    }

    print!("{}", t.render());
}

fn print_json(entries: &[FileEntry], va: &ViewArgs) {
    println!("[");
    for (i, entry) in entries.iter().enumerate() {
        let effects_json = format!("[{}]",
            entry.effects.iter()
                .map(|e| format!("\"{}\"", e))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let hash_json = entry.snap_hash.as_deref().unwrap_or("null");
        let hash_val = if hash_json == "null" {
            "null".to_string()
        } else {
            format!("\"{}\"", hash_json)
        };

        print!("  {{");
        print!("\"path\": \"{}\", ", entry.path.replace('\\', "/"));
        print!("\"kind\": \"{}\", ", entry.kind.label());
        if va.show_size { print!("\"size\": {}, ", entry.size); }
        if va.show_effects { print!("\"effects\": {}, ", effects_json); }
        if va.show_hash { print!("\"hash\": {}", hash_val); }
        print!("}}");
        if i + 1 < entries.len() { print!(","); }
        println!();
    }
    println!("]");
}

pub fn print_help() {
    eprintln!("cjcl view — Deterministic, effect-aware directory listing");
    eprintln!();
    eprintln!("Usage: cjcl view [path] [flags]");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  -r, --recursive     Recurse into subdirectories");
    eprintln!("  --no-header         Suppress header line");
    eprintln!("  --no-hash           Hide snap file hashes");
    eprintln!("  --no-effects        Hide effect annotations");
    eprintln!("  --no-size           Hide file sizes");
    eprintln!("  --plain             Plain text output (no color)");
    eprintln!("  --json              JSON output");
    eprintln!("  --table             Table output");
    eprintln!("  --color             Color output (default)");
    eprintln!("  --output <mode>     Output mode: color, plain, json, table");
}
