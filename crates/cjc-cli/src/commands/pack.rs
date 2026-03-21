//! `cjc pack` — Reproducible packaging.
//!
//! Bundles a CJC script and its referenced artifacts (.snap files, data files)
//! into a self-contained package directory with a deterministic manifest.
//!
//! Package identity is stable when inputs are unchanged — same files produce
//! the same manifest hash.

use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use crate::output::{self, OutputMode};

pub struct PackArgs {
    pub file: String,
    pub output_dir: Option<String>,
    pub include: Vec<String>,
    pub output: OutputMode,
    pub dry_run: bool,
}

impl Default for PackArgs {
    fn default() -> Self {
        Self {
            file: String::new(),
            output_dir: None,
            include: Vec::new(),
            output: OutputMode::Color,
            dry_run: false,
        }
    }
}

pub fn parse_args(args: &[String]) -> PackArgs {
    let mut pa = PackArgs::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                i += 1;
                if i < args.len() { pa.output_dir = Some(args[i].clone()); }
            }
            "--include" => {
                i += 1;
                if i < args.len() { pa.include.push(args[i].clone()); }
            }
            "--dry-run" => pa.dry_run = true,
            "--plain" => pa.output = OutputMode::Plain,
            "--json" => pa.output = OutputMode::Json,
            "--color" => pa.output = OutputMode::Color,
            other if !other.starts_with('-') => pa.file = other.to_string(),
            other => {
                eprintln!("error: unknown flag `{}` for `cjc pack`", other);
                process::exit(1);
            }
        }
        i += 1;
    }
    if pa.file.is_empty() {
        eprintln!("error: `cjc pack` requires a .cjc file argument");
        process::exit(1);
    }
    pa
}

struct PackEntry {
    rel_path: String,
    size: u64,
    hash: String,
}

pub fn run(args: &[String]) {
    let pa = parse_args(args);
    let entry_path = Path::new(&pa.file);

    if !entry_path.exists() {
        eprintln!("error: file `{}` not found", pa.file);
        process::exit(1);
    }

    let filename = pa.file.replace('\\', "/");
    let stem = entry_path.file_stem().and_then(|s| s.to_str()).unwrap_or("package");
    let output_dir = pa.output_dir.clone().unwrap_or_else(|| format!("{}.pack", stem));

    // Discover files to include
    let mut entries: Vec<PackEntry> = Vec::new();

    // Always include the main script
    add_entry(&mut entries, entry_path, entry_path.parent().unwrap_or(Path::new(".")));

    // Auto-discover: scan source for snap_load/snap_save references
    if let Ok(source) = fs::read_to_string(entry_path) {
        for line in source.lines() {
            // Look for string literals that might be file paths
            for quote_char in ['"', '\''] {
                let mut rest = line;
                while let Some(start) = rest.find(quote_char) {
                    let after = &rest[start + 1..];
                    if let Some(end) = after.find(quote_char) {
                        let literal = &after[..end];
                        // Check if it looks like a referenced file
                        if (literal.ends_with(".snap") || literal.ends_with(".csv")
                            || literal.ends_with(".tsv") || literal.ends_with(".json"))
                        {
                            let ref_path = entry_path.parent()
                                .unwrap_or(Path::new("."))
                                .join(literal);
                            if ref_path.exists() {
                                add_entry(&mut entries, &ref_path,
                                    entry_path.parent().unwrap_or(Path::new(".")));
                            }
                        }
                        rest = &after[end + 1..];
                    } else {
                        break;
                    }
                }
            }
        }
    }

    // Include explicitly specified files
    for inc in &pa.include {
        let inc_path = Path::new(inc);
        if inc_path.exists() {
            add_entry(&mut entries, inc_path, entry_path.parent().unwrap_or(Path::new(".")));
        } else {
            eprintln!("warning: --include file `{}` not found, skipping", inc);
        }
    }

    // Deterministic sort
    entries.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));

    // Deduplicate
    entries.dedup_by(|a, b| a.rel_path == b.rel_path);

    // Compute manifest hash (hash of all entry hashes concatenated)
    let manifest_input: String = entries.iter()
        .map(|e| format!("{}:{}", e.rel_path, e.hash))
        .collect::<Vec<_>>()
        .join("\n");
    let manifest_hash = cjc_snap::hash::sha256(manifest_input.as_bytes());
    let manifest_hash_hex = manifest_hash.iter().map(|b| format!("{:02x}", b)).collect::<String>();

    let total_size: u64 = entries.iter().map(|e| e.size).sum();

    if pa.dry_run {
        // Just report what would be packed
        match pa.output {
            OutputMode::Json => {
                println!("{{");
                println!("  \"entry\": \"{}\",", filename);
                println!("  \"output_dir\": \"{}\",", output_dir.replace('\\', "/"));
                println!("  \"manifest_hash\": \"{}\",", manifest_hash_hex);
                println!("  \"total_size\": {},", total_size);
                println!("  \"files\": [");
                for (i, e) in entries.iter().enumerate() {
                    print!("    {{\"path\": \"{}\", \"size\": {}, \"hash\": \"{}\"}}",
                        e.rel_path, e.size, &e.hash[..16]);
                    if i + 1 < entries.len() { print!(","); }
                    println!();
                }
                println!("  ]");
                println!("}}");
            }
            _ => {
                eprintln!("Dry run — would create `{}`:", output_dir.replace('\\', "/"));
                eprintln!();
                let mut t = crate::table::Table::new(vec!["File", "Size", "Hash"]);
                for e in &entries {
                    t.add_row_owned(vec![e.rel_path.clone(), output::format_size(e.size), e.hash[..16].to_string()]);
                }
                eprint!("{}", t.render());
                eprintln!("Total: {} files, {}", entries.len(), output::format_size(total_size));
                eprintln!("Manifest hash: {}", &manifest_hash_hex[..16]);
            }
        }
        return;
    }

    // Create package directory
    let out_path = PathBuf::from(&output_dir);
    if let Err(e) = fs::create_dir_all(&out_path) {
        eprintln!("error: could not create `{}`: {}", output_dir, e);
        process::exit(1);
    }

    // Copy files
    for entry in &entries {
        let dest = out_path.join(&entry.rel_path);
        if let Some(parent) = dest.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let src = Path::new(&pa.file).parent().unwrap_or(Path::new(".")).join(&entry.rel_path);
        if let Err(e) = fs::copy(&src, &dest) {
            eprintln!("warning: could not copy `{}`: {}", entry.rel_path, e);
        }
    }

    // Write manifest
    let mut manifest = String::new();
    manifest.push_str(&format!("# CJC Package Manifest\n"));
    manifest.push_str(&format!("entry: {}\n", filename));
    manifest.push_str(&format!("manifest_hash: {}\n", manifest_hash_hex));
    manifest.push_str(&format!("files: {}\n", entries.len()));
    manifest.push_str(&format!("total_size: {}\n", total_size));
    manifest.push_str("---\n");
    for e in &entries {
        manifest.push_str(&format!("{} {} {}\n", e.rel_path, e.size, e.hash));
    }

    let manifest_path = out_path.join("MANIFEST");
    if let Err(e) = fs::write(&manifest_path, &manifest) {
        eprintln!("error: could not write manifest: {}", e);
        process::exit(1);
    }

    match pa.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"action\": \"pack\",");
            println!("  \"entry\": \"{}\",", filename);
            println!("  \"output_dir\": \"{}\",", output_dir.replace('\\', "/"));
            println!("  \"files\": {},", entries.len());
            println!("  \"total_size\": {},", total_size);
            println!("  \"manifest_hash\": \"{}\"", manifest_hash_hex);
            println!("}}");
        }
        _ => {
            let mut t = crate::table::Table::new(vec!["File", "Size"]);
            for e in &entries {
                t.add_row_owned(vec![e.rel_path.clone(), output::format_size(e.size)]);
            }
            eprint!("{}", t.render());
            eprintln!("{} — packed {} files ({}) into `{}`",
                output::colorize(pa.output, output::BOLD_GREEN, "packed"),
                entries.len(), output::format_size(total_size),
                output_dir.replace('\\', "/"));
            eprintln!("Manifest: {}", &manifest_hash_hex[..16]);
        }
    }
}

fn add_entry(entries: &mut Vec<PackEntry>, path: &Path, base: &Path) {
    let data = match fs::read(path) {
        Ok(d) => d,
        Err(_) => return,
    };
    let hash = cjc_snap::hash::sha256(&data);
    let hash_hex = hash.iter().map(|b| format!("{:02x}", b)).collect::<String>();

    let rel_path = path.strip_prefix(base)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string()
        .replace('\\', "/");

    entries.push(PackEntry {
        rel_path,
        size: data.len() as u64,
        hash: hash_hex,
    });
}

pub fn print_help() {
    eprintln!("cjc pack — Reproducible packaging");
    eprintln!();
    eprintln!("Usage: cjc pack <file.cjc> [flags]");
    eprintln!();
    eprintln!("Bundles a script and its referenced artifacts into a package.");
    eprintln!("Auto-discovers .snap, .csv, .tsv, .json references in source.");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  -o, --output <dir>    Output directory (default: <name>.pack/)");
    eprintln!("  --include <file>      Additional file to include (repeatable)");
    eprintln!("  --dry-run             Show what would be packed without writing");
    eprintln!("  --plain               Plain text output");
    eprintln!("  --json                JSON output");
    eprintln!("  --color               Color output (default)");
}
