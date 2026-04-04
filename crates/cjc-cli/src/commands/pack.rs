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
    pub verify: bool,
    pub manifest_only: bool,
    pub list: bool,
    pub repro_check: bool,
}

impl Default for PackArgs {
    fn default() -> Self {
        Self {
            file: String::new(),
            output_dir: None,
            include: Vec::new(),
            output: OutputMode::Color,
            dry_run: false,
            verify: false,
            manifest_only: false,
            list: false,
            repro_check: false,
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
            "--verify" => pa.verify = true,
            "--manifest-only" => pa.manifest_only = true,
            "--list" => pa.list = true,
            "--repro-check" => pa.repro_check = true,
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
    // --list operates on an existing .pack directory, file can be the dir itself
    if pa.list {
        if pa.file.is_empty() {
            eprintln!("error: `cjc pack --list` requires a .pack directory argument");
            process::exit(1);
        }
        return pa;
    }
    // --repro-check operates on an existing .pack directory
    if pa.repro_check {
        if pa.file.is_empty() {
            eprintln!("error: `cjc pack --repro-check` requires a .pack directory argument");
            process::exit(1);
        }
        return pa;
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

    // --list: list contents of an existing .pack directory
    if pa.list {
        run_list(&pa);
        return;
    }

    // --repro-check: re-pack and compare manifest hash
    if pa.repro_check {
        run_repro_check(&pa);
        return;
    }

    let entry_path = Path::new(&pa.file);

    if !entry_path.exists() {
        eprintln!("error: file or directory `{}` not found", pa.file);
        process::exit(1);
    }

    let filename = pa.file.replace('\\', "/");
    let stem = entry_path.file_stem().and_then(|s| s.to_str()).unwrap_or("package");
    let output_dir = pa.output_dir.clone().unwrap_or_else(|| format!("{}.pack", stem));

    // Discover files to include
    let mut entries: Vec<PackEntry> = Vec::new();

    if entry_path.is_dir() {
        // Directory mode: walk and include all packable files
        discover_dir_files(entry_path, entry_path, &mut entries);
    } else {
        // File mode: include the main script
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
                            if literal.ends_with(".snap") || literal.ends_with(".csv")
                                || literal.ends_with(".tsv") || literal.ends_with(".json")
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
    }

    // Include explicitly specified files
    for inc in &pa.include {
        let inc_path = Path::new(inc);
        if inc_path.exists() {
            if inc_path.is_dir() {
                discover_dir_files(inc_path, inc_path, &mut entries);
            } else {
                add_entry(&mut entries, inc_path, entry_path.parent().unwrap_or(Path::new(".")));
            }
        } else {
            eprintln!("warning: --include file `{}` not found, skipping", inc);
        }
    }

    // Warn if no files were discovered
    if entries.is_empty() {
        eprintln!("warning: no packable files found in `{}`", pa.file);
    }

    // Deterministic sort
    entries.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));

    // Deduplicate
    entries.dedup_by(|a, b| a.rel_path == b.rel_path);

    // Compute manifest hash (hash of all entry hashes concatenated)
    let manifest_hash_hex = compute_manifest_hash(&entries);

    let total_size: u64 = entries.iter().map(|e| e.size).sum();

    // --manifest-only: output MANIFEST content and exit
    if pa.manifest_only {
        let manifest = build_manifest_string(&entries, &filename, &manifest_hash_hex, total_size);
        print!("{}", manifest);
        return;
    }

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
    let base_dir = if entry_path.is_dir() {
        entry_path.to_path_buf()
    } else {
        entry_path.parent().unwrap_or(Path::new(".")).to_path_buf()
    };
    for entry in &entries {
        let dest = out_path.join(&entry.rel_path);
        if let Some(parent) = dest.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let src = base_dir.join(&entry.rel_path);
        if let Err(e) = fs::copy(&src, &dest) {
            eprintln!("warning: could not copy `{}`: {}", entry.rel_path, e);
        }
    }

    // Write manifest
    let manifest = build_manifest_string(&entries, &filename, &manifest_hash_hex, total_size);

    let manifest_path = out_path.join("MANIFEST");
    if let Err(e) = fs::write(&manifest_path, &manifest) {
        eprintln!("error: could not write manifest: {}", e);
        process::exit(1);
    }

    // --verify: re-read the packed directory and verify manifest hash matches
    if pa.verify {
        let verified = verify_pack(&out_path, &manifest_hash_hex);
        if !verified {
            eprintln!("error: verification failed — manifest hash mismatch after packing");
            process::exit(1);
        }
    }

    match pa.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"action\": \"pack\",");
            println!("  \"entry\": \"{}\",", filename);
            println!("  \"output_dir\": \"{}\",", output_dir.replace('\\', "/"));
            println!("  \"files\": {},", entries.len());
            println!("  \"total_size\": {},", total_size);
            println!("  \"manifest_hash\": \"{}\",", manifest_hash_hex);
            if pa.verify {
                println!("  \"verified\": true");
            }
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
            if pa.verify {
                eprintln!("{}", output::colorize(pa.output, output::BOLD_GREEN, "verified"));
            }
        }
    }
}

/// Build the MANIFEST file content string.
fn build_manifest_string(entries: &[PackEntry], filename: &str, manifest_hash_hex: &str, total_size: u64) -> String {
    let mut manifest = String::new();
    manifest.push_str(&format!("# CJC Package Manifest\n"));
    manifest.push_str(&format!("entry: {}\n", filename));
    manifest.push_str(&format!("manifest_hash: {}\n", manifest_hash_hex));
    manifest.push_str(&format!("files: {}\n", entries.len()));
    manifest.push_str(&format!("total_size: {}\n", total_size));
    manifest.push_str("---\n");
    for e in entries {
        manifest.push_str(&format!("{} {} {}\n", e.rel_path, e.size, e.hash));
    }
    manifest
}

/// Compute the manifest hash from sorted entries.
fn compute_manifest_hash(entries: &[PackEntry]) -> String {
    let manifest_input: String = entries.iter()
        .map(|e| format!("{}:{}", e.rel_path, e.hash))
        .collect::<Vec<_>>()
        .join("\n");
    let manifest_hash = cjc_snap::hash::sha256(manifest_input.as_bytes());
    manifest_hash.iter().map(|b| format!("{:02x}", b)).collect::<String>()
}

/// Verify a packed directory by re-reading all files and comparing manifest hash.
fn verify_pack(pack_dir: &Path, expected_hash: &str) -> bool {
    let manifest_path = pack_dir.join("MANIFEST");
    let manifest_content = match fs::read_to_string(&manifest_path) {
        Ok(c) => c,
        Err(_) => return false,
    };

    // Parse file entries from manifest (lines after "---")
    let mut in_entries = false;
    let mut entries: Vec<PackEntry> = Vec::new();
    for line in manifest_content.lines() {
        if line == "---" {
            in_entries = true;
            continue;
        }
        if !in_entries { continue; }
        let parts: Vec<&str> = line.splitn(3, ' ').collect();
        if parts.len() != 3 { continue; }
        let rel_path = parts[0].to_string();
        // Re-read the actual file and compute its hash
        let file_path = pack_dir.join(&rel_path);
        let data = match fs::read(&file_path) {
            Ok(d) => d,
            Err(_) => return false,
        };
        let hash = cjc_snap::hash::sha256(&data);
        let hash_hex = hash.iter().map(|b| format!("{:02x}", b)).collect::<String>();
        entries.push(PackEntry {
            rel_path,
            size: data.len() as u64,
            hash: hash_hex,
        });
    }

    // Sort deterministically
    entries.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));

    let actual_hash = compute_manifest_hash(&entries);
    actual_hash == expected_hash
}

/// --list: list contents of an existing .pack directory by reading its MANIFEST.
fn run_list(pa: &PackArgs) {
    let pack_dir = Path::new(&pa.file);
    let manifest_path = pack_dir.join("MANIFEST");

    if !manifest_path.exists() {
        eprintln!("error: no MANIFEST found in `{}`", pa.file);
        process::exit(1);
    }

    let content = match fs::read_to_string(&manifest_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: could not read MANIFEST: {}", e);
            process::exit(1);
        }
    };

    // Parse header and entries
    let mut entry_name = String::new();
    let mut manifest_hash = String::new();
    let mut in_entries = false;
    let mut file_entries: Vec<(String, u64, String)> = Vec::new(); // (rel_path, size, hash)

    for line in content.lines() {
        if line.starts_with("entry: ") {
            entry_name = line["entry: ".len()..].to_string();
        } else if line.starts_with("manifest_hash: ") {
            manifest_hash = line["manifest_hash: ".len()..].to_string();
        } else if line == "---" {
            in_entries = true;
            continue;
        }
        if !in_entries { continue; }
        let parts: Vec<&str> = line.splitn(3, ' ').collect();
        if parts.len() == 3 {
            let size: u64 = parts[1].parse().unwrap_or(0);
            file_entries.push((parts[0].to_string(), size, parts[2].to_string()));
        }
    }

    match pa.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"pack_dir\": \"{}\",", pa.file.replace('\\', "/"));
            println!("  \"entry\": \"{}\",", entry_name);
            println!("  \"manifest_hash\": \"{}\",", manifest_hash);
            println!("  \"files\": [");
            for (i, (path, size, hash)) in file_entries.iter().enumerate() {
                print!("    {{\"path\": \"{}\", \"size\": {}, \"hash\": \"{}\"}}",
                    path, size, &hash[..hash.len().min(16)]);
                if i + 1 < file_entries.len() { print!(","); }
                println!();
            }
            println!("  ]");
            println!("}}");
        }
        _ => {
            eprintln!("Package: {}", pa.file.replace('\\', "/"));
            eprintln!("Entry:   {}", entry_name);
            eprintln!("Hash:    {}", if manifest_hash.len() >= 16 { &manifest_hash[..16] } else { &manifest_hash });
            eprintln!();
            let mut t = crate::table::Table::new(vec!["File", "Size", "Hash"]);
            let total_size: u64 = file_entries.iter().map(|(_, s, _)| *s).sum();
            for (path, size, hash) in &file_entries {
                t.add_row_owned(vec![
                    path.clone(),
                    output::format_size(*size),
                    hash[..hash.len().min(16)].to_string(),
                ]);
            }
            eprint!("{}", t.render());
            eprintln!("{} files, {}", file_entries.len(), output::format_size(total_size));
        }
    }
}

/// --repro-check: re-pack from source and compare manifest hash against existing.
fn run_repro_check(pa: &PackArgs) {
    let pack_dir = Path::new(&pa.file);
    let manifest_path = pack_dir.join("MANIFEST");

    if !manifest_path.exists() {
        eprintln!("error: no MANIFEST found in `{}`", pa.file);
        process::exit(1);
    }

    let content = match fs::read_to_string(&manifest_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: could not read MANIFEST: {}", e);
            process::exit(1);
        }
    };

    // Extract stored manifest hash
    let mut stored_hash = String::new();
    for line in content.lines() {
        if line.starts_with("manifest_hash: ") {
            stored_hash = line["manifest_hash: ".len()..].to_string();
            break;
        }
    }

    if stored_hash.is_empty() {
        eprintln!("error: no manifest_hash found in MANIFEST");
        process::exit(1);
    }

    // Re-read all files in the pack directory (excluding MANIFEST itself),
    // compute their hashes, and derive a new manifest hash.
    let mut in_entries = false;
    let mut entries: Vec<PackEntry> = Vec::new();

    for line in content.lines() {
        if line == "---" {
            in_entries = true;
            continue;
        }
        if !in_entries { continue; }
        let parts: Vec<&str> = line.splitn(3, ' ').collect();
        if parts.len() != 3 { continue; }
        let rel_path = parts[0].to_string();
        let file_path = pack_dir.join(&rel_path);
        let data = match fs::read(&file_path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("error: could not read `{}`: {}", rel_path, e);
                process::exit(1);
            }
        };
        let hash = cjc_snap::hash::sha256(&data);
        let hash_hex = hash.iter().map(|b| format!("{:02x}", b)).collect::<String>();
        entries.push(PackEntry {
            rel_path,
            size: data.len() as u64,
            hash: hash_hex,
        });
    }

    entries.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
    let recomputed_hash = compute_manifest_hash(&entries);

    let matches = recomputed_hash == stored_hash;

    match pa.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"pack_dir\": \"{}\",", pa.file.replace('\\', "/"));
            println!("  \"stored_hash\": \"{}\",", stored_hash);
            println!("  \"recomputed_hash\": \"{}\",", recomputed_hash);
            println!("  \"reproducible\": {}", matches);
            println!("}}");
        }
        _ => {
            if matches {
                eprintln!("{} — package `{}` is reproducible (hash: {})",
                    output::colorize(pa.output, output::BOLD_GREEN, "pass"),
                    pa.file.replace('\\', "/"),
                    &stored_hash[..stored_hash.len().min(16)]);
            } else {
                eprintln!("{} — package `{}` is NOT reproducible",
                    output::colorize(pa.output, output::BOLD_RED, "fail"),
                    pa.file.replace('\\', "/"));
                eprintln!("  stored:     {}", &stored_hash[..stored_hash.len().min(16)]);
                eprintln!("  recomputed: {}", &recomputed_hash[..recomputed_hash.len().min(16)]);
            }
        }
    }

    if !matches {
        process::exit(1);
    }
}

/// Extensions considered packable when walking a directory.
const PACKABLE_EXTENSIONS: &[&str] = &[
    "cjc", "csv", "tsv", "jsonl", "ndjson", "json", "snap", "lock", "toml",
];

/// Recursively discover packable files in a directory.
fn discover_dir_files(dir: &Path, base: &Path, entries: &mut Vec<PackEntry>) {
    let read_dir = match fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(_) => return,
    };

    let mut dir_entries: Vec<_> = read_dir.filter_map(|e| e.ok()).collect();
    dir_entries.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

    for entry in dir_entries {
        let path = entry.path();
        if path.is_dir() {
            // Skip hidden dirs and common build artifacts
            let dir_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if dir_name.starts_with('.') || dir_name == "target" || dir_name == "node_modules" {
                continue;
            }
            discover_dir_files(&path, base, entries);
        } else if path.is_file() {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if PACKABLE_EXTENSIONS.contains(&ext) {
                add_entry(entries, &path, base);
            }
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
    eprintln!("  --verify              Verify manifest hash after packing");
    eprintln!("  --manifest-only       Output MANIFEST content only, don't create package");
    eprintln!("  --list <dir.pack>     List contents of an existing .pack directory");
    eprintln!("  --repro-check <dir>   Re-hash packed files and compare to stored manifest");
    eprintln!("  --plain               Plain text output");
    eprintln!("  --json                JSON output");
    eprintln!("  --color               Color output (default)");
}
