//! `cjcl inspect` — Deep inspection of structured computational artifacts.
//!
//! Inspects .snap files, .csv/.tsv datasets, .jsonl/.ndjson files, .cjcl source
//! files, binary metadata formats (parquet, arrow, sqlite), safe model file
//! inspection (.pkl, .onnx, .joblib), and generic files.
//!
//! Reports structure, shape, dtype, statistics, and content hashes.
//! Never mutates inspected artifacts. Model files are NEVER deserialized or executed.
//! All output is deterministic.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::process;
use crate::output::{self, OutputMode};
use crate::formats;

// ── Column statistics ───────────────────────────────────────────────

struct ColStats {
    name: String,
    null_count: u64,
    numeric_count: u64,
    string_count: u64,
    bool_count: u64,
    min: f64,
    max: f64,
    // Kahan summation for mean
    sum: f64,
    sum_comp: f64,
    count: u64,
    // Deep-mode: Kahan summation for sum-of-squares (for variance/std)
    sum_sq: f64,
    sum_sq_comp: f64,
    // Deep-mode: unique value tracking
    unique_values: Option<BTreeSet<String>>,
}

impl ColStats {
    fn new(name: String, deep: bool) -> Self {
        Self {
            name,
            null_count: 0,
            numeric_count: 0,
            string_count: 0,
            bool_count: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            sum: 0.0,
            sum_comp: 0.0,
            count: 0,
            sum_sq: 0.0,
            sum_sq_comp: 0.0,
            unique_values: if deep { Some(BTreeSet::new()) } else { None },
        }
    }

    fn observe(&mut self, val: &str) {
        if val.is_empty() || val == "NA" || val == "NaN" || val == "null" || val == "None" {
            self.null_count += 1;
            return;
        }
        // Track uniques if deep mode
        if let Some(ref mut uniques) = self.unique_values {
            uniques.insert(val.to_string());
        }
        // Check for boolean values before numeric (since "true"/"false" don't parse as f64)
        let lower = val.trim().to_ascii_lowercase();
        if lower == "true" || lower == "false" {
            self.bool_count += 1;
            return;
        }
        if let Ok(v) = val.parse::<f64>() {
            if !v.is_nan() {
                self.numeric_count += 1;
                self.count += 1;
                // Kahan summation for sum
                let y = v - self.sum_comp;
                let t = self.sum + y;
                self.sum_comp = (t - self.sum) - y;
                self.sum = t;
                // Kahan summation for sum of squares
                let v2 = v * v;
                let y2 = v2 - self.sum_sq_comp;
                let t2 = self.sum_sq + y2;
                self.sum_sq_comp = (t2 - self.sum_sq) - y2;
                self.sum_sq = t2;
                if v < self.min { self.min = v; }
                if v > self.max { self.max = v; }
            } else {
                self.null_count += 1;
            }
        } else {
            self.string_count += 1;
        }
    }

    fn mean(&self) -> Option<f64> {
        if self.count > 0 { Some(self.sum / self.count as f64) } else { None }
    }

    fn variance(&self) -> Option<f64> {
        if self.count > 1 {
            let n = self.count as f64;
            let mean = self.sum / n;
            // Var = E[X^2] - (E[X])^2, with Bessel correction
            let var = (self.sum_sq / n - mean * mean) * n / (n - 1.0);
            Some(if var < 0.0 { 0.0 } else { var })
        } else {
            None
        }
    }

    fn std_dev(&self) -> Option<f64> {
        self.variance().map(|v| v.sqrt())
    }

    fn unique_count(&self) -> Option<usize> {
        self.unique_values.as_ref().map(|u| u.len())
    }
}

fn infer_dtype(col: &ColStats) -> &'static str {
    // Check for bool first: if all string values are "true"/"false" variants, it's bool
    if col.bool_count > 0 && col.numeric_count == 0 && col.string_count == 0 { "bool" }
    else if col.string_count > 0 && col.numeric_count == 0 && col.bool_count == 0 { "string" }
    else if col.numeric_count > 0 && col.string_count == 0 && col.bool_count == 0 {
        if col.min == col.min.floor() && col.max == col.max.floor() { "int" }
        else { "float" }
    }
    else if col.numeric_count > 0 && col.string_count > 0 { "mixed" }
    else if col.bool_count > 0 && col.string_count > 0 { "mixed" }
    else { "unknown" }
}

// ── Args ────────────────────────────────────────────────────────────

pub struct InspectArgs {
    pub file: String,
    pub output: OutputMode,
    pub show_stats: bool,
    pub show_hash: bool,
    pub max_preview: usize,
    // Second-mode flags
    pub deep: bool,
    pub header_only: bool,
    pub schema_only: bool,
    pub hash_explicit: bool,
    pub manifest: bool,
    pub compare_file: Option<String>,
}

impl Default for InspectArgs {
    fn default() -> Self {
        Self {
            file: String::new(),
            output: OutputMode::Color,
            show_stats: true,
            show_hash: true,
            max_preview: 5,
            deep: false,
            header_only: false,
            schema_only: false,
            hash_explicit: false,
            manifest: false,
            compare_file: None,
        }
    }
}

pub fn parse_args(args: &[String]) -> InspectArgs {
    let mut ia = InspectArgs::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--no-stats" => ia.show_stats = false,
            "--no-hash" => ia.show_hash = false,
            "--preview" => {
                i += 1;
                if i < args.len() { ia.max_preview = args[i].parse().unwrap_or(5); }
            }
            "--plain" => ia.output = OutputMode::Plain,
            "--json" => ia.output = OutputMode::Json,
            "--color" => ia.output = OutputMode::Color,
            "--deep" => ia.deep = true,
            "--header-only" => ia.header_only = true,
            "--schema-only" => ia.schema_only = true,
            "--hash" => ia.hash_explicit = true,
            "--manifest" => ia.manifest = true,
            "--compare" => {
                i += 1;
                if i < args.len() {
                    ia.compare_file = Some(args[i].clone());
                } else {
                    eprintln!("error: --compare requires a file argument");
                    process::exit(1);
                }
            }
            other if !other.starts_with('-') => {
                if ia.file.is_empty() {
                    ia.file = other.to_string();
                } else {
                    eprintln!("error: unexpected argument `{}` for `cjcl inspect`", other);
                    process::exit(1);
                }
            }
            other => {
                eprintln!("error: unknown flag `{}` for `cjcl inspect`", other);
                process::exit(1);
            }
        }
        i += 1;
    }
    if ia.file.is_empty() {
        eprintln!("error: `cjcl inspect` requires a file argument");
        process::exit(1);
    }
    ia
}

// ── Entry point ─────────────────────────────────────────────────────

pub fn run(args: &[String]) {
    let ia = parse_args(args);
    let path = Path::new(&ia.file);

    if !path.exists() {
        eprintln!("error: file `{}` not found", ia.file);
        process::exit(1);
    }

    // Handle --compare mode
    if let Some(ref other_file) = ia.compare_file {
        let other_path = Path::new(other_file);
        if !other_path.exists() {
            eprintln!("error: comparison file `{}` not found", other_file);
            process::exit(1);
        }
        run_compare(&ia, path, other_path);
        return;
    }

    // Handle --manifest mode
    if ia.manifest {
        run_manifest(path);
        return;
    }

    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let ext_lower = ext.to_ascii_lowercase();
    match ext_lower.as_str() {
        "snap" => inspect_snap(&ia, path),
        "csv" | "tsv" => inspect_csv(&ia, path, if ext_lower == "tsv" { '\t' } else { ',' }),
        "jsonl" | "ndjson" => inspect_jsonl(&ia, path),
        "parquet" | "feather" | "arrow" | "ipc" | "sqlite" | "db" | "sqlite3" => {
            inspect_binary_metadata(&ia, path)
        }
        "pkl" | "pickle" | "onnx" | "joblib" => inspect_model(&ia, path),
        "cjcl" => inspect_cjc(&ia, path),
        _ => inspect_generic(&ia, path),
    }
}

// ── Snap inspection (unchanged) ─────────────────────────────────────

fn inspect_snap(ia: &InspectArgs, path: &Path) {
    let data = match fs::read(path) {
        Ok(d) => d,
        Err(e) => { eprintln!("error: could not read `{}`: {}", path.display(), e); process::exit(1); }
    };

    let hash = cjc_snap::hash::sha256(&data);
    let hash_hex = hash.iter().map(|b| format!("{:02x}", b)).collect::<String>();

    // Try decoding
    let decode_result = cjc_snap::snap_decode_v2(&data)
        .or_else(|_| cjc_snap::snap_decode(&data));

    match ia.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"file\": \"{}\",", path.display().to_string().replace('\\', "/"));
            println!("  \"type\": \"snap\",");
            println!("  \"size\": {},", data.len());
            println!("  \"sha256\": \"{}\",", hash_hex);
            match &decode_result {
                Ok(val) => {
                    println!("  \"value_type\": \"{}\",", val.type_name());
                    println!("  \"decoded\": true");
                }
                Err(e) => {
                    println!("  \"decoded\": false,");
                    println!("  \"error\": \"{:?}\"", e);
                }
            }
            println!("}}");
        }
        _ => {
            let mut t = crate::table::Table::new(vec!["Property", "Value"]);
            t.add_row_owned(vec!["File".into(), path.display().to_string().replace('\\', "/")]);
            t.add_row_owned(vec!["Type".into(), "snap".into()]);
            t.add_row_owned(vec!["Size".into(), output::format_size(data.len() as u64)]);
            if ia.show_hash {
                t.add_row_owned(vec!["SHA-256".into(), hash_hex]);
            }
            match &decode_result {
                Ok(val) => {
                    t.add_row_owned(vec!["Value type".into(), val.type_name().to_string()]);
                    let preview = format!("{}", val);
                    let truncated = if preview.len() > 200 {
                        format!("{}...", &preview[..200])
                    } else {
                        preview
                    };
                    t.add_row_owned(vec!["Preview".into(), truncated]);
                }
                Err(e) => {
                    t.add_row_owned(vec!["Decode".into(), format!("FAILED: {:?}", e)]);
                }
            }
            eprint!("{}", t.render());
        }
    }
}

// ── CSV/TSV inspection ──────────────────────────────────────────────

fn inspect_csv(ia: &InspectArgs, path: &Path, delimiter: char) {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => { eprintln!("error: could not read `{}`: {}", path.display(), e); process::exit(1); }
    };

    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        eprintln!("empty file");
        return;
    }

    let headers: Vec<&str> = lines[0].split(delimiter).map(|s| s.trim()).collect();
    let ncols = headers.len();
    let nrows = lines.len() - 1; // exclude header

    let mut cols: Vec<ColStats> = headers.iter().map(|h| ColStats::new(h.to_string(), ia.deep)).collect();

    if !ia.header_only {
        for line in &lines[1..] {
            let fields: Vec<&str> = line.split(delimiter).collect();
            for (ci, col) in cols.iter_mut().enumerate() {
                let val = fields.get(ci).map(|s| s.trim()).unwrap_or("");
                col.observe(val);
            }
        }
    }

    let file_size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let type_label = if delimiter == '\t' { "tsv" } else { "csv" };

    match ia.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"file\": \"{}\",", path.display().to_string().replace('\\', "/"));
            println!("  \"type\": \"{}\",", type_label);
            println!("  \"rows\": {},", nrows);
            println!("  \"columns\": {},", ncols);
            println!("  \"size\": {},", file_size);
            if !ia.header_only {
                println!("  \"schema\": [");
                for (i, col) in cols.iter().enumerate() {
                    let dtype = infer_dtype(col);
                    print!("    {{\"name\": \"{}\", \"dtype\": \"{}\", \"nulls\": {}", col.name, dtype, col.null_count);
                    if col.count > 0 && ia.show_stats && !ia.schema_only {
                        let mean = col.sum / col.count as f64;
                        print!(", \"min\": {}, \"max\": {}, \"mean\": {}",
                            output::format_f64(col.min, 6),
                            output::format_f64(col.max, 6),
                            output::format_f64(mean, 6));
                        if ia.deep {
                            if let Some(var) = col.variance() {
                                print!(", \"variance\": {}", output::format_f64(var, 6));
                            }
                            if let Some(std) = col.std_dev() {
                                print!(", \"std\": {}", output::format_f64(std, 6));
                            }
                            if let Some(uc) = col.unique_count() {
                                print!(", \"unique\": {}", uc);
                            }
                        }
                    }
                    print!("}}");
                    if i + 1 < cols.len() { print!(","); }
                    println!();
                }
                println!("  ]");
            }
            println!("}}");
        }
        _ => {
            let mut t = crate::table::Table::new(vec!["Property", "Value"]);
            t.add_row_owned(vec!["File".into(), path.display().to_string().replace('\\', "/")]);
            t.add_row_owned(vec!["Type".into(), type_label.into()]);
            t.add_row_owned(vec!["Rows".into(), format!("{}", nrows)]);
            t.add_row_owned(vec!["Columns".into(), format!("{}", ncols)]);
            t.add_row_owned(vec!["Size".into(), output::format_size(file_size)]);
            eprint!("{}", t.render());

            if !ia.header_only && ia.show_stats && !ia.schema_only {
                eprintln!();
                let stat_headers = if ia.deep {
                    vec!["Column", "Type", "Nulls", "Min", "Max", "Mean", "Std", "Unique"]
                } else {
                    vec!["Column", "Type", "Nulls", "Min", "Max", "Mean"]
                };
                let mut st = crate::table::Table::new(stat_headers);
                for col in &cols {
                    let dtype = infer_dtype(col);
                    let (min, max, mean) = if col.count > 0 {
                        (output::format_f64(col.min, 4), output::format_f64(col.max, 4),
                         output::format_f64(col.sum / col.count as f64, 4))
                    } else {
                        ("-".into(), "-".into(), "-".into())
                    };
                    let mut row = vec![col.name.clone(), dtype.into(), format!("{}", col.null_count), min, max, mean];
                    if ia.deep {
                        let std_str = col.std_dev().map(|s| output::format_f64(s, 4)).unwrap_or_else(|| "-".into());
                        let unique_str = col.unique_count().map(|u| format!("{}", u)).unwrap_or_else(|| "-".into());
                        row.push(std_str);
                        row.push(unique_str);
                    }
                    st.add_row_owned(row);
                }
                eprint!("{}", st.render());
            } else if !ia.header_only && ia.schema_only {
                eprintln!();
                let mut st = crate::table::Table::new(vec!["Column", "Type", "Nulls"]);
                for col in &cols {
                    let dtype = infer_dtype(col);
                    st.add_row_owned(vec![col.name.clone(), dtype.into(), format!("{}", col.null_count)]);
                }
                eprint!("{}", st.render());
            }
        }
    }
}

// ── JSONL inspection ────────────────────────────────────────────────

fn inspect_jsonl(ia: &InspectArgs, path: &Path) {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => { eprintln!("error: could not read `{}`: {}", path.display(), e); process::exit(1); }
    };

    let data = formats::load_jsonl(&content);
    let nrows = data.nrows();
    let ncols = data.ncols();
    let file_size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    // Compute per-column stats from the tabular data
    let mut cols: Vec<ColStats> = data.headers.iter()
        .map(|h| ColStats::new(h.clone(), ia.deep))
        .collect();

    if !ia.header_only {
        for row in &data.rows {
            for (ci, col) in cols.iter_mut().enumerate() {
                let val = row.get(ci).map(|s| s.as_str()).unwrap_or("");
                col.observe(val);
            }
        }
    }

    match ia.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"file\": \"{}\",", path.display().to_string().replace('\\', "/"));
            println!("  \"type\": \"jsonl\",");
            println!("  \"rows\": {},", nrows);
            println!("  \"columns\": {},", ncols);
            println!("  \"size\": {},", file_size);
            if !ia.header_only {
                println!("  \"schema\": [");
                for (i, col) in cols.iter().enumerate() {
                    let dtype = infer_dtype(col);
                    print!("    {{\"name\": \"{}\", \"dtype\": \"{}\", \"nulls\": {}", col.name, dtype, col.null_count);
                    if col.count > 0 && ia.show_stats && !ia.schema_only {
                        let mean = col.sum / col.count as f64;
                        print!(", \"min\": {}, \"max\": {}, \"mean\": {}",
                            output::format_f64(col.min, 6),
                            output::format_f64(col.max, 6),
                            output::format_f64(mean, 6));
                        if ia.deep {
                            if let Some(var) = col.variance() {
                                print!(", \"variance\": {}", output::format_f64(var, 6));
                            }
                            if let Some(std) = col.std_dev() {
                                print!(", \"std\": {}", output::format_f64(std, 6));
                            }
                            if let Some(uc) = col.unique_count() {
                                print!(", \"unique\": {}", uc);
                            }
                        }
                    }
                    print!("}}");
                    if i + 1 < cols.len() { print!(","); }
                    println!();
                }
                println!("  ]");
            }
            println!("}}");
        }
        _ => {
            let mut t = crate::table::Table::new(vec!["Property", "Value"]);
            t.add_row_owned(vec!["File".into(), path.display().to_string().replace('\\', "/")]);
            t.add_row_owned(vec!["Type".into(), "jsonl".into()]);
            t.add_row_owned(vec!["Rows".into(), format!("{}", nrows)]);
            t.add_row_owned(vec!["Columns".into(), format!("{}", ncols)]);
            t.add_row_owned(vec!["Size".into(), output::format_size(file_size)]);
            eprint!("{}", t.render());

            if !ia.header_only && ia.show_stats && !ia.schema_only {
                eprintln!();
                let stat_headers = if ia.deep {
                    vec!["Column", "Type", "Nulls", "Min", "Max", "Mean", "Std", "Unique"]
                } else {
                    vec!["Column", "Type", "Nulls", "Min", "Max", "Mean"]
                };
                let mut st = crate::table::Table::new(stat_headers);
                for col in &cols {
                    let dtype = infer_dtype(col);
                    let (min, max, mean) = if col.count > 0 {
                        (output::format_f64(col.min, 4), output::format_f64(col.max, 4),
                         output::format_f64(col.sum / col.count as f64, 4))
                    } else {
                        ("-".into(), "-".into(), "-".into())
                    };
                    let mut row = vec![col.name.clone(), dtype.into(), format!("{}", col.null_count), min, max, mean];
                    if ia.deep {
                        let std_str = col.std_dev().map(|s| output::format_f64(s, 4)).unwrap_or_else(|| "-".into());
                        let unique_str = col.unique_count().map(|u| format!("{}", u)).unwrap_or_else(|| "-".into());
                        row.push(std_str);
                        row.push(unique_str);
                    }
                    st.add_row_owned(row);
                }
                eprint!("{}", st.render());
            } else if !ia.header_only && ia.schema_only {
                eprintln!();
                let mut st = crate::table::Table::new(vec!["Column", "Type", "Nulls"]);
                for col in &cols {
                    let dtype = infer_dtype(col);
                    st.add_row_owned(vec![col.name.clone(), dtype.into(), format!("{}", col.null_count)]);
                }
                eprint!("{}", st.render());
            }
        }
    }
}

// ── Binary metadata inspection (Parquet, Arrow, SQLite) ─────────────

fn inspect_binary_metadata(ia: &InspectArgs, path: &Path) {
    let meta = formats::extract_metadata(path);
    let file_size = meta.size;

    let hash_hex = if ia.show_hash || ia.hash_explicit {
        let data = fs::read(path).unwrap_or_default();
        let hash = cjc_snap::hash::sha256(&data);
        hash.iter().map(|b| format!("{:02x}", b)).collect::<String>()
    } else {
        String::new()
    };

    match ia.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"file\": \"{}\",", path.display().to_string().replace('\\', "/"));
            println!("  \"type\": \"{}\",", meta.format.label().to_ascii_lowercase());
            println!("  \"format\": \"{}\",", meta.format.label());
            println!("  \"size\": {},", file_size);
            if !hash_hex.is_empty() {
                println!("  \"sha256\": \"{}\",", hash_hex);
            }
            if let Some(ref magic) = meta.magic_bytes {
                println!("  \"magic_bytes\": \"{}\",", magic);
            }
            println!("  \"metadata\": {{");
            let entries: Vec<_> = meta.header_info.iter().collect();
            for (i, (k, v)) in entries.iter().enumerate() {
                print!("    \"{}\": \"{}\"", k, v);
                if i + 1 < entries.len() { print!(","); }
                println!();
            }
            println!("  }},");
            println!("  \"limitations\": [");
            for (i, lim) in meta.limitations.iter().enumerate() {
                print!("    \"{}\"", lim);
                if i + 1 < meta.limitations.len() { print!(","); }
                println!();
            }
            println!("  ]");
            println!("}}");
        }
        _ => {
            let mut t = crate::table::Table::new(vec!["Property", "Value"]);
            t.add_row_owned(vec!["File".into(), path.display().to_string().replace('\\', "/")]);
            t.add_row_owned(vec!["Format".into(), meta.format.label().to_string()]);
            t.add_row_owned(vec!["Size".into(), output::format_size(file_size)]);
            if !hash_hex.is_empty() {
                t.add_row_owned(vec!["SHA-256".into(), hash_hex]);
            }
            if let Some(ref magic) = meta.magic_bytes {
                t.add_row_owned(vec!["Magic bytes".into(), magic.clone()]);
            }
            eprint!("{}", t.render());

            // Metadata key-value table
            if !meta.header_info.is_empty() {
                eprintln!();
                let mut mt = crate::table::Table::new(vec!["Key", "Value"]);
                for (k, v) in &meta.header_info {
                    mt.add_row_owned(vec![k.clone(), v.clone()]);
                }
                eprint!("{}", mt.render());
            }

            // Limitations
            if !meta.limitations.is_empty() {
                eprintln!();
                for lim in &meta.limitations {
                    eprintln!("  Limitation: {}", lim);
                }
            }
        }
    }
}

// ── Model file inspection (.pkl, .onnx, .joblib) ────────────────────

fn inspect_model(ia: &InspectArgs, path: &Path) {
    let data = match fs::read(path) {
        Ok(d) => d,
        Err(e) => { eprintln!("error: could not read `{}`: {}", path.display(), e); process::exit(1); }
    };

    let file_size = data.len() as u64;
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();

    // SHA-256 hash (always for model files)
    let hash = cjc_snap::hash::sha256(&data);
    let hash_hex = hash.iter().map(|b| format!("{:02x}", b)).collect::<String>();

    // Magic byte signature
    let magic_display = if data.len() >= 8 {
        data[..8].iter().map(|b| format!("{:02x}", b)).collect::<Vec<_>>().join(" ")
    } else if !data.is_empty() {
        data.iter().map(|b| format!("{:02x}", b)).collect::<Vec<_>>().join(" ")
    } else {
        "(empty)".to_string()
    };

    // Format detection result
    let format_result = formats::detect_format(path);
    let format_description = match ext.as_str() {
        "onnx" => {
            // Check for protobuf magic byte (0x08 is field 1, varint type)
            if !data.is_empty() && data[0] == 0x08 {
                "ONNX model (protobuf-based)"
            } else {
                "ONNX model (protobuf-based, magic byte not confirmed)"
            }
        }
        "pkl" | "pickle" => {
            if !data.is_empty() && data[0] == 0x80 {
                "Pickle serialized object"
            } else {
                "Pickle serialized object (protocol 0/1 or non-standard)"
            }
        }
        "joblib" => "Joblib serialized object",
        _ => format_result.label(),
    };

    // Pickle protocol version
    let pickle_protocol: Option<u8> = if (ext == "pkl" || ext == "pickle") && data.len() >= 2 && data[0] == 0x80 {
        Some(data[1])
    } else {
        None
    };

    let meta = formats::extract_metadata(path);

    match ia.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"file\": \"{}\",", path.display().to_string().replace('\\', "/"));
            println!("  \"extension\": \"{}\",", ext);
            println!("  \"size\": {},", file_size);
            println!("  \"sha256\": \"{}\",", hash_hex);
            println!("  \"magic_bytes\": \"{}\",", magic_display);
            println!("  \"format_detection\": \"{}\",", format_description);
            if let Some(proto) = pickle_protocol {
                println!("  \"pickle_protocol\": {},", proto);
            }
            // Include extracted metadata
            if !meta.header_info.is_empty() {
                println!("  \"metadata\": {{");
                let entries: Vec<_> = meta.header_info.iter().collect();
                for (i, (k, v)) in entries.iter().enumerate() {
                    print!("    \"{}\": \"{}\"", k, v);
                    if i + 1 < entries.len() { print!(","); }
                    println!();
                }
                println!("  }},");
            }
            println!("  \"safety\": \"model files are not deserialized or executed\",");
            println!("  \"limitation\": \"Full model inspection requires specialized tools\"");
            println!("}}");
        }
        _ => {
            let mut t = crate::table::Table::new(vec!["Property", "Value"]);
            t.add_row_owned(vec!["File".into(), path.display().to_string().replace('\\', "/")]);
            t.add_row_owned(vec!["Extension".into(), ext.clone()]);
            t.add_row_owned(vec!["Size".into(), output::format_size(file_size)]);
            t.add_row_owned(vec!["SHA-256".into(), hash_hex]);
            t.add_row_owned(vec!["Magic bytes".into(), magic_display]);
            t.add_row_owned(vec!["Format".into(), format_description.to_string()]);
            if let Some(proto) = pickle_protocol {
                t.add_row_owned(vec!["Pickle protocol".into(), format!("{}", proto)]);
            }
            eprint!("{}", t.render());

            // Additional metadata from extract_metadata
            let relevant: Vec<_> = meta.header_info.iter()
                .filter(|(k, _)| *k != "file_size_bytes")
                .collect();
            if !relevant.is_empty() {
                eprintln!();
                let mut mt = crate::table::Table::new(vec!["Key", "Value"]);
                for (k, v) in relevant {
                    mt.add_row_owned(vec![k.clone(), v.clone()]);
                }
                eprint!("{}", mt.render());
            }

            eprintln!();
            eprintln!("  Safety: model files are not deserialized or executed");
            eprintln!("  Limitation: Full model inspection requires specialized tools");
        }
    }
}

// ── CJC source inspection (unchanged) ───────────────────────────────

fn inspect_cjc(ia: &InspectArgs, path: &Path) {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => { eprintln!("error: could not read `{}`: {}", path.display(), e); process::exit(1); }
    };

    let (program, diags) = cjc_parser::parse_source(&source);
    let file_size = source.len() as u64;
    let line_count = source.lines().count();

    // Extract declarations
    let mut functions = Vec::new();
    let mut structs = Vec::new();
    let mut lets = 0u64;
    let mut effects = std::collections::BTreeSet::new();

    for decl in &program.declarations {
        match &decl.kind {
            cjc_ast::DeclKind::Fn(f) => {
                let params: Vec<String> = f.params.iter().map(|p| {
                    format!("{}: {}", p.name.name, format_type_expr(&p.ty))
                }).collect();
                let sig = format!("fn {}({})", f.name.name, params.join(", "));
                functions.push(sig);
                if f.is_nogc { effects.insert("nogc".to_string()); }
                if let Some(ref effs) = f.effect_annotation {
                    for e in effs { effects.insert(e.clone()); }
                }
            }
            cjc_ast::DeclKind::Struct(s) => structs.push(s.name.name.clone()),
            cjc_ast::DeclKind::Let(_) => lets += 1,
            _ => {}
        }
    }

    match ia.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"file\": \"{}\",", path.display().to_string().replace('\\', "/"));
            println!("  \"type\": \"cjc\",");
            println!("  \"size\": {},", file_size);
            println!("  \"lines\": {},", line_count);
            println!("  \"parse_errors\": {},", diags.diagnostics.len());
            println!("  \"functions\": {},", functions.len());
            println!("  \"structs\": {},", structs.len());
            println!("  \"let_bindings\": {}", lets);
            println!("}}");
        }
        _ => {
            let mut t = crate::table::Table::new(vec!["Property", "Value"]);
            t.add_row_owned(vec!["File".into(), path.display().to_string().replace('\\', "/")]);
            t.add_row_owned(vec!["Type".into(), "cjcl source".into()]);
            t.add_row_owned(vec!["Size".into(), output::format_size(file_size)]);
            t.add_row_owned(vec!["Lines".into(), format!("{}", line_count)]);
            t.add_row_owned(vec!["Parse errors".into(), format!("{}", diags.diagnostics.len())]);
            t.add_row_owned(vec!["Functions".into(), format!("{}", functions.len())]);
            t.add_row_owned(vec!["Structs".into(), format!("{}", structs.len())]);
            t.add_row_owned(vec!["Let bindings".into(), format!("{}", lets)]);
            if !effects.is_empty() {
                let eff_str: Vec<String> = effects.into_iter().collect();
                t.add_row_owned(vec!["Effects".into(), eff_str.join(", ")]);
            }
            eprint!("{}", t.render());

            if !functions.is_empty() {
                eprintln!("\nFunctions:");
                for f in &functions {
                    eprintln!("  {}", f);
                }
            }
            if !structs.is_empty() {
                eprintln!("\nStructs:");
                for s in &structs {
                    eprintln!("  {}", s);
                }
            }
        }
    }
}

// ── Generic file inspection (unchanged) ─────────────────────────────

fn inspect_generic(ia: &InspectArgs, path: &Path) {
    let meta = match fs::metadata(path) {
        Ok(m) => m,
        Err(e) => { eprintln!("error: could not read `{}`: {}", path.display(), e); process::exit(1); }
    };

    let size = meta.len();
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("none");

    let hash_hex = if ia.show_hash {
        let data = fs::read(path).unwrap_or_default();
        let hash = cjc_snap::hash::sha256(&data);
        hash.iter().map(|b| format!("{:02x}", b)).collect::<String>()
    } else {
        "-".to_string()
    };

    match ia.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"file\": \"{}\",", path.display().to_string().replace('\\', "/"));
            println!("  \"type\": \"generic\",");
            println!("  \"extension\": \"{}\",", ext);
            println!("  \"size\": {},", size);
            if ia.show_hash { println!("  \"sha256\": \"{}\",", hash_hex); }
            println!("  \"is_text\": {}", is_likely_text(path));
            println!("}}");
        }
        _ => {
            let mut t = crate::table::Table::new(vec!["Property", "Value"]);
            t.add_row_owned(vec!["File".into(), path.display().to_string().replace('\\', "/")]);
            t.add_row_owned(vec!["Extension".into(), ext.to_string()]);
            t.add_row_owned(vec!["Size".into(), output::format_size(size)]);
            if ia.show_hash {
                t.add_row_owned(vec!["SHA-256".into(), hash_hex]);
            }
            t.add_row_owned(vec!["Text file".into(), format!("{}", is_likely_text(path))]);
            eprint!("{}", t.render());
        }
    }
}

// ── Manifest mode ───────────────────────────────────────────────────

/// Output a machine-readable manifest line: `<hash> <size> <type> <path>`
fn run_manifest(path: &Path) {
    let data = fs::read(path).unwrap_or_default();
    let file_size = data.len() as u64;
    let hash = cjc_snap::hash::sha256(&data);
    let hash_hex = hash.iter().map(|b| format!("{:02x}", b)).collect::<String>();

    let format = formats::detect_format(path);
    let type_label = if format == formats::DataFormat::Unknown {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("unknown");
        ext.to_string()
    } else {
        format.label().to_ascii_lowercase()
    };

    println!("{} {} {} {}", hash_hex, file_size, type_label, path.display().to_string().replace('\\', "/"));
}

// ── Compare mode ────────────────────────────────────────────────────

/// Compare two files' metadata side by side.
fn run_compare(ia: &InspectArgs, path_a: &Path, path_b: &Path) {
    let data_a = fs::read(path_a).unwrap_or_default();
    let data_b = fs::read(path_b).unwrap_or_default();

    let hash_a = cjc_snap::hash::sha256(&data_a);
    let hash_b = cjc_snap::hash::sha256(&data_b);
    let hash_hex_a = hash_a.iter().map(|b| format!("{:02x}", b)).collect::<String>();
    let hash_hex_b = hash_b.iter().map(|b| format!("{:02x}", b)).collect::<String>();

    let format_a = formats::detect_format(path_a);
    let format_b = formats::detect_format(path_b);

    let size_a = data_a.len() as u64;
    let size_b = data_b.len() as u64;

    let hashes_match = hash_hex_a == hash_hex_b;

    match ia.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"file_a\": \"{}\",", path_a.display().to_string().replace('\\', "/"));
            println!("  \"file_b\": \"{}\",", path_b.display().to_string().replace('\\', "/"));
            println!("  \"size_a\": {},", size_a);
            println!("  \"size_b\": {},", size_b);
            println!("  \"format_a\": \"{}\",", format_a.label());
            println!("  \"format_b\": \"{}\",", format_b.label());
            println!("  \"sha256_a\": \"{}\",", hash_hex_a);
            println!("  \"sha256_b\": \"{}\",", hash_hex_b);
            println!("  \"identical\": {}", hashes_match);
            println!("}}");
        }
        _ => {
            let mut t = crate::table::Table::new(vec!["Property", "File A", "File B"]);
            t.add_row_owned(vec![
                "File".into(),
                path_a.display().to_string().replace('\\', "/"),
                path_b.display().to_string().replace('\\', "/"),
            ]);
            t.add_row_owned(vec![
                "Format".into(),
                format_a.label().to_string(),
                format_b.label().to_string(),
            ]);
            t.add_row_owned(vec![
                "Size".into(),
                output::format_size(size_a),
                output::format_size(size_b),
            ]);
            t.add_row_owned(vec![
                "SHA-256".into(),
                hash_hex_a,
                hash_hex_b,
            ]);
            t.add_row_owned(vec![
                "Identical".into(),
                format!("{}", hashes_match),
                format!("{}", hashes_match),
            ]);
            eprint!("{}", t.render());
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

fn is_likely_text(path: &Path) -> bool {
    let data = match fs::read(path) {
        Ok(d) => d,
        Err(_) => return false,
    };
    // Sample first 512 bytes for null bytes
    let sample = &data[..data.len().min(512)];
    !sample.contains(&0u8)
}

pub fn print_help() {
    eprintln!("cjcl inspect — Deep inspection of structured computational artifacts");
    eprintln!();
    eprintln!("Usage: cjcl inspect <file> [flags]");
    eprintln!();
    eprintln!("Supported file types:");
    eprintln!("  .snap                Snap binary artifacts (decode + stats)");
    eprintln!("  .csv / .tsv          Dataset inspection (schema + column stats)");
    eprintln!("  .jsonl / .ndjson     JSON Lines inspection (schema + column stats)");
    eprintln!("  .parquet             Parquet metadata inspection");
    eprintln!("  .feather / .arrow    Arrow IPC metadata inspection");
    eprintln!("  .sqlite / .db        SQLite metadata inspection");
    eprintln!("  .pkl / .onnx         Safe model file inspection (no deserialization)");
    eprintln!("  .joblib              Safe model file inspection (no deserialization)");
    eprintln!("  .cjcl                Source file analysis (functions, structs, effects)");
    eprintln!("  (other)              Generic file inspection (size, hash)");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  --no-stats           Skip column/value statistics");
    eprintln!("  --no-hash            Skip SHA-256 hash computation");
    eprintln!("  --preview <N>        Max preview items (default: 5)");
    eprintln!("  --plain              Plain text output");
    eprintln!("  --json               JSON output");
    eprintln!("  --color              Color output (default)");
    eprintln!("  --deep               Compute additional stats (variance, std, unique count)");
    eprintln!("  --header-only        Only show file metadata, skip per-column stats");
    eprintln!("  --schema-only        Only show schema/type info, skip numeric stats");
    eprintln!("  --hash               Explicitly request SHA-256 hash");
    eprintln!("  --manifest           Output machine-readable: <hash> <size> <type> <path>");
    eprintln!("  --compare <file>     Compare two files' metadata side by side");
}

fn format_type_expr(ty: &cjc_ast::TypeExpr) -> String {
    match &ty.kind {
        cjc_ast::TypeExprKind::Named { name, args } => {
            if args.is_empty() {
                name.name.clone()
            } else {
                format!("{}<...>", name.name)
            }
        }
        cjc_ast::TypeExprKind::Tuple(elems) => {
            let parts: Vec<String> = elems.iter().map(format_type_expr).collect();
            format!("({})", parts.join(", "))
        }
        cjc_ast::TypeExprKind::Fn { params, ret } => {
            let ps: Vec<String> = params.iter().map(format_type_expr).collect();
            format!("fn({}) -> {}", ps.join(", "), format_type_expr(ret))
        }
        cjc_ast::TypeExprKind::Array { elem, .. } => {
            format!("[{}; _]", format_type_expr(elem))
        }
        cjc_ast::TypeExprKind::ShapeLit(_) => "[shape]".to_string(),
    }
}
