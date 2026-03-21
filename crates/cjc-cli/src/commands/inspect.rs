//! `cjc inspect` — Deep inspection of structured computational artifacts.
//!
//! Inspects .snap files, .csv/.tsv datasets, .cjc source files, and generic
//! files, reporting structure, shape, dtype, statistics, and content hashes.
//!
//! Never mutates inspected artifacts. All output is deterministic.

use std::fs;
use std::path::Path;
use std::process;
use crate::output::{self, OutputMode};

struct CsvColStats {
    name: String,
    null_count: u64,
    numeric_count: u64,
    string_count: u64,
    min: f64,
    max: f64,
    sum: f64,
    sum_comp: f64,
    count: u64,
}

pub struct InspectArgs {
    pub file: String,
    pub output: OutputMode,
    pub show_stats: bool,
    pub show_hash: bool,
    pub max_preview: usize,
}

impl Default for InspectArgs {
    fn default() -> Self {
        Self {
            file: String::new(),
            output: OutputMode::Color,
            show_stats: true,
            show_hash: true,
            max_preview: 5,
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
            other if !other.starts_with('-') => ia.file = other.to_string(),
            other => {
                eprintln!("error: unknown flag `{}` for `cjc inspect`", other);
                process::exit(1);
            }
        }
        i += 1;
    }
    if ia.file.is_empty() {
        eprintln!("error: `cjc inspect` requires a file argument");
        process::exit(1);
    }
    ia
}

pub fn run(args: &[String]) {
    let ia = parse_args(args);
    let path = Path::new(&ia.file);

    if !path.exists() {
        eprintln!("error: file `{}` not found", ia.file);
        process::exit(1);
    }

    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "snap" => inspect_snap(&ia, path),
        "csv" | "tsv" => inspect_csv(&ia, path, if ext == "tsv" { '\t' } else { ',' }),
        "cjc" => inspect_cjc(&ia, path),
        _ => inspect_generic(&ia, path),
    }
}

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

    let mut cols: Vec<CsvColStats> = headers.iter().map(|h| CsvColStats {
        name: h.to_string(),
        null_count: 0, numeric_count: 0, string_count: 0,
        min: f64::INFINITY, max: f64::NEG_INFINITY,
        sum: 0.0, sum_comp: 0.0, count: 0,
    }).collect();

    for line in &lines[1..] {
        let fields: Vec<&str> = line.split(delimiter).collect();
        for (ci, col) in cols.iter_mut().enumerate() {
            let val = fields.get(ci).map(|s| s.trim()).unwrap_or("");
            if val.is_empty() || val == "NA" || val == "NaN" || val == "null" || val == "None" {
                col.null_count += 1;
                continue;
            }
            if let Ok(v) = val.parse::<f64>() {
                if !v.is_nan() {
                    col.numeric_count += 1;
                    col.count += 1;
                    // Kahan summation
                    let y = v - col.sum_comp;
                    let t = col.sum + y;
                    col.sum_comp = (t - col.sum) - y;
                    col.sum = t;
                    if v < col.min { col.min = v; }
                    if v > col.max { col.max = v; }
                } else {
                    col.null_count += 1;
                }
            } else {
                col.string_count += 1;
            }
        }
    }

    let file_size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    match ia.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"file\": \"{}\",", path.display().to_string().replace('\\', "/"));
            println!("  \"type\": \"csv\",");
            println!("  \"rows\": {},", nrows);
            println!("  \"columns\": {},", ncols);
            println!("  \"size\": {},", file_size);
            println!("  \"schema\": [");
            for (i, col) in cols.iter().enumerate() {
                let dtype = infer_dtype(col);
                print!("    {{\"name\": \"{}\", \"dtype\": \"{}\", \"nulls\": {}", col.name, dtype, col.null_count);
                if col.count > 0 && ia.show_stats {
                    let mean = col.sum / col.count as f64;
                    print!(", \"min\": {}, \"max\": {}, \"mean\": {}",
                        output::format_f64(col.min, 6),
                        output::format_f64(col.max, 6),
                        output::format_f64(mean, 6));
                }
                print!("}}");
                if i + 1 < cols.len() { print!(","); }
                println!();
            }
            println!("  ]");
            println!("}}");
        }
        _ => {
            let mut t = crate::table::Table::new(vec!["Property", "Value"]);
            t.add_row_owned(vec!["File".into(), path.display().to_string().replace('\\', "/")]);
            t.add_row_owned(vec!["Type".into(), "csv".into()]);
            t.add_row_owned(vec!["Rows".into(), format!("{}", nrows)]);
            t.add_row_owned(vec!["Columns".into(), format!("{}", ncols)]);
            t.add_row_owned(vec!["Size".into(), output::format_size(file_size)]);
            eprint!("{}", t.render());

            if ia.show_stats {
                eprintln!();
                let mut st = crate::table::Table::new(vec!["Column", "Type", "Nulls", "Min", "Max", "Mean"]);
                for col in &cols {
                    let dtype = infer_dtype(col);
                    let (min, max, mean) = if col.count > 0 {
                        (output::format_f64(col.min, 4), output::format_f64(col.max, 4),
                         output::format_f64(col.sum / col.count as f64, 4))
                    } else {
                        ("-".into(), "-".into(), "-".into())
                    };
                    st.add_row_owned(vec![col.name.clone(), dtype.into(), format!("{}", col.null_count), min, max, mean]);
                }
                eprint!("{}", st.render());
            }
        }
    }
}

fn infer_dtype(col: &CsvColStats) -> &'static str {
    if col.string_count > 0 && col.numeric_count == 0 { "string" }
    else if col.numeric_count > 0 && col.string_count == 0 {
        if col.min == col.min.floor() && col.max == col.max.floor() { "int" }
        else { "float" }
    }
    else if col.numeric_count > 0 && col.string_count > 0 { "mixed" }
    else { "unknown" }
}

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
            t.add_row_owned(vec!["Type".into(), "cjc source".into()]);
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
    eprintln!("cjc inspect — Deep inspection of structured computational artifacts");
    eprintln!();
    eprintln!("Usage: cjc inspect <file> [flags]");
    eprintln!();
    eprintln!("Supported file types:");
    eprintln!("  .snap          Snap binary artifacts (decode + stats)");
    eprintln!("  .csv / .tsv    Dataset inspection (schema + column stats)");
    eprintln!("  .cjc           Source file analysis (functions, structs, effects)");
    eprintln!("  (other)        Generic file inspection (size, hash)");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  --no-stats       Skip column/value statistics");
    eprintln!("  --no-hash        Skip SHA-256 hash computation");
    eprintln!("  --preview <N>    Max preview items (default: 5)");
    eprintln!("  --plain          Plain text output");
    eprintln!("  --json           JSON output");
    eprintln!("  --color          Color output (default)");
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
