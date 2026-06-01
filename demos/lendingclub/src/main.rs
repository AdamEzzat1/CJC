//! LendingClub Locke demo — CLI entry point.
//!
//! Usage:
//!
//! ```text
//! lendingclub_demo \
//!     --input  demos/lendingclub/data/accepted_2007_to_2018Q4.csv.gz \
//!     --output demos/lendingclub/out/report.json \
//!     [--max-rows 500000]
//! ```
//!
//! `--max-rows` is optional; without it the full ~2.5 M-row CSV is loaded
//! (peaks at ~3-4 GB resident, see handoff §2). For a laptop budget the
//! sample-size guide is: 500 K rows ≈ 1.5 GB peak, 200 K ≈ 700 MB. The
//! leakage signal does not require the full dataset to fire.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use cjc_locke::json_emit::emit_locke_report_json;

use lendingclub_demo::{
    binarize_loan_status, finding_counts_by_code, load_csv_gz, run_locke_audit,
    SOURCE_TARGET_COLUMN, TARGET_COLUMN,
};

#[derive(Debug)]
struct Args {
    input: PathBuf,
    output: PathBuf,
    max_rows: Option<usize>,
}

fn parse_args() -> Result<Args, String> {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut max_rows: Option<usize> = None;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--input" | "-i" => {
                input = Some(PathBuf::from(it.next().ok_or("--input needs a value")?));
            }
            "--output" | "-o" => {
                output = Some(PathBuf::from(it.next().ok_or("--output needs a value")?));
            }
            "--max-rows" => {
                let s = it.next().ok_or("--max-rows needs a value")?;
                max_rows = Some(s.parse().map_err(|e| format!("--max-rows: {}", e))?);
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {}", other)),
        }
    }
    Ok(Args {
        input: input.ok_or("--input required")?,
        output: output.ok_or("--output required")?,
        max_rows,
    })
}

fn print_usage() {
    eprintln!(
        "lendingclub_demo --input <path.csv.gz> --output <report.json> [--max-rows N]"
    );
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("argument error: {}", e);
            print_usage();
            return ExitCode::from(2);
        }
    };

    eprintln!("[load] reading {}", args.input.display());
    let t0 = Instant::now();
    let (raw_df, n_bytes) = match load_csv_gz(&args.input, args.max_rows) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("load error: {}", e);
            return ExitCode::FAILURE;
        }
    };
    eprintln!(
        "[load] {:.1} MB decompressed, {} rows × {} cols, {:.1}s",
        n_bytes as f64 / 1_048_576.0,
        raw_df.nrows(),
        raw_df.ncols(),
        t0.elapsed().as_secs_f64()
    );

    if raw_df.get_column(SOURCE_TARGET_COLUMN).is_none() {
        eprintln!(
            "fatal: input does not contain `{}` column — is this the LendingClub accepted-loans CSV?",
            SOURCE_TARGET_COLUMN
        );
        return ExitCode::FAILURE;
    }

    eprintln!(
        "[derive] binarizing `{}` → `{}` (Charged Off | Default = 1, Fully Paid = 0; other rows dropped)",
        SOURCE_TARGET_COLUMN, TARGET_COLUMN
    );
    let t1 = Instant::now();
    let df = match binarize_loan_status(raw_df) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("derive error: {}", e);
            return ExitCode::FAILURE;
        }
    };
    let target_default_count = match df.get_column(TARGET_COLUMN) {
        Some(cjc_data::Column::Bool(v)) => v.iter().filter(|b| **b).count(),
        _ => 0,
    };
    let target_default_rate = if df.nrows() == 0 {
        0.0
    } else {
        target_default_count as f64 / df.nrows() as f64
    };
    eprintln!(
        "[derive] kept {} rows × {} cols, {:.2}s; target_default=1 on {} rows ({:.4})",
        df.nrows(),
        df.ncols(),
        t1.elapsed().as_secs_f64(),
        target_default_count,
        target_default_rate,
    );

    eprintln!("[locke] running validate() + detect_target_leakage + detect_id_like_columns");
    let t2 = Instant::now();
    let report = run_locke_audit(&df);
    eprintln!(
        "[locke] {} findings ({} error, {} warning, {} notice, {} info), {:.2}s",
        report.findings.len(),
        report.severity_counts.error,
        report.severity_counts.warning,
        report.severity_counts.notice,
        report.severity_counts.info,
        t2.elapsed().as_secs_f64()
    );

    let counts = finding_counts_by_code(&report.findings);
    eprintln!("[locke] findings by code:");
    for (code, n) in &counts {
        eprintln!("        {} × {}", code, n);
    }

    let json = emit_locke_report_json(&report);
    if let Some(parent) = args.output.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            eprintln!("create_dir_all({}): {}", parent.display(), e);
            return ExitCode::FAILURE;
        }
    }
    if let Err(e) = std::fs::write(&args.output, json.as_bytes()) {
        eprintln!("write {}: {}", args.output.display(), e);
        return ExitCode::FAILURE;
    }
    eprintln!(
        "[done] wrote {} ({} bytes) — total wall {:.2}s",
        args.output.display(),
        json.len(),
        t0.elapsed().as_secs_f64()
    );
    ExitCode::SUCCESS
}
