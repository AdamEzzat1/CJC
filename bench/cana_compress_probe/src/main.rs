//! Phase E probe CLI.
//!
//! ```text
//! cargo run --release -p cana-compress-probe                # all three prototypes
//! cargo run --release -p cana-compress-probe -- <ckpt.bin>  # explicit checkpoint path
//! ```

use std::fs;
use std::path::PathBuf;

use cana_compress_probe::{
    checkpoint_candidates, measure_checkpoint, measure_disk_artifact, measure_trace_subject,
    render_report, CheckpointOutcome, DISK_ARTIFACTS, TRACE_SUBJECTS,
};
use cana_diagnostics::workspace_root;

fn main() {
    println!("=================================================================");
    println!("Phase E — compression prototypes (before/after bytes, bounded error)");
    println!("=================================================================\n");

    // -- Prototype 1: trace streams --------------------------------------
    let mut traces = Vec::new();
    for name in TRACE_SUBJECTS {
        match measure_trace_subject(name) {
            Ok(t) => {
                println!(
                    "trace {:<22} {:>9} events  canonical RLE {:.2}x motif {:.2}x | delta RLE {:.2}x motif {:.2}x",
                    t.subject,
                    t.measured_events,
                    t.canonical.rle_ratio(),
                    t.canonical.motif_ratio(),
                    t.delta.rle_ratio(),
                    t.delta.motif_ratio(),
                );
                traces.push(t);
            }
            Err(e) => {
                eprintln!("HARD ERROR (trace {name}): {e}");
                std::process::exit(1);
            }
        }
    }

    // -- Prototype 2: checkpoint ------------------------------------------
    let explicit: Option<PathBuf> = std::env::args().nth(1).map(PathBuf::from);
    let candidates = match &explicit {
        Some(p) => vec![p.clone()],
        None => checkpoint_candidates(),
    };
    let mut checkpoint: Option<CheckpointOutcome> = None;
    for cand in &candidates {
        if cand.exists() {
            match measure_checkpoint(cand) {
                Ok(c) => {
                    let (before, after) = c.totals();
                    println!(
                        "\ncheckpoint {} : tensor payload {} -> {} bytes ({:.2}x), {} matrices",
                        c.source,
                        before,
                        after,
                        before as f64 / after.max(1) as f64,
                        c.matrices.len()
                    );
                    checkpoint = Some(c);
                    break;
                }
                Err(e) => {
                    eprintln!("HARD ERROR (checkpoint {}): {e}", cand.display());
                    std::process::exit(1);
                }
            }
        }
    }
    if checkpoint.is_none() {
        println!("\ncheckpoint: SKIPPED (no artifact found; tried {} candidate paths)", candidates.len());
    }

    // -- Prototype 3: disk artifacts ---------------------------------------
    let mut disks = Vec::new();
    for rel in DISK_ARTIFACTS {
        match measure_disk_artifact(rel) {
            Ok(d) => {
                println!(
                    "disk {:<48} RLE {:.2}x motif {:.2}x",
                    d.path,
                    d.outcome.rle_ratio(),
                    d.outcome.motif_ratio()
                );
                disks.push(d);
            }
            Err(e) => {
                eprintln!("HARD ERROR (disk {rel}): {e}");
                std::process::exit(1);
            }
        }
    }

    // -- Report -------------------------------------------------------------
    let out_dir = workspace_root().join("bench_results/cana_compress_probe");
    fs::create_dir_all(&out_dir).expect("create bench_results/cana_compress_probe");
    let report = render_report(&traces, checkpoint.as_ref(), &disks);
    let report_path = out_dir.join("REPORT.md");
    fs::write(&report_path, report).expect("write REPORT.md");
    println!("\nReport: {}", report_path.display());
}
