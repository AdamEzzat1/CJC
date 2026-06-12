//! Phase D diagnostics CLI — parent orchestrator + self-respawned child.
//!
//! ```text
//! cargo run --release -p cana-diagnostics                  # full protocol
//! cargo run --release -p cana-diagnostics -- --quick       # smoke profile
//! cargo run --release -p cana-diagnostics -- --subject mem_grad_a1 --subject fp_hot
//! cana_diagnostics --child <subject> <a|b> <iters>         # internal child mode
//! ```

use cana_diagnostics::{
    format_child_line, parse_plan, run_child_workload, run_diagnostics, subjects, Arm, RunOptions,
    Verdict,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // -- Child mode: run one phase, print one line, exit ------------------
    if args.get(1).map(String::as_str) == Some("--child") {
        match child_main(&args[2..]) {
            Ok(()) => return,
            Err(e) => {
                eprintln!("child error: {e}");
                std::process::exit(1);
            }
        }
    }

    // -- Parent mode -------------------------------------------------------
    // Debug builds time the wrong thing; refuse unless explicitly allowed
    // (children inherit the parent's binary, so a release parent always
    // spawns release children).
    let allow_debug = args.iter().any(|a| a == "--allow-debug");
    if cfg!(debug_assertions) && !allow_debug {
        eprintln!(
            "cana-diagnostics: refusing to measure in a debug build — \
             run with --release (or pass --allow-debug for development)"
        );
        std::process::exit(2);
    }

    let child_exe = std::env::current_exe().expect("cannot resolve own binary path");
    let mut opts = if args.iter().any(|a| a == "--quick") {
        RunOptions::quick(child_exe)
    } else {
        RunOptions::standard(child_exe)
    };
    let mut it = args.iter().peekable();
    while let Some(a) = it.next() {
        if a == "--subject" {
            match it.next() {
                Some(name) => opts.subject_filter.push(name.clone()),
                None => {
                    eprintln!("--subject requires a name");
                    std::process::exit(2);
                }
            }
        }
    }

    println!("=================================================================");
    println!(
        "Phase D — silicon diagnostics ({} subjects available; filter: {})",
        subjects().len(),
        if opts.subject_filter.is_empty() {
            "none — running all".to_string()
        } else {
            opts.subject_filter.join(", ")
        }
    );
    println!(
        "Protocol: {} warmup + {} measured phases/arm, ~{:.1} s sustained target",
        opts.warmup_phases,
        opts.measured_phases,
        opts.phase_target_micros as f64 / 1e6
    );
    println!("Determinism gates run FIRST; any failure aborts before timing.");
    println!("=================================================================\n");

    match run_diagnostics(&opts) {
        Ok(outcomes) => {
            println!("\n----------------------------------------------------------------");
            println!(
                "{:<24} | {:>7} | {:>12} | {:>22} | {:>12}",
                "subject", "plans≠", "modeled B/A", "wall B/A med [lo,hi]", "verdict"
            );
            println!("{}", "-".repeat(92));
            for o in &outcomes {
                println!(
                    "{:<24} | {:>7} | {:>12.5} | {:>7.4} [{:.4},{:.4}] | {:>12}",
                    o.name,
                    if o.plans_differ { "yes" } else { "no" },
                    o.modeled_ratio_b,
                    o.wall_ratio.1,
                    o.wall_ratio.0,
                    o.wall_ratio.2,
                    o.wall_verdict.label()
                );
            }
            let wins = outcomes
                .iter()
                .filter(|o| o.wall_verdict == Verdict::Win)
                .count();
            let regs = outcomes
                .iter()
                .filter(|o| o.wall_verdict == Verdict::Regression)
                .count();
            println!(
                "\nWall-clock verdicts: {wins} WIN / {regs} REGRESSION / {} inconclusive \
                 across {} subjects",
                outcomes.len() - wins - regs,
                outcomes.len()
            );
            println!(
                "Artifacts: {}",
                opts.out_dir.join("REPORT.md").display()
            );
        }
        Err(e) => {
            eprintln!("\nHARD ERROR (no timing was trusted): {e}");
            std::process::exit(1);
        }
    }
}

fn child_main(rest: &[String]) -> Result<(), String> {
    let name = rest.first().ok_or("--child requires <subject>")?;
    let arm = Arm::parse(rest.get(1).ok_or("--child requires <a|b>")?.as_str())
        .ok_or("arm must be 'a' or 'b'")?;
    let iters: u64 = rest
        .get(2)
        .ok_or("--child requires <iters>")?
        .parse()
        .map_err(|e| format!("bad iters: {e}"))?;
    let plan_path = rest.get(3).ok_or("--child requires <plan-file>")?;
    let subject = subjects()
        .into_iter()
        .find(|s| s.name == *name)
        .ok_or_else(|| format!("unknown subject {name}"))?;
    let plan_text = std::fs::read_to_string(plan_path)
        .map_err(|e| format!("cannot read plan file {plan_path}: {e}"))?;
    let plan = parse_plan(&plan_text).ok_or_else(|| format!("malformed plan file {plan_path}"))?;
    let m = run_child_workload(&subject, arm, iters, &plan)?;
    println!("{}", format_child_line(&subject.name, arm, &m));
    Ok(())
}
