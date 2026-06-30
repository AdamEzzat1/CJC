//! Physics ML benchmark harness — Phases 1 + 2 + 3.
//!
//! PDEs: heat 1D, wave 1D, viscous Burgers 1D, Allen-Cahn 1D, KdV 1D.
//!
//! Captures accuracy, determinism, and runtime metrics; writes:
//!   target/physics_ml_bench/results.json
//!   target/physics_ml_bench/summary.md
//!
//! Determinism is verified per benchmark by running twice with the same
//! seed and comparing bit-hashes of `final_params` plus bit-equality of
//! the reported metrics.
//!
//! AST/MIR agreement is structural for these builtins (both executors
//! dispatch into the same Rust function), so the harness reports it as
//! `trivial` rather than running redundant work.
//!
//! Run:  cargo run --release -p physics-ml-bench

use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use cjc_ad::pinn::{
    AllenCahnConfig, BurgersConfig, HeatConfig, KdvConfig, PinnResult, WaveConfig,
    allen_cahn_ic_grid, kdv_reference_grid, pinn_allen_cahn_train,
    pinn_burgers_train, pinn_heat_1d_nn_train, pinn_kdv_train,
    pinn_l2_max_errors, pinn_mlp_eval_grid, pinn_wave_train,
};

// ── Determinism hash (matches tests/physics_ml/common.rs) ────────────

fn splitmix64_mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

fn bit_hash_f64(xs: &[f64]) -> u64 {
    let mut h: u64 = 0xCBF29CE484222325;
    for &x in xs {
        h ^= splitmix64_mix(x.to_bits());
        h = h.wrapping_mul(0x100000001B3);
    }
    h
}

// ── Benchmark record ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct BenchRecord {
    name: &'static str,
    equation: &'static str,
    reference_kind: &'static str,
    train_steps: usize,
    seed: u64,
    final_loss: f64,
    l2_error: f64,
    max_error: f64,
    residual_norm: f64,
    runtime_s: f64,
    param_hash: u64,
    deterministic: bool,
    ast_mir_agreement: &'static str,
    l2_threshold: f64,
    max_threshold: f64,
    residual_threshold: f64,
    pass: bool,
}

impl BenchRecord {
    fn to_json(&self) -> String {
        format!(
            "{{\"name\":\"{}\",\"equation\":\"{}\",\"reference_kind\":\"{}\",\
             \"train_steps\":{},\"seed\":{},\
             \"final_loss\":{},\"l2_error\":{},\"max_error\":{},\
             \"residual_norm\":{},\"runtime_s\":{:.6},\
             \"param_hash\":\"0x{:016x}\",\"deterministic\":{},\
             \"ast_mir_agreement\":\"{}\",\
             \"l2_threshold\":{},\"max_threshold\":{},\
             \"residual_threshold\":{},\"pass\":{}}}",
            self.name, self.equation, self.reference_kind,
            self.train_steps, self.seed,
            self.final_loss, self.l2_error, self.max_error,
            self.residual_norm, self.runtime_s,
            self.param_hash, self.deterministic,
            self.ast_mir_agreement,
            self.l2_threshold, self.max_threshold,
            self.residual_threshold, self.pass,
        )
    }
}

// ── Per-PDE harness ──────────────────────────────────────────────────

fn finalize_record(
    name: &'static str,
    equation: &'static str,
    reference_kind: &'static str,
    train_steps: usize,
    seed: u64,
    r1: &PinnResult,
    r2: &PinnResult,
    elapsed: Duration,
    l2_thr: f64,
    max_thr: f64,
    residual_thr: f64,
) -> BenchRecord {
    let h1 = bit_hash_f64(&r1.final_params);
    let h2 = bit_hash_f64(&r2.final_params);
    let deterministic = h1 == h2
        && r1.l2_error.unwrap_or(f64::NAN).to_bits()
            == r2.l2_error.unwrap_or(f64::NAN).to_bits()
        && r1.max_error.unwrap_or(f64::NAN).to_bits()
            == r2.max_error.unwrap_or(f64::NAN).to_bits()
        && r1.mean_residual.to_bits() == r2.mean_residual.to_bits();
    let final_loss = r1.history.last().map(|h| h.total_loss).unwrap_or(f64::NAN);
    let l2 = r1.l2_error.unwrap_or(f64::NAN);
    let max_e = r1.max_error.unwrap_or(f64::NAN);
    let pass = deterministic
        && l2.is_finite() && l2 < l2_thr
        && max_e.is_finite() && max_e < max_thr
        && r1.mean_residual.is_finite() && r1.mean_residual < residual_thr;
    BenchRecord {
        name, equation, reference_kind, train_steps, seed,
        final_loss, l2_error: l2, max_error: max_e,
        residual_norm: r1.mean_residual,
        runtime_s: elapsed.as_secs_f64(),
        param_hash: h1, deterministic,
        ast_mir_agreement: "trivial (both executors dispatch into the same Rust trainer)",
        l2_threshold: l2_thr, max_threshold: max_thr, residual_threshold: residual_thr,
        pass,
    }
}

fn run_heat_1d(epochs: usize, seed: u64, l2_thr: f64, max_thr: f64, residual_thr: f64) -> BenchRecord {
    let cfg = HeatConfig {
        epochs, n_collocation: 64, n_ic: 50, n_bc: 25, seed,
        ..HeatConfig::default()
    };
    let start = Instant::now();
    let r1 = pinn_heat_1d_nn_train(&cfg);
    let elapsed = start.elapsed();
    let r2 = pinn_heat_1d_nn_train(&cfg);
    finalize_record(
        "pinn_heat_1d",
        "u_t = α·u_xx, α=0.01, exact: exp(-α·π²·t)·sin(π·x)",
        "analytical (full grid t∈[0,1])",
        epochs, seed, &r1, &r2, elapsed, l2_thr, max_thr, residual_thr,
    )
}

fn run_wave_1d(epochs: usize, seed: u64, l2_thr: f64, max_thr: f64, residual_thr: f64) -> BenchRecord {
    let cfg = WaveConfig {
        epochs, n_collocation: 64, n_ic: 50, n_bc: 25, seed,
        ..WaveConfig::default()
    };
    let start = Instant::now();
    let r1 = pinn_wave_train(&cfg);
    let elapsed = start.elapsed();
    let r2 = pinn_wave_train(&cfg);
    finalize_record(
        "pinn_wave_1d",
        "u_tt = c²·u_xx, c=1.0, exact: sin(π·x)·cos(c·π·t)",
        "analytical, mid-time slice t=0.5 only",
        epochs, seed, &r1, &r2, elapsed, l2_thr, max_thr, residual_thr,
    )
}

fn run_burgers_1d(epochs: usize, seed: u64, l2_thr: f64, max_thr: f64, residual_thr: f64) -> BenchRecord {
    let cfg = BurgersConfig {
        epochs, n_collocation: 64, n_ic: 50, n_bc: 25, seed,
        ..BurgersConfig::default()
    };
    let start = Instant::now();
    let r1 = pinn_burgers_train(&cfg);
    let elapsed = start.elapsed();
    let r2 = pinn_burgers_train(&cfg);
    finalize_record(
        "pinn_burgers_1d",
        "u_t + u·u_x = ν·u_xx, ν=0.01/π (no closed-form solution)",
        "trainer-reported IC reproduction at t=0 vs -sin(π·x)",
        epochs, seed, &r1, &r2, elapsed, l2_thr, max_thr, residual_thr,
    )
}

/// Build a `BenchRecord` with externally-computed accuracy metrics.
///
/// Used for Allen-Cahn / KdV where the trainer returns `l2_error: None`,
/// so the harness evaluates the trained network at a reference grid via
/// `pinn_mlp_eval_grid` and compares to a known target (KdV soliton or
/// IC reproduction for Allen-Cahn).
fn finalize_record_external(
    name: &'static str,
    equation: &'static str,
    reference_kind: &'static str,
    train_steps: usize,
    seed: u64,
    r1: &PinnResult,
    r2: &PinnResult,
    elapsed: Duration,
    layer_sizes: &[usize],
    inputs: &[f64],
    targets: &[f64],
    l2_thr: f64,
    max_thr: f64,
    residual_thr: f64,
) -> BenchRecord {
    let h1 = bit_hash_f64(&r1.final_params);
    let h2 = bit_hash_f64(&r2.final_params);
    let pred1 = pinn_mlp_eval_grid(layer_sizes, &r1.final_params, inputs);
    let pred2 = pinn_mlp_eval_grid(layer_sizes, &r2.final_params, inputs);
    let (l2_1, max_1) = pinn_l2_max_errors(&pred1, targets);
    let (l2_2, max_2) = pinn_l2_max_errors(&pred2, targets);
    let deterministic = h1 == h2
        && l2_1.to_bits() == l2_2.to_bits()
        && max_1.to_bits() == max_2.to_bits()
        && r1.mean_residual.to_bits() == r2.mean_residual.to_bits();
    let final_loss = r1.history.last().map(|h| h.total_loss).unwrap_or(f64::NAN);
    let pass = deterministic
        && l2_1.is_finite() && l2_1 < l2_thr
        && max_1.is_finite() && max_1 < max_thr
        && r1.mean_residual.is_finite() && r1.mean_residual < residual_thr;
    BenchRecord {
        name, equation, reference_kind, train_steps, seed,
        final_loss, l2_error: l2_1, max_error: max_1,
        residual_norm: r1.mean_residual,
        runtime_s: elapsed.as_secs_f64(),
        param_hash: h1, deterministic,
        ast_mir_agreement: "trivial (both executors dispatch into the same Rust trainer)",
        l2_threshold: l2_thr, max_threshold: max_thr, residual_threshold: residual_thr,
        pass,
    }
}

fn run_allen_cahn_1d(epochs: usize, seed: u64, l2_thr: f64, max_thr: f64, residual_thr: f64) -> BenchRecord {
    let cfg = AllenCahnConfig {
        epochs, n_collocation: 64, n_ic: 50, n_bc: 25, seed,
        ..AllenCahnConfig::default()
    };
    let start = Instant::now();
    let r1 = pinn_allen_cahn_train(&cfg);
    let elapsed = start.elapsed();
    let r2 = pinn_allen_cahn_train(&cfg);
    let (inputs, targets) = allen_cahn_ic_grid(50);
    finalize_record_external(
        "pinn_allen_cahn_1d",
        "u_t = ε²·u_xx + u - u³, ε=0.01 (no closed-form solution)",
        "IC reproduction at t=0 vs x²·cos(π·x) (50 points, externally evaluated)",
        epochs, seed, &r1, &r2, elapsed,
        &cfg.layer_sizes, &inputs, &targets,
        l2_thr, max_thr, residual_thr,
    )
}

fn run_kdv_1d(epochs: usize, seed: u64, l2_thr: f64, max_thr: f64, residual_thr: f64) -> BenchRecord {
    let cfg = KdvConfig {
        epochs, n_collocation: 64, n_ic: 50, n_bc: 25, seed,
        ..KdvConfig::default()
    };
    let start = Instant::now();
    let r1 = pinn_kdv_train(&cfg);
    let elapsed = start.elapsed();
    let r2 = pinn_kdv_train(&cfg);
    // KdV 1-soliton with c=1: full-domain reference grid 50×11 over [-5,5]×[0,1].
    let (inputs, targets) = kdv_reference_grid(-5.0, 5.0, 50, 0.0, 1.0, 11, 1.0);
    finalize_record_external(
        "pinn_kdv_1d",
        "u_t + 6·u·u_x + u_xxx = 0, exact: 0.5·sech²(0.5·(x-t))",
        "analytical 1-soliton, full grid 50×11 over [-5,5]×[0,1]",
        epochs, seed, &r1, &r2, elapsed,
        &cfg.layer_sizes, &inputs, &targets,
        l2_thr, max_thr, residual_thr,
    )
}

// ── Output ───────────────────────────────────────────────────────────

fn write_results_json(records: &[BenchRecord], out_dir: &Path) -> std::io::Result<()> {
    let path = out_dir.join("results.json");
    let mut s = String::from("{\n  \"phase\": 3,\n  \"benchmarks\": [\n");
    for (i, r) in records.iter().enumerate() {
        s.push_str("    ");
        s.push_str(&r.to_json());
        if i + 1 < records.len() { s.push(','); }
        s.push('\n');
    }
    s.push_str("  ]\n}\n");
    fs::write(path, s)
}

fn write_summary_md(records: &[BenchRecord], out_dir: &Path) -> std::io::Result<()> {
    let path = out_dir.join("summary.md");
    let mut s = String::from("# Physics ML Benchmark Suite — Phases 1 + 2 + 3\n\n");
    s.push_str("Deterministic, reference-checked PINN benchmarks for the post-hardening physics-ML stack.\n\n");
    s.push_str("## Results\n\n");
    s.push_str("| Benchmark | Equation | Reference | Steps | Seed | Final Loss | L2 | Max | Residual | Runtime (s) | Param Hash | Determ. | Pass |\n");
    s.push_str("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    for r in records {
        s.push_str(&format!(
            "| {} | {} | {} | {} | {} | {:.4e} | {:.4e} | {:.4e} | {:.4e} | {:.3} | `0x{:016x}` | {} | {} |\n",
            r.name, r.equation, r.reference_kind, r.train_steps, r.seed,
            r.final_loss, r.l2_error, r.max_error, r.residual_norm,
            r.runtime_s, r.param_hash,
            if r.deterministic { "✓" } else { "✗" },
            if r.pass { "PASS" } else { "FAIL" },
        ));
    }
    s.push_str("\n## Determinism\n\n");
    s.push_str("Each benchmark is executed twice with identical seed; the hash above \
        is from the first run. The `Determ.` column is `✓` only when both runs \
        produce bit-identical `final_params`, `l2_error`, `max_error`, and \
        `mean_residual`.\n\n");
    s.push_str("## Reference Solutions\n\n");
    s.push_str("- **heat 1D**: trainer evaluates against `exp(-α·π²·t)·sin(π·x)` on a \
        full (x,t) grid. True full-domain L2.\n");
    s.push_str("- **wave 1D**: trainer evaluates against `sin(π·x)·cos(c·π·t)` at \
        the mid-time slice `t=0.5` only (50 spatial points). Representative \
        but not full-domain.\n");
    s.push_str("- **burgers 1D**: no closed-form analytical solution. Trainer \
        reports IC reproduction error at `t=0` against `-sin(π·x)`. \
        `mean_residual` is the global convergence indicator.\n");
    s.push_str("- **allen-cahn 1D**: no closed-form analytical solution. Harness \
        evaluates the trained network externally via `pinn_mlp_eval_grid` and \
        compares to the IC `x²·cos(π·x)` at t=0 (50 points). `mean_residual` is \
        the most meaningful global signal. Phase 3b will add an implicit-FD \
        reference for full space-time L2.\n");
    s.push_str("- **kdv 1D**: analytical 1-soliton `0.5·sech²(0.5·(x-t))` matches \
        the IC `0.5·sech²(x/2)` with `c=1`. Harness evaluates the trained network \
        externally on a 50×11 (x,t) grid over [-5,5]×[0,1] — full-domain RMSE.\n\n");
    s.push_str("## AST/MIR Agreement\n\n");
    s.push_str("All Phase 1 + 2 trainers are exposed as builtins that pass through \
        to the same Rust function in both executors (`cjc-eval/src/lib.rs:2426+` \
        and `cjc-mir-exec/src/lib.rs:1910+`). They are *structurally* identical \
        and cannot diverge numerically. Cross-executor `.cjcl` roundtrip parity \
        is checked separately by running each example through `cjcl run` and \
        `cjcl run --mir-opt` and diffing — Phase 1 verified this for heat 1D \
        with diff exit 0; the same gate applies to wave + burgers.\n");
    fs::write(path, s)
}

fn main() -> std::io::Result<()> {
    let out_dir = Path::new("target/physics_ml_bench");
    fs::create_dir_all(out_dir)?;

    println!("Physics ML benchmark suite — Phases 1 + 2 + 3");
    println!("==============================================");

    let mut records = Vec::new();

    // ── Heat 1D ───────────────────────────────────────────────────────
    println!("\n[1/10] heat 1D smoke (500 epochs, seed=42) ...");
    let r = run_heat_1d(500, 42, 0.20, 0.40, 0.20);
    println!("  L2={:.4e}  max={:.4e}  res={:.4e}  rt={:.2}s  determ={}  pass={}",
        r.l2_error, r.max_error, r.residual_norm, r.runtime_s, r.deterministic, r.pass);
    records.push(r);

    println!("\n[2/10] heat 1D convergence (2000 epochs, seed=42) ...");
    let r = run_heat_1d(2_000, 42, 0.05, 0.10, 0.05);
    println!("  L2={:.4e}  max={:.4e}  res={:.4e}  rt={:.2}s  determ={}  pass={}",
        r.l2_error, r.max_error, r.residual_norm, r.runtime_s, r.deterministic, r.pass);
    records.push(r);

    // ── Wave 1D ───────────────────────────────────────────────────────
    // Calibrated post-run vs. observed (smoke L2=0.236/max=0.362/res=0.236,
    // conv L2=0.142/max=0.210/res=0.142). ~2-3× headroom.
    println!("\n[3/10] wave 1D smoke (500 epochs, seed=42) ...");
    let r = run_wave_1d(500, 42, 0.50, 1.00, 0.50);
    println!("  L2={:.4e}  max={:.4e}  res={:.4e}  rt={:.2}s  determ={}  pass={}",
        r.l2_error, r.max_error, r.residual_norm, r.runtime_s, r.deterministic, r.pass);
    records.push(r);

    println!("\n[4/10] wave 1D convergence (2000 epochs, seed=42) ...");
    let r = run_wave_1d(2_000, 42, 0.30, 0.50, 0.30);
    println!("  L2={:.4e}  max={:.4e}  res={:.4e}  rt={:.2}s  determ={}  pass={}",
        r.l2_error, r.max_error, r.residual_norm, r.runtime_s, r.deterministic, r.pass);
    records.push(r);

    // ── Burgers 1D ────────────────────────────────────────────────────
    // Calibrated post-run vs. observed (smoke IC L2=0.095/IC max=0.173/res=0.252,
    // conv IC L2=0.056/IC max=0.113/res=0.147). ~3× headroom.
    println!("\n[5/10] burgers 1D smoke (500 epochs, seed=42) ...");
    let r = run_burgers_1d(500, 42, 0.30, 0.55, 0.80);
    println!("  IC L2={:.4e}  IC max={:.4e}  res={:.4e}  rt={:.2}s  determ={}  pass={}",
        r.l2_error, r.max_error, r.residual_norm, r.runtime_s, r.deterministic, r.pass);
    records.push(r);

    println!("\n[6/10] burgers 1D convergence (2000 epochs, seed=42) ...");
    let r = run_burgers_1d(2_000, 42, 0.20, 0.40, 0.50);
    println!("  IC L2={:.4e}  IC max={:.4e}  res={:.4e}  rt={:.2}s  determ={}  pass={}",
        r.l2_error, r.max_error, r.residual_norm, r.runtime_s, r.deterministic, r.pass);
    records.push(r);

    // ── Allen-Cahn 1D ─────────────────────────────────────────────────
    // Calibrated post-run vs. observed (smoke IC L2=0.129/IC max=0.280/res=0.0265,
    // conv IC L2=0.0097/IC max=0.0177/res=0.0034). ~3× headroom.
    println!("\n[7/10] allen-cahn 1D smoke (500 epochs, seed=42) ...");
    let r = run_allen_cahn_1d(500, 42, 0.40, 0.85, 0.10);
    println!("  IC L2={:.4e}  IC max={:.4e}  res={:.4e}  rt={:.2}s  determ={}  pass={}",
        r.l2_error, r.max_error, r.residual_norm, r.runtime_s, r.deterministic, r.pass);
    records.push(r);

    println!("\n[8/10] allen-cahn 1D convergence (2000 epochs, seed=42) ...");
    let r = run_allen_cahn_1d(2_000, 42, 0.05, 0.10, 0.02);
    println!("  IC L2={:.4e}  IC max={:.4e}  res={:.4e}  rt={:.2}s  determ={}  pass={}",
        r.l2_error, r.max_error, r.residual_norm, r.runtime_s, r.deterministic, r.pass);
    records.push(r);

    // ── KdV 1D ────────────────────────────────────────────────────────
    // Calibrated post-run vs. observed (smoke L2=0.0380/max=0.118/res=3.14e-3,
    // conv L2=5.23e-3/max=1.60e-2/res=1.73e-4). ~3× headroom.
    println!("\n[9/10] kdv 1D smoke (500 epochs, seed=42) ...");
    let r = run_kdv_1d(500, 42, 0.15, 0.40, 0.02);
    println!("  L2={:.4e}  max={:.4e}  res={:.4e}  rt={:.2}s  determ={}  pass={}",
        r.l2_error, r.max_error, r.residual_norm, r.runtime_s, r.deterministic, r.pass);
    records.push(r);

    println!("\n[10/10] kdv 1D convergence (2000 epochs, seed=42) ...");
    let r = run_kdv_1d(2_000, 42, 0.02, 0.06, 1.0e-3);
    println!("  L2={:.4e}  max={:.4e}  res={:.4e}  rt={:.2}s  determ={}  pass={}",
        r.l2_error, r.max_error, r.residual_norm, r.runtime_s, r.deterministic, r.pass);
    records.push(r);

    write_results_json(&records, out_dir)?;
    write_summary_md(&records, out_dir)?;

    let n_pass = records.iter().filter(|r| r.pass).count();
    println!("\n{} / {} benchmarks passed", n_pass, records.len());
    println!("Wrote {} and {}",
        out_dir.join("results.json").display(),
        out_dir.join("summary.md").display(),
    );

    if n_pass != records.len() {
        std::process::exit(1);
    }
    Ok(())
}
