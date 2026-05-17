//! ABNG Result-Path Profiler — Phase 0.9.5 Research Phase R0
//! =========================================================
//!
//! Empirical, segment-by-segment cost measurement of the ABNG **result
//! path** on the real Diabetes-130 workload, so Research Phase R0's
//! speedups can be ranked against measured hotspots rather than code
//! reading (handoff §R0.5 — "Profile the full result path with real
//! measurements").
//!
//! The result path, as the COMMIT-5 Diabetes-130 harness drives it, is
//! one iteration of:
//!
//! ```text
//!   transform.transform(row)   -> (x, phi, y)
//!   g.encode_prefix(&x)        -> prefix
//!   g.descend(&prefix)         -> leaf_id
//!   g.train_step(&x, &phi, y)  -> leaf BLR update + state_hash + observe + append_event
//!   g.blr_update(0, &phi, &y)  -> root BLR update + state_hash + append_event
//! ```
//!
//! The profiler isolates each segment by timing the public APIs the
//! result path is composed of, plus two synthetic isolations
//! (`sha256_d2`, `cholesky_d`) that pin where the per-row hashing /
//! refactor cost actually goes. No production code is instrumented.
//!
//! ## Reading the numbers
//!
//! Each block reports **min** and **median** over several trials.
//! Under the documented machine contention (handoff §R0.4 #4 — "the
//! box was oversubscribed; the test averaged ~0.8 cores"), absolute
//! wall-clock carries a noise factor; **ratios between segments
//! measured in the same run are contention-robust** and are what the
//! R0 design doc relies on. `train_step` is timed on a graph whose
//! leaves are all pre-warmed, so it measures the steady-state O(d²)
//! rank-1 path, not the one-off O(d³) `cholesky` miss.
//!
//! Invocation:
//!     cargo run -p abng-result-profile --release
//!
//! Skips gracefully (exit 0) when the untracked dataset is absent.

use std::path::PathBuf;
use std::time::Instant;

use cjc_abng::blr::{chol_rank1_update, cholesky, cholesky_solve, BlrPrior, BlrState};
use cjc_abng::categorical::{
    CategoricalTransform, ColumnRole, RarePolicy, Schema, TransformConfig,
};
use cjc_abng::graph::{AdaptiveBeliefGraph, BLR_CHECKPOINT_INTERVAL};
use cjc_ad::pinn::Activation;

// ── Configuration ───────────────────────────────────────────────────

const N_COLUMNS: usize = 50;
const TARGET_COL: usize = 49;
/// Rows the `CategoricalTransform` is fitted on. Bigger than the result-
/// path sample so the fitted vocabulary is realistic.
const FIT_ROWS: usize = 8_000;
/// Rows transformed into `(x, phi, y)` triples that drive the result
/// path. The end-to-end blocks cycle through these.
const PROFILE_ROWS: usize = 4_000;

const ROUTE_BINS: u8 = 4;
const K_ROUTING: usize = 3;
const MAX_REAL: u32 = 8;

const BLR_PRIOR_PRECISION: f64 = 0.1;
const BLR_PRIOR_A: f64 = 1.0;
const BLR_PRIOR_B: f64 = 0.5;

const TRIAL_SEED: u64 = 42;

/// `n_train` of a 20 000-row stratified sub-sample at 80 % train — the
/// handoff's ~20K Dataset-A scaling rung. The extrapolation block
/// projects per-row costs onto this and the full-101 766 train split.
const REF_N_TRAIN_20K: usize = 16_000;
const REF_N_TRAIN_FULL: usize = 81_412;

// ── CSV + schema (mirrors the COMMIT-5 harness) ─────────────────────

fn read_csv(bytes: &[u8]) -> Result<Vec<Vec<String>>, String> {
    let text = std::str::from_utf8(bytes).map_err(|e| format!("utf-8: {e}"))?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().ok_or("empty CSV")?.split(',').collect();
    if header.len() != N_COLUMNS {
        return Err(format!("header has {} columns", header.len()));
    }
    let mut rows: Vec<Vec<String>> = Vec::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let cells: Vec<String> = line.split(',').map(|s| s.trim().to_string()).collect();
        if cells.len() == N_COLUMNS {
            rows.push(cells);
        }
    }
    Ok(rows)
}

/// The Diabetes-130 schema — verbatim from the COMMIT-5 harness so the
/// profiled `phi` width matches the benchmark exactly.
fn diabetes_schema() -> Schema {
    use ColumnRole::*;
    let med = CategoricalPhiOnly;
    let cols: &[(&str, ColumnRole)] = &[
        ("encounter_id", Ignore),
        ("patient_nbr", Ignore),
        ("race", Categorical),
        ("gender", Categorical),
        ("age", Categorical),
        ("weight", Categorical),
        ("admission_type_id", Categorical),
        ("discharge_disposition_id", Categorical),
        ("admission_source_id", Categorical),
        ("time_in_hospital", Numeric),
        ("payer_code", Categorical),
        ("medical_specialty", Categorical),
        ("num_lab_procedures", Numeric),
        ("num_procedures", Numeric),
        ("num_medications", Numeric),
        ("number_outpatient", Numeric),
        ("number_emergency", Numeric),
        ("number_inpatient", Numeric),
        ("diag_1", CategoricalPhiOnly),
        ("diag_2", CategoricalPhiOnly),
        ("diag_3", CategoricalPhiOnly),
        ("number_diagnoses", Numeric),
        ("max_glu_serum", Categorical),
        ("A1Cresult", Categorical),
        ("metformin", med),
        ("repaglinide", med),
        ("nateglinide", med),
        ("chlorpropamide", med),
        ("glimepiride", med),
        ("acetohexamide", med),
        ("glipizide", med),
        ("glyburide", med),
        ("tolbutamide", med),
        ("pioglitazone", med),
        ("rosiglitazone", med),
        ("acarbose", med),
        ("miglitol", med),
        ("troglitazone", med),
        ("tolazamide", med),
        ("examide", med),
        ("citoglipton", med),
        ("insulin", med),
        ("glyburide-metformin", med),
        ("glipizide-metformin", med),
        ("glimepiride-pioglitazone", med),
        ("metformin-rosiglitazone", med),
        ("metformin-pioglitazone", med),
        ("change", Categorical),
        ("diabetesMed", Categorical),
        ("readmitted", Target),
    ];
    Schema::new(cols.iter().map(|(n, r)| (n.to_string(), *r)).collect())
}

fn transform_config(raw_hash: [u8; 32], row_count: u64) -> TransformConfig {
    TransformConfig {
        route_bins: ROUTE_BINS,
        k_routing: K_ROUTING,
        max_real: MAX_REAL,
        rare_policy: RarePolicy::DEFAULT,
        missing_markers: vec!["?".to_string(), String::new()],
        target_positives: vec!["<30".to_string()],
        target_definition: "readmitted == '<30'".to_string(),
        raw_dataset_hash: raw_hash,
        split_seed: TRIAL_SEED,
        row_count,
    }
}

/// Pre-allocate a full `branching^depth` routing tree by BFS expansion
/// (mirrors the COMMIT-5 harness).
fn pre_allocate_full_tree(g: &mut AdaptiveBeliefGraph, branching: u8, depth: usize) {
    let mut level: Vec<u32> = vec![0];
    for _ in 0..depth {
        let mut next: Vec<u32> = Vec::with_capacity(level.len() * branching as usize);
        for &parent in &level {
            for key in 0..branching {
                next.push(g.add_node(parent, key).expect("add_node"));
            }
        }
        level = next;
    }
}

fn build_graph(transform: &CategoricalTransform) -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(TRIAL_SEED);
    let n_routing = transform.n_routing_features();
    let route_bins = transform.route_bins();
    let mut boundaries: Vec<f64> = Vec::new();
    for _ in 0..n_routing {
        for k in 1..route_bins {
            boundaries.push(k as f64 - 0.5);
        }
    }
    g.set_codebook(n_routing, route_bins as u16, &boundaries)
        .expect("codebook");
    g.set_leaf_head(transform.phi_width() as u32, vec![], 1, Activation::None)
        .expect("leaf head");
    g.set_blr_prior(BLR_PRIOR_PRECISION, BLR_PRIOR_A, BLR_PRIOR_B)
        .expect("blr prior");
    pre_allocate_full_tree(&mut g, route_bins, n_routing);
    g
}

// ── Timing helpers ──────────────────────────────────────────────────

/// (min, median) per-iteration ns over `trials` runs of `iters` each.
/// `setup` rebuilds mutable state before every trial; a one-tenth
/// warmup precedes the timed window. Median is the headline number —
/// it is robust to the high-side outliers a contended box produces;
/// min is reported alongside as the noise-floor estimate.
fn bench<S, T, F>(trials: usize, iters: usize, mut setup: S, mut body: F) -> (f64, f64)
where
    S: FnMut() -> T,
    F: FnMut(&mut T, usize),
{
    let mut samples: Vec<f64> = Vec::with_capacity(trials);
    for _ in 0..trials {
        let mut state = setup();
        for i in 0..(iters / 10).max(1) {
            body(&mut state, i);
        }
        let start = Instant::now();
        for i in 0..iters {
            body(&mut state, i);
        }
        samples.push(start.elapsed().as_nanos() as f64 / iters as f64);
    }
    samples.sort_by(|a, b| a.total_cmp(b));
    (samples[0], samples[samples.len() / 2])
}

fn fmt(ns: f64) -> String {
    if ns >= 1e6 {
        format!("{:.3} ms", ns / 1e6)
    } else if ns >= 1e3 {
        format!("{:.2} us", ns / 1e3)
    } else {
        format!("{ns:.1} ns")
    }
}

fn emit(name: &str, min_ns: f64, med_ns: f64) {
    eprintln!(
        "  {name:<28}: {:>12} median  ({:>12} min)",
        fmt(med_ns),
        fmt(min_ns),
    );
    println!(
        r#"{{"op":"{name}","median_ns":{med_ns:.2},"min_ns":{min_ns:.2}}}"#
    );
}

fn main() {
    eprintln!("=== ABNG Result-Path Profiler (Phase 0.9.5 Research Phase R0) ===");

    // ── load ──────────────────────────────────────────────────────────
    let path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"), "..", "..", "tests", "data",
        "diabetes_130", "diabetic_data.csv",
    ]
    .iter()
    .collect();
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("[skip] dataset absent at {}", path.display());
            eprintln!("       fetch UCI dataset 296 -> tests/data/diabetes_130/");
            return;
        }
    };
    let load_start = Instant::now();
    let raw_hash = cjc_snap::hash::sha256(&bytes);
    let rows = read_csv(&bytes).expect("CSV parses");
    let load_ns = load_start.elapsed().as_nanos() as f64;
    eprintln!(
        "  loaded {} rows ({:.1} MB) + parsed + hashed in {:.1} ms",
        rows.len(),
        bytes.len() as f64 / 1e6,
        load_ns / 1e6,
    );

    // ── fit the transform ─────────────────────────────────────────────
    let schema = diabetes_schema();
    let fit_rows: Vec<Vec<String>> = rows
        .iter()
        .filter(|r| r[TARGET_COL] != "?" && !r[TARGET_COL].is_empty())
        .take(FIT_ROWS)
        .cloned()
        .collect();
    let config = transform_config(raw_hash, rows.len() as u64);
    let transform =
        CategoricalTransform::fit(&schema, &fit_rows, &config).expect("transform fit");
    let d = transform.phi_width();
    eprintln!(
        "  transform fitted: phi_width d={d}  n_routing={}  route_bins={}",
        transform.n_routing_features(),
        transform.route_bins(),
    );

    // ── transform: time per-row, build the (x, phi, y) triples ────────
    let mut triples: Vec<(Vec<f64>, Vec<f64>, f64)> = Vec::with_capacity(PROFILE_ROWS);
    let mut transform_ns_total = 0.0f64;
    let mut transform_calls = 0usize;
    for r in rows.iter().take(PROFILE_ROWS * 2) {
        let start = Instant::now();
        let out = transform.transform(r);
        transform_ns_total += start.elapsed().as_nanos() as f64;
        transform_calls += 1;
        if let Ok(t) = out {
            triples.push(t);
        }
        if triples.len() >= PROFILE_ROWS {
            break;
        }
    }
    let transform_ns = transform_ns_total / transform_calls as f64;
    assert!(!triples.is_empty(), "no transformable rows");
    let nt = triples.len();
    eprintln!("  built {nt} (x, phi, y) triples");

    let template = build_graph(&transform);
    eprintln!(
        "  graph built: {} nodes, leaf BLR dimension d={d}",
        template.node_count(),
    );
    eprintln!("--- segment timings (median + min over trials) ---");

    let prior = BlrPrior::new(BLR_PRIOR_PRECISION, BLR_PRIOR_A, BLR_PRIOR_B).unwrap();
    let phi0 = triples[0].1.clone();

    // ── encode_prefix + descend ───────────────────────────────────────
    let (enc_min, enc_med) = bench(
        7,
        50_000,
        || template.clone(),
        |g, i| {
            let _ = g.encode_prefix(&triples[i % nt].0).unwrap();
        },
    );
    emit("encode_prefix", enc_min, enc_med);

    let prefixes: Vec<Vec<u8>> = triples
        .iter()
        .map(|(x, _, _)| template.encode_prefix(x).unwrap())
        .collect();
    let (desc_min, desc_med) = bench(
        7,
        50_000,
        || (),
        |_, i| {
            let _ = template.descend(&prefixes[i % nt]);
        },
    );
    emit("descend", desc_min, desc_med);

    // ── BLR update (n=1 rank-1 path), direct ──────────────────────────
    let (upd_min, upd_med) = bench(
        9,
        12_000,
        || {
            // Warm one full cholesky so the timed window is the
            // steady-state O(d²) rank-1 path, not the O(d³) miss.
            let mut s = BlrState::from_prior(&prior, d as u32);
            s.update(&phi0, &[0.3]).unwrap();
            s
        },
        |s, i| {
            let (_, phi, _) = &triples[i % nt];
            s.update(phi, &[0.3 + (i as f64) * 1e-6]).unwrap();
        },
    );
    emit("blr_update_math (n=1)", upd_min, upd_med);

    // ── BLR state_hash, direct ────────────────────────────────────────
    let warm_state = {
        let mut s = BlrState::from_prior(&prior, d as u32);
        for _ in 0..32 {
            s.update(&phi0, &[0.3]).unwrap();
        }
        s
    };
    let (sh_min, sh_med) = bench(9, 600, || (), |_, _| {
        let _ = warm_state.state_hash();
    });
    emit("blr_state_hash", sh_min, sh_med);

    // ── canonical_bytes alone (alloc + serialize, no SHA) ─────────────
    let (cb_min, cb_med) = bench(9, 3_000, || (), |_, _| {
        let _ = std::hint::black_box(warm_state.canonical_bytes());
    });
    emit("  canonical_bytes (no SHA)", cb_min, cb_med);

    // ── raw SHA-256 over a d*d*8-byte buffer (pure compression) ───────
    let sha_buf = vec![0x5Au8; d * d * 8];
    let (sha_min, sha_med) = bench(9, 600, || (), |_, _| {
        let _ = cjc_snap::hash::sha256(&sha_buf);
    });
    emit("  sha256(d*d*8 bytes)", sha_min, sha_med);

    // ── cholesky(d) — the chol_factor-miss / first-update cost ────────
    let warm_prec = warm_state.precision.to_vec();
    let (chol_min, chol_med) = bench(9, 300, || (), |_, _| {
        let _ = cholesky(&warm_prec, d).unwrap();
    });
    emit("cholesky_d (update miss)", chol_min, chol_med);

    // ── update sub-costs (Research Phase R1) ──────────────────────────
    // Where the steady O(d²) rank-1 NIG update's time goes. The 477 KB
    // `precision.to_vec()` clone, the Givens rank-1 factor update, and
    // the triangular solve are directly measurable; the remaining
    // passes (matvec, two quadratic forms, the Λ+φφᵀ build) are O(d²)
    // each and estimated by subtraction in the R1 design doc.
    let d2 = d * d;
    let clone_src = vec![0.5f64; d2];
    let (clone_min, clone_med) = bench(9, 5_000, || (), |_, _| {
        let _ = std::hint::black_box(clone_src.clone());
    });
    emit("  vec clone d*d (~precision.to_vec)", clone_min, clone_med);

    let (cr1_min, cr1_med) = bench(
        9,
        2_000,
        || cholesky(&warm_prec, d).unwrap(),
        |l, _| chol_rank1_update(l, d, &phi0),
    );
    emit("  chol_rank1_update O(d²)", cr1_min, cr1_med);

    let chol_l = cholesky(&warm_prec, d).unwrap();
    let solve_rhs = vec![0.7f64; d];
    let (cs_min, cs_med) = bench(9, 2_000, || (), |_, _| {
        let _ = cholesky_solve(&chol_l, d, &solve_rhs);
    });
    emit("  cholesky_solve O(d²)", cs_min, cs_med);

    // ── BLR predict — cache hit and cache miss ────────────────────────
    let (ph_min, ph_med, pm_min, pm_med) = {
        let mut s = BlrState::from_prior(&prior, d as u32);
        for _ in 0..64 {
            s.update(&phi0, &[0.3]).unwrap();
        }
        let _ = s.predict(&phi0).unwrap();
        let (hmin, hmed) = bench(9, 3_000, || (), |_, _| {
            let _ = s.predict(&phi0).unwrap();
        });
        let (mmin, mmed) = bench(9, 400, || (), |_, _| {
            *s.cached_l.borrow_mut() = None;
            let _ = s.predict(&phi0).unwrap();
        });
        (hmin, hmed, mmin, mmed)
    };
    emit("blr_predict (cache hit)", ph_min, ph_med);
    emit("blr_predict (cache miss)", pm_min, pm_med);

    // ── graph.observe — proxy for append_event cost ───────────────────
    let (obs_min, obs_med) = bench(
        7,
        20_000,
        || template.clone(),
        |g, i| {
            g.observe(1, 0.1 + (i as f64) * 1e-6).unwrap();
        },
    );
    emit("graph.observe (~append_event)", obs_min, obs_med);

    // ── train_step end-to-end, leaves fully pre-warmed ────────────────
    //
    // The setup clones `template` and runs one `train_step` per triple
    // so every reachable leaf has its `chol_factor` cached. The timed
    // window is then the steady-state per-row cost: O(d²) rank-1
    // update + state_hash + observe + append_event, no O(d³) miss.
    let (ts_min, ts_med) = bench(
        9,
        3_000,
        || {
            let mut g = template.clone();
            for (x, phi, y) in &triples {
                g.train_step(x, phi, *y).unwrap();
            }
            g
        },
        |g, i| {
            let (x, phi, y) = &triples[i % nt];
            g.train_step(x, phi, *y).unwrap();
        },
    );
    emit("train_step (steady, end-to-end)", ts_min, ts_med);

    // ── consistency cross-check ───────────────────────────────────────
    // train_step ≈ blr_update_math + state_hash + (encode + descend +
    // observe + validate). The last group is sub-µs..µs; train_step is
    // ms-scale, so the check is essentially update + state_hash.
    let implied = upd_med + sh_med;
    eprintln!(
        "  [check] train_step {:.3} ms  vs  update+state_hash {:.3} ms  (delta {:+.1} %)",
        ts_med / 1e6,
        implied / 1e6,
        100.0 * (ts_med - implied) / implied,
    );

    // ── per-row breakdown ─────────────────────────────────────────────
    //
    // The harness pays, per training row: transform + encode_prefix +
    // descend (once each) + train_step (leaf) + blr_update (root). The
    // root `blr_update` is the same shape of work as `train_step`'s BLR
    // path — one O(d²) update + one state_hash + one append_event — so
    // it is modelled here as a second `train_step`-equivalent.
    let segs: [(&str, f64); 5] = [
        ("transform", transform_ns),
        ("encode_prefix", enc_med),
        ("descend", desc_med),
        ("train_step (leaf)", ts_med),
        ("blr_update (root) ~= train_step", ts_med),
    ];
    let per_row: f64 = segs.iter().map(|(_, ns)| *ns).sum();

    eprintln!("--- per-row result-path breakdown (median-based) ---");
    for (name, ns) in &segs {
        eprintln!(
            "  {name:<32}: {:>10.3} us/row  ({:>5.1} %)",
            ns / 1e3,
            100.0 * ns / per_row,
        );
    }
    eprintln!("  {:<32}: {:>10.3} us/row", "TOTAL", per_row / 1e3);

    // Reference component costs. Since Phase 0.9.5 R0-3 (Tier 2
    // Option C) `train_step` does one O(d²) BLR update *every* row but
    // a full d×d `state_hash` only every BLR_CHECKPOINT_INTERVAL rows,
    // so these components do NOT linearly sum to the per-row total —
    // they are reference points, not a decomposition.
    eprintln!(
        "  reference: full state_hash      = {:>9.3} us  (Option C: 1 row in {})",
        sh_med / 1e3,
        BLR_CHECKPOINT_INTERVAL,
    );
    eprintln!(
        "  reference: O(d²) rank-1 update  = {:>9.3} us  (every row)",
        upd_med / 1e3,
    );
    eprintln!(
        "  reference: of a full state_hash, ~{:.0}% is SHA-256 compression",
        (100.0 * sha_med / sh_med).min(100.0),
    );

    // ── extrapolation ─────────────────────────────────────────────────
    eprintln!("--- extrapolation (training only, single-threaded, no contention) ---");
    for (label, n) in [
        ("20K sub-sample", REF_N_TRAIN_20K),
        ("full 101,766", REF_N_TRAIN_FULL),
    ] {
        let secs = per_row * n as f64 / 1e9;
        eprintln!(
            "  n_train={n:<7} ({label:<16}): {:.1} s CPU  ({:.1} min)",
            secs,
            secs / 60.0,
        );
    }

    println!(
        r#"{{"op":"summary","d":{d},"per_row_ns":{per_row:.0},"state_hash_ns":{sh_med:.0},"sha256_d2_ns":{sha_med:.0},"canonical_bytes_ns":{cb_med:.0},"blr_update_ns":{upd_med:.0},"train_step_ns":{ts_med:.0}}}"#
    );
    eprintln!("=== Done ===");
}
