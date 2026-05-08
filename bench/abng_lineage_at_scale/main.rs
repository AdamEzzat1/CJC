//! ABNG lineage attestation at scale (Phase 0.6 Item 2)
//! ======================================================
//!
//! Establishes wall-clock baselines for the full lineage flow at 10^4
//! rows: train + stamp + predict_snap pack + predict_snap unpack +
//! serialize + replay. Mirrors the structure of
//! `tests/test_abng_lineage_attestation.rs` but at production-realistic
//! row count, so Item 4's batch observe and Item 3's smart-replay
//! fast-forward can be measured against this baseline.
//!
//! Truth (clinical-trial dose-response toy):
//!     response = 0.2 + 0.6 * dose + 0.1 * dose^2
//!
//! Invocation:
//!     cargo run -p abng-lineage-at-scale --release > abng_lineage_at_scale.jsonl

use cjc_abng::graph::AdaptiveBeliefGraph;
use cjc_abng::predict_snap;
use cjc_abng::serialize::{replay, serialize};
use cjc_ad::pinn::Activation;
use std::time::Instant;

type Row = (u32, f64, f64);

fn make_dataset(n: usize) -> Vec<Row> {
    (0..n as u32)
        .map(|i| {
            let dose = (i as f64 + 1.0) / n as f64;
            let response = 0.2 + 0.6 * dose + 0.1 * dose * dose;
            (i, dose, response)
        })
        .collect()
}

fn dataset_fingerprint(rows: &[Row]) -> [u8; 32] {
    let mut buf: Vec<u8> = Vec::with_capacity(rows.len() * 20);
    for &(pid, dose, resp) in rows {
        buf.extend_from_slice(&pid.to_be_bytes());
        buf.extend_from_slice(&dose.to_bits().to_be_bytes());
        buf.extend_from_slice(&resp.to_bits().to_be_bytes());
    }
    cjc_snap::hash::sha256(&buf)
}

fn features(dose: f64) -> [f64; 4] {
    [1.0, dose, dose * dose, dose * dose * dose]
}

fn build_graph(seed: u64) -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(seed);
    g.set_codebook(1, 4, &[0.25, 0.5, 0.75]).unwrap();
    g.set_leaf_head(1, vec![4], 1, Activation::Tanh).unwrap();
    g.set_blr_prior(2.0, 1.0, 0.5).unwrap();
    g.set_density_tracker().unwrap();
    g.set_calibration(15).unwrap();
    for byte in 0u8..4 {
        g.add_node(0, byte).unwrap();
    }
    g
}

fn route_leaf(g: &AdaptiveBeliefGraph, dose: f64) -> u32 {
    let prefix = g.encode_prefix(&[dose]).unwrap();
    g.descend(&prefix).leaf_id
}

fn run_at_scale(n_rows: usize, seed: u64) {
    let dataset = make_dataset(n_rows);

    // ── stamp + train ─────────────────────────────────────────────────
    let stamp_start = Instant::now();
    let stamp = dataset_fingerprint(&dataset);
    let stamp_elapsed = stamp_start.elapsed();

    let mut g = build_graph(seed);
    g.stamp_provenance(0, stamp).unwrap();

    let train_start = Instant::now();
    for &(_pid, dose, resp) in &dataset {
        let leaf = route_leaf(&g, dose);
        let phi = features(dose);
        g.blr_update(leaf, &phi, &[resp]).unwrap();
        g.observe(leaf, resp).unwrap();
    }
    let train_elapsed = train_start.elapsed();

    // ── predict_snap pack ─────────────────────────────────────────────
    // Pack one prediction per leaf to amortize across leaf coverage.
    let pack_start = Instant::now();
    let n_pack = 1_000usize.min(n_rows);
    let mut packed: Vec<Vec<u8>> = Vec::with_capacity(n_pack);
    for i in 0..n_pack {
        let dose = (i as f64 + 0.5) / n_pack as f64;
        let leaf = route_leaf(&g, dose);
        let phi = features(dose);
        packed.push(predict_snap::pack(&g, leaf, &phi).unwrap());
    }
    let pack_elapsed = pack_start.elapsed();

    // ── predict_snap unpack ───────────────────────────────────────────
    let unpack_start = Instant::now();
    for blob in &packed {
        let _ = predict_snap::unpack(blob).unwrap();
    }
    let unpack_elapsed = unpack_start.elapsed();

    // ── serialize ─────────────────────────────────────────────────────
    let serialize_start = Instant::now();
    let bytes = serialize(&g);
    let serialize_elapsed = serialize_start.elapsed();
    let serialize_bytes = bytes.len();

    // ── replay ────────────────────────────────────────────────────────
    let replay_start = Instant::now();
    let g2 = replay(&bytes).unwrap();
    let replay_elapsed = replay_start.elapsed();
    assert_eq!(g.chain_head, g2.chain_head, "replay lost determinism");

    println!(
        r#"{{"n_rows":{n_rows},"stamp_ms":{:.3},"train_ms":{:.3},"train_per_row_ns":{:.1},"pack_n":{n_pack},"pack_ms":{:.3},"unpack_ms":{:.3},"serialize_ms":{:.3},"serialize_bytes":{serialize_bytes},"replay_ms":{:.3}}}"#,
        stamp_elapsed.as_secs_f64() * 1000.0,
        train_elapsed.as_secs_f64() * 1000.0,
        train_elapsed.as_nanos() as f64 / n_rows as f64,
        pack_elapsed.as_secs_f64() * 1000.0,
        unpack_elapsed.as_secs_f64() * 1000.0,
        serialize_elapsed.as_secs_f64() * 1000.0,
        replay_elapsed.as_secs_f64() * 1000.0,
    );
    eprintln!(
        "  n_rows={n_rows:>6}  train={:>7.1}ms  pack={:>6.1}ms  serialize={:>7.1}ms ({} B)  replay={:>7.1}ms",
        train_elapsed.as_secs_f64() * 1000.0,
        pack_elapsed.as_secs_f64() * 1000.0,
        serialize_elapsed.as_secs_f64() * 1000.0,
        serialize_bytes,
        replay_elapsed.as_secs_f64() * 1000.0,
    );
}

fn main() {
    eprintln!("=== ABNG lineage at scale baseline (Phase 0.6 Item 2) ===");
    eprintln!("Truth: response = 0.2 + 0.6*dose + 0.1*dose^2");
    let seed = 7u64;
    for &n in &[1_000usize, 10_000, 100_000] {
        run_at_scale(n, seed);
    }
    eprintln!("=== Done ===");
}
