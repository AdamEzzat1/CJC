//! Phase 7a.3 — integration tests for the trace-based replay
//! debugging system. Gated behind `--features trace`.

#![cfg(feature = "trace")]

use std::path::PathBuf;

use cjc_cronos_gan::{
    bisect_traces, fnv1a_hash_f64s, BisectResult, CronosSeed, TemporalGan, TemporalGanConfig,
    TemporalGanMode, TemporalGanTrainer, TraceEvent, TraceWriter,
};

// ─── Helpers ─────────────────────────────────────────────────────────────

fn tmp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    // Suffix with a per-test unique component so parallel test runs
    // don't collide. The process ID is deterministic-enough within a
    // single test invocation; the test names provide the per-test
    // disambiguator.
    p.push(format!("cronos_gan_trace_{}_{}.trace", std::process::id(), name));
    let _ = std::fs::remove_file(&p);
    p
}

fn sine_io(n_steps: usize) -> (Vec<f64>, Vec<f64>) {
    let inputs: Vec<f64> = (0..n_steps).map(|t| (t as f64 * 0.4).sin()).collect();
    let targets: Vec<f64> = (0..n_steps).map(|t| ((t + 1) as f64 * 0.4).sin()).collect();
    (inputs, targets)
}

fn run_traced(path: &PathBuf, seed: u64, mode: TemporalGanMode, n_steps_to_train: u64) {
    let cfg = TemporalGanConfig::symmetric(4, 1, 1)
        .with_mode(mode)
        .with_lambda_disagreement(0.1);
    let mut gan = TemporalGan::from_seed(cfg, CronosSeed(seed)).unwrap();
    let mut trainer = TemporalGanTrainer::new(cfg, &gan, 1e-2);
    let (inputs, targets) = sine_io(12);
    let mut writer = TraceWriter::create(path).unwrap();
    for _ in 0..n_steps_to_train {
        trainer
            .step_with_trace(&mut gan, &inputs, &targets, &mut writer)
            .unwrap();
    }
    writer.flush().unwrap();
}

// ─── § Hash function determinism (1) ────────────────────────────────────

#[test]
fn fnv1a_is_byte_identical_for_same_input() {
    let values = vec![0.1_f64, -0.5, 1e-9, f64::INFINITY, f64::NEG_INFINITY, 0.0];
    let h1 = fnv1a_hash_f64s(&values);
    let h2 = fnv1a_hash_f64s(&values);
    assert_eq!(h1, h2);
    // Changing any value by a representable amount changes the hash.
    // Use `next_up` so the perturbation is the smallest ULP step
    // (still a distinct f64 bit pattern, unlike `0.1 + 1e-17` which
    // would round back to `0.1`).
    let mut perturbed = values.clone();
    perturbed[0] = f64::from_bits(values[0].to_bits() + 1);
    assert_ne!(fnv1a_hash_f64s(&perturbed), h1);
}

// ─── § Trace file shape + content (2) ───────────────────────────────────

#[test]
fn trace_writer_produces_one_line_per_step() {
    let path = tmp_path("one_line_per_step");
    run_traced(&path, 42, TemporalGanMode::Symmetric, 5);
    let contents = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = contents.lines().collect();
    assert_eq!(lines.len(), 5, "expected 5 trace lines, got {}", lines.len());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn trace_event_line_has_all_expected_fields() {
    let event = TraceEvent {
        step: 7,
        ssm_params_pre_hash: 0xabcd_0000_0000_0001,
        liquid_params_pre_hash: 0xabcd_0000_0000_0002,
        ssm_loss_bits: 0x3ff0_0000_0000_0000, // 1.0
        liquid_loss_bits: 0x4000_0000_0000_0000, // 2.0
        absolute_gap_bits: 0x3fe0_0000_0000_0000, // 0.5
        regime_shift_score_bits: 0x3fc9_9999_9999_999a, // ≈ 0.2
        ssm_params_post_hash: 0xabcd_0000_0000_0003,
        liquid_params_post_hash: 0xabcd_0000_0000_0004,
    };
    let line = event.to_line();
    // Every documented field name appears.
    for &name in &TraceEvent::FIELD_NAMES {
        assert!(
            line.contains(name),
            "field `{}` missing from line: {}",
            name,
            line
        );
    }
    // Numeric fields hex-encoded.
    assert!(line.contains("ssm_params_pre=0xabcd000000000001"));
    assert!(line.contains("ssm_loss_bits=0x3ff0000000000000"));
}

// ─── § Trace file determinism across runs (1) ───────────────────────────

#[test]
fn trace_files_byte_identical_across_runs() {
    // Same (config, seed, inputs, targets) → byte-identical trace.
    let path_a = tmp_path("byte_id_a");
    let path_b = tmp_path("byte_id_b");
    run_traced(&path_a, 42, TemporalGanMode::SsmAsGenerator, 6);
    run_traced(&path_b, 42, TemporalGanMode::SsmAsGenerator, 6);
    let a_bytes = std::fs::read(&path_a).unwrap();
    let b_bytes = std::fs::read(&path_b).unwrap();
    assert_eq!(
        a_bytes, b_bytes,
        "trace files must be byte-identical for the same (config, seed)"
    );
    let _ = std::fs::remove_file(&path_a);
    let _ = std::fs::remove_file(&path_b);
}

// ─── § bisect_traces (3) ────────────────────────────────────────────────

#[test]
fn bisect_returns_identical_for_matching_traces() {
    let path_a = tmp_path("bisect_match_a");
    let path_b = tmp_path("bisect_match_b");
    run_traced(&path_a, 42, TemporalGanMode::Symmetric, 4);
    run_traced(&path_b, 42, TemporalGanMode::Symmetric, 4);
    let result = bisect_traces(&path_a, &path_b).unwrap();
    assert_eq!(result, BisectResult::Identical { n_steps: 4 });
    let _ = std::fs::remove_file(&path_a);
    let _ = std::fs::remove_file(&path_b);
}

#[test]
fn bisect_returns_divergent_when_seeds_differ() {
    let path_a = tmp_path("bisect_div_a");
    let path_b = tmp_path("bisect_div_b");
    run_traced(&path_a, 42, TemporalGanMode::Symmetric, 4);
    run_traced(&path_b, 43, TemporalGanMode::Symmetric, 4);
    let result = bisect_traces(&path_a, &path_b).unwrap();
    match result {
        BisectResult::Divergent { step, field, .. } => {
            // Different seeds ⇒ different param init ⇒ different
            // pre-state hashes on the very first step.
            assert_eq!(step, 0);
            // The first param-hash field is the first non-`step`
            // field, so divergence shows up there.
            assert!(
                field == "ssm_params_pre" || field == "liquid_params_pre",
                "expected param-hash field divergence, got {}",
                field
            );
        }
        other => panic!("expected Divergent, got {:?}", other),
    }
    let _ = std::fs::remove_file(&path_a);
    let _ = std::fs::remove_file(&path_b);
}

#[test]
fn bisect_returns_length_mismatch_when_one_trace_is_longer() {
    let path_a = tmp_path("bisect_len_a");
    let path_b = tmp_path("bisect_len_b");
    run_traced(&path_a, 42, TemporalGanMode::Symmetric, 3);
    run_traced(&path_b, 42, TemporalGanMode::Symmetric, 5);
    let result = bisect_traces(&path_a, &path_b).unwrap();
    match result {
        BisectResult::LengthMismatch {
            prefix_steps,
            len_a,
            len_b,
        } => {
            assert_eq!(prefix_steps, 3);
            assert_eq!(len_a, 3);
            assert_eq!(len_b, 5);
        }
        other => panic!("expected LengthMismatch, got {:?}", other),
    }
    let _ = std::fs::remove_file(&path_a);
    let _ = std::fs::remove_file(&path_b);
}
