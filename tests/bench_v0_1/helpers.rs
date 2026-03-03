/// Shared helpers for the Determinism@Scale Benchmark Suite.
///
/// Output protocol:
///   BENCH:name:seed:time_ms:gc_live:out_hash
///   STAGE:name:stage_idx:stage_hash

use std::path::PathBuf;

// ---------------------------------------------------------------------------
// CJC program loading
// ---------------------------------------------------------------------------

pub fn load_cjc(filename: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("bench_v0_1")
        .join("cjc_programs")
        .join(filename);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to load {}: {}", path.display(), e))
}

// ---------------------------------------------------------------------------
// Parse helper
// ---------------------------------------------------------------------------

fn parse(src: &str) -> cjc_ast::Program {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(
        !diags.has_errors(),
        "parse errors: {:?}",
        diags.diagnostics
    );
    program
}

// ---------------------------------------------------------------------------
// Execution helpers
// ---------------------------------------------------------------------------

pub fn run_eval(src: &str, seed: u64) -> Vec<String> {
    let program = parse(src);
    let mut interp = cjc_eval::Interpreter::new(seed);
    let _ = interp.exec(&program).unwrap();
    interp.output.clone()
}

pub fn run_mir(src: &str, seed: u64) -> Vec<String> {
    let program = parse(src);
    let (_, exec) = cjc_mir_exec::run_program_with_executor(&program, seed).unwrap();
    exec.output.clone()
}

// ---------------------------------------------------------------------------
// Parsed output types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BenchLine {
    pub name: String,
    pub seed: String,
    pub time_ms: String,
    pub gc_live: String,
    pub out_hash: String,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct StageLine {
    pub name: String,
    pub stage_idx: String,
    pub stage_hash: String,
}

// ---------------------------------------------------------------------------
// Line parsers
// ---------------------------------------------------------------------------

pub fn parse_bench_lines(output: &[String]) -> Vec<BenchLine> {
    output
        .iter()
        .filter(|l| l.starts_with("BENCH:"))
        .map(|l| {
            let parts: Vec<&str> = l.splitn(6, ':').collect();
            assert!(
                parts.len() == 6,
                "Malformed BENCH line (expected 6 parts): {}",
                l
            );
            BenchLine {
                name: parts[1].to_string(),
                seed: parts[2].to_string(),
                time_ms: parts[3].to_string(),
                gc_live: parts[4].to_string(),
                out_hash: parts[5].to_string(),
            }
        })
        .collect()
}

pub fn parse_stage_lines(output: &[String]) -> Vec<StageLine> {
    output
        .iter()
        .filter(|l| l.starts_with("STAGE:"))
        .map(|l| {
            let parts: Vec<&str> = l.splitn(4, ':').collect();
            assert!(
                parts.len() == 4,
                "Malformed STAGE line (expected 4 parts): {}",
                l
            );
            StageLine {
                name: parts[1].to_string(),
                stage_idx: parts[2].to_string(),
                stage_hash: parts[3].to_string(),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Determinism assertions
// ---------------------------------------------------------------------------

/// Assert that two runs of the same program with the same seed produce
/// identical BENCH and STAGE hashes.
pub fn assert_deterministic(out1: &[String], out2: &[String]) {
    let b1 = parse_bench_lines(out1);
    let b2 = parse_bench_lines(out2);
    assert_eq!(
        b1.len(),
        b2.len(),
        "BENCH line count mismatch: {} vs {}",
        b1.len(),
        b2.len()
    );
    for (a, b) in b1.iter().zip(b2.iter()) {
        assert_eq!(
            a.out_hash, b.out_hash,
            "BENCH hash mismatch for '{}': {} vs {}",
            a.name, a.out_hash, b.out_hash
        );
    }

    let s1 = parse_stage_lines(out1);
    let s2 = parse_stage_lines(out2);
    assert_eq!(
        s1.len(),
        s2.len(),
        "STAGE line count mismatch: {} vs {}",
        s1.len(),
        s2.len()
    );
    for (a, b) in s1.iter().zip(s2.iter()) {
        assert_eq!(
            a.stage_hash, b.stage_hash,
            "STAGE hash mismatch for '{}' idx {}: {} vs {}",
            a.name, a.stage_idx, a.stage_hash, b.stage_hash
        );
    }
}

/// Assert that two runs with different seeds produce different BENCH hashes
/// (sanity check — different seeds should give different randomness).
pub fn assert_seeds_differ(out_a: &[String], out_b: &[String]) {
    let ba = parse_bench_lines(out_a);
    let bb = parse_bench_lines(out_b);
    let any_differ = ba
        .iter()
        .zip(bb.iter())
        .any(|(a, b)| a.out_hash != b.out_hash);
    assert!(
        any_differ,
        "Different seeds should produce at least one different BENCH hash"
    );
}

/// Assert eval and MIR outputs are identical (parity check).
pub fn assert_parity(eval_out: &[String], mir_out: &[String]) {
    // Filter to only BENCH: and STAGE: lines for comparison
    // (timing lines will differ, so we compare hashes only)
    let eval_bench = parse_bench_lines(eval_out);
    let mir_bench = parse_bench_lines(mir_out);
    assert_eq!(
        eval_bench.len(),
        mir_bench.len(),
        "Parity: BENCH line count mismatch"
    );
    for (e, m) in eval_bench.iter().zip(mir_bench.iter()) {
        assert_eq!(
            e.out_hash, m.out_hash,
            "Parity: BENCH hash mismatch for '{}': eval={} mir={}",
            e.name, e.out_hash, m.out_hash
        );
    }

    let eval_stage = parse_stage_lines(eval_out);
    let mir_stage = parse_stage_lines(mir_out);
    assert_eq!(
        eval_stage.len(),
        mir_stage.len(),
        "Parity: STAGE line count mismatch"
    );
    for (e, m) in eval_stage.iter().zip(mir_stage.iter()) {
        assert_eq!(
            e.stage_hash, m.stage_hash,
            "Parity: STAGE hash mismatch for '{}' idx {}: eval={} mir={}",
            e.name, e.stage_idx, e.stage_hash, m.stage_hash
        );
    }
}
