//! CJC quantum benchmark harness (BENCHMARK_PLAN.md).
//!
//! ```text
//! quantum_compare run   --suite baseline --out bench_results/quantum_compare/<run> [--reps 5] [--filter W1]
//! quantum_compare child --suite baseline --case W1_n20 --path rust     (internal: fresh-process replay)
//! ```
//!
//! For every case and path (`rust` = direct library API with QASM import,
//! `eval` / `mir` = the `.cjcl` program in each executor), `run` records one
//! JSON line (schema `cjc-quantum-bench/v1`) with phase timings, the SHA-256
//! of the output bytes, a fresh-process replay check, cross-path agreement,
//! and the child's peak memory. It also writes each case's `.qasm`, `.stim`,
//! and `.cjcl` forms plus CJC's outputs, which the Python drivers in
//! `externals/` use for the Qiskit Aer and Stim columns.

mod gen;
mod sys;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use cjc_quantum::gates::Gate;
use cjc_quantum::mps::Mps;
use cjc_quantum::stabilizer::StabilizerState;
use cjc_repro::dmath;
use cjc_runtime::complex::ComplexF64;
use cjc_runtime::value::Value;

use gen::{Case, Family, SAMPLE_SEED, SHOTS};

/// `.cjcl` paths are skipped above these sizes (reported as `skipped_size`).
/// A dense `q_probs` result is 2^n `Value`s, and very long programs mostly
/// measure the parser.
const CJCL_MAX_DENSE_QUBITS: usize = 22;
const CJCL_MAX_GATES: usize = 120_000;
/// Amplitude dumps for external accuracy checks (16 · 2^n bytes).
const DUMP_MAX_DENSE_QUBITS: usize = 22;

// ---------------------------------------------------------------------------
// One execution of a case on a path
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Exec {
    build: f64,
    execute: f64,
    observe: f64,
    /// Primary output bytes (canonical, BENCHMARK_PLAN §6).
    out: Vec<u8>,
    /// Dense only: SAMPLE_SEED-seeded shots as LE u64.
    samples: Vec<u8>,
    /// Oracle data (not hashed): amplitudes, peek_z, or ⟨Z_i⟩.
    oracle: Vec<u8>,
    accuracy: Option<(String, f64)>,
}

fn f64s_le(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn mps_h() -> [[ComplexF64; 2]; 2] {
    // Identical to dispatch.rs `mps_h`, so the paths agree bit for bit.
    let isq2 = 1.0 / 2.0f64.sqrt();
    [
        [ComplexF64::real(isq2), ComplexF64::real(isq2)],
        [ComplexF64::real(isq2), ComplexF64::real(-isq2)],
    ]
}

fn mps_ry(theta: f64) -> [[ComplexF64; 2]; 2] {
    // Identical to dispatch.rs `mps_ry`.
    let c = ComplexF64::real(dmath::cos(theta / 2.0));
    let s = ComplexF64::real(dmath::sin(theta / 2.0));
    [[c, ComplexF64::real(-s.re)], [s, c]]
}

fn run_rust(case: &Case, qasm: &str) -> Result<Exec, String> {
    let mut e = Exec::default();
    match case.family {
        Family::Dense => {
            let t = Instant::now();
            let circ = cjc_quantum::qasm::from_qasm(qasm)?;
            e.build = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let sv = circ.execute()?;
            e.execute = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let probs = sv.probabilities();
            let mut rng = SAMPLE_SEED;
            let shots = cjc_quantum::measure::sample_basis_states(&sv, SHOTS, &mut rng);
            if case.multi_observe {
                // Mirrors the W6 program's q_measure(c, 3) on a copy.
                let mut copy = sv.clone();
                let mut r = 3u64;
                cjc_quantum::measure::measure_all(&mut copy, &mut r)?;
            }
            e.observe = t.elapsed().as_secs_f64();
            e.out = f64s_le(&probs);
            e.samples = shots.iter().flat_map(|&s| (s as u64).to_le_bytes()).collect();
            if case.n <= DUMP_MAX_DENSE_QUBITS {
                e.oracle = sv.amplitudes.iter().flat_map(|a| {
                    let mut b = a.re.to_le_bytes().to_vec();
                    b.extend_from_slice(&a.im.to_le_bytes());
                    b
                }).collect();
            }
            if case.workload == "W1_ghz_dense" {
                // Analytic oracle: P(0…0) = P(1…1) = 1/2, all else 0.
                let last = probs.len() - 1;
                let err = probs
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| if i == 0 || i == last { (p - 0.5).abs() } else { p.abs() })
                    .fold(0.0, f64::max);
                e.accuracy = Some(("analytic-ghz".into(), err));
            }
        }
        Family::Clifford => {
            let t = Instant::now();
            let mut s = StabilizerState::new(case.n);
            for g in &case.gates {
                match *g {
                    Gate::H(q) => s.h(q),
                    Gate::S(q) => s.s(q),
                    Gate::X(q) => s.x(q),
                    Gate::Y(q) => s.y(q),
                    Gate::Z(q) => s.z(q),
                    Gate::CNOT(a, b) => s.cnot(a, b),
                    _ => return Err("non-Clifford gate".into()),
                }
            }
            e.execute = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let peek: Vec<u8> = (0..case.n).map(|q| s.peek_z(q) as u8).collect();
            // Seeded measure-all, qubit q with seed q (mirrors the .cjcl program).
            let record: Vec<u8> = (0..case.n)
                .map(|q| {
                    let mut rng = q as u64;
                    s.measure(q, &mut rng)
                })
                .collect();
            e.observe = t.elapsed().as_secs_f64();
            e.out = record;
            e.oracle = peek;
        }
        Family::Mps { chi } => {
            let t = Instant::now();
            let mut m = Mps::with_max_bond(case.n, chi);
            for g in &case.gates {
                match *g {
                    Gate::H(q) => m.apply_single_qubit(q, mps_h()),
                    Gate::Ry(q, th) => m.apply_single_qubit(q, mps_ry(th)),
                    Gate::CNOT(a, b) => m.apply_cnot_adjacent(a, b),
                    _ => return Err("gate not on the MPS surface".into()),
                }
            }
            e.execute = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let z: Vec<f64> = (0..case.n)
                .map(|q| cjc_quantum::qml::mps_single_z_expectation(&m, q))
                .collect();
            e.observe = t.elapsed().as_secs_f64();
            e.out = f64s_le(&z);
            e.oracle = e.out.clone();
            if case.workload == "W4_mps_ghz" {
                e.accuracy = Some(("analytic-ghz-z".into(), z.iter().map(|x| x.abs()).fold(0.0, f64::max)));
            }
        }
    }
    Ok(e)
}

/// Canonical bytes of a `.cjcl` result: floats as LE f64, ints as one byte
/// (Clifford outcomes).
fn value_bytes(v: &Value) -> Result<Vec<u8>, String> {
    match v {
        Value::Array(a) => {
            let mut out = Vec::new();
            for x in a.iter() {
                match x {
                    Value::Float(f) => out.extend_from_slice(&f.to_le_bytes()),
                    Value::Int(i) => out.push(*i as u8),
                    other => return Err(format!("unexpected element {}", other.type_name())),
                }
            }
            Ok(out)
        }
        other => Err(format!("expected an array result, got {}", other.type_name())),
    }
}

fn run_cjcl(src: &str, mir: bool) -> Result<Exec, String> {
    let t = Instant::now();
    let (program, diags) = cjc_parser::parse_source(src);
    if diags.has_errors() {
        return Err("generated .cjcl failed to parse".into());
    }
    let build = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let v = if mir {
        cjc_mir_exec::run_program_with_executor(&program, 42)
            .map(|(v, _)| v)
            .map_err(|e| format!("{:?}", e))?
    } else {
        cjc_eval::Interpreter::new(42).exec(&program).map_err(|e| format!("{:?}", e))?
    };
    let execute = t.elapsed().as_secs_f64();
    Ok(Exec { build, execute, out: value_bytes(&v)?, ..Default::default() })
}

fn exec_path(case: &Case, path: &str, qasm: &str, cjcl: &str) -> Result<Exec, String> {
    match path {
        "rust" => run_rust(case, qasm),
        "eval" => run_cjcl(cjcl, false),
        "mir" => run_cjcl(cjcl, true),
        other => Err(format!("unknown path {}", other)),
    }
}

fn cjcl_skip_reason(case: &Case) -> Option<String> {
    if case.family == Family::Dense && case.n > CJCL_MAX_DENSE_QUBITS {
        return Some(format!(
            "q_probs returns 2^{} Values (~72 B each, ~{:.1} GB); .cjcl dense paths run up to n = {}",
            case.n,
            (1u64 << case.n) as f64 * 72.0 / 1e9,
            CJCL_MAX_DENSE_QUBITS
        ));
    }
    if case.gates.len() > CJCL_MAX_GATES {
        return Some(format!(
            "{} gates = {} lines of generated .cjcl; .cjcl paths run up to {} gates",
            case.gates.len(),
            case.gates.len(),
            CJCL_MAX_GATES
        ));
    }
    None
}

// ---------------------------------------------------------------------------
// Statistics and records
// ---------------------------------------------------------------------------

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let pos = p * (sorted.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    percentile(&v, 0.5)
}

fn num(x: Option<f64>) -> String {
    match x {
        Some(v) if v.is_finite() => format!("{:e}", v),
        _ => "null".into(),
    }
}

fn opt_bool(b: Option<bool>) -> String {
    b.map(|b| b.to_string()).unwrap_or_else(|| "null".into())
}

fn params_json(case: &Case) -> String {
    let chi = match case.family {
        Family::Mps { chi } => chi.to_string(),
        _ => "null".into(),
    };
    let shots = if case.family == Family::Dense { SHOTS.to_string() } else { "null".into() };
    format!(
        "{{\"n_qubits\": {}, \"depth\": {}, \"circuit_seed\": {}, \"n_gates\": {}, \"shots\": {}, \"chi_max\": {}}}",
        case.n,
        case.depth,
        case.seed,
        case.gates.len(),
        shots,
        chi
    )
}

fn method(case: &Case) -> &'static str {
    match case.family {
        Family::Dense => "statevector",
        Family::Clifford => "stabilizer",
        Family::Mps { .. } => "matrix_product_state",
    }
}

fn output_kind(case: &Case) -> &'static str {
    match case.family {
        Family::Dense => "probabilities_f64_le",
        Family::Clifford => "measure_all_seed_q_u8",
        Family::Mps { .. } => "z_expectations_f64_le",
    }
}

struct PathOutcome {
    path: &'static str,
    status: String,
    notes: String,
    reps: usize,
    times: Vec<f64>,
    phases: (f64, f64, f64),
    sha: Option<String>,
    samples_sha: Option<String>,
    accuracy: Option<(String, f64)>,
    replay_ok: Option<bool>,
    peak_rss: Option<u64>,
}

#[allow(clippy::too_many_arguments)]
fn record(run_id: &str, meta: &sys::Meta, case: &Case, o: &PathOutcome, cross: Option<bool>, out_len: usize) -> String {
    let mut t = o.times.clone();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (med, iqr) = if t.is_empty() {
        (None, None)
    } else {
        (Some(percentile(&t, 0.5)), Some(percentile(&t, 0.75) - percentile(&t, 0.25)))
    };
    let has = !t.is_empty();
    let (oracle, err) = match &o.accuracy {
        Some((n, e)) => (sys::json_str(n), num(Some(*e))),
        None => ("null".into(), "null".into()),
    };
    format!(
        concat!(
            "{{\"schema\": \"cjc-quantum-bench/v1\", \"run_id\": {}, \"workload\": \"{}\", \"case_id\": \"{}\", ",
            "\"params\": {}, ",
            "\"simulator\": {{\"name\": \"cjc-quantum\", \"path\": \"{}\", \"backend\": \"rust\", \"method\": \"{}\", ",
            "\"precision\": \"f64\", \"threads\": 1, \"version\": \"{}\", \"git_sha\": {}, \"dirty\": {}}}, ",
            "\"timing\": {{\"reps\": {}, \"warmup\": {}, \"median_s\": {}, \"iqr_s\": {}, ",
            "\"phases_median_s\": {{\"build\": {}, \"execute\": {}, \"observe\": {}}}}}, ",
            "\"memory\": {{\"peak_rss_bytes\": {}}}, ",
            "\"output\": {{\"kind\": \"{}\", \"sha256\": {}, \"len\": {}, \"samples_sha256\": {}}}, ",
            "\"accuracy\": {{\"oracle\": {}, \"linf\": {}, \"fidelity\": null}}, ",
            "\"determinism\": {{\"replay_ok\": {}, \"cross_executor_ok\": {}, \"replay_runs\": 2}}, ",
            "\"machine\": {{\"cpu\": {}, \"threads\": {}, \"os\": {}}}, ",
            "\"toolchain\": {{\"rustc\": {}, \"profile\": \"release\"}}, ",
            "\"status\": \"{}\", \"notes\": {}}}"
        ),
        sys::json_str(run_id),
        case.workload,
        case.id,
        params_json(case),
        o.path,
        method(case),
        env!("CARGO_PKG_VERSION"),
        sys::json_str(&meta.git_sha),
        meta.dirty,
        o.reps,
        if has { 1 } else { 0 },
        num(med),
        num(iqr),
        num(has.then_some(o.phases.0)),
        num(has.then_some(o.phases.1)),
        num(has.then_some(o.phases.2)),
        o.peak_rss.map(|b| b.to_string()).unwrap_or_else(|| "null".into()),
        output_kind(case),
        o.sha.as_deref().map(sys::json_str).unwrap_or_else(|| "null".into()),
        out_len,
        o.samples_sha.as_deref().map(sys::json_str).unwrap_or_else(|| "null".into()),
        oracle,
        err,
        opt_bool(o.replay_ok),
        opt_bool(cross),
        sys::json_str(&meta.cpu),
        sys::json_str(&meta.threads),
        sys::json_str(&meta.os),
        sys::json_str(&meta.rustc),
        o.status,
        sys::json_str(&o.notes),
    )
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

fn suite(name: &str) -> Vec<Case> {
    match name {
        "smoke" => gen::suite_smoke(),
        "baseline" => gen::suite_baseline(),
        other => panic!("unknown suite '{}' (smoke, baseline)", other),
    }
}

fn arg(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned()
}

/// Run one case once in a fresh process: prints `HASH <sha> RSS <bytes>`.
fn child(args: &[String]) {
    let s = arg(args, "--suite").expect("--suite");
    let id = arg(args, "--case").expect("--case");
    let path = arg(args, "--path").expect("--path");
    let case = suite(&s).into_iter().find(|c| c.id == id).expect("unknown case");
    let (qasm, cjcl) = (gen::to_qasm(&case), gen::to_cjcl(&case));
    match exec_path(&case, &path, &qasm, &cjcl) {
        Ok(e) => println!(
            "HASH {} RSS {}",
            sys::sha256_hex(&e.out),
            sys::peak_rss_bytes().unwrap_or(0)
        ),
        Err(err) => println!("ERROR {}", err),
    }
}

fn replay_in_child(suite_name: &str, case: &Case, path: &str) -> (Option<String>, Option<u64>) {
    let exe = std::env::current_exe().expect("current_exe");
    let out = Command::new(exe)
        .args(["child", "--suite", suite_name, "--case", &case.id, "--path", path])
        .output();
    let text = match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout).to_string(),
        Err(_) => return (None, None),
    };
    let parts: Vec<&str> = text.split_whitespace().collect();
    if parts.len() == 4 && parts[0] == "HASH" {
        (Some(parts[1].to_string()), parts[3].parse().ok().filter(|&b| b > 0))
    } else {
        (None, None)
    }
}

fn run(args: &[String]) {
    let suite_name = arg(args, "--suite").unwrap_or_else(|| "smoke".into());
    let out_dir = PathBuf::from(arg(args, "--out").expect("--out <dir>"));
    let reps: usize = arg(args, "--reps").map(|r| r.parse().expect("--reps")).unwrap_or(5);
    let filter = arg(args, "--filter");
    let meta = sys::meta();
    let run_id = format!("{}-{}", suite_name, meta.git_sha);
    fs::create_dir_all(out_dir.join("work")).expect("create out dir");
    let results = out_dir.join("results.jsonl");
    let manifest = out_dir.join("manifest.jsonl");
    let mut res_lines = String::new();
    let mut man_lines = String::new();

    for case in suite(&suite_name) {
        if let Some(f) = &filter {
            if !case.id.contains(f.as_str()) {
                continue;
            }
        }
        let work = out_dir.join("work").join(&case.id);
        fs::create_dir_all(&work).unwrap();
        let qasm = gen::to_qasm(&case);
        let cjcl = gen::to_cjcl(&case);
        let skip_cjcl = cjcl_skip_reason(&case);
        write(&work.join("circuit.qasm"), qasm.as_bytes());
        if let Some(stim) = gen::to_stim(&case) {
            write(&work.join("circuit.stim"), stim.as_bytes());
        }
        if skip_cjcl.is_none() {
            write(&work.join("circuit.cjcl"), cjcl.as_bytes());
        }
        eprintln!("== {} ({} gates)", case.id, case.gates.len());

        let mut outcomes: Vec<PathOutcome> = Vec::new();
        let mut out_len = 0usize;
        for path in ["rust", "eval", "mir"] {
            if path != "rust" {
                if let Some(reason) = &skip_cjcl {
                    outcomes.push(PathOutcome {
                        path, status: "skipped_size".into(), notes: reason.clone(), reps: 0,
                        times: vec![], phases: (0.0, 0.0, 0.0), sha: None, samples_sha: None,
                        accuracy: None, replay_ok: None, peak_rss: None,
                    });
                    continue;
                }
            }
            // Warm-up run: also the source of hashes and oracle dumps.
            let first = match exec_path(&case, path, &qasm, &cjcl) {
                Ok(e) => e,
                Err(err) => {
                    outcomes.push(PathOutcome {
                        path, status: "error".into(), notes: err, reps: 0, times: vec![],
                        phases: (0.0, 0.0, 0.0), sha: None, samples_sha: None, accuracy: None,
                        replay_ok: None, peak_rss: None,
                    });
                    continue;
                }
            };
            let warm_total = first.build + first.execute + first.observe;
            // Long cases get fewer repetitions; the record states how many.
            let n_reps = if warm_total > 30.0 { reps.min(3) } else { reps };
            let sha = sys::sha256_hex(&first.out);
            let samples_sha = (!first.samples.is_empty()).then(|| sys::sha256_hex(&first.samples));
            if path == "rust" {
                out_len = first.out.len();
                if !first.oracle.is_empty() {
                    let name = match case.family {
                        Family::Dense => "cjc_amplitudes_c128le.bin",
                        Family::Clifford => "cjc_peek_z_i8.bin",
                        Family::Mps { .. } => "cjc_z_expectations_f64le.bin",
                    };
                    write(&work.join(name), &first.oracle);
                }
                if case.family == Family::Clifford {
                    // The seeded measure-all record, replayed in Stim via postselect_z.
                    write(&work.join("cjc_measure_record_u8.bin"), &first.out);
                }
            }
            let (mut tt, mut b, mut x, mut ob) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let mut consistent = true;
            for _ in 0..n_reps {
                let e = exec_path(&case, path, &qasm, &cjcl).expect("rep failed after warm-up");
                consistent &= sys::sha256_hex(&e.out) == sha;
                tt.push(e.build + e.execute + e.observe);
                b.push(e.build);
                x.push(e.execute);
                ob.push(e.observe);
            }
            let (child_sha, rss) = replay_in_child(&suite_name, &case, path);
            let replay_ok = Some(consistent && child_sha.as_deref() == Some(sha.as_str()));
            eprintln!("   {:<5} median {:.4}s  sha {}  replay {:?}", path, median(tt.clone()), &sha[..12], replay_ok);
            outcomes.push(PathOutcome {
                path, status: "ok".into(), notes: String::new(), reps: n_reps, times: tt,
                phases: (median(b), median(x), median(ob)), sha: Some(sha), samples_sha,
                accuracy: first.accuracy, replay_ok, peak_rss: rss,
            });
        }

        // Cross-path agreement: every path that ran must produce the same bytes.
        let shas: Vec<&String> = outcomes.iter().filter_map(|o| o.sha.as_ref()).collect();
        let cross = if shas.len() >= 2 { Some(shas.iter().all(|s| *s == shas[0])) } else { None };
        for o in &outcomes {
            res_lines.push_str(&record(&run_id, &meta, &case, o, cross, out_len));
            res_lines.push('\n');
        }
        man_lines.push_str(&format!(
            "{{\"case_id\": \"{}\", \"workload\": \"{}\", \"family\": \"{}\", \"params\": {}, \"work_dir\": {}}}\n",
            case.id,
            case.workload,
            method(&case),
            params_json(&case),
            sys::json_str(&work.to_string_lossy()),
        ));
        // Write incrementally so a long run leaves usable partial results.
        write(&results, res_lines.as_bytes());
        write(&manifest, man_lines.as_bytes());
    }
    eprintln!("wrote {}", results.display());
}

fn write(p: &Path, bytes: &[u8]) {
    fs::write(p, bytes).unwrap_or_else(|e| panic!("write {}: {}", p.display(), e));
}

/// `kernels`: time single-gate kernels on one statevector.
///
/// Compares the reference full-scan loop, the pre-existing `simd_kernel`
/// functions (AVX2 and cache-blocked), and the strided kernel at 1 thread and
/// at the policy thread count. Every variant's output is checked bit for bit
/// against the reference before it is timed.
fn kernels(args: &[String]) {
    use cjc_quantum::statevector::Statevector;
    let out_dir = PathBuf::from(arg(args, "--out").expect("--out <dir>"));
    let reps: usize = arg(args, "--reps").map(|r| r.parse().unwrap()).unwrap_or(7);
    fs::create_dir_all(&out_dir).unwrap();
    let threads = cjc_runtime::runtime_policy::current_effective_threads();
    let h = {
        // Same constant as the library's H (one ULP away from 1.0 / sqrt(2.0)).
        let r = std::f64::consts::FRAC_1_SQRT_2;
        [[ComplexF64::real(r), ComplexF64::real(r)], [ComplexF64::real(r), ComplexF64::real(-r)]]
    };
    let mut lines = String::new();
    for n in [16usize, 20, 22, 24] {
        let mut rng = gen::SplitMix64(n as u64);
        let base: Vec<ComplexF64> = (0..1usize << n)
            .map(|_| ComplexF64::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5))
            .collect();
        for q in [0usize, n / 2, n - 1] {
            let reference = {
                let mut sv = Statevector::from_amplitudes(base.clone()).unwrap();
                cjc_quantum::gates::apply_reference(&Gate::H(q), &mut sv);
                sv.amplitudes
            };
            type Kernel<'a> = (&'a str, Box<dyn Fn(&mut Statevector) + 'a>);
            let variants: Vec<Kernel> = vec![
                ("reference_scan", Box::new(move |sv: &mut Statevector| cjc_quantum::gates::apply_reference(&Gate::H(q), sv))),
                ("simd_kernel_avx2", Box::new(move |sv: &mut Statevector| cjc_quantum::simd_kernel::apply_single_qubit_simd(sv, q, h))),
                ("simd_kernel_cached", Box::new(move |sv: &mut Statevector| cjc_quantum::simd_kernel::apply_single_qubit_cached(sv, q, h))),
                ("strided_1thread", Box::new(move |sv: &mut Statevector| cjc_quantum::kernels::apply_single_qubit_threads(&mut sv.amplitudes, q, h, 1))),
                ("strided_policy_threads", Box::new(move |sv: &mut Statevector| cjc_quantum::kernels::apply_single_qubit(&mut sv.amplitudes, q, h))),
            ];
            for (name, f) in &variants {
                let mut sv = Statevector::from_amplitudes(base.clone()).unwrap();
                f(&mut sv);
                let identical = sv.amplitudes.iter().zip(&reference).all(|(a, b)| {
                    a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits()
                });
                let mut times = Vec::new();
                for _ in 0..reps {
                    let mut sv = Statevector::from_amplitudes(base.clone()).unwrap();
                    let t = Instant::now();
                    f(&mut sv);
                    times.push(t.elapsed().as_secs_f64());
                }
                let med = median(times);
                eprintln!("n={:>2} q={:>2} {:<24} {:>10.3} ms  identical={}", n, q, name, med * 1e3, identical);
                lines.push_str(&format!(
                    "{{\"n\": {}, \"qubit\": {}, \"kernel\": \"{}\", \"median_s\": {:e}, \"reps\": {}, \"bit_identical_to_reference\": {}, \"policy_threads\": {}}}\n",
                    n, q, name, med, reps, identical, threads
                ));
            }
        }
    }
    // Whole circuits and sampling, old vs new, interleaved rep by rep in one
    // process so machine noise (thermal and power state, other load) hits
    // both sides equally. Separate benchmark runs on this laptop varied up
    // to 2x on unchanged code.
    for case in [gen::w1_ghz(22), gen::w2_random(20, 10, 1), gen::w2_random(22, 10, 1)] {
        let circ = gen::to_circuit(&case);
        let run_old = || {
            let mut sv = Statevector::new(case.n);
            for g in &case.gates {
                cjc_quantum::gates::apply_reference(g, &mut sv);
            }
            sv
        };
        let run_new = || circ.execute().unwrap();
        let (a, b) = (run_old(), run_new());
        let identical = a.amplitudes.iter().zip(&b.amplitudes).all(|(x, y)| {
            x.re.to_bits() == y.re.to_bits() && x.im.to_bits() == y.im.to_bits()
        });
        let shots_old = |sv: &Statevector| {
            let mut rng = SAMPLE_SEED;
            (0..SHOTS).map(|_| cjc_quantum::measure::sample_basis_state(sv, &mut rng)).collect::<Vec<_>>()
        };
        let shots_new = |sv: &Statevector| {
            let mut rng = SAMPLE_SEED;
            cjc_quantum::measure::sample_basis_states(sv, SHOTS, &mut rng)
        };
        let same_shots = shots_old(&b) == shots_new(&b);
        let (mut t_old, mut t_new, mut s_old, mut s_new) = (vec![], vec![], vec![], vec![]);
        for _ in 0..reps {
            let t = Instant::now();
            std::hint::black_box(run_old());
            t_old.push(t.elapsed().as_secs_f64());
            let t = Instant::now();
            std::hint::black_box(run_new());
            t_new.push(t.elapsed().as_secs_f64());
            let t = Instant::now();
            std::hint::black_box(shots_old(&b));
            s_old.push(t.elapsed().as_secs_f64());
            let t = Instant::now();
            std::hint::black_box(shots_new(&b));
            s_new.push(t.elapsed().as_secs_f64());
        }
        let (t_old, t_new, s_old, s_new) = (median(t_old), median(t_new), median(s_old), median(s_new));
        eprintln!(
            "{:<16} execute: reference {:>9.3} ms, kernels {:>9.3} ms ({:.2}x, identical={})  |  {} shots: per-shot {:>9.3} ms, batch {:>7.3} ms ({:.0}x, identical={})",
            case.id, t_old * 1e3, t_new * 1e3, t_old / t_new, identical, SHOTS, s_old * 1e3, s_new * 1e3, s_old / s_new, same_shots
        );
        lines.push_str(&format!(
            "{{\"case_id\": \"{}\", \"kind\": \"circuit_ab\", \"execute_reference_s\": {:e}, \"execute_kernels_s\": {:e}, \"amplitudes_bit_identical\": {}, \"sample_per_shot_s\": {:e}, \"sample_batch_s\": {:e}, \"shots\": {}, \"shots_identical\": {}, \"reps\": {}, \"policy_threads\": {}}}
",
            case.id, t_old, t_new, identical, s_old, s_new, SHOTS, same_shots, reps, threads
        ));
    }
    // Execution cache: the W6 program observes one circuit value three times
    // (q_probs, q_sample, q_measure). The control replaces the 2nd and 3rd
    // uses with q_copy(c), whose copy starts with an empty cache, so the
    // circuit is executed three times instead of once. Interleaved, both
    // executors; outputs must be byte-identical.
    for n in [16usize, 20] {
        let cached = gen::to_cjcl(&gen::w6_multi_observe(n, 10, 1));
        let uncached = cached
            .replace("q_sample(c,", "q_sample(q_copy(c),")
            .replace("q_measure(c,", "q_measure(q_copy(c),");
        assert_ne!(cached, uncached);
        for mir in [false, true] {
            let same = run_cjcl(&cached, mir).unwrap().out == run_cjcl(&uncached, mir).unwrap().out;
            let (mut t_c, mut t_u) = (vec![], vec![]);
            for _ in 0..reps {
                let t = Instant::now();
                std::hint::black_box(run_cjcl(&uncached, mir).unwrap());
                t_u.push(t.elapsed().as_secs_f64());
                let t = Instant::now();
                std::hint::black_box(run_cjcl(&cached, mir).unwrap());
                t_c.push(t.elapsed().as_secs_f64());
            }
            let (t_c, t_u) = (median(t_c), median(t_u));
            let path = if mir { "mir" } else { "eval" };
            eprintln!(
                "W6_n{n}_d10 {path:<4} three observations: re-executed {:>9.3} ms, cached {:>9.3} ms ({:.2}x, identical={same})",
                t_u * 1e3, t_c * 1e3, t_u / t_c
            );
            lines.push_str(&format!(
                "{{\"case_id\": \"W6_n{n}_d10_s1\", \"kind\": \"cache_ab\", \"path\": \"{path}\", \"reexecuted_s\": {:e}, \"cached_s\": {:e}, \"outputs_identical\": {same}, \"reps\": {reps}}}
",
                t_u, t_c
            ));
        }
    }
    write(&out_dir.join("kernels.jsonl"), lines.as_bytes());
    eprintln!("wrote {}", out_dir.join("kernels.jsonl").display());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("run") => run(&args),
        Some("child") => child(&args),
        Some("kernels") => kernels(&args),
        _ => {
            eprintln!("usage: quantum_compare run --suite <smoke|baseline> --out <dir> [--reps N] [--filter S]");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every smoke case produces identical bytes on all three paths, and the
    /// generated QASM round-trips to the same gates.
    #[test]
    fn smoke_paths_agree() {
        for case in gen::suite_smoke() {
            let (qasm, cjcl) = (gen::to_qasm(&case), gen::to_cjcl(&case));
            let r = run_rust(&case, &qasm).unwrap();
            let e = run_cjcl(&cjcl, false).unwrap();
            let m = run_cjcl(&cjcl, true).unwrap();
            assert_eq!(r.out, e.out, "{} rust vs eval", case.id);
            assert_eq!(r.out, m.out, "{} rust vs mir", case.id);
            let back = cjc_quantum::qasm::from_qasm(&qasm).unwrap();
            assert_eq!(back.gates().len(), case.gates.len(), "{}", case.id);
        }
    }

    #[test]
    fn clifford_peek_matches_dense_for_small_n() {
        // n = 12: a qubit is deterministic in the stabilizer picture iff its
        // dense marginal P(1) is exactly 0 or 1 (within rounding).
        let case = gen::w3_clifford(12, 6, 1);
        let r = run_rust(&case, &gen::to_qasm(&case)).unwrap();
        let sv = gen::to_circuit(&case).execute().unwrap();
        let p = sv.probabilities();
        for q in 0..12 {
            let p1: f64 = p.iter().enumerate().filter(|(i, _)| i >> q & 1 == 1).map(|(_, x)| x).sum();
            let peek = r.oracle[q] as i8;
            match peek {
                1 => assert!(p1 < 1e-9, "q{} peek +1 but P(1) = {}", q, p1),
                -1 => assert!(p1 > 1.0 - 1e-9, "q{} peek -1 but P(1) = {}", q, p1),
                _ => assert!((p1 - 0.5).abs() < 1e-9, "q{} random but P(1) = {}", q, p1),
            }
        }
    }

    #[test]
    fn stim_text_matches_gate_count() {
        let case = gen::w3_clifford(10, 3, 2);
        assert_eq!(gen::to_stim(&case).unwrap().lines().count(), case.gates.len());
        // Non-Clifford gates (T, rotations) have no Stim form.
        let dense = gen::w2_random(6, 4, 1);
        assert!(dense.gates.iter().any(|g| matches!(g, Gate::T(_) | Gate::Rx(..) | Gate::Ry(..) | Gate::Rz(..))));
        assert!(gen::to_stim(&dense).is_none());
    }
}
