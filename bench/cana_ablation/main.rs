//! Phase A6 — 5-way ablation harness emitting `CompilationProfile` rows.
//!
//! For every `(program × configuration)` pair this harness:
//!
//! 1. Compiles the program (parse → HIR → MIR → features).
//! 2. Ranks passes under the configuration's cost-model stack.
//! 3. Applies the resulting `PassPlan` and counts MIR nodes
//!    before/after (the deterministic score: size ratio, lower =
//!    better).
//! 4. Runs BOTH executors (AST tree-walk on the source program,
//!    MIR-exec on the optimized program) and compares captured print
//!    output → `parity_match`.
//! 5. Records NSS + PINN predictions for the row.
//! 6. Appends one [`CompilationProfile`] row to
//!    `bench_results/cana_ablation/profiles.cpdb`.
//!
//! ## The five ablations (Phase-A handoff §5.2)
//!
//! | id | stack |
//! |---|---|
//! | `baseline` | `LinearCostModel::trained()` only |
//! | `nss` | + `ThermalAwareCostModel` over `NssPressurePredictor` |
//! | `quantum` | + `EnergyAwarePassRanker` re-ranking (null pressures) |
//! | `nss_quantum` | + both advisory layers |
//! | `full_pinn` | `PinnPhysicalCostModel` + NSS + energy re-ranking |
//!
//! Every ablation runs the same corpus with the same seed. Wall-clock
//! is recorded as diagnostic metadata only — the score and every
//! decision input are deterministic counters (invariant #7).
//!
//! ## Program corpus
//!
//! Snapshotted from `bench/cana_ab_corpus` (which snapshotted from
//! `bench/cana_pass_ordering`). Drift is intentional: stable
//! hand-written workloads, so ablation rows stay comparable across
//! sessions.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use cjc_cana::cost_model::CostModel;
use cjc_cana::features::CanaFeatures;
use cjc_cana::legality::{
    LegalityGate, LegalityVerdict, PassSequence, PerPassLegalityGate, ProposedPass,
};
use cjc_cana::pass_ranker::{pass_plan_from, PassRanker, RankingReport, CANONICAL_PASSES};
use cjc_cana::physical_cost::PhysicalConstraints;
use cjc_cana::physical_cost::{build_physical_query, predict_physical, PhysicalCoefficients};
use cjc_cana::pinn_cost_model::PinnPhysicalCostModel;
use cjc_cana::pinn_thermal_v2::PinnThermalV2;
use cjc_cana::pressure::{NullPressurePredictor, PressurePredictor};
use cjc_cana::thermal_cost_model::ThermalAwareCostModel;
use cjc_cana::{analyze_program, LinearCostModel};
use cjc_cana_compress::pinn_bundle::read_bundle;
use cjc_cana_compress::profile_db::{
    append_row, read_all, CompilationProfile, FnProfile, PROFILE_SCHEMA_VERSION,
};
use cjc_cana_compress::EnergyAwarePassRanker;
use cjc_cana_nss::{NssPressurePredictor, RecordedPressurePredictor};
use cjc_mir::optimize::{optimize_program_with_plan, PassPlan};
use cjc_mir::MirProgram;
use cjc_mir_exec::{run_program_instrumented, trace};
use cjc_repro::KahanAccumulatorF64;

/// Train-cost-model corpus (95 programs), included by path rather than
/// snapshot-copied: training rows want feature-space BREADTH; rows
/// carry `program_hash`, so upstream corpus drift produces new rows
/// instead of silently corrupting old ones.
#[path = "../cana_train_cost_model/programs.rs"]
mod train_programs;

const SEED: u64 = 42;

/// Energy weight of one FP binop relative to one executed statement.
/// FP units burn more power than integer ALUs; 3.0 ≈ "an FP op costs
/// 4× an int op" (1 base + 3 extra). Hand-tuned v1 constant — v2's
/// trained model replaces it.
const FP_ENERGY_WEIGHT: f64 = 3.0;

// =============================================================================
// Program corpus (snapshot — see module docs)
// =============================================================================

struct Program {
    name: &'static str,
    source: &'static str,
}

const PROG_ARITH: &str = r#"
fn compute(n: i64) -> i64 {
    let a: i64 = 10 * 5 + 2;
    let b: i64 = (a + 100) * 2;
    let c: i64 = b - 50 + n;
    return c + a + b;
}
print(compute(7));
"#;

const PROG_LOOP: &str = r#"
fn sum_to(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        total = total + i;
        i = i + 1;
    }
    return total;
}
print(sum_to(1000));
"#;

const PROG_NESTED: &str = r#"
fn nested(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let mut j: i64 = 0;
        while j < n {
            total = total + i * j;
            j = j + 1;
        }
        i = i + 1;
    }
    return total;
}
print(nested(30));
"#;

const PROG_MANY_FN: &str = r#"
fn add1(x: i64) -> i64 { return x + 1; }
fn add2(x: i64) -> i64 { return x + 2; }
fn add3(x: i64) -> i64 { return x + 3; }
fn mul2(x: i64) -> i64 { return x * 2; }
fn mul3(x: i64) -> i64 { return x * 3; }
fn driver() -> i64 {
    let mut r: i64 = 0;
    r = add1(r);
    r = add2(r);
    r = add3(r);
    r = mul2(r);
    r = mul3(r);
    return r;
}
print(driver());
"#;

const PROG_MIXED: &str = r#"
fn classify(n: i64) -> i64 {
    let mut sum: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let inc: i64 = if i * 2 > n { i } else { 0 };
        sum = sum + inc;
        i = i + 1;
    }
    return sum;
}
print(classify(40));
"#;

const PROG_FLOAT: &str = r#"
fn polynomial(x: f64) -> f64 {
    let a: f64 = 3.14;
    let b: f64 = 2.71;
    let c: f64 = 1.41;
    return a * x * x + b * x + c;
}
print(polynomial(1.5));
"#;

const PROG_RECURSIVE: &str = r#"
fn factorial(n: i64) -> i64 {
    let result: i64 = if n <= 1 { 1 } else { n * factorial(n - 1) };
    return result;
}
print(factorial(10));
"#;

const PROG_LARGE: &str = r#"
fn count_evens(n: i64) -> i64 {
    let mut c: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        if i * 2 / 2 == i {
            c = c + 1;
        }
        i = i + 1;
    }
    return c;
}
fn count_squares(n: i64) -> i64 {
    let mut c: i64 = 0;
    let mut i: i64 = 0;
    while i * i < n {
        c = c + 1;
        i = i + 1;
    }
    return c;
}
fn sum_to(n: i64) -> i64 {
    let mut s: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        s = s + i;
        i = i + 1;
    }
    return s;
}
fn combined(n: i64) -> i64 {
    let a: i64 = count_evens(n);
    let b: i64 = count_squares(n);
    let c: i64 = sum_to(n);
    return a + b + c;
}
print(combined(50));
"#;

/// FP-hot workload — ADDED for the Option-B re-run (not part of the
/// original cana_ab_corpus snapshot). The Track-2 ablation concluded
/// the corpus had no program whose thermal signal could differentiate
/// the stacks; this one runs dense float arithmetic inside a nested
/// loop, so the recorded FP-op density (→ Thermal) is high while the
/// integer programs' stays near zero.
const PROG_FP_HOT: &str = r#"
fn horner(x: f64, n: i64) -> f64 {
    let mut acc: f64 = 0.0;
    let mut i: i64 = 0;
    while i < n {
        let mut j: i64 = 0;
        let mut p: f64 = 1.0;
        while j < 16 {
            p = p * x + 0.5;
            acc = acc + p * 0.001;
            j = j + 1;
        }
        i = i + 1;
    }
    return acc;
}
print(horner(1.01, 200));
"#;

const PROGRAMS: &[Program] = &[
    Program {
        name: "arith",
        source: PROG_ARITH,
    },
    Program {
        name: "loop",
        source: PROG_LOOP,
    },
    Program {
        name: "nested",
        source: PROG_NESTED,
    },
    Program {
        name: "many_fn",
        source: PROG_MANY_FN,
    },
    Program {
        name: "mixed",
        source: PROG_MIXED,
    },
    Program {
        name: "float",
        source: PROG_FLOAT,
    },
    Program {
        name: "recursive",
        source: PROG_RECURSIVE,
    },
    Program {
        name: "large",
        source: PROG_LARGE,
    },
    Program {
        name: "fp_hot",
        source: PROG_FP_HOT,
    },
];

// =============================================================================
// Workload assembly: static snapshot + thermal-gradient family + train corpus
// =============================================================================

/// Owned workload — generated and path-included programs aren't
/// `'static`, so the harness iterates these instead of `Program`.
struct Workload {
    name: String,
    source: String,
}

/// Generate the thermal-gradient family: per loop iteration, `fp_k` of
/// 10 work statements are float ops and `10 - fp_k` are integer ops,
/// so the recorded FP-op density (→ Thermal) forms a gradient across
/// the family instead of fp_hot's 0/1 step. Crossed with loop size and
/// nesting depth for feature-space spread.
fn thermal_gradient_workloads() -> Vec<Workload> {
    let mut out = Vec::new();
    for &fp_k in &[1u32, 3, 5, 7, 9] {
        for &outer in &[64i64, 256, 1024] {
            for &depth in &[1u32, 2] {
                let int_k = 10 - fp_k;
                let mut body = String::new();
                for f in 0..fp_k {
                    body.push_str(&format!("            facc = facc + 0.5{f:01};\n"));
                }
                for i in 0..int_k {
                    body.push_str(&format!("            iacc = iacc + i * {};\n", i + 3));
                }
                let source = if depth == 1 {
                    format!(
                        r#"
fn work(n: i64) -> i64 {{
    let mut facc: f64 = 0.0;
    let mut iacc: i64 = 0;
    let mut i: i64 = 0;
    while i < n {{
{body}            i = i + 1;
    }}
    print(facc);
    return iacc;
}}
print(work({outer}));
"#
                    )
                } else {
                    format!(
                        r#"
fn work(n: i64) -> i64 {{
    let mut facc: f64 = 0.0;
    let mut iacc: i64 = 0;
    let mut o: i64 = 0;
    while o < n {{
        let mut i: i64 = 0;
        while i < 8 {{
{body}            i = i + 1;
        }}
        o = o + 1;
    }}
    print(facc);
    return iacc;
}}
print(work({outer}));
"#
                    )
                };
                out.push(Workload {
                    name: format!("grad_f{fp_k}0_d{depth}_n{outer}"),
                    source,
                });
            }
        }
    }
    out
}

/// Shared tensor builder for the tensor family: deterministic values,
/// one `Tensor.from_vec` alloc, 2 scalar FP binops per element.
const TENSOR_BUILDER: &str = r#"
fn build(n: i64, scale: f64) -> Tensor {
    let mut buf: Any = [];
    let mut i: i64 = 0;
    while i < n * n {
        buf = array_push(buf, scale * float(i % 7) + 0.25);
        i = i + 1;
    }
    return Tensor.from_vec(buf, [n, n]);
}
"#;

/// Tensor workload family (Phase A1 fix). The pre-A1 corpus had ZERO
/// tensor programs, so the tensor-FP accounting (runtime counter +
/// TypeMix/MemoryProxy/physical_cost) would have zero training
/// variance — the retrained head could only ignore the new signal.
/// Four hot-loop shapes (matmul, element-wise, method-form reduction,
/// tensor+scalar mix) cover each accounting path the A1 probe found
/// blind, and a graded `tensor_tg_k{0..4}` sub-family sweeps the
/// tensor-vs-scalar FP share at fixed size (the tensor analog of the
/// `grad_` family). Eval↔MIR parity for these shapes is locked by
/// `tests/test_tensor_fp_accounting.rs` before they enter the harness.
fn tensor_workloads() -> Vec<Workload> {
    let mut out = Vec::new();
    out.push(Workload {
        name: "tensor_mm_n16_i50".to_string(),
        source: format!(
            r#"{TENSOR_BUILDER}
fn mm_hot(a: Tensor, b: Tensor, iters: i64) -> f64 {{
    let mut s: f64 = 0.0;
    let mut i: i64 = 0;
    while i < iters {{
        let c: Tensor = matmul(a, b);
        s = c.sum();
        i = i + 1;
    }}
    return s;
}}
let a: Tensor = build(16, 0.5);
let b: Tensor = build(16, 0.25);
print(mm_hot(a, b, 50));
"#
        ),
    });
    out.push(Workload {
        name: "tensor_ew_n32_i200".to_string(),
        source: format!(
            r#"{TENSOR_BUILDER}
fn ew_hot(a: Tensor, b: Tensor, iters: i64) -> Tensor {{
    let mut c: Tensor = a + b;
    let mut i: i64 = 1;
    while i < iters {{
        c = a * b;
        c = c + a;
        i = i + 1;
    }}
    return c;
}}
let a: Tensor = build(32, 0.5);
let b: Tensor = build(32, 0.25);
let r: Tensor = ew_hot(a, b, 200);
print(r.sum());
"#
        ),
    });
    out.push(Workload {
        name: "tensor_red_n64_i100".to_string(),
        source: format!(
            r#"{TENSOR_BUILDER}
fn red_hot(a: Tensor, iters: i64) -> f64 {{
    let mut s: f64 = 0.0;
    let mut i: i64 = 0;
    while i < iters {{
        s = a.sum();
        i = i + 1;
    }}
    return s;
}}
let a: Tensor = build(64, 0.5);
print(red_hot(a, 100));
"#
        ),
    });
    out.push(Workload {
        name: "tensor_mix_n16_i50".to_string(),
        source: format!(
            r#"{TENSOR_BUILDER}
fn mix_hot(a: Tensor, b: Tensor, iters: i64) -> f64 {{
    let mut acc: f64 = 0.0;
    let mut i: i64 = 0;
    while i < iters {{
        let c: Tensor = matmul(a, b);
        acc = acc + 0.001;
        i = i + 1;
    }}
    return acc;
}}
let a: Tensor = build(16, 0.5);
let b: Tensor = build(16, 0.25);
print(mix_hot(a, b, 50));
"#
        ),
    });
    // Graded sub-family: per loop iteration, `k` tensor scalar-mul ops
    // and `4 - k` scalar float adds — sweeps tensor share 0%→100%.
    for k in 0u32..=4 {
        let mut body = String::new();
        for _ in 0..k {
            body.push_str("        u = u * 0.999;\n");
        }
        for j in 0..(4 - k) {
            body.push_str(&format!("        facc = facc + 0.5{j:01};\n"));
        }
        out.push(Workload {
            name: format!("tensor_tg_k{k}"),
            source: format!(
                r#"{TENSOR_BUILDER}
fn work(t: Tensor, n: i64) -> f64 {{
    let mut facc: f64 = 0.0;
    let mut u: Tensor = t * 1.0;
    let mut i: i64 = 0;
    while i < n {{
{body}        i = i + 1;
    }}
    print(facc);
    return u.sum();
}}
let t: Tensor = build(16, 0.5);
print(work(t, 128));
"#
            ),
        });
    }
    out
}

/// Memory-gradient family (Phase A item 4). What the recorded memory
/// label CAN see (verified against the executor): `heap_bytes_in_use =
/// gc_live × 4096 + arena_alloc_count × 64`, where `gc_live` moves only
/// via the explicit `gc_alloc` builtin and `arena_alloc_count` counts
/// EXECUTED `Let` statements whose escape analysis classified them
/// `AllocHint::Arena` — flat 64 bytes per execution, size-blind,
/// cumulative (never decremented). Rc-buffer memory — arrays, tensors,
/// strings, i.e. the actual memory consumers — is INVISIBLE to the
/// label (`cjc-mir-exec/src/lib.rs` heap proxy; same structural
/// blindness pattern A1 found for thermal). This family therefore
/// sweeps per-iteration arena-let executions ×4 per step — the one
/// honest dial a `.cjcl` program has — and the resulting label spread
/// is MEASURED, not assumed; the expected outcome is a small gradient
/// that documents the label's ceiling, not std > 0.05.
fn memory_gradient_workloads() -> Vec<Workload> {
    let mut out = Vec::new();
    for k in 1u32..=5 {
        let iters = 4i64.pow(k) * 64; // 256 .. 65,536
        out.push(Workload {
            name: format!("mem_grad_a{k}"),
            source: format!(
                r#"
fn churn(n: i64) -> i64 {{
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {{
        let cell: Any = [i, i + 1];
        let pair: Any = (i, i * 2);
        total = total + i;
        i = i + 1;
    }}
    return total;
}}
print(churn({iters}));
"#
            ),
        });
    }
    out
}

/// Frozen holdout set (Phase A item 7). Ten NEW programs, never used
/// in any training or tuning decision; the trainer excludes the
/// `holdout_` prefix from BOTH the train and the FNV-split test sets
/// and reports them only as a separate shadow cohort at promotion
/// gates. Shapes deliberately overlap no existing family member:
/// different constants, different loop structures, different
/// tensor/scalar mixes.
fn holdout_workloads() -> Vec<Workload> {
    let mk = |name: &str, source: &str| Workload {
        name: name.to_string(),
        source: source.to_string(),
    };
    vec![
        mk(
            "holdout_int_collatz",
            r#"
fn steps(start: i64) -> i64 {
    let mut x: i64 = start;
    let mut c: i64 = 0;
    while x > 1 {
        if x / 2 * 2 == x {
            x = x / 2;
        } else {
            x = 3 * x + 1;
        }
        c = c + 1;
    }
    return c;
}
print(steps(27));
"#,
        ),
        mk(
            "holdout_int_gcd_grid",
            r#"
fn gcd(a: i64, b: i64) -> i64 {
    let mut x: i64 = a;
    let mut y: i64 = b;
    while y != 0 {
        let t: i64 = y;
        y = x - x / y * y;
        x = t;
    }
    return x;
}
fn grid(n: i64) -> i64 {
    let mut s: i64 = 0;
    let mut i: i64 = 1;
    while i <= n {
        let mut j: i64 = 1;
        while j <= n {
            s = s + gcd(i, j);
            j = j + 1;
        }
        i = i + 1;
    }
    return s;
}
print(grid(24));
"#,
        ),
        mk(
            "holdout_fp_logistic",
            r#"
fn iterate(r: f64, n: i64) -> f64 {
    let mut x: f64 = 0.37;
    let mut i: i64 = 0;
    while i < n {
        x = r * x * (1.0 - x);
        i = i + 1;
    }
    return x;
}
print(iterate(3.61, 4000));
"#,
        ),
        mk(
            "holdout_fp_mandel_cell",
            r#"
fn escape_steps(cr: f64, ci: f64, cap: i64) -> i64 {
    let mut zr: f64 = 0.0;
    let mut zi: f64 = 0.0;
    let mut i: i64 = 0;
    while i < cap {
        let zr2: f64 = zr * zr - zi * zi + cr;
        zi = 2.0 * zr * zi + ci;
        zr = zr2;
        if zr * zr + zi * zi > 4.0 {
            return i;
        }
        i = i + 1;
    }
    return cap;
}
print(escape_steps(-0.7436, 0.1318, 3000));
"#,
        ),
        mk(
            "holdout_fp_int_alternate",
            r#"
fn alternate(n: i64) -> f64 {
    let mut acc: f64 = 0.0;
    let mut parity: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        if parity == 0 {
            acc = acc + 1.0 / (1.0 + float(i));
            parity = 1;
        } else {
            parity = 0;
        }
        i = i + 1;
    }
    return acc;
}
print(alternate(2500));
"#,
        ),
        mk(
            "holdout_tensor_scale_chain",
            r#"
fn build(n: i64) -> Tensor {
    let mut buf: Any = [];
    let mut i: i64 = 0;
    while i < n * n {
        buf = array_push(buf, 0.125 * float(i % 11) + 0.5);
        i = i + 1;
    }
    return Tensor.from_vec(buf, [n, n]);
}
fn chain(t: Tensor, iters: i64) -> f64 {
    let mut u: Tensor = t * 1.0;
    let mut i: i64 = 0;
    while i < iters {
        u = u * 1.0009;
        u = u - 0.0004;
        i = i + 1;
    }
    return u.sum();
}
let t: Tensor = build(24);
print(chain(t, 160));
"#,
        ),
        mk(
            "holdout_tensor_mm_rect",
            r#"
fn build_rect(rows: i64, cols: i64, scale: f64) -> Tensor {
    let mut buf: Any = [];
    let mut i: i64 = 0;
    while i < rows * cols {
        buf = array_push(buf, scale * float(i % 5) + 0.2);
        i = i + 1;
    }
    return Tensor.from_vec(buf, [rows, cols]);
}
fn mm(a: Tensor, b: Tensor, iters: i64) -> f64 {
    let mut s: f64 = 0.0;
    let mut i: i64 = 0;
    while i < iters {
        let c: Tensor = matmul(a, b);
        s = c.sum();
        i = i + 1;
    }
    return s;
}
let a: Tensor = build_rect(8, 24, 0.4);
let b: Tensor = build_rect(24, 12, 0.3);
print(mm(a, b, 40));
"#,
        ),
        mk(
            "holdout_mixed_ema",
            r#"
fn ema(n: i64) -> f64 {
    let mut level: f64 = 10.0;
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let sample: f64 = float(i % 17) * 0.3;
        level = 0.9 * level + 0.1 * sample;
        total = total + i % 3;
        i = i + 1;
    }
    print(total);
    return level;
}
print(ema(3000));
"#,
        ),
        mk(
            "holdout_deep_calls",
            r#"
fn leaf(x: i64) -> i64 { return x * 2 + 1; }
fn mid(x: i64) -> i64 { return leaf(x) + leaf(x + 1); }
fn top(x: i64) -> i64 { return mid(x) + mid(x + 2); }
fn drive(n: i64) -> i64 {
    let mut s: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        s = s + top(i);
        i = i + 1;
    }
    return s;
}
print(drive(800));
"#,
        ),
        mk(
            "holdout_alloc_pulse",
            r#"
fn pulse(n: i64) -> i64 {
    let mut keep: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let burst: Any = [i, i + 1, i + 2, i + 3];
        let tag: Any = (i, i * i);
        keep = keep + i % 7;
        i = i + 1;
    }
    return keep;
}
print(pulse(5000));
"#,
        ),
    ]
}

/// Assemble the full workload list: 9 static snapshot programs +
/// 30 thermal-gradient programs + the 95-program train corpus +
/// 9 tensor-family programs (Phase A1) + 5 memory-gradient programs
/// (Phase A4) + 10 frozen holdout programs (Phase A7). Order matters:
/// the row-hash canary indexes `workloads[8]` (fp_hot), so new
/// families append AFTER the static set.
fn all_workloads() -> Vec<Workload> {
    let mut out: Vec<Workload> = PROGRAMS
        .iter()
        .map(|p| Workload {
            name: p.name.to_string(),
            source: p.source.to_string(),
        })
        .collect();
    out.extend(thermal_gradient_workloads());
    for p in train_programs::PROGRAMS {
        out.push(Workload {
            name: format!("train_{}", p.name),
            source: p.source.to_string(),
        });
    }
    out.extend(tensor_workloads());
    out.extend(memory_gradient_workloads());
    out.extend(holdout_workloads());
    out
}

// =============================================================================
// Ablation configurations
// =============================================================================

/// Synthetic-predictor configurations (Option A) — the original
/// Track-2 set, kept for cross-session comparability.
const CONFIG_IDS: &[&str] = &["baseline", "nss", "quantum", "nss_quantum", "full_pinn"];

/// Recorded-trace configurations (Option B). Same stacks as the three
/// pressure-consuming synthetic configs, but the predictor is a
/// [`RecordedPressurePredictor`] built from a real instrumented run of
/// the program. `baseline` and `quantum` don't consume pressure, so
/// they have no recorded variant.
///
/// The `_t50` / `_c80` / `_c60` variants sweep the thermal threshold /
/// hard cap — a legitimate ablation axis now that recorded thermal
/// forms a gradient: different caps trip on different subsets of the
/// gradient family, which is exactly the label variance v2 training
/// needs.
const CONFIG_IDS_RECORDED: &[&str] = &[
    "nss_rec",
    "nss_rec_t50",
    "nss_quantum_rec",
    "full_pinn_rec",
    "full_pinn_rec_c80",
    "full_pinn_rec_c60",
    // PINN v2: same stack as full_pinn_rec but the closed-form thermal
    // is replaced by the TRAINED head (CPB0 bundle, shadow-gate
    // PROMOTE) — measures the PLAN-level consequence of promotion.
    "full_pinn_v2_rec",
];

/// Forced-plan diagnostic configs (PINN v2 §5 follow-up): apply a fixed
/// pass list to EVERY function, bypassing cost-model ranking but NOT
/// legality — each `(function, pass)` pair is individually gate-checked
/// and rejected pairs are dropped from the plan. Purpose: generate
/// energy-label variance across plans (pre-v2, only 39/1474 rows
/// diverged from baseline because ranked plans mostly tie it), so a
/// future energy head has something to train on. `force_none` pins the
/// fully-unoptimized anchor (empty per-function plans).
const CONFIG_IDS_FORCED: &[&str] = &[
    "force_none",
    "force_cf",
    "force_sr",
    "force_dce",
    "force_cse",
    "force_licm",
    "force_unroll",
    "force_all",
];

/// The pass list a forced config applies to every function; `None` for
/// ranked configs.
fn forced_passes(config: &str) -> Option<Vec<&'static str>> {
    Some(match config {
        "force_none" => vec![],
        "force_cf" => vec!["constant_fold"],
        "force_sr" => vec!["strength_reduce"],
        "force_dce" => vec!["dce"],
        "force_cse" => vec!["cse"],
        "force_licm" => vec!["licm"],
        "force_unroll" => vec!["loop_unroll"],
        "force_all" => CANONICAL_PASSES.to_vec(),
        _ => return None,
    })
}

/// Rank `mir` under one ablation configuration. Returns the report plus
/// the cost-model identity that drove it. `recorded` backs the `*_rec`
/// configs.
fn rank_under(
    config: &str,
    mir: &MirProgram,
    features: &CanaFeatures,
    recorded: &RecordedPressurePredictor,
    v2_head: &PinnThermalV2,
) -> (RankingReport, String, u32) {
    match config {
        "baseline" => {
            let model = LinearCostModel::trained();
            let (id, ver) = (model.name().to_string(), model.version());
            let report = PassRanker::new(model, PerPassLegalityGate::new()).rank(mir, features);
            (report, id, ver)
        }
        "nss" => {
            let model = ThermalAwareCostModel::new(
                LinearCostModel::trained(),
                NssPressurePredictor::default(),
            );
            let (id, ver) = (model.name().to_string(), model.version());
            let report = PassRanker::new(model, PerPassLegalityGate::new()).rank(mir, features);
            (report, id, ver)
        }
        "nss_rec" => {
            let model = ThermalAwareCostModel::new(LinearCostModel::trained(), recorded.clone());
            let (id, ver) = (model.name().to_string(), model.version());
            let report = PassRanker::new(model, PerPassLegalityGate::new()).rank(mir, features);
            (report, id, ver)
        }
        "nss_rec_t50" => {
            let model = ThermalAwareCostModel::new(LinearCostModel::trained(), recorded.clone())
                .with_thermal_threshold(0.5);
            let (id, ver) = (model.name().to_string(), model.version());
            let report = PassRanker::new(model, PerPassLegalityGate::new()).rank(mir, features);
            (report, id, ver)
        }
        "quantum" => {
            let model = LinearCostModel::trained();
            let (id, ver) = (model.name().to_string(), model.version());
            let adapter = EnergyAwarePassRanker::new(
                PassRanker::new(model, PerPassLegalityGate::new()),
                Box::new(NullPressurePredictor),
            );
            (adapter.rank(mir, features), id, ver)
        }
        "nss_quantum" => {
            let model = LinearCostModel::trained();
            let (id, ver) = (model.name().to_string(), model.version());
            let adapter = EnergyAwarePassRanker::new(
                PassRanker::new(model, PerPassLegalityGate::new()),
                Box::new(NssPressurePredictor::default()),
            );
            (adapter.rank(mir, features), id, ver)
        }
        "nss_quantum_rec" => {
            let model = LinearCostModel::trained();
            let (id, ver) = (model.name().to_string(), model.version());
            let adapter = EnergyAwarePassRanker::new(
                PassRanker::new(model, PerPassLegalityGate::new()),
                Box::new(recorded.clone()),
            );
            (adapter.rank(mir, features), id, ver)
        }
        "full_pinn" => {
            let model = PinnPhysicalCostModel::new(
                LinearCostModel::trained(),
                NssPressurePredictor::default(),
            );
            let (id, ver) = (model.name().to_string(), model.version());
            let adapter = EnergyAwarePassRanker::new(
                PassRanker::new(model, PerPassLegalityGate::new()),
                Box::new(NssPressurePredictor::default()),
            );
            (adapter.rank(mir, features), id, ver)
        }
        "full_pinn_rec" => {
            let model = PinnPhysicalCostModel::new(LinearCostModel::trained(), recorded.clone());
            let (id, ver) = (model.name().to_string(), model.version());
            let adapter = EnergyAwarePassRanker::new(
                PassRanker::new(model, PerPassLegalityGate::new()),
                Box::new(recorded.clone()),
            );
            (adapter.rank(mir, features), id, ver)
        }
        "full_pinn_v2_rec" => {
            let model = PinnPhysicalCostModel::new(LinearCostModel::trained(), recorded.clone())
                .with_thermal_head(v2_head.clone());
            let (id, ver) = (model.name().to_string(), model.version());
            let adapter = EnergyAwarePassRanker::new(
                PassRanker::new(model, PerPassLegalityGate::new()),
                Box::new(recorded.clone()),
            );
            (adapter.rank(mir, features), id, ver)
        }
        "full_pinn_rec_c80" | "full_pinn_rec_c60" => {
            let cap = if config.ends_with("c80") { 0.80 } else { 0.60 };
            let model = PinnPhysicalCostModel::new(LinearCostModel::trained(), recorded.clone())
                .with_constraints(PhysicalConstraints {
                    max_thermal_pressure: cap,
                    ..PhysicalConstraints::default()
                });
            let (id, ver) = (model.name().to_string(), model.version());
            let adapter = EnergyAwarePassRanker::new(
                PassRanker::new(model, PerPassLegalityGate::new()),
                Box::new(recorded.clone()),
            );
            (adapter.rank(mir, features), id, ver)
        }
        other => panic!("unknown ablation config {other}"),
    }
}

// =============================================================================
// Per-(program × config) experiment
// =============================================================================

fn total_expr_count(features: &CanaFeatures) -> u64 {
    features
        .per_fn
        .values()
        .map(|f| f.memory.expr_count as u64)
        .fold(0u64, |a, b| a.saturating_add(b))
}

/// Unified output of planning one experiment, whether the plan came
/// from a cost-model ranking or a forced pass list.
struct PlannedExperiment {
    plan: PassPlan,
    cost_model_id: String,
    cost_model_version: u32,
    recommended_count: u32,
    dropped_count: u32,
    legality_approved: bool,
    legality_violation_count: u32,
}

/// Produce the pass plan for `config`. Ranked configs go through
/// `rank_under`; `force_*` configs apply a fixed list to EVERY function
/// with per-`(function, pass)` legality checks — the gate retains final
/// authority, rejected pairs are dropped (counted in `dropped_count`),
/// and the row's `legality_approved` records whether the gate filtered
/// anything. Every featurized function is inserted into the plan map
/// (an empty list = run NOTHING for that function — without the
/// insertion, `optimize_program_with_plan` falls back to the FULL
/// default sequence).
fn plan_under(
    config: &str,
    mir: &MirProgram,
    features: &CanaFeatures,
    recorded: &RecordedPressurePredictor,
    v2_head: &PinnThermalV2,
) -> PlannedExperiment {
    if let Some(forced) = forced_passes(config) {
        let gate = PerPassLegalityGate::new();
        let mut plan = PassPlan::empty();
        let mut kept = 0u32;
        let mut dropped = 0u32;
        let mut violation_count = 0u32;
        for fn_name in features.per_fn.keys() {
            let mut passes: Vec<String> = Vec::new();
            for p in &forced {
                let mut seq = PassSequence::default();
                seq.per_function
                    .insert(fn_name.clone(), vec![ProposedPass::Run(p.to_string())]);
                match gate.verify(mir, &seq, features) {
                    LegalityVerdict::Approved => {
                        passes.push(p.to_string());
                        kept = kept.saturating_add(1);
                    }
                    LegalityVerdict::Rejected(v) => {
                        dropped = dropped.saturating_add(1);
                        violation_count = violation_count.saturating_add(v.len() as u32);
                    }
                }
            }
            plan.per_function.insert(fn_name.clone(), passes);
        }
        return PlannedExperiment {
            plan,
            cost_model_id: "forced_plan".to_string(),
            cost_model_version: 0,
            recommended_count: kept,
            dropped_count: dropped,
            legality_approved: violation_count == 0,
            legality_violation_count: violation_count,
        };
    }

    let (report, cost_model_id, cost_model_version) =
        rank_under(config, mir, features, recorded, v2_head);
    let (legality_approved, legality_violation_count) = match &report.verdict {
        LegalityVerdict::Approved => (true, 0u32),
        LegalityVerdict::Rejected(v) => (false, v.len() as u32),
    };
    PlannedExperiment {
        plan: pass_plan_from(&report.sequence),
        cost_model_id,
        cost_model_version,
        recommended_count: report.total_recommended() as u32,
        dropped_count: report.total_dropped() as u32,
        legality_approved,
        legality_violation_count,
    }
}

/// One `(workload × config)` experiment. Returns the row plus the RAW
/// energy proxy of the optimized run; the caller normalizes `score`
/// against the program's `baseline` config before persisting.
fn run_experiment(
    prog: &Workload,
    config: &str,
    recorded: &RecordedPressurePredictor,
    v2_head: &PinnThermalV2,
) -> (CompilationProfile, f64) {
    let wall_start = Instant::now();

    // -- Compile + plan (ranked or forced) -----------------------------------
    let (ast, diags) = cjc_parser::parse_source(&prog.source);
    assert!(
        !diags.has_errors(),
        "parse errors in {}: {:?}",
        prog.name,
        diags.diagnostics
    );
    let mut al = cjc_hir::AstLowering::new();
    let hir = al.lower_program(&ast);
    let mut h2m = cjc_mir::HirToMir::new();
    let mir = h2m.lower_program(&hir);
    let features = analyze_program(&mir).features;

    let planned = plan_under(config, &mir, &features, recorded, v2_head);
    let plan = planned.plan;
    let mut optimized = optimize_program_with_plan(&mir, &plan);
    cjc_mir::escape::annotate_program(&mut optimized);
    let compile_wall_micros = wall_start.elapsed().as_micros() as u64;

    // -- Phase A item 5: run the EXISTING verifiers on every optimized
    //    program. They existed but were never exercised here; mandatory
    //    before forced/selected plans get more aggressive. Hard panic =
    //    the regen gate fails loudly, never a silently-poisoned corpus.
    if let Err(errors) = cjc_mir::nogc_verify::verify_nogc(&optimized) {
        panic!(
            "NoGC verifier rejected optimized {}/{}: {:?}",
            prog.name, config, errors
        );
    }
    let mir_legality = cjc_mir::verify::verify_mir_legality(&optimized);
    assert!(
        mir_legality.is_ok(),
        "MIR legality verifier rejected optimized {}/{}: {:?}",
        prog.name,
        config,
        mir_legality.errors()
    );

    // -- Deterministic size metric (kept in the row as structural info;
    //    no longer the score) ------------------------------------------------
    let mir_nodes_before = total_expr_count(&features);
    let optimized_features = cjc_cana::features::extract(&optimized);
    let mir_nodes_after = total_expr_count(&optimized_features);

    // Unroll-explosion guard (Phase A item 5). The research doc
    // proposed 1.5×; the first gated regen MEASURED that bound wrong:
    // the ranked BASELINE plan fully unrolls countable 8-trip loops,
    // legitimately growing `grad_f10_d2_n64` 97 → 605 nodes (6.24×).
    // Full unroll of trip count N is ~N× body growth by design, so the
    // runaway-duplication cap sits above the designed maximum: 16×.
    // The end-of-run report prints the measured corpus-wide max so the
    // bound stays evidence-tightened, not guessed.
    let size_ratio = if mir_nodes_before > 0 {
        mir_nodes_after as f64 / mir_nodes_before as f64
    } else {
        1.0
    };
    assert!(
        size_ratio <= 16.0,
        "code-size explosion in {}/{}: {} -> {} nodes (ratio {size_ratio:.3} > 16)",
        prog.name,
        config,
        mir_nodes_before,
        mir_nodes_after
    );

    // -- NSS predictions (full per-function maps; schema v3 persists
    //    them per function, not just the max) --------------------------------
    // Recorded configs record the recorded predictor's view; synthetic
    // configs record Option A's — the row reflects what the ranker saw.
    // Membership test, NOT `ends_with("_rec")`: the `_t50`/`_c80`/`_c60`
    // variants rank under recorded pressures too, and a suffix match
    // silently stamped them with Option-A labels (caught by the v2 §2.1
    // data-sanity pass).
    let (nss_cpu_map, nss_memory_map, nss_thermal_map) =
        if CONFIG_IDS_RECORDED.contains(&config) {
            (
                recorded.predict_cpu_saturation(&mir, &features),
                recorded.predict_memory_peak(&mir, &features),
                recorded.predict_thermal(&mir, &features),
            )
        } else {
            let nss = NssPressurePredictor::default();
            (
                nss.predict_cpu_saturation(&mir, &features),
                nss.predict_memory_peak(&mir, &features),
                nss.predict_thermal(&mir, &features),
            )
        };
    let max_of = |m: &BTreeMap<String, f64>| m.values().copied().fold(0.0f64, f64::max);
    let (nss_cpu_max, nss_memory_max, nss_thermal_max) = (
        max_of(&nss_cpu_map),
        max_of(&nss_memory_map),
        max_of(&nss_thermal_map),
    );

    // -- Workload estimates + PINN predictions + per-function profiles ------
    // Neutral pass ("dce" has identity physical amplification) so the
    // row captures the program's intrinsic workload, not a per-pass
    // variant. Schema v3 (Phase A items 2+3): the per-function query
    // values, the loop features from CfgMetrics (which existed but
    // never reached rows), and the per-function pressure labels all
    // persist alongside the program-level sums/maxes.
    let coeffs = PhysicalCoefficients::default();
    let mut est_flops = 0u64;
    let mut est_read = 0u64;
    let mut est_written = 0u64;
    let mut est_alloc = 0u64;
    let mut est_ws = 0u64;
    let mut est_float_ops = 0u64;
    let mut pinn_energy_max = 0.0f64;
    let mut pinn_thermal_max = 0.0f64;
    let mut pinn_bandwidth_max = 0.0f64;
    let mut per_function: Vec<(String, FnProfile)> = Vec::new();
    for (fn_name, ff) in &features.per_fn {
        let q = build_physical_query(fn_name, "dce", ff);
        est_flops = est_flops.saturating_add(q.flops_estimate);
        est_read = est_read.saturating_add(q.bytes_read_estimate);
        est_written = est_written.saturating_add(q.bytes_written_estimate);
        est_alloc = est_alloc.saturating_add(q.allocation_bytes_estimate);
        est_ws = est_ws.saturating_add(q.working_set_bytes_estimate);
        est_float_ops = est_float_ops.saturating_add(q.float_ops_estimate);
        if let Some(est) = predict_physical(&q, &coeffs) {
            pinn_energy_max = pinn_energy_max.max(est.energy_estimate);
            pinn_thermal_max = pinn_thermal_max.max(est.thermal_pressure);
            pinn_bandwidth_max = pinn_bandwidth_max.max(est.bandwidth_pressure);
        }
        per_function.push((
            fn_name.clone(),
            FnProfile {
                flops: q.flops_estimate,
                bytes_read: q.bytes_read_estimate,
                bytes_written: q.bytes_written_estimate,
                alloc_bytes: q.allocation_bytes_estimate,
                working_set: q.working_set_bytes_estimate,
                float_ops: q.float_ops_estimate,
                countable_loop_count: ff.cfg.countable_loop_count,
                max_loop_depth: ff.cfg.max_loop_depth,
                nss_cpu: nss_cpu_map.get(fn_name).copied().unwrap_or(0.0),
                nss_memory: nss_memory_map.get(fn_name).copied().unwrap_or(0.0),
                nss_thermal: nss_thermal_map.get(fn_name).copied().unwrap_or(0.0),
            },
        ));
    }

    // -- Parity + energy: AST-eval vs INSTRUMENTED MIR-exec on the
    //    OPTIMIZED program. The same run serves both purposes — the
    //    instrumented-vs-uninstrumented output identity is locked by
    //    tests/test_mir_exec_instrumented.rs, so enabling tracing here
    //    cannot perturb the parity verdict.
    let mut interp = cjc_eval::Interpreter::new(SEED);
    let eval_result = interp.exec(&ast);

    trace::with_trace(|c| {
        c.reset();
        c.enable();
    });
    let mut exec = cjc_mir_exec::MirExecutor::new(SEED);
    exec.scan_ast_imports(&ast);
    let exec_result = exec.exec(&optimized);
    let opt_events = trace::with_trace(|c| {
        c.disable();
        let e = c.take();
        c.reset();
        e
    });

    let parity_match = match (&eval_result, &exec_result) {
        (Ok(_), Ok(_)) => Some(interp.output == exec.output),
        _ => Some(false),
    };

    // Deterministic modeled energy of the OPTIMIZED run (§5.3 metric 5):
    //   energy = executed_statements + FP_ENERGY_WEIGHT · fp_ops + heap_pages
    // Plans that eliminate executed work (unroll fewer cond evals, CF/DCE
    // fewer statements) lower it; the FP term prices the thermal
    // dimension a size-ratio metric was structurally blind to.
    let mut instr_total: u64 = 0;
    let mut heap_max: u64 = 0;
    let mut fp_acc = KahanAccumulatorF64::new();
    for ev in &opt_events {
        instr_total = instr_total.saturating_add(ev.instruction_count as u64);
        heap_max = heap_max.max(ev.heap_bytes_in_use);
        let fp_in_window = ev.thermal_intensity * ev.instruction_count as f64;
        fp_acc.add(fp_in_window);
    }
    let fp_total = fp_acc.finalize();
    let fp_term = FP_ENERGY_WEIGHT * fp_total;
    let heap_term = heap_max as f64 / 4096.0;
    let energy_partial = instr_total as f64 + fp_term;
    let raw_energy = energy_partial + heap_term;

    let pass_sequence: Vec<(String, Vec<String>)> = plan
        .per_function
        .iter()
        .map(|(f, seq)| (f.clone(), seq.clone()))
        .collect();

    let row = CompilationProfile {
        schema_version: PROFILE_SCHEMA_VERSION,
        program_name: prog.name.to_string(),
        program_hash: features.program_hash.0,
        feature_hash: features.feature_hash.0,
        sidecar_bundle_hash: 0, // no sidecar in the ablation harness (yet)
        config_id: config.to_string(),
        cost_model_id: planned.cost_model_id,
        cost_model_version: planned.cost_model_version,
        pass_sequence,
        per_function,
        estimated_flops: est_flops,
        estimated_bytes_read: est_read,
        estimated_bytes_written: est_written,
        estimated_alloc_bytes: est_alloc,
        estimated_working_set: est_ws,
        estimated_float_ops: est_float_ops,
        nss_predicted_cpu_max: nss_cpu_max,
        nss_predicted_memory_max: nss_memory_max,
        nss_predicted_thermal_max: nss_thermal_max,
        pinn_predicted_energy_max: pinn_energy_max,
        pinn_predicted_thermal_max: pinn_thermal_max,
        pinn_predicted_bandwidth_max: pinn_bandwidth_max,
        mir_nodes_before,
        mir_nodes_after,
        recommended_count: planned.recommended_count,
        dropped_count: planned.dropped_count,
        legality_approved: planned.legality_approved,
        legality_violation_count: planned.legality_violation_count,
        parity_match,
        compile_wall_micros,
        // Placeholder — the caller overwrites with the
        // baseline-relative energy ratio before persisting.
        score: raw_energy,
    };
    (row, raw_energy)
}

// =============================================================================
// Main — run all experiments, emit rows, print comparison + §5.2 gate
// =============================================================================

fn main() {
    let out_dir = PathBuf::from("bench_results/cana_ablation");
    fs::create_dir_all(&out_dir).expect("create bench_results/cana_ablation");
    let db_path = out_dir.join("profiles.cpdb");
    // Fresh file per invocation: the harness is deterministic, so
    // re-running appends identical rows; truncating keeps the corpus
    // duplicate-free for training. (Cross-session accumulation can
    // concatenate archives.)
    let _ = fs::remove_file(&db_path);

    // PINN v2 trained head — offline-trained weights, loaded read-only
    // (training never runs during compilation). The bundle is committed,
    // so a missing file means a broken checkout, not a fresh one.
    let v2_head = read_bundle(&PathBuf::from(
        "bench_results/cana_train_pinn/pinn_thermal_v2.cpb",
    ))
    .expect("CPB0 bundle missing/corrupt — run `cargo run --release -p cana-train-pinn -- train`")
    .head;

    let workloads = all_workloads();
    let n_configs = CONFIG_IDS.len() + CONFIG_IDS_RECORDED.len() + CONFIG_IDS_FORCED.len();
    println!("=================================================================");
    println!(
        "Phase A6 v4 — {} programs × {} configs = {} experiments (seed {SEED})",
        workloads.len(),
        n_configs,
        workloads.len() * n_configs,
    );
    println!("Score = energy(optimized run) / energy(baseline config), lower = better");
    println!("=================================================================\n");

    // Per program: one instrumented run (Option B) feeds the recorded
    // predictor used by the *_rec configs.
    let mut recorded_preds: BTreeMap<String, RecordedPressurePredictor> = BTreeMap::new();
    for prog in &workloads {
        let (ast, diags) = cjc_parser::parse_source(&prog.source);
        assert!(!diags.has_errors(), "parse errors in {}", prog.name);
        let (_val, exec, events) = run_program_instrumented(&ast, SEED).expect("instrumented run");
        let recorded = RecordedPressurePredictor::from_recorded_events(
            events,
            exec.trace_node_assignments().clone(),
        );
        recorded_preds.insert(prog.name.clone(), recorded);
    }

    // rows[program][config] = profile (score already baseline-relative).
    let mut rows: BTreeMap<String, BTreeMap<&str, CompilationProfile>> = BTreeMap::new();
    for prog in &workloads {
        let recorded = &recorded_preds[&prog.name];
        // Baseline first — its raw energy normalizes the others.
        let (mut base_row, base_energy) = run_experiment(prog, "baseline", recorded, &v2_head);
        let normalizer = base_energy.max(1.0);
        base_row.score = base_energy / normalizer; // 1.0 by construction
        append_row(&db_path, &base_row).expect("append profile row");
        rows.entry(prog.name.clone())
            .or_default()
            .insert("baseline", base_row);
        for config in CONFIG_IDS
            .iter()
            .chain(CONFIG_IDS_RECORDED.iter())
            .chain(CONFIG_IDS_FORCED.iter())
            .filter(|c| **c != "baseline")
        {
            let (mut row, raw_energy) = run_experiment(prog, config, recorded, &v2_head);
            row.score = raw_energy / normalizer;
            append_row(&db_path, &row).expect("append profile row");
            rows.entry(prog.name.clone())
                .or_default()
                .insert(config, row);
        }
    }

    // -- Thermal-gradient verification ----------------------------------------
    // The gradient family must produce a SPREAD of recorded thermal
    // values, not fp_hot's 0/1 step — this is the label-variance
    // prerequisite for v2 training.
    println!("Thermal gradient (recorded max thermal per gradient program):");
    let mut gradient_thermals: Vec<(String, f64)> = rows
        .iter()
        .filter(|(p, _)| p.starts_with("grad_"))
        .map(|(p, per_config)| {
            let t = per_config
                .get("full_pinn_rec")
                .map(|r| r.nss_predicted_thermal_max)
                .unwrap_or(f64::NAN);
            (p.clone(), t)
        })
        .collect();
    gradient_thermals.sort_by(|a, b| a.1.total_cmp(&b.1));
    for chunk in gradient_thermals.chunks(3) {
        let line: Vec<String> = chunk
            .iter()
            .map(|(p, t)| format!("{p:<22} {t:>6.3}"))
            .collect();
        println!("  {}", line.join("   "));
    }
    let distinct_bands = {
        let mut bands: Vec<u32> = gradient_thermals
            .iter()
            .filter(|(_, t)| t.is_finite())
            .map(|(_, t)| (t * 10.0).floor() as u32)
            .collect();
        bands.sort_unstable();
        bands.dedup();
        bands.len()
    };
    println!("  → {distinct_bands} distinct 0.1-wide thermal bands across the gradient family");

    // -- Score spread summary -------------------------------------------------
    println!(
        "\nPer-config score statistics across {} programs:",
        rows.len()
    );
    println!(
        "{:<18} | {:>8} | {:>8} | {:>8} | {:>10}",
        "config", "min", "mean", "max", "≠baseline"
    );
    println!("{}", "-".repeat(64));
    for config in CONFIG_IDS
        .iter()
        .chain(CONFIG_IDS_RECORDED.iter())
        .chain(CONFIG_IDS_FORCED.iter())
    {
        let mut acc = KahanAccumulatorF64::new();
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut n = 0u32;
        let mut differs = 0u32;
        for per_config in rows.values() {
            if let Some(r) = per_config.get(config) {
                acc.add(r.score);
                min = min.min(r.score);
                max = max.max(r.score);
                n += 1;
                if (r.score - 1.0).abs() > 1e-9 {
                    differs += 1;
                }
            }
        }
        let mean = acc.finalize() / n.max(1) as f64;
        println!(
            "{:<18} | {:>8.4} | {:>8.4} | {:>8.4} | {:>10}",
            config, min, mean, max, differs
        );
    }

    // -- Code-size ratio report (Phase A item 5) ------------------------------
    // Measured corpus-wide max — the evidence base for the explosion
    // cap above (16×). If this number creeps toward the cap, the cap
    // is doing its job; tighten or investigate, don't raise it blindly.
    let mut max_ratio = 0.0f64;
    let mut max_ratio_at = String::new();
    for (prog, per_config) in &rows {
        for (config, r) in per_config {
            if r.mir_nodes_before > 0 {
                let ratio = r.mir_nodes_after as f64 / r.mir_nodes_before as f64;
                if ratio > max_ratio {
                    max_ratio = ratio;
                    max_ratio_at = format!("{prog}/{config}");
                }
            }
        }
    }
    println!(
        "\nCode-size ratio (nodes_after/nodes_before): corpus max {max_ratio:.3} at {max_ratio_at} (hard cap 16)"
    );

    // -- Differentiation check: did real traces change ANY plan? ------------
    let mut diverged: Vec<String> = Vec::new();
    let mut diff_lines_printed = 0usize;
    const MAX_DIFF_LINES: usize = 24;
    println!("\nPlan diffs (synthetic vs recorded, first {MAX_DIFF_LINES} lines):");
    for (prog, per_config) in &rows {
        for (syn, rec) in [
            ("nss", "nss_rec"),
            ("nss", "nss_rec_t50"),
            ("nss_quantum", "nss_quantum_rec"),
            ("full_pinn", "full_pinn_rec"),
            ("full_pinn", "full_pinn_rec_c80"),
            ("full_pinn", "full_pinn_rec_c60"),
            ("full_pinn", "full_pinn_v2_rec"),
        ] {
            if per_config[syn].pass_sequence != per_config[rec].pass_sequence {
                diverged.push(format!("{prog}:{rec}"));
                // Show exactly which passes the real pressure withheld
                // (or added) per function — union of both plans, so a
                // function dropped ENTIRELY (PINN hard limit zeroing
                // every benefit) still prints.
                let syn_map: BTreeMap<&String, &Vec<String>> = per_config[syn]
                    .pass_sequence
                    .iter()
                    .map(|(f, p)| (f, p))
                    .collect();
                let rec_map: BTreeMap<&String, &Vec<String>> = per_config[rec]
                    .pass_sequence
                    .iter()
                    .map(|(f, p)| (f, p))
                    .collect();
                let empty: Vec<String> = Vec::new();
                let mut all_fns: Vec<&String> =
                    syn_map.keys().chain(rec_map.keys()).copied().collect();
                all_fns.sort();
                all_fns.dedup();
                for func in all_fns {
                    let syn_passes = syn_map.get(func).copied().unwrap_or(&empty);
                    let rec_passes = rec_map.get(func).copied().unwrap_or(&empty);
                    if syn_passes != rec_passes && diff_lines_printed < MAX_DIFF_LINES {
                        println!(
                            "  [plan diff] {prog}/{func} under {rec}: {:?} -> {:?}",
                            syn_passes, rec_passes
                        );
                        diff_lines_printed += 1;
                    }
                }
            }
        }
    }

    // -- PINN v1 head vs v2 trained head: did promotion change PLANS? --------
    let mut v2_plan_diffs = 0usize;
    let mut v2_score_diffs = 0usize;
    for per_config in rows.values() {
        let v1 = &per_config["full_pinn_rec"];
        let v2 = &per_config["full_pinn_v2_rec"];
        if v1.pass_sequence != v2.pass_sequence {
            v2_plan_diffs += 1;
        }
        if (v1.score - v2.score).abs() > 1e-12 {
            v2_score_diffs += 1;
        }
    }
    println!(
        "\nPINN v2 head vs v1 closed form (full_pinn_v2_rec vs full_pinn_rec): \
         {v2_plan_diffs}/{} plans differ, {v2_score_diffs}/{} scores differ",
        rows.len(),
        rows.len()
    );

    // -- §5.2 promotion gate over the recorded cohort -------------------------
    // full_pinn_rec vs the best NON-PINN RANKED config (cap variants are
    // PINN too — comparing PINN against itself would inflate ties; the
    // force_* diagnostics are not real ranking candidates and would
    // poison the comparison). Lower = better.
    let margin = 0.1;
    let mut pinn_wins = 0usize;
    let mut ties = 0usize;
    for per_config in rows.values() {
        let pinn = per_config["full_pinn_rec"].score;
        let best_other = per_config
            .iter()
            .filter(|(c, _)| !c.starts_with("full_pinn") && !c.starts_with("force_"))
            .map(|(_, r)| r.score)
            .fold(f64::INFINITY, f64::min);
        if pinn <= best_other - margin {
            pinn_wins += 1;
        } else if (pinn - best_other).abs() < 1e-12 {
            ties += 1;
        }
    }
    let total = rows.len();
    let parity_all = rows
        .values()
        .flat_map(|m| m.values())
        .all(|r| r.parity_match == Some(true));

    println!("\n----------------------------------------------------------------");
    println!(
        "Plan divergence (synthetic vs recorded): {}",
        if diverged.is_empty() {
            "NONE — recorded pressures did not change any plan".to_string()
        } else {
            format!("{} config-program pairs", diverged.len())
        }
    );
    println!(
        "§5.2 gate (full_pinn_rec): wins (≥{margin} margin): {pinn_wins}/{total}   ties: {ties}/{total}"
    );
    println!(
        "Parity (all rows): {}",
        if parity_all { "100%" } else { "FAILED" }
    );

    // Row-hash stability canary: re-run one recorded experiment on the
    // fp_hot program (index 8 in the static set — the one with maximal
    // thermal signal, so the canary covers the most instrumented path).
    let canary = &workloads[8];
    let (mut again, raw) =
        run_experiment(canary, "full_pinn_rec", &recorded_preds[&canary.name], &v2_head);
    let (_, base_raw) =
        run_experiment(canary, "baseline", &recorded_preds[&canary.name], &v2_head);
    again.score = raw / base_raw.max(1.0);
    let stable = rows[&canary.name]["full_pinn_rec"].row_hash() == again.row_hash();
    println!(
        "Row-hash stability (double-run, wall-clock excluded): {}",
        if stable { "byte-identical" } else { "DRIFT" }
    );

    let back = read_all(&db_path).expect("read back profile db");
    println!("\nProfile DB: {} rows at {}", back.len(), db_path.display());
    let row_target_met = back.len() >= 1000;
    println!(
        "≥1000-row corpus prerequisite: {}",
        if row_target_met { "MET" } else { "NOT MET" }
    );
    let verdict = if pinn_wins * 10 >= total * 6 && row_target_met {
        "PINN v2 promotion gate: WOULD PASS (≥60% wins + 1000 rows)"
    } else {
        "PINN v2 promotion gate: NOT MET — see §5.4"
    };
    println!("{verdict}");
    assert!(parity_all, "parity must hold across every ablation row");
    assert!(stable, "row hash must be double-run stable");
    assert!(
        distinct_bands >= 4,
        "thermal gradient must span ≥4 distinct bands, got {distinct_bands}"
    );
    assert!(row_target_met, "corpus must reach 1000 rows");
}
