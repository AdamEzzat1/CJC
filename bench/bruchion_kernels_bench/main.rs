//! `bruchion_kernels_bench` — the CJC-side benchmark record for the Bruchion kernel pack.
//!
//! What it measures: the Rust fallback path against the Bruchion kernel path for the
//! routed entry points (`kernel::relu_raw`, `dispatch::axpy`, `dispatch::dot_kahan`,
//! `ml::mse_loss`, `kernel::matmul_raw`, `dispatch::adam_step_raw`,
//! `dispatch::heat1d_residual_grad`), **through the runtime switch** the interpreter uses
//! (`runtime_policy::set_bruchion_kernels`), on the same buffers, in the same process.
//!
//! Protocol (the `cana_diagnostics` one, without the child processes): the iteration
//! count is calibrated once on the fallback arm so every arm does identical work; one
//! warm-up phase per arm; then the measured phases interleave the three arms —
//! `A1` (fallback), `A2` (fallback again: the A/A control), `B` (kernel) — so a drift in
//! the machine's state lands on all three. Each arm reports `median [min, max]` ns per
//! call over the phases; the kernel's ratio band is the most conservative one the two
//! bands allow (`lo = B.min / A1.max`, `hi = B.max / A1.min`); the A/A band is the same
//! for `A2 / A1`. The verdict is **faster only when the whole band sits below 1.0**,
//! **slower only when it sits above**, and the row also says whether the kernel's median
//! ratio is inside the A/A spread, in which case the run cannot tell the two apart.
//!
//! Every row carries a digest of the arm's outputs (FNV-1a over the bits): a timing row
//! whose bits diverged is not a comparison, and the run fails on one.
//!
//! The hard wall the other CJC benches declare applies here too: nothing measured feeds a
//! hash, a decision, or a stable field. Timings are artifacts, in
//! `bench_results/bruchion_kernels/`.
//!
//! Run through `bench/bruchion_kernels_bench/run.ps1`, which gates on the machine's load
//! and stamps the gate's readings into the provenance; or directly:
//!
//! ```text
//! $env:BRUCHION_KERNELS_DIR = "<kernel dir>"
//! cargo run -p bruchion-kernels-bench --release --features bruchion-kernels -- --out bench_results/bruchion_kernels
//! ```
//!
//! Without the feature the kernel arm is not compiled in; the report says so and the
//! fallback arm and its A/A still run, which is a record of the harness's own spread.

use std::fmt::Write as _;
use std::hint::black_box;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use cjc_runtime::bruchion::dispatch;
use cjc_runtime::{kernel, ml, runtime_policy};

struct Opts {
    out: PathBuf,
    n: usize,
    phases: usize,
    warmup: usize,
    phase_micros: u64,
    max_iters: u64,
    seed: u64,
}

fn parse_opts() -> Opts {
    let mut o = Opts {
        out: PathBuf::from("bench_results/bruchion_kernels"),
        n: 1 << 16,
        phases: 5,
        warmup: 1,
        phase_micros: 500_000,
        max_iters: 1_000_000,
        seed: 42,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let v = |i: &mut usize| -> String {
            *i += 1;
            args.get(*i).cloned().unwrap_or_else(|| panic!("missing value after {}", args[*i - 1]))
        };
        match args[i].as_str() {
            "--out" => o.out = PathBuf::from(v(&mut i)),
            "--n" => o.n = v(&mut i).parse().expect("--n"),
            "--phases" => o.phases = v(&mut i).parse().expect("--phases"),
            "--warmup" => o.warmup = v(&mut i).parse().expect("--warmup"),
            "--phase-micros" => o.phase_micros = v(&mut i).parse().expect("--phase-micros"),
            "--max-iters" => o.max_iters = v(&mut i).parse().expect("--max-iters"),
            "--seed" => o.seed = v(&mut i).parse().expect("--seed"),
            other => panic!("unknown argument {other}"),
        }
        i += 1;
    }
    o
}

// ---- inputs ----------------------------------------------------------------------

/// SplitMix64: the same generator the parity tests use, so a run is reproducible from
/// its seed alone.
struct Split(u64);
impl Split {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [-1, 1), scaled by 2^k for k in -8..8, with an exact zero every 17th
    /// element and a negative zero every 23rd — the parity tests' distribution.
    fn f64s(&mut self, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                if i % 23 == 22 {
                    return -0.0;
                }
                if i % 17 == 16 {
                    return 0.0;
                }
                let u = (self.next() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
                let k = (self.next() % 16) as i32 - 8;
                u * cjc_repro::powi_f64(2.0, k)
            })
            .collect()
    }
    fn unit(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| (self.next() >> 11) as f64 / (1u64 << 53) as f64).collect()
    }
}

fn fnv1a(bits: impl Iterator<Item = u64>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bits {
        for byte in b.to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

// ---- allocations -------------------------------------------------------------------

/// Every heap allocation (and reallocation) in the process is counted, so each row can
/// say how many a single call makes on each arm — the memory audit's number, next to the
/// time. Counting is a relaxed atomic increment per allocation; it is the same overhead
/// on every arm and it is stated in the provenance.
static ALLOC_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

struct CountingAlloc;

unsafe impl std::alloc::GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::alloc::System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        std::alloc::System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new_size: usize) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::alloc::System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn allocations_now() -> u64 {
    ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}

// ---- workloads ---------------------------------------------------------------------

/// One workload: a call to time, and a digest of what the call produced.
struct Workload {
    name: String,
    /// Whether the switch changes what the call does. A status-quo row (the unfused
    /// chain the kernel replaces) is not routed: it runs the two fallback arms only and
    /// is read against the routed row it stands beside, never as a kernel ratio.
    routed: bool,
    /// Elements per call, for the ns-per-element column (0: report ns per call only).
    elems: usize,
    call: Box<dyn FnMut()>,
    digest: Box<dyn FnMut() -> u64>,
}

fn workloads(o: &Opts) -> Vec<Workload> {
    let n = o.n;
    let mut g = Split(o.seed);
    let x = g.f64s(n);
    let y0 = g.f64s(n);
    let mut ws: Vec<Workload> = Vec::new();

    // relu: out = max(x, 0), through kernel::relu_raw.
    {
        let xs = x.clone();
        let out = std::rc::Rc::new(std::cell::RefCell::new(vec![0.0f64; n]));
        let o1 = out.clone();
        ws.push(Workload {
            routed: true,
            name: "relu".into(),
            elems: n,
            call: Box::new(move || kernel::relu_raw(&xs, &mut o1.borrow_mut())),
            digest: Box::new(move || fnv1a(out.borrow().iter().map(|v| v.to_bits()))),
        });
    }
    // axpy: y = 1.5 * x + y, y restored from y0 before every call on every arm, so each
    // call computes the same values and the restore's cost lands on all three arms.
    {
        let xs = x.clone();
        let y0c = y0.clone();
        let y = std::rc::Rc::new(std::cell::RefCell::new(y0.clone()));
        let y1 = y.clone();
        ws.push(Workload {
            routed: true,
            name: "axpy".into(),
            elems: n,
            call: Box::new(move || {
                let mut yy = y1.borrow_mut();
                yy.copy_from_slice(&y0c);
                dispatch::axpy(1.5, &xs, &mut yy);
            }),
            digest: Box::new(move || fnv1a(y.borrow().iter().map(|v| v.to_bits()))),
        });
    }
    // dot_kahan and mse: a scalar each; the digest is the last result's bits.
    {
        let (xs, ys) = (x.clone(), y0.clone());
        let last = std::rc::Rc::new(std::cell::Cell::new(0.0f64));
        let l1 = last.clone();
        ws.push(Workload {
            routed: true,
            name: "dot_kahan".into(),
            elems: n,
            call: Box::new(move || l1.set(black_box(dispatch::dot_kahan(&xs, &ys)))),
            digest: Box::new(move || last.get().to_bits()),
        });
    }
    {
        let (xs, ys) = (x.clone(), y0.clone());
        let last = std::rc::Rc::new(std::cell::Cell::new(0.0f64));
        let l1 = last.clone();
        ws.push(Workload {
            routed: true,
            name: "mse".into(),
            elems: n,
            call: Box::new(move || l1.set(black_box(ml::mse_loss(&xs, &ys).expect("mse")))),
            digest: Box::new(move || last.get().to_bits()),
        });
    }
    // matmul, through kernel::matmul_raw (Kahan per output element on both sides).
    for &(m, k, nn) in &[(64usize, 17usize, 33usize), (128, 128, 128)] {
        let a = g.f64s(m * k);
        let b = g.f64s(k * nn);
        let c = std::rc::Rc::new(std::cell::RefCell::new(vec![0.0f64; m * nn]));
        let c1 = c.clone();
        ws.push(Workload {
            routed: true,
            name: format!("matmul {m}x{k}x{nn}"),
            elems: m * k * nn,
            call: Box::new(move || kernel::matmul_raw(&a, &b, &mut c1.borrow_mut(), m, k, nn)),
            digest: Box::new(move || fnv1a(c.borrow().iter().map(|v| v.to_bits()))),
        });
    }
    // adam_step_raw at a fixed step (t = 7): params, m, v restored before every call.
    // This is the dispatch entry, where `1 - beta^t` is hoisted on both arms; the
    // `ml::adam_step` fallback loop recomputes `powf` per element, which is a CJC-side
    // cost, not the kernel's, and is not what this row compares.
    {
        let p0 = g.f64s(n);
        let grads = g.f64s(n);
        let m0 = g.f64s(n);
        let v0: Vec<f64> = g.unit(n);
        let st = std::rc::Rc::new(std::cell::RefCell::new((p0.clone(), m0.clone(), v0.clone())));
        let s1 = st.clone();
        ws.push(Workload {
            routed: true,
            name: "adam_step (t = 7)".into(),
            elems: n,
            call: Box::new(move || {
                let mut s = s1.borrow_mut();
                let (p, m, v) = &mut *s;
                p.copy_from_slice(&p0);
                m.copy_from_slice(&m0);
                v.copy_from_slice(&v0);
                dispatch::adam_step_raw(p, &grads, m, v, 1e-3, 0.9, 0.999, 1e-8, 7.0);
            }),
            digest: Box::new(move || {
                let s = st.borrow();
                fnv1a(s.0.iter().chain(s.1.iter()).chain(s.2.iter()).map(|v| v.to_bits()))
            }),
        });
    }
    // heat1d residual + gradient at the parity tests' largest shape and one bigger.
    for &(nc, np) in &[(1000usize, 9usize), (8192, 9)] {
        let xc = g.unit(nc);
        let coeffs: Vec<f64> = g.f64s(np).iter().map(|c| c * 0.01).collect();
        let f: Vec<f64> = xc.iter().map(|&xv| -std::f64::consts::PI * std::f64::consts::PI * (std::f64::consts::PI * xv).sin()).collect();
        let bufs = std::rc::Rc::new(std::cell::RefCell::new((vec![0.0f64; nc], vec![0.0f64; np], 0.0f64)));
        let b1 = bufs.clone();
        ws.push(Workload {
            routed: true,
            name: format!("heat1d_residual_grad {nc}x{np}"),
            elems: nc * np,
            call: Box::new(move || {
                let mut b = b1.borrow_mut();
                let (r, grad, loss) = &mut *b;
                *loss = dispatch::heat1d_residual_grad(&xc, &coeffs, &f, r, grad);
            }),
            digest: Box::new(move || {
                let b = bufs.borrow();
                fnv1a(b.0.iter().chain(b.1.iter()).map(|v| v.to_bits()).chain(std::iter::once(b.2.to_bits())))
            }),
        });
    }
    // mse_loss_grad: the fused loss and gradient of mean((pred - target)^2), one pass into
    // a caller buffer, routed. Beside it, unrouted, the status quo it replaces: the
    // GradGraph chain `sub -> mul -> mean -> backward` on the same two tensors, which is
    // what a CJC program's `mean((pred - target)^2)` with a gradient costs today. The two
    // rows produce the same bits (the cjc-ad parity test holds them to it); the digests
    // here are asserted equal across phases within a row, not across the rows.
    {
        let (xs, ys) = (x.clone(), y0.clone());
        let out = std::rc::Rc::new(std::cell::RefCell::new((0.0f64, vec![0.0f64; n])));
        let o1 = out.clone();
        ws.push(Workload {
            routed: true,
            name: "mse_loss_grad".into(),
            elems: n,
            call: Box::new(move || {
                let mut o = o1.borrow_mut();
                let (loss, grad) = &mut *o;
                *loss = black_box(ml::mse_loss_grad(&xs, &ys, grad).expect("mse_loss_grad"));
            }),
            digest: Box::new(move || {
                let o = out.borrow();
                fnv1a(std::iter::once(o.0.to_bits()).chain(o.1.iter().map(|v| v.to_bits())))
            }),
        });
        let pt = cjc_runtime::tensor::Tensor::from_vec(x.clone(), &[n]).expect("tensor");
        let tt = cjc_runtime::tensor::Tensor::from_vec(y0.clone(), &[n]).expect("tensor");
        let out = std::rc::Rc::new(std::cell::RefCell::new((0.0f64, vec![0.0f64; n])));
        let o1 = out.clone();
        ws.push(Workload {
            routed: false,
            name: "mse+grad via GradGraph (status quo)".into(),
            elems: n,
            call: Box::new(move || {
                let mut g = cjc_ad::GradGraph::new();
                let p = g.parameter(pt.clone());
                let t = g.parameter(tt.clone());
                let diff = g.sub(p, t);
                let sq = g.mul(diff, diff);
                let loss = g.mean(sq);
                g.backward(loss);
                let mut o = o1.borrow_mut();
                o.0 = g.tensor(loss).to_vec()[0];
                o.1.copy_from_slice(&g.grad(p).expect("a gradient").to_vec());
                black_box(&o.1);
            }),
            digest: Box::new(move || {
                let o = out.borrow();
                fnv1a(std::iter::once(o.0.to_bits()).chain(o.1.iter().map(|v| v.to_bits())))
            }),
        });
    }
    ws
}

// ---- protocol ----------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    A1,
    A2,
    B,
}
impl Arm {
    fn label(self) -> &'static str {
        match self {
            Arm::A1 => "fallback",
            Arm::A2 => "fallback (A/A)",
            Arm::B => "kernel",
        }
    }
    fn kernel_on(self) -> bool {
        matches!(self, Arm::B)
    }
}

struct Band {
    min: f64,
    med: f64,
    max: f64,
}
fn band(xs: &[f64]) -> Band {
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Band { min: s[0], med: s[s.len() / 2], max: s[s.len() - 1] }
}
/// B/A: the median ratio plus the most conservative bounds the two bands allow.
fn ratio_band(a: &Band, b: &Band) -> (f64, f64, f64) {
    (b.min / a.max.max(f64::MIN_POSITIVE), b.med / a.med.max(f64::MIN_POSITIVE), b.max / a.min.max(f64::MIN_POSITIVE))
}

struct Row {
    name: String,
    routed: bool,
    elems: usize,
    iters: u64,
    /// Heap allocations one call makes on the fallback arm and, when run, the kernel arm.
    allocs: (u64, Option<u64>),
    a1: Band,
    a2: Band,
    b: Option<Band>,
    digests: (u64, u64, Option<u64>),
    phases: Vec<(usize, Arm, f64)>,
}

fn run_workload(w: &mut Workload, o: &Opts, kernel_arm: bool) -> Row {
    let kernel_arm = kernel_arm && w.routed;
    let arms: Vec<Arm> = if kernel_arm { vec![Arm::A1, Arm::A2, Arm::B] } else { vec![Arm::A1, Arm::A2] };
    // Calibrate once on the fallback arm.
    runtime_policy::set_bruchion_kernels(false);
    (w.call)();
    let t0 = Instant::now();
    (w.call)();
    let single_ns = t0.elapsed().as_nanos().max(1) as u64;
    let iters = ((o.phase_micros * 1000) / single_ns).clamp(1, o.max_iters);
    // Allocations of one call per arm, after a first call on that arm has warmed any
    // lazily built state.
    let mut allocs_of = |on: bool| -> u64 {
        runtime_policy::set_bruchion_kernels(on);
        (w.call)();
        let before = allocations_now();
        (w.call)();
        let after = allocations_now();
        runtime_policy::set_bruchion_kernels(false);
        after - before
    };
    let allocs_a = allocs_of(false);
    let allocs_b = if kernel_arm { Some(allocs_of(true)) } else { None };
    let mut phase = |arm: Arm| -> f64 {
        runtime_policy::set_bruchion_kernels(arm.kernel_on());
        let t0 = Instant::now();
        for _ in 0..iters {
            (w.call)();
        }
        let ns = t0.elapsed().as_nanos() as f64 / iters as f64;
        runtime_policy::set_bruchion_kernels(false);
        ns
    };
    for _ in 0..o.warmup {
        for &arm in &arms {
            let _ = phase(arm);
        }
    }
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); 3];
    let mut digests: [Option<u64>; 3] = [None; 3];
    let mut phases = Vec::new();
    for p in 0..o.phases {
        for &arm in &arms {
            let ns = phase(arm);
            let idx = arm as usize;
            samples[idx].push(ns);
            phases.push((p, arm, ns));
            let d = (w.digest)();
            match digests[idx] {
                None => digests[idx] = Some(d),
                Some(prev) => assert_eq!(prev, d, "{}: {} produced different bits in two phases", w.name, arm.label()),
            }
        }
    }
    Row {
        name: w.name.clone(),
        routed: w.routed,
        elems: w.elems,
        iters,
        allocs: (allocs_a, allocs_b),
        a1: band(&samples[0]),
        a2: band(&samples[1]),
        b: if kernel_arm { Some(band(&samples[2])) } else { None },
        digests: (digests[0].unwrap(), digests[1].unwrap(), digests[2]),
        phases,
    }
}

// ---- provenance --------------------------------------------------------------------

fn sh(cmd: &str, args: &[&str]) -> String {
    Command::new(cmd)
        .args(args)
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

fn kernel_sha256(dir: &str) -> String {
    let meta = std::fs::read_to_string(PathBuf::from(dir).join("metadata.json")).unwrap_or_default();
    match meta.find("\"kernel_sha256\"") {
        Some(i) => {
            let rest = &meta[i + 15..];
            let start = rest.find('"').map(|j| j + 1).unwrap_or(0);
            let end = rest[start..].find('"').map(|j| start + j).unwrap_or(start);
            rest[start..end].to_string()
        }
        None => String::from("(not found)"),
    }
}

fn provenance(o: &Opts, feature: bool) -> Vec<(String, String)> {
    let git = |args: &[&str]| sh("git", args);
    // The record's own output directory is untracked until it is committed; it is the one
    // path that does not make the tree dirty. Anything else untracked or modified does.
    let out_prefix = o.out.to_string_lossy().replace('\\', "/");
    let dirty = git(&["status", "--porcelain"])
        .lines()
        .any(|l| !l.get(3..).unwrap_or("").replace('\\', "/").starts_with(out_prefix.trim_end_matches('/')));
    let kdir = std::env::var("BRUCHION_KERNELS_DIR").unwrap_or_else(|_| "(unset)".into());
    let unix = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let mut p = vec![
        ("repo".to_string(), git(&["rev-parse", "--show-toplevel"])),
        ("branch".to_string(), git(&["rev-parse", "--abbrev-ref", "HEAD"])),
        ("commit".to_string(), git(&["rev-parse", "HEAD"])),
        ("dirty tree".to_string(), dirty.to_string()),
        ("rustc".to_string(), sh("rustc", &["-vV"]).lines().filter(|l| l.starts_with("rustc ") || l.starts_with("host:")).collect::<Vec<_>>().join("; ")),
        ("profile".to_string(), if cfg!(debug_assertions) { "debug (NOT a record: the fallback is not auto-vectorized)".into() } else { "release".into() }),
        ("target".to_string(), format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)),
        ("logical cores".to_string(), runtime_policy::detect_cores().to_string()),
        ("runtime policy".to_string(), runtime_policy::get().summary()),
        ("feature bruchion-kernels".to_string(), feature.to_string()),
        ("BRUCHION_KERNELS_DIR".to_string(), kdir.clone()),
        ("kernel_sha256".to_string(), if feature { kernel_sha256(&kdir) } else { "(no kernel arm)".into() }),
        ("f64::powi is binary exponentiation on this target".to_string(), dispatch::f64_powi_is_binary_exponentiation().to_string()),
        ("elements per call (n)".to_string(), o.n.to_string()),
        ("protocol".to_string(), format!("iterations calibrated once on the fallback arm to ~{} us per phase (max {}); {} warm-up phase(s) per arm; {} measured phases, arms interleaved A1/A2/B; statistic: ns per call, median [min, max] over phases", o.phase_micros, o.max_iters, o.warmup, o.phases)),
        ("seed".to_string(), o.seed.to_string()),
        ("unix time".to_string(), unix.to_string()),
        ("launcher".to_string(), std::env::var("BRUCHION_BENCH_LAUNCHER").unwrap_or_else(|_| "(direct: no load gate)".into())),
        ("load gate".to_string(), std::env::var("BRUCHION_BENCH_GATE").unwrap_or_else(|_| "(none: the machine's load was not measured)".into())),
    ];
    if let Ok(boot) = std::env::var("BRUCHION_BENCH_BOOT") {
        p.push(("last boot".to_string(), boot));
    }
    p
}

// ---- output ------------------------------------------------------------------------

fn fmt_band(b: &Band, elems: usize) -> String {
    if elems > 0 {
        format!("{:.4} [{:.4}, {:.4}] ns/elem", b.med / elems as f64, b.min / elems as f64, b.max / elems as f64)
    } else {
        format!("{:.0} [{:.0}, {:.0}] ns", b.med, b.min, b.max)
    }
}

fn main() {
    let o = parse_opts();
    let feature = cfg!(feature = "bruchion-kernels");
    eprintln!("bruchion_kernels_bench: n = {}, phases = {}, kernel arm compiled in: {}", o.n, o.phases, feature);
    if feature && !runtime_policy::get().bruchion_kernels {
        runtime_policy::set_bruchion_kernels(true);
        assert!(dispatch::enabled(), "the switch did not enable the kernels");
        runtime_policy::set_bruchion_kernels(false);
    }
    let prov = provenance(&o, feature);
    let mut rows = Vec::new();
    let mut ws = workloads(&o);
    for w in ws.iter_mut() {
        eprint!("  {:<32}", w.name);
        let row = run_workload(w, &o, feature);
        eprintln!("A1 {}  A2 {}  B {}", fmt_band(&row.a1, row.elems), fmt_band(&row.a2, row.elems), row.b.as_ref().map(|b| fmt_band(b, row.elems)).unwrap_or_else(|| "-".into()));
        rows.push(row);
    }
    let peak_rss_kb = cjc_runtime::builtins::peak_rss_kb();

    // The report.
    let mut md = String::new();
    let _ = writeln!(md, "# Bruchion kernels: the CJC-side record\n");
    let _ = writeln!(md, "The Rust fallback path against the Bruchion kernel path, routed through `runtime_policy::set_bruchion_kernels`, on the same buffers, in one process. Every row's arms produced the same bits (asserted). The hard wall: nothing measured here feeds a hash, a decision, or a stable field.\n");
    let _ = writeln!(md, "## Provenance\n");
    for (k, v) in &prov {
        let _ = writeln!(md, "- {k}: {v}");
    }
    let _ = writeln!(md, "- peak RSS at exit: {peak_rss_kb} KiB\n");
    let _ = writeln!(md, "## Results\n");
    let _ = writeln!(md, "`median [min, max]` per arm over the measured phases; ratio band = kernel / fallback with the most conservative bounds the two bands allow; A/A band = the second fallback arm over the first. A row is **faster** only when the whole kernel band sits below 1.0 and **slower** only when it sits above; \"inside band\" otherwise. \"within A/A\" means the kernel's median ratio is no further from 1 than the A/A band reaches, so this run cannot tell the arms apart.\n");
    let _ = writeln!(md, "\"allocs/call\" is the number of heap allocations one call makes on the fallback arm and on the kernel arm (counted by the process's global allocator). A row marked *status quo* is not routed: it is the unfused chain the row above it replaces, timed on the fallback arms only, and it is read against that row, not as a kernel ratio.\n");
    let _ = writeln!(md, "| workload | iters/phase | fallback | fallback (A/A) | kernel | A/A band | kernel band (lo, med, hi) | allocs/call (fallback, kernel) | verdict |");
    let _ = writeln!(md, "|---|---:|---:|---:|---:|---:|---:|---:|---|");
    let mut jsonl = String::new();
    let mut csv = String::from("workload,phase,arm,ns_per_call\n");
    let mut bits_failures = 0;
    for r in &rows {
        let (aa_lo, aa_med, aa_hi) = ratio_band(&r.a1, &r.a2);
        let aa_reach = (aa_lo - 1.0).abs().max((aa_hi - 1.0).abs());
        let (kb, verdict) = match &r.b {
            Some(b) => {
                let (lo, med, hi) = ratio_band(&r.a1, b);
                let same_bits = r.digests.2 == Some(r.digests.0);
                if !same_bits {
                    bits_failures += 1;
                }
                let mut v = if !same_bits {
                    "BITS DIFFER".to_string()
                } else if hi < 1.0 {
                    format!("kernel faster, {:.2}x", 1.0 / med)
                } else if lo > 1.0 {
                    format!("kernel slower, {:.2}x", med)
                } else {
                    "inside band".to_string()
                };
                if same_bits && (med - 1.0).abs() <= aa_reach {
                    v.push_str(" (within A/A)");
                }
                (format!("{lo:.3}, {med:.3}, {hi:.3}"), v)
            }
            None if !r.routed => ("-".to_string(), "status quo, not routed".to_string()),
            None => ("-".to_string(), "kernel arm not compiled in".to_string()),
        };
        let allocs = match r.allocs.1 {
            Some(b) => format!("{}, {}", r.allocs.0, b),
            None => format!("{}, -", r.allocs.0),
        };
        let _ = writeln!(
            md,
            "| {} | {} | {} | {} | {} | {:.3}, {:.3}, {:.3} | {} | {} | {} |",
            r.name,
            r.iters,
            fmt_band(&r.a1, r.elems),
            fmt_band(&r.a2, r.elems),
            r.b.as_ref().map(|b| fmt_band(b, r.elems)).unwrap_or_else(|| "-".into()),
            aa_lo,
            aa_med,
            aa_hi,
            kb,
            allocs,
            verdict
        );
        let b_json = match &r.b {
            Some(b) => format!("{{\"min\":{},\"med\":{},\"max\":{}}}", b.min, b.med, b.max),
            None => "null".into(),
        };
        let _ = writeln!(
            jsonl,
            "{{\"workload\":\"{}\",\"routed\":{},\"elems\":{},\"iters\":{},\"a1\":{{\"min\":{},\"med\":{},\"max\":{}}},\"a2\":{{\"min\":{},\"med\":{},\"max\":{}}},\"b\":{},\"allocs_a1\":{},\"allocs_b\":{},\"digest_a1\":\"{:016x}\",\"digest_a2\":\"{:016x}\",\"digest_b\":{},\"verdict\":\"{}\"}}",
            r.name, r.routed, r.elems, r.iters, r.a1.min, r.a1.med, r.a1.max, r.a2.min, r.a2.med, r.a2.max, b_json,
            r.allocs.0, r.allocs.1.map(|a| a.to_string()).unwrap_or_else(|| "null".into()), r.digests.0, r.digests.1,
            r.digests.2.map(|d| format!("\"{d:016x}\"")).unwrap_or_else(|| "null".into()), verdict
        );
        for (p, arm, ns) in &r.phases {
            let _ = writeln!(csv, "{},{},{},{}", r.name, p, arm.label(), ns);
        }
    }
    let _ = writeln!(md, "\n## Reading it\n");
    let _ = writeln!(md, "- Within-machine ratios only; never compare absolute numbers across machines or across runs on a machine whose load was not gated.");
    let _ = writeln!(md, "- A verdict inside the A/A band is not a result either way.");
    let _ = writeln!(md, "- The `adam_step` row compares the dispatch entry, where `1 - beta^t` is hoisted on both arms; `ml::adam_step`'s fallback recomputes `powf` per element, a CJC-side cost this row does not charge to either side.");
    let _ = writeln!(md, "- Digests (FNV-1a over the outputs' bits) are in `rows.jsonl`; per-phase timings in `phases.csv`.");
    std::fs::create_dir_all(&o.out).expect("out dir");
    std::fs::write(o.out.join("REPORT.md"), md).expect("REPORT.md");
    std::fs::write(o.out.join("rows.jsonl"), jsonl).expect("rows.jsonl");
    std::fs::write(o.out.join("phases.csv"), csv).expect("phases.csv");
    let mut prov_txt = String::new();
    for (k, v) in &prov {
        let _ = writeln!(prov_txt, "{k}: {v}");
    }
    std::fs::write(o.out.join("provenance.txt"), prov_txt).expect("provenance.txt");
    eprintln!("wrote {}", o.out.display());
    if bits_failures > 0 {
        eprintln!("{bits_failures} row(s) produced different bits on the kernel arm: NOT a record");
        std::process::exit(1);
    }
}
