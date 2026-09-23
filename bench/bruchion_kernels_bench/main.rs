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
    // matmul, through kernel::matmul_raw (Kahan per output element on both sides). The
    // first two shapes are the original rows; the rest are the second research stack's
    // shape family around the reproduced 64x17x33 win: the same k and n with a longer m,
    // the roles rotated, a small square, a short k, a long k.
    // The third stack's sweep (2026-09-23) follows the original rows: k at (64, _, 33), m
    // at (_, 17, 33), n at (16, 16, _), and four shapes with every dimension below 64 —
    // the only shapes `Tensor::matmul` sends to this loop (it takes the tiled engine at
    // 64 and above). The shape reaches each call through `black_box`, so the linker cannot
    // specialize a row's loop on its dimensions.
    for &(m, k, nn) in &[
        (64usize, 17usize, 33usize), (128, 128, 128), (256, 17, 33), (33, 64, 17), (17, 33, 64), (32, 32, 32), (64, 64, 8), (16, 16, 1024),
        (64, 4, 33), (64, 8, 33), (64, 16, 33), (64, 24, 33), (64, 32, 33), (64, 48, 33), (64, 64, 33), (64, 128, 33),
        (8, 17, 33), (16, 17, 33), (32, 17, 33), (63, 17, 33),
        (16, 16, 16), (16, 16, 64), (16, 16, 256),
        (8, 8, 8), (48, 17, 33), (32, 48, 16), (63, 63, 63),
    ] {
        let a = g.f64s(m * k);
        let b = g.f64s(k * nn);
        let c = std::rc::Rc::new(std::cell::RefCell::new(vec![0.0f64; m * nn]));
        let c1 = c.clone();
        ws.push(Workload {
            routed: true,
            name: format!("matmul {m}x{k}x{nn}"),
            elems: m * k * nn,
            call: Box::new(move || {
                let (m, k, nn) = black_box((m, k, nn));
                kernel::matmul_raw(&a, &b, &mut c1.borrow_mut(), m, k, nn)
            }),
            digest: Box::new(move || fnv1a(c.borrow().iter().map(|v| v.to_bits()))),
        });
    }
    // adam_step_raw at a fixed step (t = 7): params, m, v restored before every call.
    // This is the dispatch entry, where `1 - beta^t` is hoisted on both arms; the
    // `ml::adam_step` fallback loop recomputes `powf` per element, which is a CJC-side
    // cost, not the kernel's, and is not what this row compares. Routed again now that
    // the pack's square root is the hardware instruction.
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
    // matmul in TiledMatmul's order: the engine itself on the fallback arms (allocating
    // its result, as it does under Tensor::matmul, then copied into the row's buffer), the
    // tiled kernel into the same buffer on the kernel arm. Same bits (the parity test).
    // The expectation, stated before measuring: parity or a loss against the engine's
    // 4-wide AVX2 axpy micro-kernel, the pack's twin being two lanes at the locked -O2.
    for &(m, k, nn) in &[(128usize, 128usize, 128usize), (256, 256, 256)] {
        let a = g.f64s(m * k);
        let b = g.f64s(k * nn);
        let c = std::rc::Rc::new(std::cell::RefCell::new(vec![0.0f64; m * nn]));
        let c1 = c.clone();
        ws.push(Workload {
            routed: true,
            name: format!("matmul tiled {m}x{k}x{nn}"),
            elems: m * k * nn,
            call: Box::new(move || dispatch::matmul_tiled(&a, &b, &mut c1.borrow_mut(), m, k, nn)),
            digest: Box::new(move || fnv1a(c.borrow().iter().map(|v| v.to_bits()))),
        });
    }
    // The call-path probe. Both arms of the first row run the KERNEL: the fallback arm
    // calls the bare ffi symbol, the kernel arm goes through the routed entry
    // `kernel::relu_raw` (two switch reads, the dispatch call, the slice narrowing), so
    // the ratio is the call path's cost and nothing else. The second row puts the Rust
    // body on the fallback arm against the bare ffi on the kernel arm: the loop gap with
    // no call path in it. Together they say where the recorded `relu` gap lives.
    #[cfg(feature = "bruchion-kernels")]
    {
        use cjc_runtime::bruchion::ffi;
        let xs = x.clone();
        let out = std::rc::Rc::new(std::cell::RefCell::new(vec![0.0f64; n]));
        let o1 = out.clone();
        ws.push(Workload {
            routed: true,
            name: "relu call path: bare ffi vs kernel::relu_raw".into(),
            elems: n,
            call: Box::new(move || {
                let mut o = o1.borrow_mut();
                if dispatch::enabled() {
                    kernel::relu_raw(&xs, &mut o);
                } else {
                    // SAFETY: two live buffers of n doubles.
                    unsafe { ffi::cjc_relu_f64(n as i64, xs.as_ptr(), o.as_mut_ptr()) }
                }
            }),
            digest: Box::new(move || fnv1a(out.borrow().iter().map(|v| v.to_bits()))),
        });
        let xs = x.clone();
        let out = std::rc::Rc::new(std::cell::RefCell::new(vec![0.0f64; n]));
        let o1 = out.clone();
        ws.push(Workload {
            routed: true,
            name: "relu loop: Rust body vs bare ffi".into(),
            elems: n,
            call: Box::new(move || {
                let mut o = o1.borrow_mut();
                if dispatch::enabled() {
                    // SAFETY: two live buffers of n doubles.
                    unsafe { ffi::cjc_relu_f64(n as i64, xs.as_ptr(), o.as_mut_ptr()) }
                } else {
                    dispatch::relu_fallback(&xs, &mut o);
                }
            }),
            digest: Box::new(move || fnv1a(out.borrow().iter().map(|v| v.to_bits()))),
        });
    }
    // mse_loss_grad: the fused loss and gradient of mean((pred - target)^2), one pass into
    // a caller buffer, routed. Beside it the GradGraph chain `sub -> mul -> mean ->
    // backward` on the same two tensors, twice: with `mean` (the fused `MseMean` node,
    // whose backward is one pass through `dispatch::mse_grad_scaled` and so the switch —
    // routed) and with `mean_unfused` (the plain chain, the status quo a CJC program's
    // `mean((pred - target)^2)` with a gradient cost before the node — unrouted). The
    // three rows produce the same bits (the cjc-ad parity test holds them to it); the
    // digests here are asserted equal across phases within a row, not across the rows.
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
        let (pt2, tt2) = (pt.clone(), tt.clone());
        let out = std::rc::Rc::new(std::cell::RefCell::new((0.0f64, vec![0.0f64; n])));
        let o1 = out.clone();
        ws.push(Workload {
            routed: true,
            name: "mse+grad via GradGraph, fused mean".into(),
            elems: n,
            call: Box::new(move || {
                let mut g = cjc_ad::GradGraph::new();
                let p = g.parameter(pt2.clone());
                let t = g.parameter(tt2.clone());
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
        let out = std::rc::Rc::new(std::cell::RefCell::new((0.0f64, vec![0.0f64; n])));
        let o1 = out.clone();
        ws.push(Workload {
            routed: false,
            name: "mse+grad via GradGraph, unfused (status quo)".into(),
            elems: n,
            call: Box::new(move || {
                let mut g = cjc_ad::GradGraph::new();
                let p = g.parameter(pt.clone());
                let t = g.parameter(tt.clone());
                let diff = g.sub(p, t);
                let sq = g.mul(diff, diff);
                let loss = g.mean_unfused(sq);
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
    // ---- the second research stack's rows (2026-09-22, night) ----
    // conv1d through kernel::conv1d_raw: one Kahan dot per output element. Unrouted; the
    // row is here for its allocation count (the form before 2026-09-22 allocated a Vec
    // per output element).
    {
        let (sig_len, ch, ks) = (4096usize, 8usize, 9usize);
        let signal = g.f64s(sig_len);
        let filters = g.f64s(ch * ks);
        let bias = g.f64s(ch);
        let out_len = sig_len - ks + 1;
        let out = std::rc::Rc::new(std::cell::RefCell::new(vec![0.0f64; ch * out_len]));
        let o1 = out.clone();
        ws.push(Workload {
            routed: false,
            name: format!("conv1d {sig_len}x{ch}x{ks} (kernel::conv1d_raw)"),
            elems: ch * out_len * ks,
            call: Box::new(move || kernel::conv1d_raw(&signal, &filters, &bias, &mut o1.borrow_mut(), sig_len, ch, ks)),
            digest: Box::new(move || fnv1a(out.borrow().iter().map(|v| v.to_bits()))),
        });
    }
    // The PINN end to end: piml_heat_1d_train at the pinn_routing_bits shape with 1000
    // collocation points, 9 parameters and 20 epochs. Routed: the physics loss goes
    // through dispatch::heat1d_residual_grad under the switch; the digest is over the
    // final parameters, and the elems figure is nominal (points x params x epochs) so
    // the ns-per-elem column stays comparable across records.
    //
    // Two rows. The first passes its arguments as literals, as record 8 did: under fat
    // LTO the linker inlines the trainer into this one call site and specializes it on
    // degree 8 and 1000 points (the parameter loop unrolls, `powi` becomes multiplies),
    // which no interpreter call can do. The second passes the same arguments through
    // `black_box`: the trainer as a program reaches it, with runtime shapes. The two
    // fallback arms differ by that specialization alone, and it is the second row that
    // decides routing (2026-09-23).
    {
        let out = std::rc::Rc::new(std::cell::RefCell::new(Vec::<f64>::new()));
        let o1 = out.clone();
        ws.push(Workload {
            routed: true,
            name: "piml_heat_1d_train 1000x9, 20 epochs (end to end)".into(),
            elems: 1000 * 9 * 20,
            call: Box::new(move || {
                let r = cjc_ad::pinn::piml_heat_1d_train(8, 16, 1000, 0.05, 20, 1e-3, 10.0, 100.0, 1);
                *o1.borrow_mut() = r.final_params;
            }),
            digest: Box::new(move || fnv1a(out.borrow().iter().map(|v| v.to_bits()))),
        });
        let out = std::rc::Rc::new(std::cell::RefCell::new(Vec::<f64>::new()));
        let o1 = out.clone();
        ws.push(Workload {
            routed: true,
            name: "piml_heat_1d_train 1000x9, 20 epochs (end to end, arguments opaque)".into(),
            elems: 1000 * 9 * 20,
            call: Box::new(move || {
                let (degree, n_data, n_colloc, noise, epochs, lr, pw, bw, seed) =
                    black_box((8usize, 16usize, 1000usize, 0.05f64, 20usize, 1e-3f64, 10.0f64, 100.0f64, 1u64));
                let r = cjc_ad::pinn::piml_heat_1d_train(degree, n_data, n_colloc, noise, epochs, lr, pw, bw, seed);
                *o1.borrow_mut() = r.final_params;
            }),
            digest: Box::new(move || fnv1a(out.borrow().iter().map(|v| v.to_bits()))),
        });
    }
    // Quantum: one Hadamard on qubit 9 of a 20-qubit state (2^20 amplitudes, 16 MiB),
    // through the crate's Gate::apply — a loop over every index with a branch on half of
    // them — and the same update as a strided pair loop written here with the crate's own
    // complex arithmetic in the same pair order: the same bits (asserted when the rows are
    // built). Both unrouted, both restore the state from a base copy before the gate (the
    // copy is in both rows' time); a CJC-side loop-shape comparison, not a kernel.
    {
        use cjc_quantum::gates::Gate;
        use cjc_quantum::statevector::Statevector;
        use cjc_runtime::complex::ComplexF64;
        let nq = 20usize;
        let q = 9usize;
        let mut sv = Statevector::new(nq);
        for i in 0..nq {
            Gate::H(i).apply(&mut sv).expect("H");
            Gate::Ry(i, 0.1 * (i as f64 + 1.0)).apply(&mut sv).expect("Ry");
        }
        let base: Vec<ComplexF64> = sv.amplitudes.clone();
        fn h_strided(amps: &mut [ComplexF64], q: usize) {
            // h_matrix() of cjc_quantum::gates, and apply_single_qubit's pair arithmetic,
            // over the pairs in the same ascending order without the per-index branch.
            let s = ComplexF64::real(std::f64::consts::FRAC_1_SQRT_2);
            let ms = ComplexF64::real(-std::f64::consts::FRAC_1_SQRT_2);
            let u = [[s, s], [s, ms]];
            let bit = 1usize << q;
            let mut lo = 0usize;
            while lo < amps.len() {
                for i in lo..lo + bit {
                    let j = i | bit;
                    let a0 = amps[i];
                    let a1 = amps[j];
                    amps[i] = u[0][0].mul_fixed(a0).add(u[0][1].mul_fixed(a1));
                    amps[j] = u[1][0].mul_fixed(a0).add(u[1][1].mul_fixed(a1));
                }
                lo += 2 * bit;
            }
        }
        // The two forms agree bit for bit, or the rows are not comparable.
        {
            let mut a = Statevector::new(nq);
            a.amplitudes.copy_from_slice(&base);
            Gate::H(q).apply(&mut a).expect("H");
            let mut b = base.clone();
            h_strided(&mut b, q);
            assert!(a.amplitudes.iter().zip(&b).all(|(x, y)| x.re.to_bits() == y.re.to_bits() && x.im.to_bits() == y.im.to_bits()), "the strided Hadamard is not Gate::apply's bits");
        }
        let work = std::rc::Rc::new(std::cell::RefCell::new(sv));
        let (w1, w2) = (work.clone(), work.clone());
        let (base1, base2) = (base.clone(), base);
        ws.push(Workload {
            routed: false,
            name: format!("q_h on {nq} qubits (Gate::apply, a branch per index)"),
            elems: 1 << nq,
            call: Box::new(move || {
                let mut s = w1.borrow_mut();
                s.amplitudes.copy_from_slice(&base1);
                Gate::H(q).apply(&mut s).expect("H");
            }),
            digest: Box::new(move || fnv1a(w2.borrow().amplitudes.iter().flat_map(|c| [c.re.to_bits(), c.im.to_bits()]))),
        });
        let work = std::rc::Rc::new(std::cell::RefCell::new(Statevector::new(nq)));
        let (w1, w2) = (work.clone(), work.clone());
        ws.push(Workload {
            routed: false,
            name: format!("q_h on {nq} qubits (strided pair loop, same bits)"),
            elems: 1 << nq,
            call: Box::new(move || {
                let mut s = w1.borrow_mut();
                s.amplitudes.copy_from_slice(&base2);
                h_strided(&mut s.amplitudes, q);
            }),
            digest: Box::new(move || fnv1a(w2.borrow().amplitudes.iter().flat_map(|c| [c.re.to_bits(), c.im.to_bits()]))),
        });
    }
    // ---- the third research stack's rows (2026-09-23) ----
    //
    // Tensor::matmul end to end, at shapes below the tile threshold: the language's own
    // entry (two `to_vec` copies, the result tensor) over `kernel::matmul_raw`, which it
    // now calls (before, it ran its own copy of that loop and no routing reached it).
    // Routed; the digest is over the result's bits.
    for &(m, k, nn) in &[(8usize, 8usize, 8usize), (32, 32, 32), (48, 17, 33), (63, 63, 63)] {
        let at = cjc_runtime::tensor::Tensor::from_vec(g.f64s(m * k), &[m, k]).expect("tensor");
        let bt = cjc_runtime::tensor::Tensor::from_vec(g.f64s(k * nn), &[k, nn]).expect("tensor");
        let out = std::rc::Rc::new(std::cell::RefCell::new(None::<cjc_runtime::tensor::Tensor>));
        let o1 = out.clone();
        ws.push(Workload {
            routed: true,
            name: format!("Tensor::matmul {m}x{k}x{nn} (end to end)"),
            elems: m * k * nn,
            call: Box::new(move || {
                let (at, bt) = black_box((&at, &bt));
                *o1.borrow_mut() = Some(at.matmul(bt).expect("matmul"));
            }),
            digest: Box::new(move || fnv1a(out.borrow().as_ref().expect("a result").to_vec().iter().map(|v| v.to_bits()))),
        });
    }
    // The per-output-element binned accumulator, replaced by one scratch per call
    // (`BinnedScratchF64`): conv2d, linear and sum_axis, each beside the form before
    // (stated here, inline: `BinnedAccumulatorF64::new()` per output, which zeroes 40 KiB
    // and walks 2048 bins for every output element). Both rows of a pair are unrouted
    // CJC-side forms; a pair is read against itself. The other sites that changed the
    // same way (mean_axis, var_axis, einsum) share the mechanism and are not timed.
    {
        use cjc_runtime::accumulator::BinnedAccumulatorF64;
        use cjc_runtime::tensor::Tensor;
        // conv2d: 2 images, 3 channels, 28x28, 8 filters of 3x3, stride 1 — 10,816 outputs
        // of 27 products each.
        let (n, c_in, h_in, w_in, c_out, kh, kw, stride) = (2usize, 3usize, 28usize, 28usize, 8usize, 3usize, 3usize, 1usize);
        let (h_out, w_out) = ((h_in - kh) / stride + 1, (w_in - kw) / stride + 1);
        let input = g.f64s(n * c_in * h_in * w_in);
        let filters = g.f64s(c_out * c_in * kh * kw);
        let bias = g.f64s(c_out);
        #[allow(clippy::too_many_arguments)]
        fn conv2d_before(input: &[f64], filters: &[f64], bias: &[f64], out: &mut [f64],
                         n: usize, c_in: usize, h_in: usize, w_in: usize, c_out: usize, kh: usize, kw: usize, stride: usize) {
            let h_out = (h_in - kh) / stride + 1;
            let w_out = (w_in - kw) / stride + 1;
            for bn in 0..n {
                for co in 0..c_out {
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let mut acc = BinnedAccumulatorF64::new();
                            for ci in 0..c_in {
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        let ih = oh * stride + ki;
                                        let iw = ow * stride + kj;
                                        acc.add(input[((bn * c_in + ci) * h_in + ih) * w_in + iw] * filters[((co * c_in + ci) * kh + ki) * kw + kj]);
                                    }
                                }
                            }
                            out[((bn * c_out + co) * h_out + oh) * w_out + ow] = acc.finalize() + bias[co];
                        }
                    }
                }
            }
        }
        let out = std::rc::Rc::new(std::cell::RefCell::new(vec![0.0f64; n * c_out * h_out * w_out]));
        let elems = n * c_out * h_out * w_out * c_in * kh * kw;
        {
            let (i1, f1, b1, o1) = (input.clone(), filters.clone(), bias.clone(), out.clone());
            let o2 = out.clone();
            ws.push(Workload {
                routed: false,
                name: format!("conv2d {n}x{c_in}x{h_in}x{w_in} * {c_out}x{kh}x{kw} (kernel::conv2d_raw, one scratch per call)"),
                elems,
                call: Box::new(move || {
                    let (n, c_in, h_in, w_in, c_out, kh, kw, stride) = black_box((n, c_in, h_in, w_in, c_out, kh, kw, stride));
                    kernel::conv2d_raw(&i1, &f1, &b1, &mut o1.borrow_mut(), n, c_in, h_in, w_in, c_out, kh, kw, stride)
                }),
                digest: Box::new(move || fnv1a(o2.borrow().iter().map(|v| v.to_bits()))),
            });
        }
        {
            let (i1, f1, b1, o1) = (input.clone(), filters.clone(), bias.clone(), out.clone());
            let o2 = out.clone();
            ws.push(Workload {
                routed: false,
                name: format!("conv2d {n}x{c_in}x{h_in}x{w_in} * {c_out}x{kh}x{kw} (an accumulator per output, the form before)"),
                elems,
                call: Box::new(move || {
                    let (n, c_in, h_in, w_in, c_out, kh, kw, stride) = black_box((n, c_in, h_in, w_in, c_out, kh, kw, stride));
                    conv2d_before(&i1, &f1, &b1, &mut o1.borrow_mut(), n, c_in, h_in, w_in, c_out, kh, kw, stride)
                }),
                digest: Box::new(move || fnv1a(o2.borrow().iter().map(|v| v.to_bits()))),
            });
        }
        // linear: [64, 128] x [32, 128]^T + [32] — 2,048 outputs of 128 products, through
        // Tensor::linear and through the form before on the same tensors (its `to_vec`
        // copies and result tensor included, as Tensor::linear has them).
        let (outer, inf, outf) = (64usize, 128usize, 32usize);
        let xt = Tensor::from_vec(g.f64s(outer * inf), &[outer, inf]).expect("tensor");
        let wt = Tensor::from_vec(g.f64s(outf * inf), &[outf, inf]).expect("tensor");
        let bt = Tensor::from_vec(g.f64s(outf), &[outf]).expect("tensor");
        fn linear_before(x: &Tensor, w: &Tensor, b: &Tensor) -> Tensor {
            let (data, w, b) = (x.to_vec(), w.to_vec(), b.to_vec());
            let out_features = b.len();
            let in_features = x.shape()[x.shape().len() - 1];
            let outer = data.len() / in_features;
            let mut result = vec![0.0f64; outer * out_features];
            for row in 0..outer {
                for j in 0..out_features {
                    let mut acc = BinnedAccumulatorF64::new();
                    for p in 0..in_features {
                        acc.add(data[row * in_features + p] * w[j * in_features + p]);
                    }
                    result[row * out_features + j] = acc.finalize() + b[j];
                }
            }
            Tensor::from_vec(result, &[outer, out_features]).expect("tensor")
        }
        let out = std::rc::Rc::new(std::cell::RefCell::new(None::<Tensor>));
        {
            let (x1, w1, b1, o1) = (xt.clone(), wt.clone(), bt.clone(), out.clone());
            let o2 = out.clone();
            ws.push(Workload {
                routed: false,
                name: format!("Tensor::linear [{outer}, {inf}] x [{outf}, {inf}] (one scratch per call)"),
                elems: outer * inf * outf,
                call: Box::new(move || {
                    let (x, w, b) = black_box((&x1, &w1, &b1));
                    *o1.borrow_mut() = Some(x.linear(w, b).expect("linear"));
                }),
                digest: Box::new(move || fnv1a(o2.borrow().as_ref().expect("a result").to_vec().iter().map(|v| v.to_bits()))),
            });
        }
        {
            let (x1, w1, b1, o1) = (xt, wt, bt, out.clone());
            let o2 = out.clone();
            ws.push(Workload {
                routed: false,
                name: format!("Tensor::linear [{outer}, {inf}] x [{outf}, {inf}] (an accumulator per output, the form before)"),
                elems: outer * inf * outf,
                call: Box::new(move || {
                    let (x, w, b) = black_box((&x1, &w1, &b1));
                    *o1.borrow_mut() = Some(linear_before(x, w, b));
                }),
                digest: Box::new(move || fnv1a(o2.borrow().as_ref().expect("a result").to_vec().iter().map(|v| v.to_bits()))),
            });
        }
        // sum_axis: [256, 64] along axis 1 — 256 outputs of 64 values.
        let (rows, cols) = (256usize, 64usize);
        let st = Tensor::from_vec(g.f64s(rows * cols), &[rows, cols]).expect("tensor");
        fn sum_axis1_before(t: &Tensor) -> Tensor {
            let data = t.to_vec();
            let (rows, cols) = (t.shape()[0], t.shape()[1]);
            let mut result = vec![0.0f64; rows];
            for r in 0..rows {
                let mut acc = BinnedAccumulatorF64::new();
                for c in 0..cols {
                    acc.add(data[r * cols + c]);
                }
                result[r] = acc.finalize();
            }
            Tensor::from_vec(result, &[rows]).expect("tensor")
        }
        let out = std::rc::Rc::new(std::cell::RefCell::new(None::<Tensor>));
        {
            let (s1, o1) = (st.clone(), out.clone());
            let o2 = out.clone();
            ws.push(Workload {
                routed: false,
                name: format!("Tensor::sum_axis [{rows}, {cols}] axis 1 (one scratch per call)"),
                elems: rows * cols,
                call: Box::new(move || {
                    let s = black_box(&s1);
                    *o1.borrow_mut() = Some(s.sum_axis(1).expect("sum_axis"));
                }),
                digest: Box::new(move || fnv1a(o2.borrow().as_ref().expect("a result").to_vec().iter().map(|v| v.to_bits()))),
            });
        }
        {
            let (s1, o1) = (st, out.clone());
            let o2 = out.clone();
            ws.push(Workload {
                routed: false,
                name: format!("Tensor::sum_axis [{rows}, {cols}] axis 1 (an accumulator per output, the form before)"),
                elems: rows * cols,
                call: Box::new(move || {
                    let s = black_box(&s1);
                    *o1.borrow_mut() = Some(sum_axis1_before(s));
                }),
                digest: Box::new(move || fnv1a(o2.borrow().as_ref().expect("a result").to_vec().iter().map(|v| v.to_bits()))),
            });
        }
        // OptimizerState.step as the interpreters run it: `ml::optimizer_step_tensors`
        // (the gradient borrowed) beside the form before (both tensors copied, the code
        // each interpreter carried). SGD without momentum, so the result is the same
        // bits every call and the digest holds across phases; the point of the pair is
        // the allocation column.
        let pt = Tensor::from_vec(g.f64s(n), &[n]).expect("tensor");
        let gt = Tensor::from_vec(g.f64s(n), &[n]).expect("tensor");
        fn step_before(state: &mut dyn std::any::Any, params_t: &Tensor, grads_t: &Tensor) -> Tensor {
            let mut params = params_t.to_vec();
            let grads = grads_t.to_vec();
            if let Some(sgd) = state.downcast_mut::<ml::SgdState>() {
                ml::sgd_step(&mut params, &grads, sgd);
            } else {
                panic!("an SGD state");
            }
            Tensor::from_vec(params, params_t.shape()).expect("tensor")
        }
        let out = std::rc::Rc::new(std::cell::RefCell::new(None::<Tensor>));
        {
            let (p1, g1, o1) = (pt.clone(), gt.clone(), out.clone());
            let o2 = out.clone();
            let mut state: Box<dyn std::any::Any> = Box::new(ml::SgdState::new(n, 1e-3, 0.0));
            ws.push(Workload {
                routed: false,
                name: "OptimizerState.step, SGD (ml::optimizer_step_tensors, the gradient borrowed)".into(),
                elems: n,
                call: Box::new(move || {
                    let (p, gr) = black_box((&p1, &g1));
                    *o1.borrow_mut() = Some(ml::optimizer_step_tensors(state.as_mut(), p, gr).expect("step"));
                }),
                digest: Box::new(move || fnv1a(o2.borrow().as_ref().expect("a result").to_vec().iter().map(|v| v.to_bits()))),
            });
        }
        {
            let (p1, g1, o1) = (pt, gt, out.clone());
            let o2 = out.clone();
            let mut state: Box<dyn std::any::Any> = Box::new(ml::SgdState::new(n, 1e-3, 0.0));
            ws.push(Workload {
                routed: false,
                name: "OptimizerState.step, SGD (both tensors copied, the form before)".into(),
                elems: n,
                call: Box::new(move || {
                    let (p, gr) = black_box((&p1, &g1));
                    *o1.borrow_mut() = Some(step_before(state.as_mut(), p, gr));
                }),
                digest: Box::new(move || fnv1a(o2.borrow().as_ref().expect("a result").to_vec().iter().map(|v| v.to_bits()))),
            });
        }
        // GradGraph zero_grad + backward + clip_grad on a held graph: the slots are
        // written in place now (before: a zeros tensor per parameter per zero_grad and a
        // new tensor per parameter per clip_grad, by construction — that form is not
        // kept, so this row has no partner; its allocation count is the record).
        {
            let mut gg = cjc_ad::GradGraph::new();
            let p = gg.parameter(Tensor::from_vec(g.f64s(n), &[n]).expect("tensor"));
            let t = gg.parameter(Tensor::from_vec(g.f64s(n), &[n]).expect("tensor"));
            let d = gg.sub(p, t);
            let sq = gg.mul(d, d);
            let loss = gg.mean_unfused(sq);
            gg.backward(loss);
            let graph = std::rc::Rc::new(std::cell::RefCell::new(gg));
            let (g1, g2) = (graph.clone(), graph);
            ws.push(Workload {
                routed: false,
                name: "GradGraph zero_grad + backward + clip_grad, 2 parameters (in place)".into(),
                elems: n,
                call: Box::new(move || {
                    let mut gg = g1.borrow_mut();
                    gg.zero_grad();
                    gg.backward(loss);
                    gg.clip_grad(0.5);
                }),
                digest: Box::new(move || fnv1a(g2.borrow().grad(p).expect("a gradient").to_vec().iter().map(|v| v.to_bits()))),
            });
        }
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
    // `sh` trims the whole output, so a first porcelain line that begins with a space
    // (" D path": a tracked file deleted, as the launcher's archive step does to the
    // previous record) has lost its status column; a fixed-offset slice then took the
    // path's first letter off and called the tree dirty (the record at a24189b says so,
    // wrongly). Each line is trimmed on its own and split after its status word.
    let dirty_paths: Vec<String> = git(&["status", "--porcelain"])
        .lines()
        .map(|l| l.trim_start().splitn(2, ' ').nth(1).unwrap_or("").trim_start().replace('\\', "/"))
        .filter(|p| !p.is_empty() && !p.starts_with(out_prefix.trim_end_matches('/')))
        .collect();
    let dirty = !dirty_paths.is_empty();
    let kdir = std::env::var("BRUCHION_KERNELS_DIR").unwrap_or_else(|_| "(unset)".into());
    let unix = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let mut p = vec![
        ("repo".to_string(), git(&["rev-parse", "--show-toplevel"])),
        ("branch".to_string(), git(&["rev-parse", "--abbrev-ref", "HEAD"])),
        ("commit".to_string(), git(&["rev-parse", "HEAD"])),
        ("dirty tree".to_string(), dirty.to_string()),
        ("dirty paths".to_string(), if dirty { dirty_paths.join(", ") } else { "(none outside the record directory)".into() }),
        ("rustc".to_string(), sh("rustc", &["-vV"]).lines().filter(|l| l.starts_with("rustc ") || l.starts_with("host:")).collect::<Vec<_>>().join("; ")),
        ("profile".to_string(), if cfg!(debug_assertions) { "debug (NOT a record: the fallback is not auto-vectorized)".into() } else { "release".into() }),
        ("target".to_string(), format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)),
        ("logical cores".to_string(), runtime_policy::detect_cores().to_string()),
        ("runtime policy".to_string(), runtime_policy::get().summary()),
        ("feature bruchion-kernels".to_string(), feature.to_string()),
        ("routing set, kernel arm".to_string(), dispatch::KernelOps(runtime_policy::get().bruchion_kernel_ops).names()),
        ("routing set, default".to_string(), dispatch::KernelOps::MEASURED.names()),
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
    // The kernel arm measures every kernel, the ones the default routing set leaves out
    // included (their rows are why it leaves them out): the set is widened to all for the
    // whole run, and the arms flip only the switch.
    runtime_policy::set_bruchion_kernel_ops(dispatch::KernelOps::ALL.0);
    if feature && !runtime_policy::get().bruchion_kernels {
        runtime_policy::set_bruchion_kernels(true);
        assert!(dispatch::enabled(), "the switch did not enable the kernels");
        assert!(dispatch::KernelOp::EVERY.iter().all(|&op| dispatch::routes(op)), "the kernel arm must route every operation");
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
            // A `|` in a name would split its table row (the record at 1ce0e15 shows it).
            r.name.replace('|', "\\|"),
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
