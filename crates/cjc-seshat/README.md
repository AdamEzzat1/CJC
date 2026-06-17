# cjc-seshat

**Seshat** — a deterministic cross-language causal profiler for Python/Rust
systems. It explains, in one trace, where a mixed Python/Rust program spends
CPU, memory, copies, async stalls, GIL contention, Rust lock contention, and
deterministic performance variance — and *why*, causally, rather than as a flat
list of self-times.

> Named for the Egyptian goddess of writing, measurement, and record-keeping.
> Slots into CJC-Lang's mythological-naming convention (`cjc-locke`,
> `cjc-dharht`, `cjc-abng`).

## Why

Modern Python projects use Rust for speed (PyO3/maturin), but profiling the
*seam* between them is still awkward: Rust tools (`perf`, `samply`, `pprof`) and
Python tools (Scalene, Memray, py-spy) each see only one side. Seshat's unique
capability is to show **one merged timeline** where Python frames, Rust frames,
native frames, async tasks, and FFI boundary crossings appear together — see
`seshat merge` (feature 13), which stitches a Python `.seshat` and a Rust
`.seshat` into a single unified trace.

## The 13 features

| # | Feature | Entry point |
|---|---------|-------------|
| 1 | Cross-language flamegraph | `analyze::flamegraph` |
| 2 | Python↔Rust boundary cost | `analyze::boundary` |
| 3 | Copy detector (no false positives) | `analyze::copy` |
| 4 | GIL + Rust lock contention | `analyze::contention` |
| 5 | Async-aware stalls (Tokio/asyncio) | `analyze::async_stall` |
| 6 | Memory ownership map | `analyze::ownership` |
| 7 | Peak-memory explanation | `analyze::peak` |
| 8 | Recommendations with evidence | `analyze_trace().recommendations` |
| 9 | Determinism / variance profiling | `analyze::variance` |
| 10 | Thermal / power-aware mode | `analyze::thermal` |
| 11 | Data-pipeline profiler | `analyze::pipeline` |
| 12 | "What changed?" regression diff | `diff` |
| 13 | **Python+Rust trace merge** | `merge` |

## Architecture: collection ⟂ analysis

A profiler measures time, and time is nondeterministic; CJC-Lang demands
bit-identical output. Seshat resolves this the way `cjc-cana` does — by splitting
the system in two:

- **Collection** (feature `collect-live`): the nondeterministic, platform-specific
  probes (Rust `GlobalAlloc` shim, CPython frame walk, perf_event counters).
  They only *produce* a `Trace`.
- **Analysis** (the default surface): a **pure, deterministic function of a
  recorded `Trace`**. Same trace → byte-identical report, every run, every
  platform.

The unit of reproducibility is the `.seshat` trace (`serialize` / `replay`).
Ordering/structure use a **logical clock** (event index), reproducible
attribution uses **sample counts**, and wall-clock is a single advisory scalar
excluded from the content hash. See [`docs/seshat/SESHAT_DETERMINISM.md`](../../docs/seshat/SESHAT_DETERMINISM.md).

## Quickstart (Rust)

```rust
use cjc_seshat::{Trace, FrameKind, OwnershipDomain, analyze_trace, render};

let mut b = Trace::builder(/* run_id */ 42);
let main = b.intern_frame(FrameKind::Py, "main", "app.py", 10);
let bnd  = b.intern_frame(FrameKind::FfiBoundary, "pyo3::call", "ffi.rs", 1);
let work = b.intern_frame(FrameKind::Rust, "process_batch", "lib.rs", 88);
for _ in 0..38 { b.sample_running(&[main, bnd, work]); } // 38% crossing the seam
for _ in 0..62 { b.sample_running(&[main, work]); }
b.copy(OwnershipDomain::RustHeap, OwnershipDomain::NumPy, 1 << 20, bnd);
let trace = b.finish();

let report = analyze_trace(&trace);
assert_eq!(report.flamegraph.total_samples, 100);
println!("{}", render::text(&report));        // human report
let _svg  = render::flamegraph_svg(&report);  // dependency-free SVG
let _json = render::json(&report);            // deterministic, content-addressed
```

## `.cjcl`-facing builtins (write-only markers)

A CJC-Lang program can annotate its own execution into a Seshat trace. These are
write-only event emitters (they never feed analysis back into program state),
generalizing the older `profile_zone_*` builtins. They work identically in
`cjc-eval` and `cjc-mir-exec`.

| Builtin | Purpose |
|---|---|
| `seshat_reset()` | reset the per-thread trace sink |
| `seshat_zone_start(name) -> handle` | open a pipeline-stage zone |
| `seshat_zone_stop(handle)` | close a zone |
| `seshat_mark_boundary(name)` | record a Py↔Rust / FFI crossing |
| `seshat_mark_copy(from, to, bytes)` | record a cross-domain copy |
| `seshat_alloc_tag(domain, bytes)` | tag an allocation's ownership domain |
| `seshat_event_count() -> n` | introspection: events recorded so far |
| `seshat_dump_trace(path) -> n` | flush the sink to a `.seshat` file |

```text
seshat_reset();
let z: i64 = seshat_zone_start("compute");
seshat_alloc_tag("rust", 4096);
seshat_mark_copy("rust", "numpy", 4096);   // flagged avoidable (zero-copy compatible)
seshat_zone_stop(z);
seshat_dump_trace("run.seshat");
```

## Recording a real trace (Rust, feature `collect-live`)

Seshat can capture a real `.seshat` from a running Rust process. Install the
allocator shim and bracket your work in zones; heap traffic is captured
automatically and a background thread samples the zone stack.

```rust
#[global_allocator]
static GLOBAL: cjc_seshat::collect::SeshatAlloc = cjc_seshat::collect::SeshatAlloc;

use cjc_seshat::collect::{Recorder, zone, mark_copy};
use cjc_seshat::OwnershipDomain;

let rec = Recorder::start();
{
    let _z = zone("compute");
    // ... real Rust work; every allocation is captured ...
    mark_copy(OwnershipDomain::RustHeap, OwnershipDomain::NumPy, 8 * 400_000);
}
let trace = rec.finish();                 // a real .seshat
std::fs::write("run.seshat", cjc_seshat::serialize(&trace)).unwrap();
```

## CLI

```bash
# Record a real in-process Rust workload (proof of live capture):
cargo run -p cjc-seshat --features collect-live --bin seshat -- record-demo run.seshat

# Attribute Rust allocations to real functions (no manual zones) via native
# unwinding — adds a backtrace per allocation, so it is opt-in:
cargo run -p cjc-seshat --features collect-live --bin seshat -- record-demo run.seshat --unwind

# Analyze any .seshat (pure — no feature needed):
cargo run -p cjc-seshat --bin seshat -- analyze run.seshat            # text report
cargo run -p cjc-seshat --bin seshat -- analyze run.seshat --json     # deterministic JSON
cargo run -p cjc-seshat --bin seshat -- analyze run.seshat --svg fg.svg
cargo run -p cjc-seshat --bin seshat -- diff base.seshat cand.seshat  # regression diff

# Merge a Python trace and a Rust trace into one unified cross-language trace
# (Rust nests under the matching Py↔Rust boundary frame):
cargo run -p cjc-seshat --bin seshat -- merge py.seshat rust.seshat --under myext --out merged.seshat
cargo run -p cjc-seshat --bin seshat -- analyze merged.seshat        # Python + Rust in one report
```

## Status

- **Shipped (deterministic core):** trace model (`.seshat`), all 12 analyses,
  JSON/text/SVG renderers, the `seshat_*` builtin surface wired into both
  executors, the `seshat` CLI (`analyze`/`diff`), and the full deterministic test
  suite (unit + proptest + bolero + determinism + executor parity).
- **Shipped (Rust live capture, feature `collect-live`):** a real Rust in-process
  collector — `GlobalAlloc` shim capturing actual heap traffic, RAII `zone`
  scopes, a zone-stack wall-clock sampler, `Recorder` session, and
  `seshat record-demo`. With `CaptureConfig { alloc_stacks: true }` (CLI
  `--unwind`) **allocations are attributed to real Rust functions via native
  unwinding** (the `backtrace` crate, scoped strictly to this feature — the
  default build stays dependency-free). `collect::native_sample()` takes a
  synchronous CPU sample of the **calling thread's real native call stack** (no
  manual zones), and `collect::mark_host(name)` declares the host boundary for
  merge correlation. Honest scope: *allocation-site* unwinding and *explicit*
  native sampling are automatic; *automatic* cross-thread CPU sampling (SIGPROF /
  thread-suspend) is deferred.
- **Shipped (trace merge):** `seshat merge <host> <native> [native2 ...]` (pure,
  deterministic) stitches a Python and one-or-more Rust traces into one unified
  trace, grafting each Rust subtree under the matching Py↔Rust boundary —
  correlated by an explicit token (`mark_host` / `--under`) or, failing that, the
  most-sampled boundary.
- **Shipped (thermal capture):** the Python recorder's `seshat[thermal]` extra
  (psutil) samples `cpu_freq()` → Counter events → the analyzer's thermal mode
  (throttle detection). Optional extra; the core stays zero-dep.
- **Shipped (cross-language capture, [`python-seshat/`](../../python-seshat)):**
  a pure-stdlib Python recorder using `sys.setprofile` to capture **real Python
  frames + the Python↔native (Rust/C) boundary**, writing the same `.seshat`
  format the Rust CLI analyzes. No PyO3/maturin — the file is the interface.
  Proven by `tests/python_bridge.rs` against a committed Python-produced fixture.
- **Deferred:** *automatic* cross-thread CPU-time native sampling (SIGPROF /
  `SuspendThread`+`StackWalk` — large, unsafe, platform-specific; explicit
  `native_sample()` is shipped), cache-miss / IPC hardware counters (psutil gives
  frequency only), exact (non-heuristic) GIL detection (needs a C extension that
  would defeat the pure-stdlib recorder), and *automatic* token injection across
  the seam (explicit `mark_host` tokens are shipped).

## Tests

```bash
cargo test -p cjc-seshat                                # core: unit + integration + proptest + bolero
cargo test -p cjc-seshat --features collect-live --test collect   # live-capture well-formedness
cargo test --test seshat_parity                         # AST-eval ↔ MIR-exec builtin parity
```
