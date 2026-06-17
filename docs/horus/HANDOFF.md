# Horus — Handoff for the Next Session

Purpose: pick up `horus`, the CJC-Lang binding layer for the **Polytrace**
profiler. Everything to start cold is here. (For the profiler engine itself, see
`docs/seshat/HANDOFF_NEXT.md` + `crates/polytrace/README.md`.)

---

## 0. Status snapshot (read first)

`horus` is the thin glue crate that exposes the standalone `polytrace` profiler to
the CJC-Lang runtime. The split is deliberate:

- **`polytrace`** — the standalone, **publishable**, dependency-free profiler
  engine (trace model, 13 analyses, `merge`, serialize/replay, renderers, the
  `collect-live` collectors). Versioned independently (**0.1.0**) for crates.io.
  Carries **no `cjc-runtime` dependency**.
- **`horus`** — the CJC-Lang binding: the write-only `seshat_*` `.cjcl` builtins
  over polytrace's trace model, routed from `cjc-eval` + `cjc-mir-exec`. Depends
  on `polytrace` + `cjc-runtime`. `publish = false`. Follows the workspace version
  (0.1.11). Re-exports all of `polytrace` (`pub use polytrace::*`) so it's a
  drop-in for code that used the old combined `cjc-seshat` crate.

**Current state: green.** `horus` 6 unit + 1 bolero fuzz target pass; executor
parity `seshat_parity` 4/4; `cjc-eval`/`cjc-mir-exec` compile clean. All merged to
**`master`** (the rename + split is commit `2eedc2c`).

---

## 1. Where things live

| Thing | Path |
|---|---|
| Horus crate | `crates/horus/{Cargo.toml,src/lib.rs,src/dispatch.rs,tests/fuzz_builtins.rs}` |
| The `.cjcl` builtins | `crates/horus/src/dispatch.rs` (`dispatch_seshat`, `SESHAT_BUILTINS`) |
| Executor wiring | `cjc-eval/src/lib.rs:2820`, `cjc-mir-exec/src/lib.rs:241` (`CallDispatch::Seshat`), `:2354`, `:2862` |
| Executor manifests | `cjc-eval/Cargo.toml:26`, `cjc-mir-exec/Cargo.toml:30` (`horus.workspace = true`) |
| Executor↔executor parity | `tests/seshat/parity.rs` (root `[[test]] name = "seshat_parity"`, Cargo.toml:195) |
| Profiler engine (re-exported) | `crates/polytrace/` (its own README + tests) |
| Workspace membership | `Cargo.toml:64-65` (members), `:139` (`horus` dep), `:191-192` (path+version) |

---

## 2. Build & test (worktree; reuse the parent target dir)

```bash
CARGO_TARGET_DIR=<parent>/target cargo test  -p horus            # 6 unit + 1 fuzz
CARGO_TARGET_DIR=<parent>/target cargo test  --test seshat_parity # AST-eval ↔ MIR-exec parity
CARGO_TARGET_DIR=<parent>/target cargo build -p cjc-eval -p cjc-mir-exec  # wiring compiles
CARGO_TARGET_DIR=<parent>/target cargo test  -p polytrace        # the engine (sanity)
```

---

## 3. Architecture invariants — DO NOT break

1. **The split stays clean.** Capture/analysis logic belongs in `polytrace`;
   `horus` is *only* the CJC-Lang glue. Never add a `cjc-runtime` dependency to
   `polytrace` (it must stay publishable + dependency-free).
2. **Satellite dispatch, no cycle.** `dispatch_seshat` is routed from both
   executors *after* `cjc_runtime::dispatch_builtin` and the other satellites
   (`dispatch_quantum`/`dispatch_grad_graph`/`dispatch_abng`/`dispatch_locke`) —
   the established pattern that keeps `cjc-runtime → horus` from being a cycle.
3. **Write-only + deterministic.** Every builtin records into the per-thread sink
   and returns a deterministic value (sequential zone handle, event count, or 0).
   **None returns trace-derived analysis into program state** — that's what keeps
   them from perturbing control flow, and what guarantees AST-eval ↔ MIR-exec
   parity. Zone handles come from the builder's monotonic counter, so both
   executors observe identical handles.
4. **`Value` enum / MIR untouched.** Builtins marshal through `cjc_runtime::Value`
   at the boundary only; no new `Value` variant.
5. **Boundary hardening.** Arity/type/domain/negative-byte checks return `Err`,
   never panic (bolero-fuzzed in `tests/fuzz_builtins.rs`).

---

## 4. The `.cjcl` builtin surface (8, all write-only)

| Builtin | Args → returns | Effect |
|---|---|---|
| `seshat_reset()` | → 0 | reset the per-thread sink |
| `seshat_zone_start(name)` | → handle (seq, from 1) | open a pipeline-stage zone |
| `seshat_zone_stop(handle)` | → 0 | close a zone |
| `seshat_mark_boundary(name)` | → 0 | record a Py↔Rust/FFI crossing |
| `seshat_mark_copy(from, to, bytes)` | → 0 | cross-domain copy (flags avoidable) |
| `seshat_alloc_tag(domain, bytes)` | → 0 | tag an allocation's ownership domain |
| `seshat_event_count()` | → n | events recorded so far |
| `seshat_dump_trace(path)` | → num_events | flush the sink to a `.seshat` file |

Domains accepted by `from/to/domain`: `pyheap rustheap mmap numpy arrow tensor gpu
nativeext` (case-insensitive; `polytrace::OwnershipDomain::from_str`).

---

## 5. The work to pick up (prioritized)

### A. Naming alignment — the #1 obvious inconsistency
The crate is `horus`, the engine is `polytrace`, but the builtins are still
`seshat_*` and the parity test/dir are `seshat_parity` / `tests/seshat/`. Decide a
consistent scheme and rename:
- **Option 1 (recommended):** builtins → `horus_*` (matches the binding). 
- **Option 2:** builtins → `polytrace_*` (matches the engine the user calls into).
- Whichever: also rename `SESHAT_BUILTINS`, `dispatch_seshat`, the executor
  routing comments/`CallDispatch::Seshat` variant, `tests/seshat/` →
  `tests/horus/`, and the `[[test]]` entry in `Cargo.toml`.
- This is new surface (no external `.cjcl` users yet), so back-compat aliases are
  optional — but trivial to add (keep `seshat_*` forwarding to the new names).
- **Touch points to update together:** `crates/horus/src/dispatch.rs`,
  `cjc-eval/src/lib.rs` (1 call site), `cjc-mir-exec/src/lib.rs` (3 sites + the
  `CallDispatch` variant), `tests/seshat/parity.rs`, root `Cargo.toml` (`[[test]]`),
  `crates/polytrace/README.md` (the builtin table).

### B. Whole-stack name unification (decide once)
Three names span the stack: `polytrace` (crate), `horus` (binding),
`python-seshat` (recorder, PyPI pkg `seshat-profiler`). Decide whether to unify
(e.g. `python-polytrace`, builtins `polytrace_*`) or keep `horus`/`seshat` as
deliberate sub-brands. Pairs with (A).

### C. Horus has no README
`polytrace` has one; `horus` doesn't. Add `crates/horus/README.md` (what it is, the
builtin table, the write-only rule, a `.cjcl` example).

### D. Optional surface expansion (guard the write-only rule)
`horus` re-exports all of `polytrace`, so you *could* expose more to `.cjcl`. Safe
additions stay write-only (e.g. a `seshat_counter(freq_mhz)` thermal marker, or a
native-frame marker). **Anything that returns a *report/analysis* into program
state breaks invariant #3 and parity** — don't, or gate it behind a clearly
non-deterministic, explicitly-labelled path. Decide before coding.

### E. Publish coordination
`polytrace` is the publish target (0.1.0, see `docs/seshat/HANDOFF_NEXT.md` /
the publish steps). `horus` is `publish = false` and uses a path dep on
`polytrace`. If you ever publish `horus`, `polytrace` must be on crates.io first
and the `version = "0.1.0"` path dep must match the published version.

---

## 6. Gotchas

- **Thread-local sink + cargo's test thread pool.** `SINK` is `thread_local!`;
  cargo runs tests on a pool, so a sink can outlive one test. Always call
  `seshat_reset()` / `dispatch::reset()` at the top of a recording or test (the
  parity test does this via `horus::dispatch::reset`).
- **Parity is the gate.** Any new builtin must produce byte-identical sink state in
  `cjc-eval` and `cjc-mir-exec` — add a case to `tests/seshat/parity.rs`.
- **Determinism comes from handles.** Don't introduce wall-clock, RNG, or
  filesystem-dependent return values; `seshat_dump_trace` returns `num_events`
  (deterministic), not a byte/path count, on purpose.

---

## 7. Pointers
- Profiler engine + its remaining gaps: `docs/seshat/HANDOFF_NEXT.md`,
  `docs/seshat/SESHAT_DESIGN.md`, `crates/polytrace/README.md`.
- The split commit: `git show 2eedc2c` (rename `cjc-seshat` → `polytrace` + carve
  out `horus`). Everything is on `master`.
