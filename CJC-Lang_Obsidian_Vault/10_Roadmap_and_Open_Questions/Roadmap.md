---
title: Roadmap
tags: [roadmap, planning]
status: Grounded in docs/spec/stage3_roadmap.md and CLAUDE.md
---

# Roadmap

Synthesized from `docs/spec/stage3_roadmap.md`, `docs/CJC_Optimization_and_Roadmap.md`, the Stage 2.x progress notes, and the Prime Directives in `CLAUDE.md`.

Items are grouped by **priority** (P0 highest). Status labels: ✅ Done · 🔄 In progress · 📋 Planned · ❓ Implied but not scheduled.

## Stage 3 — Infrastructure & Language Completeness

### P0 — Infrastructure hardening

| ID | Item | Status | Crates |
|---|---|---|---|
| S3-P0-01..04 | ADR materialization, error code registry, test coverage matrix | ✅ Done | `docs/` |
| S3-P0-05 | Split `cjc-runtime/src/lib.rs` into submodules (≤80 LOC re-exports) | 🔄 | `cjc-runtime` |
| S3-P0-06 | Proptest infrastructure (`tests/prop_tests/`) | 🔄 | root |
| S3-P0-07 | End-to-end fixture runner (`tests/fixtures/`) | 🔄 | root |

### P1 — Performance and compiler depth

| ID | Item | Status | Crates |
|---|---|---|---|
| S3-P1-01 | Vec COW for `Value::Array` / `Value::Tuple` (ADR-0009) | 🔄 | runtime + both executors |
| S3-P1-02 | Optional rayon-parallel matmul (`cjc-runtime/parallel` feature, [[Binned Accumulator]]) | 🔄 | `cjc-runtime` |
| S3-P1-03 | Extended numeric types: i8..u128, f16, Complex | 🔄 | `cjc-types`, `cjc-runtime` |
| S3-P1-04 | Structural collection types: `Set<T>`, `Queue<T>` | 🔄 | `cjc-runtime`, `cjc-types` |
| S3-P1-05 | `Option` / `Result` / `Range` / `Slice` as first-class Type variants (runtime still `Value::Enum`) | 🔄 | `cjc-types` |
| S3-P1-06 | SSA: phi nodes + use-def chains + dominator tree (ADR-0012) — see [[SSA Form]], [[Dominator Tree]] | 🔄 | `cjc-mir` |
| S3-P1-07 | [[TCO Extension]] — conditional branches + mutual recursion | 🔄 | `cjc-mir-exec` |
| S3-P1-08 | Shape inference pipeline (E0500, E0501, E0502) | 🔄 | `cjc-types` |

### P2 — Domain capability (ML types)

| ID | Item | Status | Crates |
|---|---|---|---|
| S3-P2-01 | `DType` enum + `Value::DTypeVal` | 🔄 | `cjc-runtime` |
| S3-P2-02 | `QuantizedTensor` (INT8/INT4) with `dequantize()` | 🔄 | `cjc-runtime`, `cjc-mir-exec` |
| S3-P2-03 | `MaskTensor` (bit-packed attention masks) | 🔄 | `cjc-runtime`, `cjc-mir-exec` |
| S3-P2-04 | `SparseTensor` method dispatch (`matvec`, `to_dense`, `nnz`) | 🔄 | `cjc-runtime`, `cjc-mir-exec` |

### P3 — Deferred infrastructure

| ID | Item | Status |
|---|---|---|
| S3-P3-01 | LLVM / Cranelift native backend (subset) — new crate `cjc-codegen` | 📋 |
| S3-P3-02 | Language Server Protocol (LSP) — new crate `cjc-lsp` | 📋 |
| S3-P3-03 | Scope stack `SmallVec` optimization | 📋 |
| S3-P3-04 | W0010 lint (recursive call not in tail position) | 📋 |

## Language features from CLAUDE.md

These are the **feature implementation scope** items called out in the project's top-level `CLAUDE.md`. They are planned but not all scheduled into a Stage 3 task ID:

1. **[[If as Expression]]** — currently `if` is a statement; lift it to `ExprKind::IfExpr` with type-unified branches. Touches AST, type checker, HIR, MIR, and both executors. ([[cjc-eval]] + [[cjc-mir-exec]])
2. **MIR integration for autodiff** — gradients must flow through MIR operations deterministically. **Phase 3c (2026-04-26): partial — language-level `grad_graph_*` primitives now expose the GradGraph arena to user `.cjcl` source**, with byte-equal AST↔MIR parity across the full PINN training loop (see [[ADR-0016 Language-Level GradGraph Primitives]] and [[PINN in Pure CJC-Lang]]). Open: native higher-order AD (`grad_graph_grad_of`) — Phase 3d will replace the FD residual fallback. See [[Autodiff]].
3. **Default parameters**: `fn solve(x: f64, tol: f64 = 1e-6)`. Requires parser, signature, call-site lowering, and MIR default argument insertion.
4. **Variadic functions**: `fn sum(...values: f64)`. Lowers to a deterministic array; no dynamic allocation surprises.
5. **Numerical solver stubs**: `ode_step()`, `pde_step()`, `symbolic_derivative()`. Not full solvers — hooks for [[Bastion]].
6. **Sparse eigensolvers**: Lanczos, Arnoldi. Deterministic iteration order required. See [[Sparse Linear Algebra]].
7. **Multi-file module system**: `mod math;` / `import stats.linear`. See [[Module System]] — infrastructure partially exists but is not wired as the default path.
8. **Decorators**: `@log`, `@timed` as a language feature. Parser → AST → HIR → MIR → runtime wrapper execution.

## Green Compute (Runtime Policy)

The "make CJC-Lang greener — more energy and thermally efficient" initiative.
Framing: *efficient trustworthy computation*, not brute-force throughput. The
niche is more efficient per **trustworthy** result (scientific, medical,
regulated, reproducible, edge), not hyperscale TFLOPs. Design:
[[ADR-0025 Runtime Policy Layer]]; concept: [[Runtime Policy Layer]];
performance-recovery research: [[Green Compute Performance Recovery]].

| ID | Item | Status | Crates |
|---|---|---|---|
| GC-01 | `RuntimePolicy` struct + `cool`/`balanced`/`max-perf` thermal profiles | ✅ Done | `cjc-runtime` |
| GC-02 | rayon thread cap (`--threads`/`--profile`) — set-once at CLI startup, laptop-safe `balanced` default | ✅ Done | `cjc-runtime`, `cjc-cli` |
| GC-03 | Deterministic energy estimate (`energy_estimate`, workload-counts not wall time) + 15 policy/energy builtins | ✅ Done | `cjc-runtime` |
| GC-09 | **Race-to-idle adaptive cap** — burst full / throttle when sustained; recovers burst perf, keeps sustained bound; `--no-adaptive` for fixed schedule ([[ADR-0026 Race-to-Idle Adaptive Scheduling]]) | ✅ Done | `cjc-runtime`, `cjc-cli` |
| GC-04 | Wire `numeric_mode` into live reduction dispatch (today: recorded preference only) | 📋 Planned | `cjc-runtime` |
| GC-05 | Instrumented FLOP/byte counter for whole-program energy accounting | 📋 Planned | `cjc-runtime` |
| GC-06a | **Fused elementwise kernels** — `fused_axpy`/`fused_mul_sub`/`fused_sub_sq` single-pass, eliminate intermediate allocs (~40% less traffic), bit-identical to unfused; compounds with race-to-idle ([[ADR-0027 Fused Elementwise Kernels]]) | ✅ Done | `cjc-runtime` |
| GC-06b | **Adaptive layout abstraction + deterministic hash-table** — memory + scaling headroom for the next dataset size; unblocks **Phase 1.0** of [[TidyView Architecture]] / [[ADR-0023 ABNG Adaptive Belief Radix Graph (Phase 0.1)|ABNG]] at real scale (memory traffic is energy) | 📋 Planned | `cjc-data`, `cjc-abng` |
| GC-07 | Hot/cold path split (fast numerical path vs. cold audit/SVG/Merkle work) via `audit_mode` hook | ❓ Implied | `cjc-runtime` |
| GC-08 | Energy/perf telemetry: `joules_per_epoch`, `AUC_per_joule`, `trust_adjusted_auc_per_joule` (user code / [[Bastion]] on top of `energy_estimate`) | ❓ Implied | `cjc-data`, libs |

## Stage 4 preview

Listed in `stage3_roadmap.md` as post-Stage-3 targets:

1. LLVM / Cranelift native backend.
2. LSP server with VS Code extension.
3. Concurrent execution model (`Channel<T>`, `Arc<T>`, `Mutex<T>`).
4. File I/O (`File`, `read_file`, `write_file`).
5. JSON support.

## Prohibited changes (anti-patterns)

From `stage3_roadmap.md` §Anti-Patterns — these must **not** be changed while implementing the roadmap:

1. Do not change public function signatures in `cjc-mir-exec/src/lib.rs`.
2. Do not add `rayon` to workspace dependencies — optional feature in `cjc-runtime` only.
3. Do not change `KahanAccumulatorF64` in the serial matmul path.
4. Do not break `Value::Enum` representation — `Option`/`Result` remain `Value::Enum { variant: "Some"/"None"/"Ok"/"Err" }` at runtime even when they become first-class Type variants.
5. Do not rename existing test files.
6. Do not remove `#[cfg(test)]` blocks during module splits — move them to the target file.

## Regression gates (must pass before any task is marked Done)

```bash
cargo test --workspace
cargo test milestone_2_4 -- parity      # AST-eval == MIR-exec
cargo test milestone_2_4 -- nogc        # no-alloc proof paths
cargo test milestone_2_4 -- optimizer
cargo test fixtures                      # after S3-P0-07
cargo test prop_tests                    # after S3-P0-06
cargo test --workspace --features cjc-runtime/parallel  # after S3-P1-02
```

Baseline test count at Stage 2.4: **535**. The count must only increase — any decrease means a test was deleted, not refactored.

## Related

- [[Roadmap Dependency Graph]]
- [[Open Questions]]
- [[Documentation Gaps]]
- [[Current State of CJC-Lang]]
- [[Version History]]
