---
title: Current State of CJC-Lang
tags: [foundation, status]
status: Snapshot as of 2026-04-09
---

# Current State of CJC-Lang

This is a **reality-mapped** status note. It distinguishes what is actually implemented and tested from what is aspirational or planned. Claims are grounded in the repository state at `C:\Users\adame\CJC` on 2026-04-09.

Version: **v0.1.4** — see [[Version History]].

## Summary table

| Subsystem | Status | Notes |
|---|---|---|
| Lexer ([[cjc-lexer]]) | **Implemented** | All literal forms, format strings, regex literals, byte strings |
| Parser ([[cjc-parser]]) | **Implemented** | Pratt parser with error recovery |
| AST ([[cjc-ast]]) | **Implemented** | Visitor pattern, structural validation |
| Type system ([[cjc-types]]) | **Implemented** | Hindley-Milner unification, effect registry |
| Diagnostics ([[cjc-diag]]) | **Implemented** | Error codes E0001–E8xxx, spans, severity |
| HIR lowering ([[cjc-hir]]) | **Implemented** | [[Capture Analysis]], desugaring |
| MIR lowering ([[cjc-mir]]) | **Implemented** | CFG, SSA, dominators, loop analysis |
| MIR optimizer | **Implemented** | CF, DCE, CSE, SR, LICM, SCCP |
| NoGC verifier | **Implemented** | Call-graph fixpoint |
| Escape analysis | **Implemented** | Stack/Arena/Rc classification |
| cjc-eval (v1 interpreter) | **Implemented** | Tree-walk |
| cjc-mir-exec (v2 register VM) | **Implemented** | Register-machine, trampolined tail calls |
| Parity gates | **Implemented** | Eval ≡ MIR-exec enforced |
| cjc-dispatch | **Implemented** | Multi-dispatch by specificity |
| cjc-runtime builtins | **Implemented** | 363 in `cjc-runtime/src/builtins.rs` + 83 in `cjc-quantum/src/dispatch.rs` = **446 dispatch arms / 441 unique names** (verified 2026-04-09) |
| Tensor runtime | **Implemented** | COW buffers, dtype, SIMD kernels |
| Linear algebra | **Implemented** | matmul, det, solve, lstsq, eigh, svd, QR, Cholesky |
| Autodiff — forward mode | **Implemented** | Dual numbers |
| Autodiff — reverse mode | **Implemented** | Tape-based `ComputeGraph` |
| Autodiff — MIR integration | **Planned** | Not yet wired into [[cjc-mir-exec]] |
| Kahan / Binned accumulators | **Implemented** | `cjc-repro` |
| SplitMix64 RNG | **Implemented** | `cjc-repro` |
| cjc-data DataFrame DSL | **Implemented** | ~73 tidy operations |
| cjc-vizor | **Implemented** | 80 chart types, deterministic SVG/BMP |
| cjc-regex | **Implemented** | Thompson NFA, no backtracking |
| cjc-snap | **Implemented** | Binary format, SHA-256, CSR, DataFrame persistence |
| cjc-quantum | **Implemented** (research-grade) | Statevector, MPS, density, DMRG, VQE, QAOA, QEC, stabilizer |
| cjc-module | **Implemented** | 1,183 LOC, wired via `cjcl run --multi-file` → `run_program_with_modules` (verified 2026-04-09) — see [[Module System]] |
| cjc-cli | **Implemented** | 30+ subcommands |
| cjc-analyzer (LSP) | **Experimental** | Skeleton, not integrated with full type checker |
| Closures | **Implemented** | With capture analysis |
| Pattern matching | **Implemented** | Structural destructuring of tuples and structs |
| For loops | **Implemented** | Desugar to while |
| `if` as expression | **Implemented** | Verified in both executors 2026-04-09 — see [[If as Expression]] |
| Default parameters | **Planned** | Not in parser |
| Variadic functions | **Implemented** | `fn f(...args: f64)` — verified 2026-04-09 via `tests/test_variadic.rs` (11/11 passing) — see [[Variadic Functions]] |
| Decorators | **Planned** | Not in parser |
| Browser / WASM target | **Planned** | Not started |
| LLVM / native backend | **Planned** | Not started |

See [[Roadmap]] for the forward-looking view and [[Open Questions]] for unresolved design questions.

## A note on the module system (resolved 2026-04-09)

`README.md` and older CLAUDE.md drafts described `cjc-module` as "incomplete." Direct inspection shows it is **fully wired**:

- `crates/cjc-module/src/lib.rs` — 1,183 lines: `ModuleId`, `ModuleInfo`, `ModuleGraph`, `build_module_graph`, `merge_programs`, `check_visibility`, `build_import_aliases`, 17+ tests.
- `crates/cjc-parser/src/lib.rs:872` — parses `import` declarations.
- `crates/cjc-ast/` — has `Import(ImportDecl)` variant.
- `crates/cjc-cli/src/lib.rs:680-759` — `--multi-file` flag calls `cjc_module::build_module_graph` then `cjc_mir_exec::run_program_with_modules`.
- `crates/cjc-mir-exec/src/lib.rs:4063` — `pub fn run_program_with_modules(entry_path, seed)` exists and is the integration seam.

The crate is **Implemented**, not "incomplete." The stale label should be removed from `README.md` and `CLAUDE.md`. See [[Module System]].

## Tests

- Test scale (verified 2026-04-09):
  - **6,715 raw `#[test]` markers** across the codebase — 1,913 unit tests in `crates/*/src/` + 4,802 integration tests in `tests/`
  - Plus **22 `proptest!` macros** (each generates many property-test cases at runtime)
  - Plus **38 `bolero::check` fuzz targets**
  - **5,353 tests actually executed** by `cargo test --workspace --release` (118 binaries, 0 failures). The ~1,360 gap is tests behind feature flags (`cjc-runtime/parallel`, quantum features, etc.) that don't compile in the default workspace build.
  - README's "3,700+" is stale; CLAUDE.md's "5,320 as of 2026-03-21" was close to the default-build figure.
- Chess RL benchmark: raw file counts give 319 tests (69 project + 84 advanced + 104 hardening + 62 playability), not the "216" previously claimed. [[Chess RL Demo]] needs updating.
- Parity gates: 50+ tests, all passing.

See [[Test Infrastructure]].

## Known not-yet-working

From the README's own "What Is Not Yet Working" section (verified 2026-04-09):

- ~~Multi-file module system~~ — **resolved**: wired via `cjcl run --multi-file` (see [[Module System]]). README text is stale.
- Default function parameters — confirmed missing
- Variadic functions — confirmed missing
- Decorators — confirmed missing
- MIR-level autodiff integration — confirmed; `cjc-ad` runs via [[cjc-eval]] but not yet [[cjc-mir-exec]]
- Browser compilation target — confirmed missing

## Related

- [[CJC-Lang Overview]]
- [[Version History]]
- [[Roadmap]]
- [[Open Questions]]
- [[Test Infrastructure]]
