---
title: Open Questions
tags: [roadmap, open-questions]
status: Partially resolved 2026-04-09
---

# Open Questions

Uncertainties encountered while building this vault. Each is phrased so a future reader (or CJC-Lang developer) can either confirm, refute, or schedule it.

Items below are ordered: **still open** first, then **resolved** at the bottom for archive.

---

## Still open

### Test count (partially answered)

- `README.md` claims **3,700+ workspace tests**.
- `CLAUDE.md` memory note says **5,320 tests** as of 2026-03-21.
- Raw `#[test]` marker count via grep is ~6,700.

**Status:** Needs an authoritative `cargo test --workspace 2>&1 | grep 'test result'` run in a clean worktree. The discrepancy between "test markers" and "tests actually run" is the feature-flag gap — some tests only compile with `--features cjc-runtime/parallel` etc.

**Recommendation:** Add a `docs/progress/test_count_YYYY-MM-DD.txt` artifact on every release and reference the most recent one from the README instead of a hardcoded number.

### Performance numbers

`docs/spec/CJC_PERFORMANCE_MANIFESTO.md` reports:

- RNN 10K steps in ~2.5s (~3,995 steps/sec)
- Transformer ~562 tokens/sec
- Binary footprint ~1.8 MB

**Status:** The manifesto now carries a `HISTORICAL DOCUMENT` header (added 2026-04-09) noting the numbers are from an unstamped pre-v0.1.4 build. A fresh benchmark against HEAD is needed before any of these can be re-published.

### AD and MIR integration

`CLAUDE.md` lists "MIR integration for autodiff" as feature #2 to implement. `cjc-ad` exists and is exercised by [[cjc-eval]] — but how much of AD currently survives the HIR → MIR lowering path?

**Status:** Phase 3c (2026-04-26) closed the user-visible part: 24 `grad_graph_*` builtins now route from both executors via `cjc_ad::dispatch_grad_graph` (satellite dispatch, mirroring `cjc-quantum`). Both backends produce byte-identical output across the full PINN training loop on the flagship demo (`examples/physics_ml/pinn_heat_1d_pure.cjcl`). What's still open: native higher-order AD (`grad_graph_grad_of`) — Phase 3c shipped a finite-difference fallback at ε=1e-3 and explicitly deferred analytical Hessians to Phase 3d. See [[ADR-0016 Language-Level GradGraph Primitives]].

### Parity gate coverage matrix

Beyond G-8 and G-10, is there a matrix listing every feature surface (quantum, sparse linalg, vizor, regex, dataframe, …) and whether it has parity coverage in both executors?

**Status:** Still open — a draft matrix is planned but not yet built.

### Quantum correctness tests

Does every quantum submodule (VQE, QAOA, DMRG, QEC, QML) have at least one correctness test against a known reference (Pauli expectation, Bell probabilities, GHZ)?

**Status:** Still open.

### LLVM backend scope

Every roadmap doc mentions LLVM / Cranelift as "future". Which IR level is targeted (AST / HIR / MIR)?

**Status:** Still open. [[ADR Index]] has no codegen ADR yet.

### Zero-dep claim under feature flags

After `cjc-runtime/parallel` ships (per [[ADR-0011 Parallel Matmul]]), is the zero-dep claim qualified to "default build" or does it remain absolute?

**Status:** Still open — needs an explicit README edit when ADR-0011 lands.

---

## Resolved (archive)

### ✓ Module system wiring — resolved 2026-04-09

- `README.md` and older `CLAUDE.md` drafts described it as "incomplete."
- **Actually:** fully wired. 1,183 LOC in `cjc-module`, `cjc-parser/src/lib.rs:872` parses `import`, `cjc-cli/src/lib.rs:680-759` wires `--multi-file`, `cjc-mir-exec/src/lib.rs:4063` has `run_program_with_modules`. Multi-file programs work today.
- **Action:** `README.md` and `CLAUDE.md` should drop the "incomplete" label. See [[Module System]].

### ✓ Builtin count — resolved 2026-04-09

- README said "221+", grep survey found ~334.
- **Actually:** 336 in `cjc-runtime/src/builtins.rs` + 83 in `cjc-quantum/src/dispatch.rs` ≈ **~419 total**. The 66 arms in `cjc-eval` and `cjc-mir-exec` are routing layers, not separate builtins.
- **Action:** [[Builtins Catalog]] should carry the ~419 figure as the authoritative count.

### ✓ `if` expression status — resolved 2026-04-09

- `CLAUDE.md` contradicted itself.
- **Actually:** `if` is already a full expression in both executors. Verified by running `let x: i64 = if 1 < 2 { 10 } else { 20 };` through `cjcl run` and `cjcl run --mir-opt` — both print `10`.
- **Action:** Remove "if AS AN EXPRESSION" from CLAUDE.md's feature implementation scope. See [[If as Expression]].

---

## Related

- [[Roadmap]]
- [[Documentation Gaps]]
- [[Current State of CJC-Lang]]
