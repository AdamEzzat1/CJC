---
title: Version History
tags: [foundation, history]
status: Summarized from CHANGELOG and commit log
---

# Version History

Grounded in the git log, `CHANGELOG.md`, and the progress docs in `docs/`.

## v0.1.4 — Rebrand (2026-04-06)
- Project renamed from **CJC** to **CJC-Lang** (Computational Jacobian Core).
- CLI command: `cjc` → `cjcl`.
- File extension: `.cjc` → `.cjcl`.
- Install command: `cargo install cjc-lang`.
- Internal crate names remain `cjc-*` for continuity.

## v0.1.3
- Fixed `cargo install cjc` binary entry point.
- Commit `3061026`: "Enable `cargo install cjc` with binary entry point."

## v0.1.2
- Data science foundation (docs/CJC_DataScience_Readiness_Audit.md).
- ML infrastructure expansion.
- 30+ CLI commands (see [[CLI Surfaces]]).
- Comprehensive test suites added.
- Commit `435940d`: "Hardening, ML infrastructure, and comprehensive test suites."

## v0.1.0 — First public
- Lexer, parser, eval, types, dispatch, runtime, AD, data DSL.
- Tree-walk interpreter only.
- See `docs/RELEASE_NOTES_v0.1.0.md`.

## Stage 2 milestones (between v0.1.0 and v0.1.2)

These are described in `docs/CJC_STAGE2_*.md` and progress files:

- **Stage 2.0** — HIR + MIR + MIR-exec infrastructure bootstrapped.
- **Stage 2.1** — Closures with [[Capture Analysis]].
- **Stage 2.2** — Match expressions + structural destructuring (tuples, structs).
- **Stage 2.3** — For loops (range iteration, desugared to while).
- **Stage 2.4** — [[NoGC Verifier]] + [[MIR Optimizer]] (CF + DCE) + [[Parity Gates]] (G-8, G-10).

## Hardening phases

The `docs/` directory contains a long tail of hardening reports and phase changelogs:

- `BETA_HARDENING_PLAN.md` / `BETA_HARDENING_CHANGELOG.md`
- `PERFORMANCE_OPTIMIZATION_CHANGELOG.md`
- `PERFORMANCE_V2_CHANGELOG.md`
- `TIDYVIEW_HARDENING_CHANGELOG.md`
- `STAGE_2_6_HARDENING.md`
- `phase_b_changelog.md`, `phase_c_changelog.md`

These are **historical** but useful as context. They are not part of the current state model.

## Recent non-version commits

- `cb7c76c` — v0.1.4: Rebrand CJC to CJC-Lang
- `e171b18` — Add .gitignore entries for pack artifacts and proptest regressions
- `9c1270c` — Add comprehensive docstrings across all 20 library crates

## Related

- [[Current State of CJC-Lang]]
- [[Roadmap]]
