# Chess RL Demo — Change-Control Audit & CJC Impact Map

## Governing Rule

> **Do not change CJC (Rust crates, CJC source strings in `cjc_source.rs`) without telling the user first. All fixes go in the HTML/JS demo layer unless impossible.**

## Audit Date

2026-03-12

## Source of Issues

Multi-role review (6 roles: UX Designer, Chess Engine Engineer, RL Research Engineer, Determinism Auditor, Software Architect, QA Engineer) graded the demo B+. ~30 issues identified.

---

## Issue Classification

### Legend

| Column | Meaning |
|--------|---------|
| **ID** | Unique identifier |
| **Layer** | `JS` = HTML/JS only, `CJC` = requires CJC change, `RUST` = Rust test infra only |
| **Risk** | `LOW` / `MED` / `HIGH` — likelihood of regression |
| **Phase** | Which execution phase addresses it |

---

### Engine Correctness

| ID | Issue | Layer | Risk | Phase | Rollback |
|----|-------|-------|------|-------|----------|
| E-1 | `legalMoves()` uses `applyMove()` (auto-queen) instead of `applyMoveWithPromo()` — promotion-sensitive legality filtering broken | JS | MED | 3 | Revert `legalMoves()` to old body |
| E-2 | No threefold repetition detection | JS | LOW | 4 | Remove position hash tracking |
| E-3 | No 50-move rule detection | JS | LOW | 4 | Remove halfmove counter |
| E-4 | No insufficient material detection (K vs K, K+B vs K, K+N vs K) | JS | LOW | 4 | Remove material check |
| E-5 | `evaluateMove()` calls `applyMove()` (auto-queen) instead of `applyMoveWithPromo()` for check detection | JS | LOW | 2 | Revert evaluateMove check logic |
| E-6 | Draw offer/acceptance is instant, no tension | JS | LOW | 4 | Remove draw UI |

### RL / Agent Strength

| ID | Issue | Layer | Risk | Phase | Rollback |
|----|-------|-------|------|-------|----------|
| R-1 | No baseline subtraction in REINFORCE — high variance | JS | LOW | 5 | Remove baseline |
| R-2 | Fixed learning rate (0.01) — no schedule | JS | LOW | 5 | Revert to fixed LR |
| R-3 | Trained weights not persisted to localStorage — lost on refresh | JS | LOW | 5 | Remove localStorage write |
| R-4 | No 1-ply minimax/tactical lookahead — agent misses obvious captures | JS | MED | 5 | Remove lookahead, revert to pure heuristic |
| R-5 | Temperature fixed at 0.5 — no annealing | JS | LOW | 5 | Revert to fixed temperature |

### Replay / Determinism

| ID | Issue | Layer | Risk | Phase | Rollback |
|----|-------|-------|------|-------|----------|
| D-1 | "Replay Stable" badge shows even after training modifies weights | JS | LOW | 6 | Hide badge logic |
| D-2 | Replay contract undefined — no doc on what "replay" means | JS | LOW | 6 | Remove badge |
| D-3 | RNG state not cleanly separated between games | JS | MED | 6 | Revert RNG init |

### Architecture / Code Quality

| ID | Issue | Layer | Risk | Phase | Rollback |
|----|-------|-------|------|-------|----------|
| A-1 | All logic in single 2157-line HTML file — no separation of concerns | JS | MED | 7 | N/A (structural) |
| A-2 | Heavy use of global `game` object — mutation everywhere | JS | MED | 7 | N/A (structural) |
| A-3 | No error boundaries — exceptions silently swallowed | JS | LOW | 7 | Remove try/catch additions |
| A-4 | localStorage unbounded — no size limits, no eviction | JS | LOW | 7 | Remove size cap |

### UX / Visual Polish

| ID | Issue | Layer | Risk | Phase | Rollback |
|----|-------|-------|------|-------|----------|
| U-1 | Board squares 48px — too small for comfortable play | JS | LOW | 8 | Revert CSS sizes |
| U-2 | Info panels cramped — need better spacing | JS | LOW | 8 | Revert CSS |
| U-3 | No drag-and-drop for moves | JS | MED | 9 | Remove drag handlers |
| U-4 | No move sounds | JS | LOW | 9 | Remove audio code |
| U-5 | No undo/takeback | JS | MED | 9 | Remove undo logic |
| U-6 | No move timer | JS | LOW | 9 | Remove timer code |
| U-7 | Promotion modal functional but could be more polished | JS | LOW | 8 | Revert modal CSS |

### Opening Explorer / Profile

| ID | Issue | Layer | Risk | Phase | Rollback |
|----|-------|-------|------|-------|----------|
| O-1 | Opening Explorer is flat table — no expandable tree | JS | LOW | 10 | Revert to flat table |
| O-2 | Profile tab requires 2 games — should show stats after 1 | JS | LOW | 10 | Revert threshold |

### Adaptive Learning / External Data

| ID | Issue | Layer | Risk | Phase | Rollback |
|----|-------|-------|------|-------|----------|
| X-1 | Adaptive style learning declared but not verified end-to-end | JS | LOW | 11 | N/A (verification only) |
| X-2 | PGN import pipeline (Rust) verified but bridge to JS opening priors not built | RUST | LOW | 11 | N/A (future extension) |

---

## CJC Impact Assessment

### Files That MUST NOT Change (without explicit user notification)

| File | Reason |
|------|--------|
| `tests/chess_rl_project/cjc_source.rs` | Authoritative CJC chess engine source (761 LOC) |
| `crates/cjc-*/src/**` | All CJC compiler crates |
| `tests/chess_rl_project/*.rs` | CJC-level chess RL tests (49 tests) |
| `tests/chess_rl_hardening/*.rs` | CJC-level hardening tests (170 tests) |
| `tests/chess_rl_playability/pgn_parser.rs` | Rust PGN parser |
| `tests/chess_rl_playability/test_pgn_import.rs` | PGN import tests |

### Files That WILL Change (JS/HTML layer only)

| File | Changes |
|------|---------|
| `examples/chess_rl_platform.html` | ALL phases 2-12 modifications |

### Files That WILL Be Created (documentation only)

| File | Phase |
|------|-------|
| `docs/chess_rl_change_control.md` | 1 (this file) |
| `docs/chess_rl_engine_correctness_audit.md` | 2 (update existing) |
| `docs/chess_rl_replay_contract.md` | 6 |
| `docs/chess_rl_competence_plan.md` | 5 (update existing) |
| `docs/chess_rl_final_audit.md` | 13 (update existing) |
| `docs/chess_rl_demo_readiness.md` | 12 (update existing) |
| `docs/chess_rl_regression_report_final.md` | 13 (update existing) |
| `docs/portfolio/chess_rl_playable_platform_summary.md` | 12 (update existing) |

---

## Verdict

**All ~30 issues can be resolved in the JS/HTML layer.** Zero CJC changes required.

- Promotion legality (E-1): JS-side — `legalMoves()` needs to call `applyMoveWithPromo()` for pawn-to-8th-rank moves
- Draw rules (E-2, E-3, E-4): JS-side — position hashing, halfmove counter, material counting
- Agent strength (R-1 through R-5): JS-side — all RL logic is in the HTML file
- Replay/determinism (D-1 through D-3): JS-side — badge logic and RNG management
- Architecture (A-1 through A-4): JS-side — code organization within the HTML file
- UX (U-1 through U-7): JS/CSS — visual and interaction changes
- Explorer/Profile (O-1, O-2): JS-side — rendering logic
- External data (X-2): Future extension — no changes needed now

**CJC is untouched. Rust test infrastructure is untouched. No regressions to existing 433 chess RL tests.**

---

## Execution Order

| Phase | Description | Issues Addressed |
|-------|-------------|-----------------|
| 1 | Change-control audit (this document) | — |
| 2 | Engine correctness audit | E-5 |
| 3 | Promotion correctness & UX | E-1 |
| 4 | Draw rules | E-2, E-3, E-4, E-6 |
| 5 | RL stabilization | R-1, R-2, R-3, R-4, R-5 |
| 6 | Replay determinism | D-1, D-2, D-3 |
| 7 | JS/HTML architecture | A-1, A-2, A-3, A-4 |
| 8 | UX polish | U-1, U-2, U-7 |
| 9 | Drag/drop, sounds, undo, timer | U-3, U-4, U-5, U-6 |
| 10 | Explorer & Profile | O-1, O-2 |
| 11 | Adaptive learning & data verification | X-1, X-2 |
| 12 | Portfolio demo readiness | — |
| 13 | Full regression & documentation | — |
