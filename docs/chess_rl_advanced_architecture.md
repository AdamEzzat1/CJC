# Chess RL Advanced Research Platform — Architecture Analysis

## Phase 1 Deliverable

### 1. Current System Architecture

```
┌─────────────────────────────────────────────────────────┐
│  CJC Source Constants (tests/chess_rl_project/cjc_source.rs)  │
│                                                         │
│  CHESS_ENV (345 LOC)   RL_AGENT (154 LOC)   TRAINING (165 LOC) │
│    init_board()          init_weights()       play_episode()    │
│    apply_move()          forward_move()       train_episode()   │
│    legal_moves()         select_action()      eval_vs_random()  │
│    terminal_status()     reinforce_update()                     │
│    encode_board()                                               │
└──────────────────────┬──────────────────────────────────┘
                       │ Compiled by
                       ▼
┌─────────────────────────────────────────────────────────┐
│  CJC Pipeline: Lexer → Parser → AST → HIR → MIR → Exec │
│                                                         │
│  cjc-mir-exec (4,072 LOC)                               │
│    MirExecutor.exec() → deterministic execution         │
│    Builtins: categorical_sample, log, exp, softmax      │
│    RNG: SplitMix64 (cjc-repro)                          │
│    Tensors: matmul, relu, transpose, softmax            │
└──────────────────────┬──────────────────────────────────┘
                       │ Output
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Test Harness (Rust)                                     │
│    chess_rl_project/ — 49 tests (7 suites)              │
│    chess_rl_hardening/ — 88 tests + 12 property + 5 fuzz │
│    Total: 149 chess RL tests                             │
└─────────────────────────────────────────────────────────┘
```

### 2. Integration Points for New Capabilities

#### 2.1 Multi-Episode Training (Phase 2)
- **Location**: New CJC source constant `MULTI_TRAINING` in `cjc_source.rs`
- **Integration**: Wraps existing `train_episode()` in a loop, collects per-episode metrics
- **Output**: JSONL trace of `{episode, reward, loss, num_moves}` per episode
- **Risk**: LOW — additive, no existing code modified

#### 2.2 Win-Rate Metrics (Phase 3)
- **Location**: New CJC function `eval_win_rate()` in `MULTI_TRAINING`
- **Integration**: Calls existing `play_episode_random()` N times, aggregates win/draw/loss
- **Output**: Win rate percentage per evaluation checkpoint
- **Risk**: LOW — additive

#### 2.3 Castling + En Passant (Phase 4)
- **Location**: Modify `CHESS_ENV` constant
- **Integration**:
  - Board state needs castling rights (4 bools) and en passant square
  - Encode as extra array elements appended to board (indices 64-68)
  - `generate_pseudo_legal()` gains castling/EP move generation
  - `apply_move()` gains castling/EP execution + rights update
  - `encode_board()` gains castling/EP feature encoding
- **Risk**: MEDIUM — modifies core chess logic; existing tests must still pass
- **Mitigation**: Keep backward-compatible board format; extra fields are additive

#### 2.4 Autodiff Gradients (Phase 5)
- **Location**: New CJC source using `grad_graph()` builtins OR manual dual-number approach
- **Integration**:
  - Current: Manual REINFORCE in `reinforce_update()` (60 lines of chain-rule code)
  - Option A: Expose `GradGraph` operations as CJC builtins (heavy wiring)
  - Option B: Keep manual gradients but use CJC's `derivative()` builtin for verification
  - Option C: Implement AD-aware training in Rust test harness, not CJC
- **Recommendation**: Option B (safest) — verify manual gradients against AD, don't replace
- **Risk**: HIGH if replacing manual gradients; LOW if verifying only

#### 2.5 MIR Trace Hooks (Phase 6)
- **Location**: `crates/cjc-mir-exec/src/lib.rs` — `MirExecutor`
- **Integration**:
  - Add `trace_hooks: Option<TraceConfig>` to MirExecutor
  - Instrument `eval_expr()` to emit trace events for Call, MatMul, Branch
  - Output: JSONL stream of `{event, fn_name, args_summary, result_summary, timestamp_ns}`
- **Risk**: LOW — opt-in instrumentation, no behavior change when disabled
- **Existing hooks**: `@trace` decorator already logs entry/exit; extend this pattern

#### 2.6 Self-Play Architecture (Phase 7)
- **Location**: New CJC source constant `SELFPLAY` in `cjc_source.rs`
- **Integration**: Two separate weight sets, alternating as white/black
- **Determinism**: Both players share same RNG stream (seeded); move order is deterministic
- **Risk**: LOW — additive CJC code + test harness

#### 2.7 Model Snapshot System (Phase 8)
- **Location**: Rust test harness using `cjc-snap`
- **Integration**:
  - After training, extract weight tensors from executor output
  - `snap()` the weights → `SnapBlob` with SHA-256 content hash
  - Save to `models/` directory as `.snap` files
  - Load via `restore()` → inject into program as initial weights
- **Risk**: LOW — cjc-snap already handles Tensor serialization

#### 2.8 League Manager (Phase 9)
- **Location**: Rust test harness
- **Integration**: Round-robin or Swiss pairings between saved model snapshots
- **Determinism**: Each match uses a fixed seed derived from (model_a_hash, model_b_hash, round)
- **Risk**: LOW — pure Rust orchestration layer

#### 2.9 ELO Rating System (Phase 10)
- **Location**: Rust test harness + JSON output
- **Integration**: Standard ELO formula (K=32), initialized at 1500
- **Output**: `trace/elo_ratings.json`
- **Risk**: LOW — pure computation on match results

#### 2.10 Deterministic Replay (Phase 11)
- **Location**: Rust test harness
- **Integration**:
  - Record: seed + model snapshot hash → full game trajectory
  - Replay: load snapshot, re-execute with same seed, verify bit-identical
  - Uses existing `cjc-snap` for model loading and `cjc-repro` for RNG
- **Risk**: LOW — leverages existing determinism guarantees

#### 2.11 Dashboard Upgrade (Phase 12)
- **Location**: `examples/chess_rl_dashboard.html`
- **Integration**: Add tabs for training curves, win rates, ELO progression, replay viewer
- **Risk**: LOW — HTML/JS only, no compiler changes

### 3. Dependency Graph

```
Phase 2 (Multi-Episode) ──→ Phase 3 (Win-Rate)
      │                          │
      ▼                          ▼
Phase 8 (Snapshots) ──→ Phase 7 (Self-Play)
      │                     │
      ▼                     ▼
Phase 9 (League) ──→ Phase 10 (ELO)
      │
      ▼
Phase 11 (Replay)

Phase 4 (Castling/EP) ──→ independent, integrate after Phase 2
Phase 5 (AD Verify)   ──→ independent, after Phase 2
Phase 6 (MIR Traces)  ──→ independent, after Phase 2
Phase 12 (Dashboard)  ──→ after all data-producing phases
Phase 13 (Regression) ──→ after all code changes
Phase 14 (Docs)       ──→ after everything
```

### 4. Files to Create/Modify

| Phase | New Files | Modified Files |
|-------|-----------|----------------|
| 2 | tests/chess_rl_advanced/multi_training.rs | tests/chess_rl_project/cjc_source.rs |
| 3 | tests/chess_rl_advanced/win_rate.rs | tests/chess_rl_project/cjc_source.rs |
| 4 | tests/chess_rl_advanced/castling_ep.rs | tests/chess_rl_project/cjc_source.rs |
| 5 | tests/chess_rl_advanced/ad_verify.rs | (none) |
| 6 | tests/chess_rl_advanced/mir_trace.rs | crates/cjc-mir-exec/src/lib.rs |
| 7 | tests/chess_rl_advanced/selfplay.rs | tests/chess_rl_project/cjc_source.rs |
| 8 | tests/chess_rl_advanced/snapshots.rs | (none) |
| 9 | tests/chess_rl_advanced/league.rs | (none) |
| 10 | tests/chess_rl_advanced/elo.rs | (none) |
| 11 | tests/chess_rl_advanced/replay.rs | (none) |
| 12 | examples/chess_rl_research_dashboard.html | (none) |
| 13 | tests/test_chess_rl_advanced.rs | (none) |
| 14 | docs/chess_rl_advanced_regression_report.md, docs/portfolio/chess_rl_advanced_portfolio_summary.md | (none) |

### 5. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Castling breaks existing board tests | HIGH | Keep board[0..63] unchanged; add rights as board[64..68] |
| AD integration introduces non-determinism | MEDIUM | Verify-only mode (Option B); don't replace manual gradients |
| MIR trace hooks slow down execution | LOW | Opt-in via TraceConfig; disabled by default |
| Snapshot format changes break replay | LOW | Version the snap format; validate hash on restore |
| Self-play RNG stream interference | MEDIUM | Fork RNG per player: rng.fork() for white, rng.fork() for black |

### 6. Test Count Projection

| Category | Current | New | Total |
|----------|---------|-----|-------|
| Chess RL project | 49 | 0 | 49 |
| Chess RL hardening | 88+12+5 | 0 | 105 |
| Chess RL advanced | 0 | ~80-120 | ~80-120 |
| **Chess RL total** | **154** | **~80-120** | **~234-274** |
