# CJC Chess RL — Playable Adaptive Research Platform

## Portfolio Summary

**One-liner:** Built a fully interactive, human-playable chess RL research platform with post-game REINFORCE training, castling/EP, PGN import parser, policy explainability, player profiling, and 433 engine tests — all in pure CJC + HTML/JS with zero external dependencies.

### Resume Bullet

> Designed and implemented a human-playable chess RL platform featuring a JS chess engine mirroring a custom language's MIR-executed engine, SplitMix64 deterministic RNG, 2-layer MLP policy agent with post-game REINFORCE gradient updates, positional evaluation heuristics, castling and en passant support, a Rust PGN parser with SAN-to-coordinate resolution, real-time "Why this move?" explainability, promotion choice UI, demo presets, deterministic replay badge, statistical player profiling, and comprehensive engine testing (433 tests, 0 failures).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Interactive Platform (HTML/JS)                                  │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌──────────────┐  │
│  │  Board    │  │  Policy  │  │  Opening  │  │  Style       │  │
│  │  (click   │  │  Panel   │  │  Explorer │  │  Profile     │  │
│  │  to move) │  │ (top-5)  │  │ (freq)    │  │  (stats)     │  │
│  └──────┬───┘  └──────────┘  └───────────┘  └──────────────┘  │
│         │                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  JS Chess Engine (mirrors CJC engine exactly)            │   │
│  │    initBoard, legalMoves, applyMove, terminalStatus      │   │
│  │    encodeBoard, inCheck, isAttackedBy, findKing          │   │
│  └──────────────────────────────────────────────────────────┘   │
│         │                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  JS Agent + SplitMix64 RNG                               │   │
│  │    2-layer MLP [66→16→1] | Softmax | Categorical Sample  │   │
│  └──────────────────────────────────────────────────────────┘   │
│         │                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Trace Recorder → JSONL (localStorage + download)        │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
          │
          │ (Rust side: validation + regression)
          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Engine Hardening Tests (37 tests, 6 suites)                    │
│    test_human_move.rs      — 11 tests (legality, check, promo) │
│    test_replay_trace.rs    — 6 tests (determinism, encoding)   │
│    test_player_profile.rs  — 4 tests (metrics, trajectories)   │
│    test_pgn_import.rs      — 5 tests (normalization, shape)    │
│    test_opening_tree.rs    — 5 tests (stability, order)        │
│    test_adaptation.rs      — 6 tests (scoring, probabilities)  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Technical Achievements

| Achievement | Detail |
|-------------|--------|
| JS engine parity | Exact mirror of CJC chess engine (same piece encoding, move generation order, board format) |
| Deterministic RNG | SplitMix64 ported to JS via BigInt for exact u64 parity with Rust implementation |
| "Why this move?" card | Factual policy explainability with top-5 candidates, confidence, material delta |
| Demo presets bar | One-click start for common play/replay/debug scenarios |
| Replay badge | Persistent seed/status display with "Replay Exact" button |
| Promotion choice | Full underpromotion UI (queen/rook/bishop/knight) for human players |
| Capture heuristic | Agent uses positional evaluation + piece-value bonus for aggressive play |
| Post-game REINFORCE | Agent improves after every game via gradient updates on MLP weights |
| Castling & en passant | JS engine supports full castling + EP (CJC backend simplified) |
| PGN import parser | Rust-side SAN resolver with board parity verification against CJC engine |
| Player profiling | Statistical profiles from stored traces (capture rate, piece preferences, W/D/L) |
| Trace compatibility | Human game JSONL traces use same format as self-play and training traces |
| Zero dependencies | Pure HTML/JS platform, no external libraries or frameworks |
| 433+ tests | Capture audit, promotion audit, movegen, PGN parser, property tests |

## Test Results

| Metric | Count |
|--------|-------|
| Capture audit tests | 12 |
| Promotion audit tests | 9 |
| PGN parser tests | 19 |
| Playability tests | 37 |
| Advanced tests | 135 |
| Hardening tests | 170 |
| Total chess RL tests | 433 |
| Failures | 0 |
| Regressions | 0 |

## Files Created

| File | Purpose |
|------|---------|
| `examples/chess_rl_platform.html` | Interactive platform (play, review, explore, profile) |
| `tests/test_chess_rl_playability.rs` | Test entry point |
| `tests/chess_rl_playability/mod.rs` | Module structure |
| `tests/chess_rl_playability/helpers.rs` | Shared test helpers |
| `tests/chess_rl_playability/pgn_parser.rs` | Rust PGN parser with SAN resolution (490+ LOC) |
| `tests/chess_rl_playability/test_human_move.rs` | Human move validation (11 tests) |
| `tests/chess_rl_playability/test_replay_trace.rs` | Trace replay determinism (6 tests) |
| `tests/chess_rl_playability/test_player_profile.rs` | Profile generation (4 tests) |
| `tests/chess_rl_playability/test_pgn_import.rs` | PGN import + parser integration (19 tests) |
| `tests/chess_rl_playability/test_opening_tree.rs` | Opening tree construction (5 tests) |
| `tests/chess_rl_playability/test_adaptation.rs` | Style adaptation (6 tests) |
| `data/external_games/sample_games.pgn` | Sample PGN file (4 games, 1 with castling) |
| `docs/chess_rl_playability_architecture.md` | Architecture document |
| `docs/chess_rl_external_data_ingestion.md` | PGN ingestion documentation |
| `docs/chess_rl_playability_regression_report.md` | Regression report |
| `docs/portfolio/chess_rl_playable_platform_summary.md` | This file |

## Follow-On Improvements

1. **Adaptive curriculum** — Agent difficulty adjustment based on player profile
2. **WASM bridge** — Run actual CJC engine in browser via WebAssembly for guaranteed parity
3. **Multi-agent tournament** — Play against snapshots from different training stages
4. **Position analysis** — FEN import for arbitrary position exploration
5. **Supervised warm-start** — Use imported PGN traces for initial weight training
