# Chess RL Playability Architecture — Phase 1 Analysis

## 1. Human-Play Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Interactive Dashboard (HTML/JS)                                 │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌──────────────┐  │
│  │  Board    │  │  Policy  │  │  Opening  │  │  Style       │  │
│  │  (click   │  │  Panel   │  │  Explorer │  │  Profile     │  │
│  │  to move) │  │          │  │           │  │  Summary     │  │
│  └──────┬───┘  └──────────┘  └───────────┘  └──────────────┘  │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  JS Chess Engine (mirrors CJC engine exactly)            │   │
│  │    initBoard()           legalMoves(board, side)         │   │
│  │    applyMove(board,f,t)  terminalStatus(board, side)     │   │
│  │    encodeBoard()         inCheck(board, side)            │   │
│  │    isAttackedBy()        findKing(board, side)           │   │
│  └──────────────────────────────────────────────────────────┘   │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  JS Agent (lightweight inference)                         │   │
│  │    SplitMix64 RNG         2-layer MLP [66→16→1]          │   │
│  │    forwardMove()          selectAction()                  │   │
│  │    Softmax + categorical  Deterministic sampling          │   │
│  └──────────────────────────────────────────────────────────┘   │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Trace Recorder                                           │   │
│  │    Per-ply: board, side, legalMoves, chosenMove           │   │
│  │    Agent turns: moveProbs, topK, confidence               │   │
│  │    Export: JSONL to localStorage or download               │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
          │
          │ (Rust side: validation + regression)
          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Rust Test Harness                                               │
│                                                                  │
│  tests/chess_rl_playability/                                     │
│    test_human_move_validation.rs  — illegal move rejection       │
│    test_replay_from_trace.rs     — trace replay determinism      │
│    test_player_profile.rs        — profile generation            │
│    test_pgn_import.rs            — PGN parsing + normalization   │
│    test_opening_tree.rs          — move-prefix tree construction │
│    test_adaptation.rs            — style-conditioned evaluation  │
│                                                                  │
│  Engine API (same as training/self-play):                        │
│    init_board(), legal_moves(), apply_move()                     │
│    terminal_status(), encode_board(), in_check()                 │
│    select_action(), forward_move()                               │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Engine/UI Boundary

The engine and UI are cleanly separated:

| Layer | Location | Language | Responsibility |
|-------|----------|----------|---------------|
| Core engine | tests/chess_rl_project/cjc_source.rs | CJC (via MIR-exec) | Authoritative chess rules, RL training |
| JS engine mirror | examples/chess_rl_platform.html | JavaScript | Browser-side chess rules (identical logic) |
| JS agent | examples/chess_rl_platform.html | JavaScript | Lightweight MLP inference for human play |
| Trace recorder | examples/chess_rl_platform.html | JavaScript | Per-ply JSONL generation |
| Test harness | tests/chess_rl_playability/ | Rust | Validation, regression, import |
| Trace storage | trace/ | JSON/JSONL | Game records, profiles, imports |

## 3. Determinism Guarantees

| Feature | Determinism Method |
|---------|-------------------|
| Agent moves | SplitMix64 RNG in JS (identical to cjc-repro) |
| Move generation | Square-ordered enumeration (0-63), identical to CJC |
| Board encoding | Same normalization: piece * side / 6.0 |
| Weight loading | JSON weight export from Rust, loaded in JS |
| Replay | Same seed + same human moves = identical trace |
| Player profiles | Deterministic statistics from stored traces |
| PGN import | Normalized to CJC board format, sorted deterministically |

## 4. Human Input / Event Flow

```
User clicks square A → highlight + show legal moves from A
User clicks square B → if legal: apply move, record trace, agent responds
                        if illegal: flash red, deselect
Agent turn → compute all legal moves
           → forward pass for each (score)
           → softmax → categorical sample (seeded RNG)
           → apply move, record trace, show policy panel
Game end → display result, offer: review / save trace / new game
```

## 5. Replay Compatibility

Human games produce the same JSONL format as self-play traces:
```json
{"ply":0,"board":[4,2,3,...],"side":1,"legal_moves":[8,16,8,24,...],"source":"human"}
{"ply":1,"board":[...],"side":1,"move":[12,28],"source":"human","type":"human_move"}
{"ply":2,"board":[...],"side":-1,"move":[52,36],"source":"agent","probs":[0.12,0.08,...],"type":"agent_move"}
```

Existing replay infrastructure (dashboard, trace demo) can load human game traces directly.

## 6. Player-Style Adaptation Architecture

```
trace/human_games/*.jsonl
         │
         ▼
┌─────────────────────────┐
│  Profile Generator      │
│  (Rust or JS)           │
│                         │
│  Extracts:              │
│  - opening frequencies  │
│  - capture timing       │
│  - avg game length      │
│  - move diversity       │
│  - piece preferences    │
│  - aggression metrics   │
└────────┬────────────────┘
         │
         ▼
trace/player_profiles/default_player_profile.json
         │
         ▼
Dashboard: "Your Style" panel
         │
         ▼
Optional: style-conditioned evaluation scheduling
```

Profile is purely statistical — no invented personality narratives.

## 7. External Dataset Ingestion Architecture

```
data/external_games/*.pgn
         │
         ▼
┌─────────────────────────┐
│  PGN Parser (Rust)       │
│  - parse headers         │
│  - parse move text       │
│  - convert to CJC board  │
│  - normalize             │
│  - reject malformed      │
└────────┬────────────────┘
         │
         ▼
trace/imported_games/*.jsonl    (normalized game traces)
trace/import_indexes/index.json (provenance metadata)
         │
         ▼
Uses:
  - Opening explorer statistics
  - Evaluation benchmark corpus
  - Supervised warm-start (optional, explicit)
  - Opening-book priors (optional, explicit)
```

## 8. Files to Create

| File | Purpose |
|------|---------|
| `examples/chess_rl_platform.html` | Main interactive platform (play, review, explore) |
| `tests/test_chess_rl_playability.rs` | Entry point for playability tests |
| `tests/chess_rl_playability/mod.rs` | Module structure |
| `tests/chess_rl_playability/helpers.rs` | Shared helpers |
| `tests/chess_rl_playability/test_human_move.rs` | Human move validation |
| `tests/chess_rl_playability/test_replay_trace.rs` | Trace replay determinism |
| `tests/chess_rl_playability/test_player_profile.rs` | Profile generation |
| `tests/chess_rl_playability/test_pgn_import.rs` | PGN parsing |
| `tests/chess_rl_playability/test_opening_tree.rs` | Opening tree construction |
| `tests/chess_rl_playability/test_adaptation.rs` | Style-conditioned evaluation |
| `docs/chess_rl_external_data_ingestion.md` | Import documentation |
| `docs/chess_rl_playability_regression_report.md` | Regression report |
| `docs/portfolio/chess_rl_playable_platform_summary.md` | Portfolio summary |

## 9. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| JS engine diverges from CJC engine | HIGH | Parity tests: same position → same legal moves |
| JS RNG diverges from SplitMix64 | HIGH | Port exact SplitMix64 algorithm to JS, verify |
| Human input breaks state machine | LOW | Strict FSM: IDLE → SELECTED → MOVED |
| PGN parsing edge cases | MEDIUM | Reject malformed, log warnings |
| Profile generation non-deterministic | LOW | Sort all inputs, use stable algorithms |
