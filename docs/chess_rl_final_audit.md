# CJC Chess RL Final Audit

## Audit Date: 2026-03-12

## Scope

Complete final audit of the CJC Chess RL Interactive Platform covering engine correctness, UI fidelity, replay stability, competence, and demo readiness.

## Engine Correctness

### Move Generation
- **Pawn**: Forward-one, forward-two from starting rank, diagonal captures only. All correct.
- **Knight**: 8 L-shaped offsets, captures and quiet moves. Correct.
- **Bishop**: 4 diagonal sliding rays with blocking. Correct.
- **Rook**: 4 straight sliding rays with blocking. Correct.
- **Queen**: Combined bishop + rook sliding. Correct.
- **King**: 8 adjacent squares, can capture undefended pieces. Correct.

### Capture Pipeline
- All piece types generate captures correctly
- Sliding pieces stop at first occupied square (capture enemy, blocked by friendly)
- Pawns only capture diagonally, never forward
- King cannot capture defended pieces (filtered by legal_moves)
- Captures reduce piece count by exactly 1
- **12 dedicated capture tests all pass**

### Promotion
- White pawn to rank 7 → queen (piece 5)
- Black pawn to rank 0 → queen (piece -5)
- Promotion-with-capture works correctly
- Multiple promotions in a game work correctly
- UI supports underpromotion (queen/rook/bishop/knight) for human players
- Agent always promotes to queen (deterministic)
- **9 dedicated promotion tests all pass**

### Legality Filtering
- legal_moves applies each pseudo-legal move and checks own king not in check
- Correctly rejects self-check moves
- Correctly allows captures that don't leave king exposed

### Terminal Detection
- Checkmate: no legal moves + in check → status 2
- Stalemate: no legal moves + not in check → status 3
- Ongoing: legal moves exist → status 0

### JS Mirror Parity
- All engine functions verified identical between CJC and JS implementations
- Same piece encoding, move generation order, promotion logic
- SplitMix64 RNG uses BigInt for exact u64 arithmetic match

## UI Fidelity

### Board Rendering
- Board matches engine state exactly (piece positions verified through move sequences)
- Legal move highlighting matches actual legal moves from engine
- Capture squares show red ring, empty targets show blue dot
- Last-move highlighting (from/to squares) correct
- Check detection highlights king square in red
- Board orientation respects settings (auto/white/black)

### Promotion UI
- Modal overlay appears when pawn reaches promotion rank
- 4 options: Queen, Rook, Bishop, Knight
- Selection completes the move with correct piece placement
- Agent promotion is automatic (queen)

### Policy Display ("Why this move?" card)
- Shows real policy probabilities from the agent's MLP + heuristic
- Top 5 candidates with actual probability bars
- Confidence percentage matches chosen move's probability
- Legal move count matches actual legal_moves output
- Material delta shown when capture occurs
- No invented strategy prose — purely factual data

### Demo Presets Bar
- 5 presets: Play Agent, Play Random, Replay Last, Debug Trace, Quick Game
- Each correctly configures settings and starts/navigates as expected
- Visible at top of page (not hidden in settings)

### Deterministic Replay Badge
- Shows current seed, opponent type, game status
- "Replay Exact" button restarts with identical settings
- Badge updates correctly on game state changes

## Replay/Export

### Trace Format
- JSONL format with one entry per line
- Entries include: ply, board state, side, move, source, type
- Agent moves include probs array and action_idx
- Promotion moves include promotion_piece field
- Game-end entries include result and total_moves

### Replay Stability
- Same seed + same human move sequence = identical game trace
- Review mode reconstructs board state correctly at every ply
- Policy data at each ply matches the original game's policy output
- Auto-play advances through game at consistent 800ms intervals

### Export
- JSONL download produces valid file
- Copy-to-clipboard works
- Session summary export includes game metadata

## Competence Assessment

### Capture Heuristic
- Agent adds 0.5x piece-value bonus to capture moves
- Result: Agent consistently captures high-value pieces when available
- Improved win rate vs random from ~50% to ~65-75% (estimated)
- Documented honestly as "MLP + capture heuristic"

### Remaining Weaknesses
- No positional evaluation
- No opening knowledge
- No endgame technique
- No look-ahead search
- Random MLP base weights (not trained)

## Adaptive Style System
- Style profile computes real statistics from stored traces
- Per-piece usage distribution, capture rate, game length
- Win/Draw/Loss record with percentages
- No anthropomorphized strategy descriptions

## External Data Ingestion
- Directory structure in place
- Format specification documented
- Provenance schema defined
- Full PGN parser is a planned follow-on

## Test Results

### New Tests Added
| Suite | Tests | Status |
|-------|-------|--------|
| test_capture_audit | 12 | All pass |
| test_promotion_audit | 9 | All pass |

### Existing Test Suites
All pre-existing tests continue to pass (no regressions).

## Documented Simplifications
1. No castling (king + rook move separately)
2. No en passant
3. Auto-queen promotion in CJC backend (UI supports underpromotion)
4. 200-halfmove draw limit (not 50-move rule)
5. No three-fold repetition detection

## Verdict

The platform is demo-ready. A reviewer can:
1. Click a demo preset to immediately start playing
2. See real policy data in the "Why this move?" card
3. Verify determinism via the replay badge
4. Observe the agent making reasonable capture decisions
5. Review completed games with per-ply analysis
6. Inspect the style profile and opening explorer

The engine is honest about its limitations and does not fabricate intelligence.
