# Chess RL — External Data Ingestion Architecture

## Overview

The CJC Chess RL platform supports importing external chess game data (PGN format) for use in opening exploration, evaluation benchmarks, and optional supervised warm-starts.

## Pipeline

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
```

## PGN Format Support

### Supported PGN Features
- Standard 7-tag roster (Event, Site, Date, Round, White, Black, Result)
- Standard algebraic notation (SAN) move text
- Game termination markers (1-0, 0-1, 1/2-1/2, *)
- Multiple games per file

### Rejected During CJC Backend Import
- Castling notation (O-O, O-O-O) — CJC backend engine lacks castling (JS engine supports it)
- En passant — CJC backend engine lacks en passant (JS engine supports it)

### Unsupported PGN Features
- Annotations, comments, NAGs ({}, ;, $)
- Variations (parenthesized alternatives)
- Non-standard piece letters

## Normalization

PGN moves are converted to the CJC board format:

| PGN Concept | CJC Equivalent |
|-------------|----------------|
| Piece at file/rank | Board index: rank*8 + file |
| Piece encoding | 0=empty, ±1..±6 (P,N,B,R,Q,K) |
| Move | (from_sq, to_sq) pair |
| Promotion | Auto-queen (piece becomes ±5) |
| Game result | 1.0 (white), -1.0 (black), 0.0 (draw) |

## Output Format

Imported games produce JSONL traces identical to self-play and human game traces:

```json
{"ply":0,"board":[4,2,3,...],"side":1,"source":"import","pgn_event":"..."}
{"ply":1,"board":[...],"side":1,"move":[12,28],"source":"import","type":"move"}
{"ply":2,"board":[...],"side":-1,"move":[52,36],"source":"import","type":"move"}
```

## Provenance Tracking

Each import produces a provenance record in `trace/import_indexes/index.json`:

```json
{
  "imports": [
    {
      "source_file": "lichess_2024_01.pgn",
      "import_date": "2025-03-12T10:00:00Z",
      "games_parsed": 150,
      "games_rejected": 3,
      "rejection_reasons": ["castling_not_supported", "malformed_san"],
      "output_files": ["trace/imported_games/lichess_2024_01_001.jsonl", "..."]
    }
  ]
}
```

## Integration Points

| Use Case | How Imported Data Is Used |
|----------|--------------------------|
| Opening Explorer | Aggregated move frequencies from imported traces |
| Evaluation Benchmark | Agent performance against known game outcomes |
| Supervised Warm-Start | Optional: use imported move distributions as training signal |
| Opening-Book Priors | Optional: bias initial move selection toward common openings |

All integration is **explicit and toggleable** — imported data never silently changes agent behavior.

## Determinism Guarantees

- PGN parsing is fully deterministic (no randomness)
- Games are sorted by (Event, Date, Round, White, Black) for stable ordering
- Duplicate detection via board-state hashing
- Import index is append-only, never silently modified

## Directory Structure

```
data/
  external_games/        # Raw PGN files (user-provided)
    README.md            # Instructions for adding PGN files

trace/
  imported_games/        # Normalized JSONL traces
  import_indexes/        # Provenance metadata
    index.json
```

## Current Status

**Implemented.** The Rust PGN parser (`tests/chess_rl_playability/pgn_parser.rs`) provides:

- Full PGN text parsing (headers, SAN move text, results, multi-game files)
- SAN-to-coordinate resolution using a Rust-side board that mirrors CJC encoding
- En passant and castling detection (games using these are rejected for CJC backend compatibility)
- Promotion handling with explicit piece selection
- JSONL trace generation matching the self-play trace format
- Board parity verified against CJC engine (`init_board`, `apply_move`)

**Test coverage:** 19 tests (5 board normalization + 14 parser integration), all passing.

**Sample PGN file:** `data/external_games/sample_games.pgn` (4 games, 1 with castling for rejection testing).

Note: The JS platform now supports castling and en passant natively, but the CJC backend engine does not. PGN games using these features are rejected during Rust-side import to maintain CJC engine parity.

## Integration with Agent Training

The interactive JS platform now performs **post-game REINFORCE training** after every human game:

- `trainFromGame()` runs gradient updates on the MLP weights using the game trace
- Weights persist across games via `game.trainedWeights`
- Temperature-controlled softmax (T=0.5) for aggressive play
- Training episode counter displayed in the UI
- "Reset Agent" button to restart from random weights

When imported data is used for training, the training configuration explicitly specifies:
- `warmstart_source`: path to imported traces
- `warmstart_epochs`: number of supervised pre-training epochs
- `opening_prior_depth`: how many plies of opening data to use
- `curriculum_mode`: how imported data informs training schedule

All parameters are optional, defaulting to pure REINFORCE from random initialization.

## Future Extensions

- FEN position import for mid-game analysis
- Lichess API integration for direct game download
- Rating-stratified imports for difficulty-based curriculum
- Bridge imported PGN traces to JS platform for opening book priors
