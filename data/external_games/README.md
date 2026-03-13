# External Chess Game Data

Place PGN files here for import into the CJC Chess RL platform.

## Supported Format

Standard PGN files with algebraic notation. Note that the CJC chess engine
does not support castling or en passant, so games using these moves will
be skipped during import.

## Usage

Import is handled by the Rust PGN parser (when implemented).
Imported games are normalized to JSONL traces in `trace/imported_games/`.
