//! PGN import / board normalization tests.
//!
//! Tests that board positions can be constructed from move sequences
//! and normalized to the CJC board format, as needed for PGN import.

use super::helpers::*;
use super::pgn_parser::{self, Board, parse_pgn, import_pgn, resolve_san, game_to_jsonl, sq_of};

/// Board can be reconstructed from algebraic-style move indices.
#[test]
fn reconstruct_board_from_move_indices() {
    let src = chess_program(r#"
        let board = init_board();
        // 1. e4 (12->28) e5 (52->36) 2. Nf3 (6->21) Nc6 (57->42)
        let board = apply_move(board, 12, 28);
        let board = apply_move(board, 52, 36);
        let board = apply_move(board, 6, 21);
        let board = apply_move(board, 57, 42);
        // Verify pieces
        print(board[28]); // e4: white pawn
        print(board[36]); // e5: black pawn
        print(board[21]); // f3: white knight
        print(board[42]); // c6: black knight
        print(board[12]); // e2: empty
        print(board[6]);  // g1: empty
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(parse_int_at(&out, 0), 1, "e4: white pawn");
    assert_eq!(parse_int_at(&out, 1), -1, "e5: black pawn");
    assert_eq!(parse_int_at(&out, 2), 2, "f3: white knight");
    assert_eq!(parse_int_at(&out, 3), -2, "c6: black knight");
    assert_eq!(parse_int_at(&out, 4), 0, "e2: empty");
    assert_eq!(parse_int_at(&out, 5), 0, "g1: empty");
}

/// Board encoding produces valid tensor shape [1, 64].
#[test]
fn encoding_produces_valid_shape() {
    let src = chess_program(r#"
        let board = init_board();
        let feat = encode_board(board, 1);
        print(feat.shape());
    "#);
    let out = run_mir(&src, 42);
    assert!(out[0].contains("[1, 64]"), "encoding should produce [1, 64] tensor, got: {}", out[0]);
}

/// Encoding is symmetric: encoding for white is negative of encoding for black.
#[test]
fn encoding_side_symmetry() {
    let src = chess_program(r#"
        let board = init_board();
        let feat_w = encode_board(board, 1);
        let feat_b = encode_board(board, -1);
        // For piece at sq 0 (white rook, 4):
        //   White: 4*1/6 = 0.666...
        //   Black: 4*(-1)/6 = -0.666...
        print(feat_w.get([0, 0]));
        print(feat_b.get([0, 0]));
    "#);
    let out = run_mir(&src, 42);
    let w = parse_float_at(&out, 0);
    let b = parse_float_at(&out, 1);
    assert!((w + b).abs() < 1e-10, "encoding for white and black should be symmetric (negated)");
}

/// Batch of moves can be validated: all moves from init position are legal.
#[test]
fn all_generated_moves_are_legal() {
    let src = chess_program(r#"
        let board = init_board();
        let moves = legal_moves(board, 1);
        let all_valid = true;
        let i = 0;
        while i < len(moves) {
            let from_sq = moves[i];
            let to_sq = moves[i + 1];
            let new_board = apply_move(board, from_sq, to_sq);
            // After a legal move, our king should not be in check
            if in_check(new_board, 1) {
                all_valid = false;
            }
            i = i + 2;
        }
        if all_valid { print("valid"); } else { print("invalid"); }
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "valid", "all generated legal moves should leave king safe");
}

/// Square coordinate helpers: rank_of, file_of, sq_of roundtrip.
#[test]
fn coordinate_roundtrip() {
    let src = chess_program(r#"
        // Test all 64 squares
        let ok = true;
        let sq = 0;
        while sq < 64 {
            let r = rank_of(sq);
            let f = file_of(sq);
            if sq_of(r, f) != sq {
                ok = false;
            }
            sq = sq + 1;
        }
        if ok { print("pass"); } else { print("fail"); }
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "pass");
}

// ─── PGN Parser Integration Tests ───

/// Parse a single PGN game with full headers.
#[test]
fn pgn_parse_single_game_with_headers() {
    let pgn = r#"[Event "Casual Game"]
[Site "Home"]
[Date "2026.03.12"]
[Round "1"]
[White "Alice"]
[Black "Bob"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0
"#;
    let games = parse_pgn(pgn);
    assert_eq!(games.len(), 1);
    assert_eq!(games[0].headers.get("Event").unwrap(), "Casual Game");
    assert_eq!(games[0].headers.get("White").unwrap(), "Alice");
    assert_eq!(games[0].headers.get("Black").unwrap(), "Bob");
    assert_eq!(games[0].moves.len(), 6);
    assert_eq!(games[0].result, "1-0");
}

/// Parse multiple PGN games from one string.
#[test]
fn pgn_parse_multiple_games() {
    let pgn = r#"[Event "Game 1"]
[Result "1-0"]

1. e4 e5 2. d4 d5 1-0

[Event "Game 2"]
[Result "0-1"]

1. d4 d5 2. Nf3 Nf6 0-1

[Event "Game 3"]
[Result "1/2-1/2"]

1. e4 e5 1/2-1/2
"#;
    let games = parse_pgn(pgn);
    assert_eq!(games.len(), 3);
    assert_eq!(games[0].result, "1-0");
    assert_eq!(games[1].result, "0-1");
    assert_eq!(games[2].result, "1/2-1/2");
}

/// SAN resolution: basic pawn moves.
#[test]
fn pgn_resolve_pawn_single_push() {
    let board = Board::initial();
    let mv = resolve_san(&board, "e4", 1).unwrap();
    assert_eq!(mv.from, sq_of(1, 4)); // e2
    assert_eq!(mv.to, sq_of(3, 4));   // e4
    assert!(mv.promotion.is_none());
}

/// SAN resolution: black pawn push.
#[test]
fn pgn_resolve_black_pawn_push() {
    let mut board = Board::initial();
    board.apply_move(sq_of(1, 4), sq_of(3, 4)); // e4
    let mv = resolve_san(&board, "e5", -1).unwrap();
    assert_eq!(mv.from, sq_of(6, 4)); // e7
    assert_eq!(mv.to, sq_of(4, 4));   // e5
}

/// SAN resolution: knight move with disambiguation.
#[test]
fn pgn_resolve_knight_disambiguated() {
    let board = Board::initial();
    // From initial position, Nc3 should resolve g1-knight (no, wait Nc3 is b1 knight)
    let mv = resolve_san(&board, "Nc3", 1).unwrap();
    assert_eq!(mv.from, sq_of(0, 1)); // b1
    assert_eq!(mv.to, sq_of(2, 2));   // c3
}

/// SAN resolution: pawn capture.
#[test]
fn pgn_resolve_pawn_capture() {
    let mut board = Board::initial();
    // Set up: e4 d5
    board.apply_move(sq_of(1, 4), sq_of(3, 4)); // e2-e4
    board.apply_move(sq_of(6, 3), sq_of(4, 3)); // d7-d5
    // exd5
    let mv = resolve_san(&board, "exd5", 1).unwrap();
    assert_eq!(mv.from, sq_of(3, 4)); // e4
    assert_eq!(mv.to, sq_of(4, 3));   // d5
}

/// SAN resolution: bishop diagonal move.
#[test]
fn pgn_resolve_bishop_move() {
    let mut board = Board::initial();
    board.apply_move(sq_of(1, 4), sq_of(3, 4)); // e4 opens diagonal
    // Bc4
    let mv = resolve_san(&board, "Bc4", 1).unwrap();
    assert_eq!(mv.from, sq_of(0, 5)); // f1
    assert_eq!(mv.to, sq_of(3, 2));   // c4
}

/// Import a game without castling/EP succeeds.
#[test]
fn pgn_import_simple_game_succeeds() {
    let pgn = r#"[Event "Test"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 1-0
"#;
    let (imported, rejected) = import_pgn(pgn);
    assert_eq!(rejected.len(), 0, "no rejections: {:?}", rejected.iter().map(|(_, r)| r.to_string()).collect::<Vec<_>>());
    assert_eq!(imported.len(), 1);
    assert_eq!(imported[0].moves.len(), 8);
    assert_eq!(imported[0].result, 1.0);
    // Board states: initial + 8 moves = 9
    assert_eq!(imported[0].board_states.len(), 9);
}

/// Import rejects games with castling.
#[test]
fn pgn_import_rejects_castling_game() {
    let pgn = r#"[Event "Test"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. O-O Nf6 1-0
"#;
    let (imported, rejected) = import_pgn(pgn);
    assert_eq!(imported.len(), 0);
    assert_eq!(rejected.len(), 1);
    assert!(rejected[0].1.to_string().contains("castling"));
}

/// JSONL output contains correct number of lines.
#[test]
fn pgn_jsonl_output_structure() {
    let pgn = r#"[Event "Test"]
[Result "1-0"]

1. e4 e5 2. d4 d5 1-0
"#;
    let (imported, _) = import_pgn(pgn);
    let jsonl = game_to_jsonl(&imported[0]);
    let lines: Vec<&str> = jsonl.lines().collect();
    // init + 4 moves + result = 6
    assert_eq!(lines.len(), 6);
    assert!(lines[0].contains("\"type\":\"init\""));
    assert!(lines[0].contains("\"source\":\"import\""));
    assert!(lines[1].contains("\"type\":\"move\""));
    assert!(lines[5].contains("\"type\":\"result\""));
}

/// Board parity: PGN parser Board::initial matches CJC init_board().
#[test]
fn pgn_board_matches_cjc_init_board() {
    // Get CJC init_board result
    let src = chess_program(r#"
        let board = init_board();
        let i = 0;
        while i < 64 {
            print(board[i]);
            i = i + 1;
        }
    "#);
    let out = run_mir(&src, 42);
    let cjc_board: Vec<i64> = out.iter().take(64).map(|s| s.trim().parse::<i64>().unwrap()).collect();

    // Rust parser initial board
    let rust_board = Board::initial();

    for sq in 0..64 {
        assert_eq!(
            rust_board.squares[sq], cjc_board[sq],
            "Board mismatch at sq {}: Rust={}, CJC={}",
            sq, rust_board.squares[sq], cjc_board[sq]
        );
    }
}

/// Move parity: PGN parser apply_move matches CJC apply_move.
#[test]
fn pgn_apply_move_matches_cjc() {
    // Apply e4 in CJC
    let src = chess_program(r#"
        let board = init_board();
        let board = apply_move(board, 12, 28);
        let i = 0;
        while i < 64 {
            print(board[i]);
            i = i + 1;
        }
    "#);
    let out = run_mir(&src, 42);
    let cjc_board: Vec<i64> = out.iter().take(64).map(|s| s.trim().parse::<i64>().unwrap()).collect();

    // Apply e4 in Rust
    let mut rust_board = Board::initial();
    rust_board.apply_move(12, 28);

    for sq in 0..64 {
        assert_eq!(
            rust_board.squares[sq], cjc_board[sq],
            "After e4 mismatch at sq {}: Rust={}, CJC={}",
            sq, rust_board.squares[sq], cjc_board[sq]
        );
    }
}

/// Promotion in PGN is handled correctly.
#[test]
fn pgn_resolve_promotion() {
    // Set up a board with white pawn on e7, empty e8
    let mut board = Board::initial();
    // Clear everything relevant and place a white pawn on e7
    for sq in 0..64 {
        board.squares[sq] = pgn_parser::EMPTY;
    }
    board.squares[sq_of(0, 4)] = pgn_parser::W_KING;   // e1
    board.squares[sq_of(7, 0)] = pgn_parser::B_KING;    // a8
    board.squares[sq_of(6, 4)] = pgn_parser::W_PAWN;    // e7 (will promote on e8)

    let mv = resolve_san(&board, "e8=Q", 1).unwrap();
    assert_eq!(mv.from, sq_of(6, 4)); // e7
    assert_eq!(mv.to, sq_of(7, 4));   // e8
    assert_eq!(mv.promotion, Some(pgn_parser::W_QUEEN));
}

/// Mixed import: some games accepted, some rejected.
#[test]
fn pgn_mixed_import() {
    let pgn = r#"[Event "No Castling Game"]
[Result "1/2-1/2"]

1. e4 e5 2. d4 d5 1/2-1/2

[Event "Castling Game"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. O-O Nf6 1-0

[Event "Another Simple Game"]
[Result "0-1"]

1. d4 d5 2. Nf3 Nf6 0-1
"#;
    let (imported, rejected) = import_pgn(pgn);
    assert_eq!(imported.len(), 2, "two non-castling games accepted");
    assert_eq!(rejected.len(), 1, "one castling game rejected");
    assert_eq!(imported[0].result, 0.0);  // draw
    assert_eq!(imported[1].result, -1.0); // black wins
}
