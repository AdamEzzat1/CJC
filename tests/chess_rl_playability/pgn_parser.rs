//! PGN Parser for CJC Chess RL Platform.
//!
//! Parses standard PGN files into structured game data, then converts
//! SAN (Standard Algebraic Notation) moves into (from_sq, to_sq) pairs
//! using the CJC board encoding (0=empty, ±1..±6 = P,N,B,R,Q,K).
//!
//! Board layout: rank*8 + file, rank 0 = white side (a1=0, h1=7, a8=56, h8=63).

use std::collections::BTreeMap;
use std::fmt;

// ─── Piece Encoding (matches CJC engine) ───

pub const EMPTY: i64 = 0;
pub const W_PAWN: i64 = 1;
pub const W_KNIGHT: i64 = 2;
pub const W_BISHOP: i64 = 3;
pub const W_ROOK: i64 = 4;
pub const W_QUEEN: i64 = 5;
pub const W_KING: i64 = 6;
pub const B_PAWN: i64 = -1;
pub const B_KNIGHT: i64 = -2;
pub const B_BISHOP: i64 = -3;
pub const B_ROOK: i64 = -4;
pub const B_QUEEN: i64 = -5;
pub const B_KING: i64 = -6;

// ─── Coordinate Helpers ───

pub fn rank_of(sq: usize) -> usize {
    sq / 8
}

pub fn file_of(sq: usize) -> usize {
    sq % 8
}

pub fn sq_of(rank: usize, file: usize) -> usize {
    rank * 8 + file
}

/// Parse algebraic square name (e.g., "e4") to board index.
pub fn parse_square(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    if bytes.len() < 2 {
        return None;
    }
    let file = match bytes[0] {
        b'a'..=b'h' => (bytes[0] - b'a') as usize,
        _ => return None,
    };
    let rank = match bytes[1] {
        b'1'..=b'8' => (bytes[1] - b'1') as usize,
        _ => return None,
    };
    Some(sq_of(rank, file))
}

/// Square index to algebraic name (e.g., 0 → "a1").
pub fn square_name(sq: usize) -> String {
    let file = (b'a' + file_of(sq) as u8) as char;
    let rank = (b'1' + rank_of(sq) as u8) as char;
    format!("{}{}", file, rank)
}

// ─── Board ───

/// A chess board matching CJC encoding.
#[derive(Clone, Debug)]
pub struct Board {
    pub squares: [i64; 64],
}

impl Board {
    /// Standard starting position.
    pub fn initial() -> Self {
        let mut squares = [EMPTY; 64];
        // White pieces (rank 0)
        squares[0] = W_ROOK;
        squares[1] = W_KNIGHT;
        squares[2] = W_BISHOP;
        squares[3] = W_QUEEN;
        squares[4] = W_KING;
        squares[5] = W_BISHOP;
        squares[6] = W_KNIGHT;
        squares[7] = W_ROOK;
        // White pawns (rank 1)
        for f in 0..8 {
            squares[sq_of(1, f)] = W_PAWN;
        }
        // Black pawns (rank 6)
        for f in 0..8 {
            squares[sq_of(6, f)] = B_PAWN;
        }
        // Black pieces (rank 7)
        squares[56] = B_ROOK;
        squares[57] = B_KNIGHT;
        squares[58] = B_BISHOP;
        squares[59] = B_QUEEN;
        squares[60] = B_KING;
        squares[61] = B_BISHOP;
        squares[62] = B_KNIGHT;
        squares[63] = B_ROOK;
        Board { squares }
    }

    /// Apply a move (from_sq, to_sq). Handles pawn promotion (auto-queen).
    pub fn apply_move(&mut self, from: usize, to: usize) {
        let piece = self.squares[from];
        self.squares[to] = piece;
        self.squares[from] = EMPTY;
        // Auto-queen promotion
        if piece == W_PAWN && rank_of(to) == 7 {
            self.squares[to] = W_QUEEN;
        } else if piece == B_PAWN && rank_of(to) == 0 {
            self.squares[to] = B_QUEEN;
        }
    }

    /// Apply a move with explicit promotion piece.
    pub fn apply_move_with_promotion(&mut self, from: usize, to: usize, promo_piece: i64) {
        self.squares[to] = promo_piece;
        self.squares[from] = EMPTY;
    }

    /// Get piece at square.
    pub fn at(&self, sq: usize) -> i64 {
        self.squares[sq]
    }

    /// Find all squares containing a specific piece.
    pub fn find_pieces(&self, piece: i64) -> Vec<usize> {
        self.squares
            .iter()
            .enumerate()
            .filter(|(_, &p)| p == piece)
            .map(|(sq, _)| sq)
            .collect()
    }

    /// Check if a square is attacked by the given side (1=white, -1=black).
    pub fn is_attacked_by(&self, sq: usize, side: i64) -> bool {
        let r = rank_of(sq) as i64;
        let f = file_of(sq) as i64;

        // Pawn attacks
        if side == 1 {
            // White pawns attack upward-diagonally
            if r > 0 {
                if f > 0 && self.at(sq_of((r - 1) as usize, (f - 1) as usize)) == W_PAWN {
                    return true;
                }
                if f < 7 && self.at(sq_of((r - 1) as usize, (f + 1) as usize)) == W_PAWN {
                    return true;
                }
            }
        } else {
            // Black pawns attack downward-diagonally
            if r < 7 {
                if f > 0 && self.at(sq_of((r + 1) as usize, (f - 1) as usize)) == B_PAWN {
                    return true;
                }
                if f < 7 && self.at(sq_of((r + 1) as usize, (f + 1) as usize)) == B_PAWN {
                    return true;
                }
            }
        }

        // Knight attacks
        let knight = if side == 1 { W_KNIGHT } else { B_KNIGHT };
        let knight_offsets: [(i64, i64); 8] = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1),
        ];
        for (dr, df) in &knight_offsets {
            let nr = r + dr;
            let nf = f + df;
            if nr >= 0 && nr < 8 && nf >= 0 && nf < 8 {
                if self.at(sq_of(nr as usize, nf as usize)) == knight {
                    return true;
                }
            }
        }

        // King attacks
        let king = if side == 1 { W_KING } else { B_KING };
        for dr in -1..=1i64 {
            for df in -1..=1i64 {
                if dr == 0 && df == 0 {
                    continue;
                }
                let nr = r + dr;
                let nf = f + df;
                if nr >= 0 && nr < 8 && nf >= 0 && nf < 8 {
                    if self.at(sq_of(nr as usize, nf as usize)) == king {
                        return true;
                    }
                }
            }
        }

        // Sliding attacks (bishop/queen diagonals, rook/queen straight)
        let bishop = if side == 1 { W_BISHOP } else { B_BISHOP };
        let rook = if side == 1 { W_ROOK } else { B_ROOK };
        let queen = if side == 1 { W_QUEEN } else { B_QUEEN };

        // Diagonal directions (bishop + queen)
        for (dr, df) in &[(1i64, 1i64), (1, -1), (-1, 1), (-1, -1)] {
            let mut nr = r + dr;
            let mut nf = f + df;
            while nr >= 0 && nr < 8 && nf >= 0 && nf < 8 {
                let p = self.at(sq_of(nr as usize, nf as usize));
                if p != EMPTY {
                    if p == bishop || p == queen {
                        return true;
                    }
                    break;
                }
                nr += dr;
                nf += df;
            }
        }

        // Straight directions (rook + queen)
        for (dr, df) in &[(1i64, 0i64), (-1, 0), (0, 1), (0, -1)] {
            let mut nr = r + dr;
            let mut nf = f + df;
            while nr >= 0 && nr < 8 && nf >= 0 && nf < 8 {
                let p = self.at(sq_of(nr as usize, nf as usize));
                if p != EMPTY {
                    if p == rook || p == queen {
                        return true;
                    }
                    break;
                }
                nr += dr;
                nf += df;
            }
        }

        false
    }

    /// Check if the given side's king is in check.
    pub fn in_check(&self, side: i64) -> bool {
        let king = if side == 1 { W_KING } else { B_KING };
        for sq in 0..64 {
            if self.squares[sq] == king {
                return self.is_attacked_by(sq, -side);
            }
        }
        false
    }

    /// Check if a move from `from` to `to` is pseudo-legal for the given side.
    /// This does NOT check if the king is left in check.
    pub fn is_pseudo_legal(&self, from: usize, to: usize, side: i64) -> bool {
        let piece = self.at(from);
        if piece == EMPTY || piece.signum() != side {
            return false;
        }
        let target = self.at(to);
        // Can't capture own piece
        if target != EMPTY && target.signum() == side {
            return false;
        }

        let abs_piece = piece.abs();
        let fr = rank_of(from) as i64;
        let ff = file_of(from) as i64;
        let tr = rank_of(to) as i64;
        let tf = file_of(to) as i64;
        let dr = tr - fr;
        let df = tf - ff;

        match abs_piece {
            1 => {
                // Pawn
                let dir = side; // +1 for white, -1 for black
                if df == 0 && target == EMPTY {
                    // Forward
                    if dr == dir {
                        return true;
                    }
                    // Double push from starting rank
                    let start_rank = if side == 1 { 1 } else { 6 };
                    if fr == start_rank && dr == 2 * dir {
                        let mid = sq_of((fr + dir) as usize, ff as usize);
                        return self.at(mid) == EMPTY;
                    }
                }
                // Diagonal capture
                if dr == dir && df.abs() == 1 && target != EMPTY {
                    return true;
                }
                false
            }
            2 => {
                // Knight
                let adr = dr.abs();
                let adf = df.abs();
                (adr == 2 && adf == 1) || (adr == 1 && adf == 2)
            }
            3 => {
                // Bishop
                if dr.abs() != df.abs() || dr == 0 {
                    return false;
                }
                self.path_clear(from, to)
            }
            4 => {
                // Rook
                if dr != 0 && df != 0 {
                    return false;
                }
                if dr == 0 && df == 0 {
                    return false;
                }
                self.path_clear(from, to)
            }
            5 => {
                // Queen
                let is_diag = dr.abs() == df.abs() && dr != 0;
                let is_straight = (dr == 0) != (df == 0);
                if !is_diag && !is_straight {
                    return false;
                }
                self.path_clear(from, to)
            }
            6 => {
                // King
                dr.abs() <= 1 && df.abs() <= 1 && (dr != 0 || df != 0)
            }
            _ => false,
        }
    }

    /// Check if path between from and to is clear (for sliding pieces).
    fn path_clear(&self, from: usize, to: usize) -> bool {
        let fr = rank_of(from) as i64;
        let ff = file_of(from) as i64;
        let tr = rank_of(to) as i64;
        let tf = file_of(to) as i64;
        let dr = (tr - fr).signum();
        let df = (tf - ff).signum();
        let mut r = fr + dr;
        let mut f = ff + df;
        while (r, f) != (tr, tf) {
            if self.at(sq_of(r as usize, f as usize)) != EMPTY {
                return false;
            }
            r += dr;
            f += df;
        }
        true
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for rank in (0..8).rev() {
            write!(f, "{} ", rank + 1)?;
            for file in 0..8 {
                let piece = self.at(sq_of(rank, file));
                let c = match piece {
                    W_PAWN => 'P', W_KNIGHT => 'N', W_BISHOP => 'B',
                    W_ROOK => 'R', W_QUEEN => 'Q', W_KING => 'K',
                    B_PAWN => 'p', B_KNIGHT => 'n', B_BISHOP => 'b',
                    B_ROOK => 'r', B_QUEEN => 'q', B_KING => 'k',
                    _ => '.',
                };
                write!(f, "{} ", c)?;
            }
            writeln!(f)?;
        }
        writeln!(f, "  a b c d e f g h")
    }
}

// ─── PGN Data Structures ───

/// A parsed PGN game.
#[derive(Debug, Clone)]
pub struct PgnGame {
    pub headers: BTreeMap<String, String>,
    pub moves: Vec<String>,    // Raw SAN move strings
    pub result: String,        // "1-0", "0-1", "1/2-1/2", "*"
}

/// A resolved move (SAN → coordinates).
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedMove {
    pub from: usize,
    pub to: usize,
    pub promotion: Option<i64>, // Piece type if promotion
}

/// Result of importing a PGN game.
#[derive(Debug, Clone)]
pub struct ImportedGame {
    pub headers: BTreeMap<String, String>,
    pub moves: Vec<ResolvedMove>,
    pub result: f64,           // 1.0 (white), -1.0 (black), 0.0 (draw)
    pub board_states: Vec<[i64; 64]>, // Board after each ply
}

/// Import rejection reason.
#[derive(Debug, Clone)]
pub enum RejectionReason {
    CastlingNotSupported,
    EnPassantDetected,
    MalformedSan(String),
    AmbiguousMove(String),
    IllegalMove(String),
}

impl fmt::Display for RejectionReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RejectionReason::CastlingNotSupported => write!(f, "castling_not_supported"),
            RejectionReason::EnPassantDetected => write!(f, "en_passant_detected"),
            RejectionReason::MalformedSan(s) => write!(f, "malformed_san: {}", s),
            RejectionReason::AmbiguousMove(s) => write!(f, "ambiguous_move: {}", s),
            RejectionReason::IllegalMove(s) => write!(f, "illegal_move: {}", s),
        }
    }
}

/// Result of parsing/importing a single game.
pub type GameResult = Result<ImportedGame, RejectionReason>;

// ─── PGN Parser ───

/// Parse a PGN string containing one or more games.
pub fn parse_pgn(input: &str) -> Vec<PgnGame> {
    let mut games = Vec::new();
    let mut headers = BTreeMap::new();
    let mut move_text = String::new();
    let mut in_moves = false;

    for line in input.lines() {
        let trimmed = line.trim();

        if trimmed.is_empty() {
            if in_moves && !move_text.trim().is_empty() {
                // End of game
                let (moves, result) = parse_move_text(&move_text);
                games.push(PgnGame { headers, moves, result });
                headers = BTreeMap::new();
                move_text.clear();
                in_moves = false;
            }
            continue;
        }

        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            // Header tag
            if in_moves && !move_text.trim().is_empty() {
                // New game started before blank line
                let (moves, result) = parse_move_text(&move_text);
                games.push(PgnGame { headers, moves, result });
                headers = BTreeMap::new();
                move_text.clear();
            }
            in_moves = false;
            if let Some((key, val)) = parse_header(trimmed) {
                headers.insert(key, val);
            }
        } else {
            // Move text
            in_moves = true;
            move_text.push(' ');
            move_text.push_str(trimmed);
        }
    }

    // Handle last game
    if !move_text.trim().is_empty() {
        let (moves, result) = parse_move_text(&move_text);
        games.push(PgnGame { headers, moves, result });
    }

    games
}

/// Parse a PGN header line like `[Event "Casual Game"]`.
fn parse_header(line: &str) -> Option<(String, String)> {
    let inner = line.trim_start_matches('[').trim_end_matches(']');
    let space_pos = inner.find(' ')?;
    let key = inner[..space_pos].to_string();
    let val = inner[space_pos + 1..].trim();
    let val = val.trim_matches('"').to_string();
    Some((key, val))
}

/// Parse move text into individual SAN moves and result.
fn parse_move_text(text: &str) -> (Vec<String>, String) {
    let mut moves = Vec::new();
    let mut result = "*".to_string();

    for token in text.split_whitespace() {
        // Skip move numbers (e.g., "1.", "12.", "1...")
        if token.ends_with('.') || token.contains("...") {
            continue;
        }
        // Check for result tokens
        match token {
            "1-0" | "0-1" | "1/2-1/2" | "*" => {
                result = token.to_string();
                continue;
            }
            _ => {}
        }
        // Skip annotations/comments
        if token.starts_with('{') || token.starts_with(';') || token.starts_with('$') {
            continue;
        }
        // This should be a SAN move
        if !token.is_empty() {
            moves.push(token.to_string());
        }
    }

    (moves, result)
}

// ─── SAN Resolution ───

/// Map SAN piece letter to CJC piece type (absolute value).
fn san_piece_type(ch: char) -> Option<i64> {
    match ch {
        'K' => Some(6),
        'Q' => Some(5),
        'R' => Some(4),
        'B' => Some(3),
        'N' => Some(2),
        _ => None,
    }
}

/// Resolve a SAN move string to (from_sq, to_sq) on the given board.
/// `side` is 1 for white, -1 for black.
pub fn resolve_san(board: &Board, san: &str, side: i64) -> Result<ResolvedMove, RejectionReason> {
    let san = san.trim();

    // Castling
    if san == "O-O" || san == "O-O-O" || san == "0-0" || san == "0-0-0" {
        return Err(RejectionReason::CastlingNotSupported);
    }

    // Strip check/checkmate indicators
    let san = san.trim_end_matches('+').trim_end_matches('#');

    // Parse promotion
    let (san, promotion) = if san.contains('=') {
        let parts: Vec<&str> = san.splitn(2, '=').collect();
        let promo_char = parts[1].chars().next().unwrap_or('Q');
        let promo_type = san_piece_type(promo_char).unwrap_or(5); // default queen
        let promo_piece = promo_type * side;
        (parts[0], Some(promo_piece))
        } else {
        (san, None)
    };

    let chars: Vec<char> = san.chars().collect();
    if chars.is_empty() {
        return Err(RejectionReason::MalformedSan(san.to_string()));
    }

    // Determine piece type and parse disambiguation + destination
    let (piece_type, disambig_file, disambig_rank, dest_sq) = if chars[0].is_uppercase() {
        // Piece move (N, B, R, Q, K)
        let piece = san_piece_type(chars[0])
            .ok_or_else(|| RejectionReason::MalformedSan(san.to_string()))?;
        parse_san_target(&chars[1..], piece)?
    } else {
        // Pawn move
        parse_pawn_san(&chars, side)?
    };

    let target_piece = piece_type * side;

    // Find candidate source squares
    let candidates = board.find_pieces(target_piece);
    let mut valid = Vec::new();

    for &from in &candidates {
        // Apply disambiguation
        if let Some(f) = disambig_file {
            if file_of(from) != f {
                continue;
            }
        }
        if let Some(r) = disambig_rank {
            if rank_of(from) != r {
                continue;
            }
        }

        // Check if move is pseudo-legal
        if !board.is_pseudo_legal(from, dest_sq, side) {
            continue;
        }

        // Check that king is not left in check
        let mut test_board = board.clone();
        if let Some(promo) = promotion {
            test_board.apply_move_with_promotion(from, dest_sq, promo);
        } else {
            test_board.apply_move(from, dest_sq);
        }
        if !test_board.in_check(side) {
            valid.push(from);
        }
    }

    match valid.len() {
        0 => Err(RejectionReason::IllegalMove(format!(
            "no legal piece found for {} (side={})",
            san, side
        ))),
        1 => Ok(ResolvedMove {
            from: valid[0],
            to: dest_sq,
            promotion,
        }),
        _ => Err(RejectionReason::AmbiguousMove(format!(
            "ambiguous: {} has {} candidates",
            san,
            valid.len()
        ))),
    }
}

/// Parse SAN target for piece moves: optional disambiguation + optional 'x' + destination.
fn parse_san_target(
    chars: &[char],
    piece: i64,
) -> Result<(i64, Option<usize>, Option<usize>, usize), RejectionReason> {
    // Remove 'x' for captures
    let chars: Vec<char> = chars.iter().copied().filter(|&c| c != 'x').collect();

    if chars.len() < 2 {
        return Err(RejectionReason::MalformedSan(
            chars.iter().collect::<String>(),
        ));
    }

    // Destination is always the last two chars
    let dest_file = chars[chars.len() - 2];
    let dest_rank = chars[chars.len() - 1];
    if !('a'..='h').contains(&dest_file) || !('1'..='8').contains(&dest_rank) {
        return Err(RejectionReason::MalformedSan(
            chars.iter().collect::<String>(),
        ));
    }
    let dest_sq = sq_of(
        (dest_rank as u8 - b'1') as usize,
        (dest_file as u8 - b'a') as usize,
    );

    // Disambiguation from remaining chars
    let disambig = &chars[..chars.len() - 2];
    let mut disambig_file = None;
    let mut disambig_rank = None;

    for &ch in disambig {
        if ('a'..='h').contains(&ch) {
            disambig_file = Some((ch as u8 - b'a') as usize);
        } else if ('1'..='8').contains(&ch) {
            disambig_rank = Some((ch as u8 - b'1') as usize);
        }
    }

    Ok((piece, disambig_file, disambig_rank, dest_sq))
}

/// Parse pawn SAN: file + optional 'x' + destination.
fn parse_pawn_san(
    chars: &[char],
    _side: i64,
) -> Result<(i64, Option<usize>, Option<usize>, usize), RejectionReason> {
    // Remove 'x' for captures
    let chars: Vec<char> = chars.iter().copied().filter(|&c| c != 'x').collect();

    if chars.len() < 2 {
        return Err(RejectionReason::MalformedSan(
            chars.iter().collect::<String>(),
        ));
    }

    // For pawn captures, first char is disambiguation file
    let disambig_file = if chars.len() > 2 && ('a'..='h').contains(&chars[0]) {
        Some((chars[0] as u8 - b'a') as usize)
    } else {
        None
    };

    // Destination is the last two chars
    let dest_file = chars[chars.len() - 2];
    let dest_rank = chars[chars.len() - 1];
    if !('a'..='h').contains(&dest_file) || !('1'..='8').contains(&dest_rank) {
        return Err(RejectionReason::MalformedSan(
            chars.iter().collect::<String>(),
        ));
    }
    let dest_sq = sq_of(
        (dest_rank as u8 - b'1') as usize,
        (dest_file as u8 - b'a') as usize,
    );

    Ok((1, disambig_file, None, dest_sq)) // piece_type=1 (pawn)
}

// ─── Game Import ───

/// Import a parsed PGN game: resolve all SAN moves to coordinates.
/// Rejects games containing castling or en passant.
pub fn import_game(game: &PgnGame) -> GameResult {
    let mut board = Board::initial();
    let mut side: i64 = 1; // White starts
    let mut resolved_moves = Vec::new();
    let mut board_states = Vec::new();

    // Record initial board state
    board_states.push(board.squares);

    for san in &game.moves {
        // Detect en passant: pawn capture to empty square
        let resolved = resolve_san(&board, san, side)?;

        // En passant detection: pawn moves diagonally to empty square
        let piece = board.at(resolved.from);
        if piece.abs() == 1 {
            let from_file = file_of(resolved.from);
            let to_file = file_of(resolved.to);
            if from_file != to_file && board.at(resolved.to) == EMPTY {
                return Err(RejectionReason::EnPassantDetected);
            }
        }

        // Apply the move
        if let Some(promo) = resolved.promotion {
            board.apply_move_with_promotion(resolved.from, resolved.to, promo);
        } else {
            board.apply_move(resolved.from, resolved.to);
        }
        board_states.push(board.squares);
        resolved_moves.push(resolved);
        side = -side;
    }

    let result = match game.result.as_str() {
        "1-0" => 1.0,
        "0-1" => -1.0,
        _ => 0.0,
    };

    Ok(ImportedGame {
        headers: game.headers.clone(),
        moves: resolved_moves,
        result,
        board_states,
    })
}

/// Import all games from PGN text. Returns (imported, rejected) lists.
pub fn import_pgn(
    pgn_text: &str,
) -> (Vec<ImportedGame>, Vec<(PgnGame, RejectionReason)>) {
    let games = parse_pgn(pgn_text);
    let mut imported = Vec::new();
    let mut rejected = Vec::new();

    for game in games {
        match import_game(&game) {
            Ok(ig) => imported.push(ig),
            Err(reason) => rejected.push((game, reason)),
        }
    }

    (imported, rejected)
}

// ─── JSONL Trace Output ───

/// Generate a JSONL trace string for an imported game.
pub fn game_to_jsonl(game: &ImportedGame) -> String {
    let mut lines = Vec::new();

    // Initial position
    let board_str = game.board_states[0]
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",");
    lines.push(format!(
        r#"{{"ply":0,"board":[{}],"side":1,"source":"import","type":"init"}}"#,
        board_str
    ));

    // Moves
    for (i, mv) in game.moves.iter().enumerate() {
        let ply = i + 1;
        let side = if i % 2 == 0 { 1 } else { -1 };
        let board_str = game.board_states[ply]
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");
        lines.push(format!(
            r#"{{"ply":{},"board":[{}],"side":{},"move":[{},{}],"source":"import","type":"move"}}"#,
            ply, board_str, side, mv.from, mv.to
        ));
    }

    // Result
    lines.push(format!(
        r#"{{"type":"result","result":{},"source":"import"}}"#,
        game.result
    ));

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_square() {
        assert_eq!(parse_square("a1"), Some(0));
        assert_eq!(parse_square("h1"), Some(7));
        assert_eq!(parse_square("a8"), Some(56));
        assert_eq!(parse_square("e4"), Some(28));
        assert_eq!(parse_square("d5"), Some(35));
    }

    #[test]
    fn test_square_name() {
        assert_eq!(square_name(0), "a1");
        assert_eq!(square_name(7), "h1");
        assert_eq!(square_name(56), "a8");
        assert_eq!(square_name(28), "e4");
    }

    #[test]
    fn test_initial_board() {
        let b = Board::initial();
        assert_eq!(b.at(0), W_ROOK);
        assert_eq!(b.at(4), W_KING);
        assert_eq!(b.at(8), W_PAWN);
        assert_eq!(b.at(48), B_PAWN);
        assert_eq!(b.at(60), B_KING);
        assert_eq!(b.at(28), EMPTY);
    }

    #[test]
    fn test_parse_pgn_single_game() {
        let pgn = r#"[Event "Test"]
[White "Alice"]
[Black "Bob"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0
"#;
        let games = parse_pgn(pgn);
        assert_eq!(games.len(), 1);
        assert_eq!(games[0].headers["Event"], "Test");
        assert_eq!(games[0].moves, vec!["e4", "e5", "Nf3", "Nc6"]);
        assert_eq!(games[0].result, "1-0");
    }

    #[test]
    fn test_parse_pgn_multiple_games() {
        let pgn = r#"[Event "Game 1"]
[Result "1-0"]

1. e4 e5 1-0

[Event "Game 2"]
[Result "0-1"]

1. d4 d5 0-1
"#;
        let games = parse_pgn(pgn);
        assert_eq!(games.len(), 2);
        assert_eq!(games[0].headers["Event"], "Game 1");
        assert_eq!(games[1].headers["Event"], "Game 2");
    }

    #[test]
    fn test_resolve_pawn_push() {
        let board = Board::initial();
        let mv = resolve_san(&board, "e4", 1).unwrap();
        assert_eq!(mv.from, sq_of(1, 4)); // e2
        assert_eq!(mv.to, sq_of(3, 4));   // e4
    }

    #[test]
    fn test_resolve_knight_move() {
        let board = Board::initial();
        let mv = resolve_san(&board, "Nf3", 1).unwrap();
        assert_eq!(mv.from, sq_of(0, 6)); // g1
        assert_eq!(mv.to, sq_of(2, 5));   // f3
    }

    #[test]
    fn test_castling_rejected() {
        let board = Board::initial();
        let result = resolve_san(&board, "O-O", 1);
        assert!(matches!(result, Err(RejectionReason::CastlingNotSupported)));
    }

    #[test]
    fn test_import_simple_game() {
        let pgn = r#"[Event "Test"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0
"#;
        let (imported, rejected) = import_pgn(pgn);
        assert_eq!(rejected.len(), 0, "no rejections expected");
        assert_eq!(imported.len(), 1);
        assert_eq!(imported[0].moves.len(), 4);
        assert_eq!(imported[0].result, 1.0);
        // Board states: initial + 4 moves = 5
        assert_eq!(imported[0].board_states.len(), 5);
    }

    #[test]
    fn test_import_rejects_castling() {
        let pgn = r#"[Event "Test"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. O-O Be7 1-0
"#;
        let (imported, rejected) = import_pgn(pgn);
        assert_eq!(imported.len(), 0);
        assert_eq!(rejected.len(), 1);
        assert!(matches!(
            rejected[0].1,
            RejectionReason::CastlingNotSupported
        ));
    }

    #[test]
    fn test_jsonl_output() {
        let pgn = r#"[Event "Test"]
[Result "1-0"]

1. e4 e5 1-0
"#;
        let (imported, _) = import_pgn(pgn);
        let jsonl = game_to_jsonl(&imported[0]);
        let lines: Vec<&str> = jsonl.lines().collect();
        assert_eq!(lines.len(), 4); // init + 2 moves + result
        assert!(lines[0].contains("\"type\":\"init\""));
        assert!(lines[1].contains("\"type\":\"move\""));
        assert!(lines[3].contains("\"type\":\"result\""));
    }
}
