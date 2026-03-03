//! Phase C test C3: Bitwise Operations

fn run_mir(src: &str) -> Vec<String> {
    let (tokens, _) = cjc_lexer::Lexer::new(src).tokenize();
    let (program, _) = cjc_parser::Parser::new(tokens).parse_program();
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    executor.output
}

#[test]
fn c3_and_mask() {
    let out = run_mir(r#"print(bit_and(15, 9));"#); // 0b1111 & 0b1001 = 0b1001 = 9
    assert_eq!(out, vec!["9"]);
}

#[test]
fn c3_or_combine() {
    let out = run_mir(r#"print(bit_or(10, 5));"#); // 0b1010 | 0b0101 = 0b1111 = 15
    assert_eq!(out, vec!["15"]);
}

#[test]
fn c3_xor_toggle() {
    let out = run_mir(r#"print(bit_xor(12, 10));"#); // 0b1100 ^ 0b1010 = 0b0110 = 6
    assert_eq!(out, vec!["6"]);
}

#[test]
fn c3_not_complement() {
    let out = run_mir(r#"print(bit_not(0));"#); // !0 = -1 (all ones)
    assert_eq!(out, vec!["-1"]);
}

#[test]
fn c3_shl_basic() {
    let out = run_mir(r#"print(bit_shl(1, 3));"#); // 1 << 3 = 8
    assert_eq!(out, vec!["8"]);
}

#[test]
fn c3_shr_basic() {
    let out = run_mir(r#"print(bit_shr(16, 2));"#); // 16 >> 2 = 4
    assert_eq!(out, vec!["4"]);
}

#[test]
fn c3_popcount_allones() {
    // -1 as i64 is all ones (64 bits) → u64 all ones → popcount = 64
    let out = run_mir(r#"print(popcount(-1));"#);
    assert_eq!(out, vec!["64"]);
}

#[test]
fn c3_popcount_zero() {
    let out = run_mir(r#"print(popcount(0));"#);
    assert_eq!(out, vec!["0"]);
}

#[test]
fn c3_popcount_specific() {
    let out = run_mir(r#"print(popcount(255));"#); // 8 ones
    assert_eq!(out, vec!["8"]);
}

#[test]
fn c3_determinism() {
    let src = r#"
let a = bit_shl(1, 7);
let b = bit_or(a, 42);
let c = popcount(b);
print(c);
"#;
    let out1 = run_mir(src);
    let out2 = run_mir(src);
    assert_eq!(out1, out2);
}

#[test]
fn c3_chess_bitboard() {
    // Typical chess bitboard operation: set bit at position, test it
    let out = run_mir(r#"
let board = 0;
let board = bit_or(board, bit_shl(1, 0));
let board = bit_or(board, bit_shl(1, 7));
let board = bit_or(board, bit_shl(1, 63));
print(popcount(board));
let has_corner = bit_and(board, bit_shl(1, 0));
print(has_corner);
"#);
    assert_eq!(out[0], "3"); // 3 bits set
    assert_eq!(out[1], "1"); // bit 0 is set
}
