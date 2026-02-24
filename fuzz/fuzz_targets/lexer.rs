#![no_main]

use libfuzzer_sys::fuzz_target;
use cjc_lexer::Lexer;

/// Fuzz the CJC lexer with arbitrary byte input.
///
/// The lexer must NEVER panic on any input. It may produce error tokens
/// or diagnostics, but must always return without crashing.
fuzz_target!(|data: &[u8]| {
    // Convert bytes to string (lossy — allows testing UTF-8 boundary behavior).
    let src = String::from_utf8_lossy(data);

    // Lex — must not panic.
    let _ = Lexer::new(&src).tokenize();
});
