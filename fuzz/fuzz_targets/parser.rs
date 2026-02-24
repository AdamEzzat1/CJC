#![no_main]

use libfuzzer_sys::fuzz_target;

/// Fuzz the CJC parser with arbitrary byte input.
///
/// The parser must NEVER panic on any input. It may produce diagnostics/errors,
/// but must always return without crashing.
fuzz_target!(|data: &[u8]| {
    // Convert bytes to string (lossy — allows testing UTF-8 boundary behavior).
    let src = String::from_utf8_lossy(data);

    // Parse — must not panic.
    let _ = cjc_parser::parse_source(&src);
});
