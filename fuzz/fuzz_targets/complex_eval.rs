#![no_main]

use libfuzzer_sys::fuzz_target;

/// Fuzz the CJC evaluator with programs that use Complex numbers.
///
/// Takes arbitrary bytes, wraps them in a Complex-using program template,
/// and executes through the MIR pipeline. The execution must NEVER panic.
fuzz_target!(|data: &[u8]| {
    // Convert bytes to string (lossy).
    let src = String::from_utf8_lossy(data);

    // Try parsing the input as raw CJC source.
    let (program, diag) = cjc_parser::parse_source(&src);

    // Only proceed with execution if parse succeeded without errors.
    if diag.has_errors() {
        return;
    }

    // Execute via MIR — must not panic even on weird programs.
    let _ = cjc_mir_exec::run_program_with_executor(&program, 42);
});
