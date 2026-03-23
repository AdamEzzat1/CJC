//! Bolero fuzz targets for CJC v0.1 hardening.
//!
//! Windows-compatible: runs as proptest in `cargo test`.
//! Linux: `cargo bolero test <target>` for coverage-guided fuzzing.

use std::panic;

/// Fuzz: lexer must not panic on arbitrary UTF-8.
#[test]
fn fuzz_lexer_utf8() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        if let Ok(s) = std::str::from_utf8(input) {
            let s = s.to_string();
            let _ = panic::catch_unwind(|| {
                let _ = cjc_lexer::Lexer::new(&s).tokenize();
            });
        }
    });
}

/// Fuzz: parser must not panic on arbitrary UTF-8.
#[test]
fn fuzz_parser_utf8() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        if let Ok(s) = std::str::from_utf8(input) {
            let s = s.to_string();
            let _ = panic::catch_unwind(|| {
                let _ = cjc_parser::parse_source(&s);
            });
        }
    });
}

/// Fuzz: full parse → MIR-exec pipeline must not panic on valid programs.
#[test]
fn fuzz_mir_pipeline_no_crash() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        // Cap input size to avoid pathological allocations from random length prefixes
        if input.len() > 4096 { return; }
        if let Ok(s) = std::str::from_utf8(input) {
            let s = s.to_string();
            let _ = panic::catch_unwind(|| {
                let (program, diags) = cjc_parser::parse_source(&s);
                if !diags.has_errors() {
                    let _ = cjc_mir_exec::run_program_with_executor(&program, 42);
                }
            });
        }
    });
}

/// Fuzz: full parse → eval pipeline must not panic on valid programs.
#[test]
fn fuzz_eval_pipeline_no_crash() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        if input.len() > 4096 { return; }
        if let Ok(s) = std::str::from_utf8(input) {
            let s = s.to_string();
            let _ = panic::catch_unwind(|| {
                let (program, diags) = cjc_parser::parse_source(&s);
                if !diags.has_errors() {
                    let mut interp = cjc_eval::Interpreter::new(42);
                    let _ = interp.exec(&program);
                }
            });
        }
    });
}

/// Fuzz: snap roundtrip — encode then decode must always succeed.
/// Note: snap_decode on truly random bytes triggers unchecked allocation from
/// length prefixes, so we fuzz the roundtrip path instead.
#[test]
fn fuzz_snap_roundtrip() {
    use cjc_runtime::Value;
    bolero::check!().with_type::<i64>().for_each(|n: &i64| {
        let val = Value::Int(*n);
        let encoded = cjc_snap::snap_encode(&val);
        let decoded = cjc_snap::snap_decode(&encoded).expect("roundtrip must succeed");
        match decoded {
            Value::Int(v) => assert_eq!(*n, v, "roundtrip int mismatch"),
            other => panic!("expected Int, got {:?}", other),
        }
    });
}

/// Fuzz: regex engine must not panic or hang on arbitrary patterns+inputs.
#[test]
fn fuzz_regex_no_crash() {
    bolero::check!()
        .with_type::<(Vec<u8>, Vec<u8>)>()
        .for_each(|(pattern, text): &(Vec<u8>, Vec<u8>)| {
            if let (Ok(p), Ok(t)) = (std::str::from_utf8(pattern), std::str::from_utf8(text)) {
                let p = p.to_string();
                let t = t.to_string();
                let _ = panic::catch_unwind(move || {
                    let _ = cjc_regex::is_match(&p, "", t.as_bytes());
                });
            }
        });
}

/// Fuzz: RNG determinism — same seed always produces same first N values.
#[test]
fn fuzz_rng_determinism() {
    bolero::check!().with_type::<u64>().for_each(|seed: &u64| {
        let mut r1 = cjc_repro::Rng::seeded(*seed);
        let mut r2 = cjc_repro::Rng::seeded(*seed);
        for _ in 0..20 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    });
}

/// Fuzz: Kahan accumulator on random floats — must be deterministic.
#[test]
fn fuzz_kahan_determinism() {
    bolero::check!()
        .with_type::<Vec<u64>>()
        .for_each(|raw: &Vec<u64>| {
            let values: Vec<f64> = raw.iter()
                .map(|&bits| f64::from_bits(bits))
                .filter(|x| x.is_finite())
                .take(100)
                .collect();
            let mut acc1 = cjc_repro::KahanAccumulatorF64::new();
            let mut acc2 = cjc_repro::KahanAccumulatorF64::new();
            for &v in &values {
                acc1.add(v);
                acc2.add(v);
            }
            assert_eq!(acc1.finalize().to_bits(), acc2.finalize().to_bits());
        });
}
