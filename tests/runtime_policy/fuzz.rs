//! Runtime Policy Layer — bolero fuzz targets.
//!
//! Three harnesses:
//!
//! 1. `fuzz_runtime_policy_structural` — decode a random byte sequence into a
//!    sequence of policy dispatch calls (reset / set_threads / set_batch_size /
//!    queries / energy). The contract: no input corrupts the dispatch layer's
//!    bookkeeping — after any sequence the policy is still readable.
//!
//! 2. `fuzz_runtime_policy_mode_strings` — feed arbitrary bytes as a mode string
//!    to each of the three string setters. Valid spellings return `Ok(Some(String))`;
//!    everything else returns `Err`. Neither path may panic, and the dispatch
//!    surface must stay usable afterwards.
//!
//! 3. `fuzz_energy_estimate` — feed arbitrary `[i64; 2]` to `energy_estimate`
//!    and assert the result is always a finite, non-negative `Float`.
//!
//! Both `bolero::check!` harnesses compile to proptest on Windows/macOS and to
//! libfuzzer/AFL under `cargo bolero`.

use std::rc::Rc;

use bolero::check;

use cjc_runtime::builtins::dispatch_builtin;
use cjc_runtime::value::Value;

#[test]
fn fuzz_runtime_policy_structural() {
    check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        let _ = dispatch_builtin("runtime_policy_reset", &[]);
        let max = input.len().min(64);
        let mut i = 0;
        while i < max {
            let op = input[i] % 8;
            let arg = if i + 1 < max { input[i + 1] as i64 } else { 0 };
            i += 2;
            match op {
                0 => {
                    let _ = dispatch_builtin("runtime_policy_reset", &[]);
                }
                1 => {
                    let _ = dispatch_builtin("runtime_policy_set_threads", &[Value::Int(arg)]);
                }
                2 => {
                    let _ = dispatch_builtin("runtime_policy_set_batch_size", &[Value::Int(arg)]);
                }
                3 => {
                    let _ = dispatch_builtin("runtime_policy_thermal_mode", &[]);
                }
                4 => {
                    let _ = dispatch_builtin("runtime_policy_threads", &[]);
                }
                5 => {
                    let _ = dispatch_builtin("runtime_policy_batch_size", &[]);
                }
                6 => {
                    let _ = dispatch_builtin("runtime_policy_summary", &[]);
                }
                _ => {
                    let _ = dispatch_builtin(
                        "energy_estimate",
                        &[Value::Int(arg), Value::Int(arg)],
                    );
                }
            }
        }
        // The dispatch surface must remain usable after any sequence.
        let alive = dispatch_builtin("runtime_policy_batch_size", &[]);
        assert!(
            matches!(alive, Ok(Some(Value::Int(_)))),
            "dispatch corrupted after structural fuzz: {alive:?}",
        );
    });
}

#[test]
fn fuzz_runtime_policy_mode_strings() {
    check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        let _ = dispatch_builtin("runtime_policy_reset", &[]);
        if input.is_empty() {
            return;
        }
        let name = match input[0] % 3 {
            0 => "runtime_policy_set_thermal_mode",
            1 => "runtime_policy_set_audit_mode",
            _ => "runtime_policy_set_numeric_mode",
        };
        let s = String::from_utf8_lossy(&input[1..]).to_string();
        let res = dispatch_builtin(name, &[Value::String(Rc::new(s.clone()))]);
        match res {
            // Accepted a valid spelling — returns the canonical mode string.
            Ok(Some(Value::String(_))) => {}
            // Rejected an unknown spelling — the expected error path.
            Err(_) => {}
            // A known builtin must never return Ok(None) or a non-string.
            other => panic!("{name}(`{s}`) returned unexpected {other:?}"),
        }
        // Dispatch must stay usable after any input.
        let alive = dispatch_builtin("runtime_policy_thermal_mode", &[]);
        assert!(
            matches!(alive, Ok(Some(Value::String(_)))),
            "dispatch corrupted after mode-string fuzz `{s}`: {alive:?}",
        );
    });
}

#[test]
fn fuzz_energy_estimate() {
    check!().with_type::<[i64; 2]>().for_each(|arr: &[i64; 2]| {
        let res = dispatch_builtin("energy_estimate", &[Value::Int(arr[0]), Value::Int(arr[1])]);
        match res {
            Ok(Some(Value::Float(j))) => {
                assert!(j.is_finite(), "energy not finite for {arr:?}: {j}");
                assert!(j >= 0.0, "energy negative for {arr:?}: {j}");
            }
            other => panic!("energy_estimate returned {other:?} for {arr:?}"),
        }
    });
}
