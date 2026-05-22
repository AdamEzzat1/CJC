//! Runtime Policy Layer — property tests.
//!
//! Properties are asserted against the shared dispatch surface
//! (`cjc_runtime::builtins::dispatch_builtin`), which is the exact path both
//! executors take. Each property runs the proptest floor of 256 cases.
//!
//! Focus areas:
//!   - the energy model is non-negative, finite, monotone, and additive;
//!   - mode setters round-trip through their getters (canonical normalization);
//!   - the advisory batch size round-trips for arbitrary sizes.

use std::rc::Rc;

use proptest::prelude::*;

use cjc_runtime::builtins::dispatch_builtin;
use cjc_runtime::value::Value;

fn energy(flops: i64, bytes: i64) -> f64 {
    match dispatch_builtin("energy_estimate", &[Value::Int(flops), Value::Int(bytes)]) {
        Ok(Some(Value::Float(j))) => j,
        other => panic!("energy_estimate returned {other:?}"),
    }
}

fn reset() {
    dispatch_builtin("runtime_policy_reset", &[]).unwrap();
}

fn set_thermal(s: &str) -> Result<Option<Value>, String> {
    dispatch_builtin(
        "runtime_policy_set_thermal_mode",
        &[Value::String(Rc::new(s.to_string()))],
    )
}

/// Set a thermal mode expected to be valid, returning the canonical spelling
/// the builtin echoes back. (`Value` is not `PartialEq`, so we compare strings.)
fn set_thermal_str(s: &str) -> String {
    match set_thermal(s) {
        Ok(Some(Value::String(r))) => r.to_string(),
        other => panic!("set_thermal_mode(`{s}`) returned {other:?}"),
    }
}

fn get_thermal() -> String {
    match dispatch_builtin("runtime_policy_thermal_mode", &[]) {
        Ok(Some(Value::String(s))) => s.to_string(),
        other => panic!("thermal_mode returned {other:?}"),
    }
}

fn set_batch(n: i64) {
    dispatch_builtin("runtime_policy_set_batch_size", &[Value::Int(n)]).unwrap();
}

fn get_batch() -> i64 {
    match dispatch_builtin("runtime_policy_batch_size", &[]) {
        Ok(Some(Value::Int(n))) => n,
        other => panic!("batch_size returned {other:?}"),
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        ..ProptestConfig::default()
    })]

    /// The energy estimate is always non-negative and finite, even for the
    /// full i64 range (negatives clamp to zero inside the kernel).
    #[test]
    fn energy_non_negative_and_finite(flops in any::<i64>(), bytes in any::<i64>()) {
        let j = energy(flops, bytes);
        prop_assert!(j.is_finite(), "energy not finite for ({flops}, {bytes}): {j}");
        prop_assert!(j >= 0.0, "energy negative for ({flops}, {bytes}): {j}");
    }

    /// More FLOPs (fixed bytes) never decreases the estimate.
    #[test]
    fn energy_monotonic_in_flops(
        f1 in 0i64..2_000_000_000,
        f2 in 0i64..2_000_000_000,
        bytes in 0i64..2_000_000_000,
    ) {
        let lo = f1.min(f2);
        let hi = f1.max(f2);
        prop_assert!(energy(lo, bytes) <= energy(hi, bytes));
    }

    /// More bytes (fixed FLOPs) never decreases the estimate.
    #[test]
    fn energy_monotonic_in_bytes(
        flops in 0i64..2_000_000_000,
        b1 in 0i64..2_000_000_000,
        b2 in 0i64..2_000_000_000,
    ) {
        let lo = b1.min(b2);
        let hi = b1.max(b2);
        prop_assert!(energy(flops, lo) <= energy(flops, hi));
    }

    /// Energy is additive across its two components — the FLOP and byte terms
    /// are independent linear contributions (exact f64 equality, same arithmetic).
    #[test]
    fn energy_is_additive(
        flops in 0i64..2_000_000_000,
        bytes in 0i64..2_000_000_000,
    ) {
        let both = energy(flops, bytes);
        let parts = energy(flops, 0) + energy(0, bytes);
        prop_assert_eq!(both, parts);
    }

    /// Setting a thermal mode (incl. aliases) then reading it back yields the
    /// canonical spelling.
    #[test]
    fn thermal_mode_round_trips(choice in 0usize..5) {
        reset();
        let (input, canonical) = match choice {
            0 => ("cool", "cool"),
            1 => ("balanced", "balanced"),
            2 => ("max-perf", "max-perf"),
            3 => ("maxperf", "max-perf"),
            _ => ("max_perf", "max-perf"),
        };
        prop_assert_eq!(set_thermal_str(input), canonical.to_string());
        prop_assert_eq!(get_thermal(), canonical.to_string());
        reset();
    }

    /// Unknown thermal mode strings are rejected (Err), never silently accepted.
    #[test]
    fn unknown_thermal_mode_is_rejected(s in "[a-z]{1,8}") {
        // Skip the three valid spellings + aliases.
        prop_assume!(!matches!(
            s.as_str(),
            "cool" | "balanced" | "maxperf"
        ));
        reset();
        let before = get_thermal();
        let res = set_thermal(&s);
        prop_assert!(res.is_err(), "expected `{s}` to be rejected, got {res:?}");
        // State is unchanged after a rejected set.
        prop_assert_eq!(get_thermal(), before);
        reset();
    }

    /// The advisory batch size round-trips for arbitrary non-negative sizes.
    #[test]
    fn batch_size_round_trips(n in 0i64..1_000_000) {
        reset();
        set_batch(n);
        prop_assert_eq!(get_batch(), n);
        reset();
    }

    /// Negative batch sizes clamp to zero rather than wrapping to a huge usize.
    #[test]
    fn negative_batch_size_clamps_to_zero(n in i64::MIN..0) {
        reset();
        set_batch(n);
        prop_assert_eq!(get_batch(), 0);
        reset();
    }
}
