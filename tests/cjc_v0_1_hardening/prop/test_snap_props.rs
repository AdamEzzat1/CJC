//! Property-based tests for snap serialization roundtrip.

use proptest::prelude::*;
use std::rc::Rc;
use cjc_runtime::Value;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Snap encode → decode is identity for integers.
    #[test]
    fn snap_roundtrip_int(n in proptest::num::i64::ANY) {
        let val = Value::Int(n);
        let encoded = cjc_snap::snap_encode(&val);
        let decoded = cjc_snap::snap_decode(&encoded);
        match decoded {
            Ok(Value::Int(v)) => prop_assert_eq!(v, n),
            Ok(other) => prop_assert!(false, "Expected Int, got {:?}", other),
            Err(e) => prop_assert!(false, "Decode failed: {}", e),
        }
    }

    /// Snap encode → decode is identity for finite floats (bit-exact).
    #[test]
    fn snap_roundtrip_float(f in prop::num::f64::NORMAL.prop_filter("finite", |x| x.is_finite())) {
        let val = Value::Float(f);
        let encoded = cjc_snap::snap_encode(&val);
        let decoded = cjc_snap::snap_decode(&encoded);
        match decoded {
            Ok(Value::Float(v)) => prop_assert_eq!(v.to_bits(), f.to_bits()),
            Ok(other) => prop_assert!(false, "Expected Float, got {:?}", other),
            Err(e) => prop_assert!(false, "Decode failed: {}", e),
        }
    }

    /// Snap encode → decode is identity for booleans.
    #[test]
    fn snap_roundtrip_bool(b: bool) {
        let val = Value::Bool(b);
        let encoded = cjc_snap::snap_encode(&val);
        let decoded = cjc_snap::snap_decode(&encoded);
        match decoded {
            Ok(Value::Bool(v)) => prop_assert_eq!(v, b),
            Ok(other) => prop_assert!(false, "Expected Bool, got {:?}", other),
            Err(e) => prop_assert!(false, "Decode failed: {}", e),
        }
    }

    /// Snap encode → decode is identity for strings.
    #[test]
    fn snap_roundtrip_string(s in "[ -~]{0,200}") {
        let val = Value::String(Rc::new(s.clone()));
        let encoded = cjc_snap::snap_encode(&val);
        let decoded = cjc_snap::snap_decode(&encoded);
        match decoded {
            Ok(Value::String(v)) => prop_assert_eq!(v.as_ref(), s.as_str()),
            Ok(other) => prop_assert!(false, "Expected String, got {:?}", other),
            Err(e) => prop_assert!(false, "Decode failed: {}", e),
        }
    }

    /// Snap encoding is deterministic: same input → same bytes.
    #[test]
    fn snap_encoding_deterministic(n in proptest::num::i64::ANY) {
        let val = Value::Int(n);
        let e1 = cjc_snap::snap_encode(&val);
        let e2 = cjc_snap::snap_encode(&val);
        prop_assert_eq!(e1, e2);
    }

    /// Arbitrary byte sequences don't crash the decoder.
    ///
    /// snap_decode used to trust length prefixes, so this test wrapped the call
    /// in catch_unwind and kept inputs under 8 bytes. That was not enough: an
    /// allocation failure aborts rather than panics, and 5 bytes
    /// (`[TAG_CATEGORICAL, u32 n_levels]`) could request ~57 GB, which killed
    /// the whole test binary on Windows CI. Counts are now capped by the
    /// remaining input (crates/cjc-snap/tests/decode_untrusted_lengths.rs), so
    /// a panic here is a real failure and inputs can be long.
    #[test]
    fn snap_decode_no_panic(bytes in proptest::collection::vec(0u8..=255, 0..256)) {
        let _ = cjc_snap::snap_decode(&bytes);
        let _ = cjc_snap::snap_decode_v2(&bytes);
    }

    /// Same, with a valid tag first so random bytes reach every decoder arm
    /// (a uniformly random first byte is an invalid tag ~90% of the time).
    #[test]
    fn snap_decode_no_panic_after_valid_tag(
        tag in 0u8..=0x18,
        rest in proptest::collection::vec(0u8..=255, 0..256),
    ) {
        let mut bytes = vec![tag];
        bytes.extend_from_slice(&rest);
        let _ = cjc_snap::snap_decode(&bytes);
    }
}
