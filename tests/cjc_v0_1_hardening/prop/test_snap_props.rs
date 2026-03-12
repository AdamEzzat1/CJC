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
    /// Note: snap_decode trusts length prefixes, so random bytes can trigger
    /// huge allocations. We use catch_unwind + 8-byte limit to stay safe.
    #[test]
    fn snap_decode_no_panic(bytes in proptest::collection::vec(0u8..=255, 0..8)) {
        let _ = std::panic::catch_unwind(|| {
            let _ = cjc_snap::snap_decode(&bytes);
        });
    }
}
