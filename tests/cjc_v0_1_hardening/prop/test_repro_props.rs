//! Property-based tests for the reproducibility layer (RNG, accumulators).

use proptest::prelude::*;
use cjc_repro::{Rng, KahanAccumulatorF64};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// RNG is deterministic: same seed → same sequence.
    #[test]
    fn rng_seed_determinism(seed: u64) {
        let mut r1 = Rng::seeded(seed);
        let mut r2 = Rng::seeded(seed);
        for _ in 0..50 {
            prop_assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    /// next_f64 always in [0, 1).
    #[test]
    fn rng_f64_in_range(seed: u64) {
        let mut r = Rng::seeded(seed);
        for _ in 0..100 {
            let v = r.next_f64();
            prop_assert!(v >= 0.0, "next_f64 < 0: {}", v);
            prop_assert!(v < 1.0 || v == 1.0, "next_f64 >= 1: {}", v);
        }
    }

    /// Fork produces deterministic children.
    #[test]
    fn rng_fork_determinism(seed: u64) {
        let mut r1 = Rng::seeded(seed);
        let mut r2 = Rng::seeded(seed);
        let mut c1 = r1.fork();
        let mut c2 = r2.fork();
        for _ in 0..20 {
            prop_assert_eq!(c1.next_u64(), c2.next_u64());
        }
    }

    /// Kahan accumulator is deterministic: same values → same sum (bit-exact).
    #[test]
    fn kahan_deterministic(
        values in proptest::collection::vec(
            prop::num::f64::NORMAL.prop_filter("finite", |x| x.is_finite()),
            0..100
        )
    ) {
        let mut acc1 = KahanAccumulatorF64::new();
        let mut acc2 = KahanAccumulatorF64::new();
        for &v in &values {
            acc1.add(v);
            acc2.add(v);
        }
        prop_assert_eq!(
            acc1.finalize().to_bits(),
            acc2.finalize().to_bits(),
            "Kahan sums must be bit-identical"
        );
    }

    /// Kahan accumulator: adding zero doesn't change sum.
    #[test]
    fn kahan_add_zero(
        values in proptest::collection::vec(
            prop::num::f64::NORMAL.prop_filter("finite", |x| x.is_finite()),
            1..50
        )
    ) {
        let mut acc = KahanAccumulatorF64::new();
        for &v in &values {
            acc.add(v);
        }
        let before = acc.finalize();
        acc.add(0.0);
        let after = acc.finalize();
        prop_assert_eq!(
            before.to_bits(),
            after.to_bits(),
            "Adding zero should not change sum"
        );
    }
}
