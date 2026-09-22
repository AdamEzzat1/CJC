//! The bits of `piml_heat_1d_train`, pinned across the routing of its physics loss to
//! the Bruchion kernel `cjc_heat1d_residual_grad_f64`.
//!
//! Every shape below was trained BEFORE the routing change (the physics loop as it stood,
//! one `sin` per collocation point per epoch, four fresh gradient vectors per epoch) and
//! its digest recorded here. The tests then hold three things:
//!
//! * the unrouted training still gives those bits (the buffer reuse and the precomputed
//!   source term changed no arithmetic);
//! * with the feature built and the switch on, the routed training gives the SAME bits
//!   (the kernel is the loop's arithmetic, on this target, at these shapes);
//! * the comparison can fail: a different seed gives a different digest, and a parameter
//!   vector with one last bit flipped is rejected. A parity test that cannot see a wrong
//!   value proves nothing, so the wrong values are asserted too.
//!
//! The digest is FNV-1a over the bits of the final parameters, every entry of the loss
//! history (all five values), and the three summary numbers, so a one-ulp change in any
//! epoch of any term moves it. `print_the_digests` (ignored) reprints the table; run it
//! with `--ignored --nocapture` to re-pin after an INTENDED change of the arithmetic, and
//! say so in the commit.

use cjc_ad::pinn::{piml_heat_1d_train, PinnResult};

/// (degree, n_data, n_colloc, noise_std, epochs, lr, physics_weight, boundary_weight, seed)
type Shape = (usize, usize, usize, f64, usize, f64, f64, f64, u64);

/// Shapes that cover the kernel's edges: the in-crate determinism test's shape; a wider
/// polynomial at more points; a tiny odd one (four parameters, nine collocation points,
/// no noise); the 1000x9 shape the cjc-runtime parity tests use, few epochs.
const SHAPES: [Shape; 4] = [
    (6, 20, 30, 0.01, 500, 1e-3, 10.0, 100.0, 42),
    (8, 30, 50, 0.01, 200, 1e-3, 1.0, 10.0, 42),
    (3, 7, 9, 0.0, 40, 1e-3, 10.0, 100.0, 7),
    (8, 16, 1000, 0.05, 20, 1e-3, 10.0, 100.0, 1),
];

/// The digests captured before the routing change, one per shape, in order.
/// Captured by `print_the_digests` at CJC `cb27c41` plus the fused mse_grad edit
/// (pinn.rs untouched), rustc 1.97.1, x86_64-pc-windows-msvc, debug profile.
const PINNED: [u64; 4] = [
    PINNED_0, PINNED_1, PINNED_2, PINNED_3,
];
const PINNED_0: u64 = 0x5302981cd111b754;
const PINNED_1: u64 = 0xd175b1dab06cf16c;
const PINNED_2: u64 = 0x3c6d94175d7992d5;
const PINNED_3: u64 = 0x4e2a376f6429ac5f;

fn train(s: Shape) -> PinnResult {
    piml_heat_1d_train(s.0, s.1, s.2, s.3, s.4, s.5, s.6, s.7, s.8)
}

fn fnv1a(bits: impl Iterator<Item = u64>) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bits {
        for byte in b.to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn digest(r: &PinnResult) -> u64 {
    let params = r.final_params.iter().map(|v| v.to_bits());
    let history = r.history.iter().flat_map(|l| {
        [l.total_loss, l.data_loss, l.physics_loss, l.boundary_loss, l.grad_norm]
            .into_iter()
            .map(|v| v.to_bits())
    });
    let summary = [
        r.mean_residual.to_bits(),
        r.l2_error.map_or(u64::MAX, |v| v.to_bits()),
        r.max_error.map_or(u64::MAX, |v| v.to_bits()),
    ]
    .into_iter();
    fnv1a(params.chain(history).chain(summary))
}

fn digest_with_one_bit_flipped(r: &PinnResult) -> u64 {
    let mut r = r.clone();
    let last = r.final_params.len() - 1;
    r.final_params[last] = f64::from_bits(r.final_params[last].to_bits() ^ 1);
    digest(&r)
}

/// Reprints the table; ignored so the pinned values only move on purpose.
#[test]
#[ignore]
fn print_the_digests() {
    for (i, &s) in SHAPES.iter().enumerate() {
        let r = train(s);
        eprintln!("const PINNED_{i}: u64 = 0x{:016x}; // {:?}", digest(&r), s);
    }
}

/// The unrouted training (the switch is off by default; the fallback loop runs whether or
/// not the feature is built) gives the bits captured before the change.
#[test]
fn the_unrouted_training_gives_the_pinned_bits() {
    assert!(!cjc_runtime::bruchion::dispatch::enabled(), "the switch is off by default");
    for (i, &s) in SHAPES.iter().enumerate() {
        let r = train(s);
        assert_eq!(digest(&r), PINNED[i], "shape {i}: {s:?}");
        assert_ne!(digest_with_one_bit_flipped(&r), PINNED[i], "shape {i}: a flipped last bit must be seen");
    }
}

/// The digest tells shapes apart: the same shape under another seed, and the same shape
/// with one more epoch, both differ from the pinned value.
#[test]
fn the_digest_sees_a_wrong_training() {
    let s = SHAPES[2];
    let other_seed = train((s.0, s.1, s.2, s.3, s.4, s.5, s.6, s.7, s.8 + 1));
    assert_ne!(digest(&other_seed), PINNED[2]);
    let one_more_epoch = train((s.0, s.1, s.2, s.3, s.4 + 1, s.5, s.6, s.7, s.8));
    assert_ne!(digest(&one_more_epoch), PINNED[2]);
}

/// With the feature built and the switch on, the physics loss and its gradient come from
/// the Bruchion kernel, and the training gives the pinned bits — and, run back to back,
/// the routed and unrouted results are equal element by element, not only by digest.
#[cfg(feature = "bruchion-kernels")]
#[test]
fn the_routed_training_gives_the_pinned_bits() {
    use cjc_runtime::runtime_policy::set_bruchion_kernels;
    for (i, &s) in SHAPES.iter().enumerate() {
        let off = train(s);
        set_bruchion_kernels(true);
        assert!(cjc_runtime::bruchion::dispatch::enabled());
        let on = train(s);
        set_bruchion_kernels(false);
        assert_eq!(digest(&on), PINNED[i], "routed, shape {i}: {s:?}");
        assert_eq!(digest(&off), digest(&on), "shape {i}");
        for (j, (a, b)) in off.final_params.iter().zip(&on.final_params).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "shape {i}, param {j}");
        }
        for (a, b) in off.history.iter().zip(&on.history) {
            assert_eq!(a.physics_loss.to_bits(), b.physics_loss.to_bits(), "shape {i}, epoch {}", a.epoch);
            assert_eq!(a.grad_norm.to_bits(), b.grad_norm.to_bits(), "shape {i}, epoch {}", a.epoch);
        }
        assert_ne!(digest_with_one_bit_flipped(&on), digest(&off), "shape {i}: a flipped last bit must be seen");
    }
}
