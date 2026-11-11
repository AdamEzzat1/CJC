//! Log-scale coordinate mapping must use `cjc_repro::dmath`, like the rest of
//! the layout engine (which already uses `det_powf`/`det_powi`).
//!
//! It used `f64::log(x, base)`, which is platform `ln(x) / ln(base)`. That has
//! two problems:
//! * The bits depend on the platform libm, so the same plot could get
//!   different pixel coordinates on Windows, Linux and macOS.
//! * Exact decades are not exact: `ln(1000)/ln(10)` is 2.9999999999999996,
//!   so a point at 1000 on a [1, 1e6] base-10 axis is drawn off its tick.
//!
//! Oracle: for decades on a base-10 or base-2 axis the exact pixel position is
//! known in closed form, `plot_x + k/n * plot_w` with integers k and n.
//! For other bases the oracle is the same formula computed with dmath directly.

use cjc_repro::dmath::DetMath;
use cjc_vizor::layout::{log_ticks, LayoutResult};

fn layout(x_min: f64, x_max: f64, base: f64) -> LayoutResult {
    LayoutResult {
        plot_x: 50.0,
        plot_y: 20.0,
        plot_w: 600.0,
        plot_h: 400.0,
        x_min,
        x_max,
        y_min: x_min,
        y_max: x_max,
        x_ticks: vec![],
        y_ticks: vec![],
        x_discrete: false,
        x_log: Some(base),
        y_log: Some(base),
    }
}

#[test]
fn base10_decades_land_exactly_on_ticks() {
    let l = layout(1.0, 1e6, 10.0);
    for k in 0..=6 {
        let x = 10f64.det_powi(k);
        let want_x = l.plot_x + (k as f64) / 6.0 * l.plot_w;
        let want_y = l.plot_y + l.plot_h - (k as f64) / 6.0 * l.plot_h;
        assert_eq!(l.map_x(x), want_x, "map_x(1e{k})");
        assert_eq!(l.map_y(x), want_y, "map_y(1e{k})");
    }
}

#[test]
fn base2_powers_land_exactly_on_ticks() {
    let l = layout(1.0, 1024.0, 2.0);
    for k in 0..=10 {
        let x = 2f64.det_powi(k);
        let want = l.plot_x + (k as f64) / 10.0 * l.plot_w;
        assert_eq!(l.map_x(x), want, "map_x(2^{k})");
    }
}

/// Bases other than 2 and 10 use `ln(x)/ln(base)`, computed with dmath.
#[test]
fn other_bases_match_dmath_bit_for_bit() {
    for &base in &[std::f64::consts::E, 3.0, 5.0] {
        let l = layout(0.5, 7.0e4, base);
        let lg = |v: f64| v.max(1e-15).det_ln() / base.det_ln();
        let (lo, hi) = (lg(l.x_min), lg(l.x_max));
        let mut x = 0.5;
        while x < 7.0e4 {
            let want = l.plot_x + (lg(x) - lo) / (hi - lo) * l.plot_w;
            assert_eq!(l.map_x(x).to_bits(), want.to_bits(), "base {base}, x {x}");
            x *= 1.37;
        }
    }
}

/// Ticks were already right (the candidate filter hides the floor/ceil
/// rounding); this pins that down so routing through dmath does not change it.
#[test]
fn log_ticks_unchanged_at_exact_decades() {
    assert_eq!(log_ticks(1000.0, 1e6, 10.0), vec![1e3, 1e4, 1e5, 1e6]);
    assert_eq!(log_ticks(1.0, 1000.0, 10.0), vec![1.0, 10.0, 100.0, 1000.0]);
}
