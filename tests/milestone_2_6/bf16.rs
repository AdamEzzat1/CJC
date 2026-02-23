// Milestone 2.6 — Bf16 (Brain Float 16) Runtime Tests
//
// Tests for the Bf16 type at the runtime level:
// - from_f32 / to_f32 round-trip
// - Arithmetic operations (add, sub, mul, div)
// - Golden bit patterns
// - Special values (0, -1, infinity)
// - Precision loss characteristics
// - Value::Bf16 display formatting

use cjc_runtime::{Bf16, Value};

// ---------------------------------------------------------------------------
// Round-trip conversion
// ---------------------------------------------------------------------------

#[test]
fn bf16_roundtrip_one() {
    let b = Bf16::from_f32(1.0);
    let f = b.to_f32();
    assert!((f - 1.0).abs() < 1e-6, "1.0 round-trip failed: got {}", f);
}

#[test]
fn bf16_roundtrip_negative() {
    let b = Bf16::from_f32(-3.5);
    let f = b.to_f32();
    assert!(
        (f - (-3.5)).abs() < 0.1,
        "-3.5 round-trip failed: got {}",
        f
    );
}

#[test]
fn bf16_roundtrip_zero() {
    let b = Bf16::from_f32(0.0);
    let f = b.to_f32();
    assert_eq!(f, 0.0, "0.0 round-trip failed");
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

#[test]
fn bf16_add() {
    let a = Bf16::from_f32(1.0);
    let b = Bf16::from_f32(2.0);
    let c = a.add(b);
    let result = c.to_f32();
    assert!(
        (result - 3.0).abs() < 0.1,
        "1.0 + 2.0 = {}, expected ~3.0",
        result
    );
}

#[test]
fn bf16_sub() {
    let a = Bf16::from_f32(5.0);
    let b = Bf16::from_f32(2.0);
    let c = a.sub(b);
    let result = c.to_f32();
    assert!(
        (result - 3.0).abs() < 0.1,
        "5.0 - 2.0 = {}, expected ~3.0",
        result
    );
}

#[test]
fn bf16_mul() {
    let a = Bf16::from_f32(3.0);
    let b = Bf16::from_f32(4.0);
    let c = a.mul(b);
    let result = c.to_f32();
    assert!(
        (result - 12.0).abs() < 0.1,
        "3.0 * 4.0 = {}, expected ~12.0",
        result
    );
}

#[test]
fn bf16_div() {
    let a = Bf16::from_f32(10.0);
    let b = Bf16::from_f32(4.0);
    let c = a.div(b);
    let result = c.to_f32();
    assert!(
        (result - 2.5).abs() < 0.1,
        "10.0 / 4.0 = {}, expected ~2.5",
        result
    );
}

// ---------------------------------------------------------------------------
// Golden bit patterns
// ---------------------------------------------------------------------------

#[test]
fn bf16_golden_bit_pattern_one() {
    // f32 1.0 = 0x3F800000
    // bf16 truncates lower 16 bits: 0x3F80
    let b = Bf16::from_f32(1.0);
    assert_eq!(
        b.0, 0x3F80,
        "Bf16(1.0) should be 0x3F80, got 0x{:04X}",
        b.0
    );
}

#[test]
fn bf16_golden_bit_pattern_neg_one() {
    // f32 -1.0 = 0xBF800000
    // bf16: 0xBF80
    let b = Bf16::from_f32(-1.0);
    assert_eq!(
        b.0, 0xBF80,
        "Bf16(-1.0) should be 0xBF80, got 0x{:04X}",
        b.0
    );
}

// ---------------------------------------------------------------------------
// Special values
// ---------------------------------------------------------------------------

#[test]
fn bf16_zero_bit_pattern() {
    let b = Bf16::from_f32(0.0);
    assert_eq!(b.0, 0x0000, "Bf16(0.0) should be 0x0000");
}

#[test]
fn bf16_infinity_handling() {
    let b = Bf16::from_f32(f32::INFINITY);
    let f = b.to_f32();
    assert!(f.is_infinite() && f > 0.0, "positive infinity round-trip");

    let b_neg = Bf16::from_f32(f32::NEG_INFINITY);
    let f_neg = b_neg.to_f32();
    assert!(
        f_neg.is_infinite() && f_neg < 0.0,
        "negative infinity round-trip"
    );
}

// ---------------------------------------------------------------------------
// Precision loss
// ---------------------------------------------------------------------------

#[test]
fn bf16_precision_loss_vs_f32() {
    // bf16 has only ~7 bits of mantissa (vs 23 for f32)
    // 1.0001 in f32 is precise, but bf16 should round it to ~1.0
    let precise_f32: f32 = 1.0001;
    let b = Bf16::from_f32(precise_f32);
    let recovered = b.to_f32();

    // The bf16 representation should NOT preserve the 0.0001 difference
    // (it only has ~2-3 decimal digits of precision)
    let error = (recovered - precise_f32).abs();
    // bf16 should lose precision; recovered should differ from original
    // by more than f32 epsilon, but be close to 1.0
    assert!(
        (recovered - 1.0).abs() < 0.02,
        "bf16(1.0001) should be close to 1.0, got {}",
        recovered
    );
    // But the error relative to 1.0001 should be nonzero
    // (unless by coincidence the truncation lands exactly)
    assert!(
        error > f32::EPSILON || recovered == 1.0,
        "expected precision loss or exact 1.0, error={}",
        error
    );
}

// ---------------------------------------------------------------------------
// Value::Bf16 display
// ---------------------------------------------------------------------------

#[test]
fn bf16_value_display() {
    let val = Value::Bf16(Bf16::from_f32(1.0));
    let displayed = format!("{}", val);
    assert_eq!(displayed, "1", "Value::Bf16(1.0) should display as '1'");
}

#[test]
fn bf16_neg_operation() {
    let a = Bf16::from_f32(7.0);
    let b = a.neg();
    let result = b.to_f32();
    assert!(
        (result - (-7.0)).abs() < 0.1,
        "neg(7.0) = {}, expected ~-7.0",
        result
    );
}
