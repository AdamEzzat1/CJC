//! Training corpus for the LinearCostModel — **v2 (60-program expansion).**
//!
//! Revised after the first-run findings (RMSE 0.13-0.18 vs mean benefits
//! 0.06-0.12 — noise floor exceeded signal). Two structural changes:
//!
//!   1. **Workload scaling.** Inner-loop iteration counts are scaled up so
//!      most programs run for ≥ 1ms wall-clock, putting per-pass benefits
//!      well above the 5-20μs scheduler-noise floor.
//!
//!   2. **Corpus expansion** from 18 → 60 programs. The new programs densify
//!      the feature space across (`expr_count`, `max_loop_depth`,
//!      `branch_count`, `alloc_sites`) corners and across pass-affinity
//!      categories. Each pass now has ~10-15 programs where it should
//!      dominate, vs 2-4 before.
//!
//! Feature-space coverage targets:
//!
//!   | Axis               | First run    | v2 corpus     |
//!   |--------------------|--------------|---------------|
//!   | max_loop_depth     | {0,1,2,3,4}  | {0,1,2,3,4}   |
//!   | expr_count buckets | small only   | 0-20 / 20-50 / 50-100 / 100+ |
//!   | branch_count       | 0-3 typical  | 0-10 typical, peak 12 |
//!   | alloc_sites        | 0 across all | 0-2 (still mostly 0 — language has no easy heap surface) |
//!
//! Programs use single-return shape (let-bound if/else) throughout — the
//! `if cond { return x; } ... return y;` shape used to trigger the
//! dominators OOB (task_9d7ae8b2, now fixed but the convention is
//! documented in the original cana_pass_ordering bench).

pub struct Program {
    pub name: &'static str,
    pub source: &'static str,
    /// Pass we expect to deliver the biggest benefit on this program.
    /// Used to sanity-check the fitted coefficients downstream.
    #[allow(dead_code)]
    pub expected_dominant_pass: &'static str,
}

// ============================================================================
// CF-favoring programs (constant arithmetic — chain folds collapse to literals)
// ============================================================================

const PROG_ARITH_TINY: &str = r#"
fn compute() -> i64 {
    let a: i64 = 5 + 3;
    return a * 2;
}
print(compute());
"#;

const PROG_ARITH_MED: &str = r#"
fn compute(n: i64) -> i64 {
    let a: i64 = 10 * 5 + 2;
    let b: i64 = (a + 100) * 2;
    let c: i64 = b - 50 + n;
    return c + a + b;
}
print(compute(7));
"#;

const PROG_ARITH_HEAVY: &str = r#"
fn compute(n: i64) -> i64 {
    let a: i64 = 1 + 2 + 3 + 4 + 5;
    let b: i64 = 10 * 20 - 30 + 40;
    let c: i64 = a * b;
    let d: i64 = c + a - b;
    let e: i64 = d * 2 + a;
    return e + n;
}
print(compute(11));
"#;

/// Many independent constant chains (~30 foldable subexpressions).
const PROG_CF_FORTY: &str = r#"
fn compute(n: i64) -> i64 {
    let a1: i64 = 1 + 2; let a2: i64 = 3 + 4; let a3: i64 = 5 + 6;
    let a4: i64 = 7 + 8; let a5: i64 = 9 + 10; let a6: i64 = 11 + 12;
    let b1: i64 = a1 * 2; let b2: i64 = a2 * 3; let b3: i64 = a3 * 4;
    let b4: i64 = a4 * 5; let b5: i64 = a5 * 6; let b6: i64 = a6 * 7;
    let c1: i64 = b1 + b2; let c2: i64 = b3 + b4; let c3: i64 = b5 + b6;
    return c1 + c2 + c3 + n;
}
print(compute(99));
"#;

/// Many constant-arithmetic operations packed into a single loop body.
/// CF folds the body; LICM hoists what remains; the loop runs unchanged.
const PROG_CF_IN_HOT_LOOP: &str = r#"
fn compute(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let k: i64 = 2 * 3 + 4 * 5 - 1;
        total = total + k + i;
        i = i + 1;
    }
    return total;
}
print(compute(5000));
"#;

/// CF on integer constants ≠ CF on float constants (different reduction
/// axes). Float version exercises the `StrictFold` reduction-axis path.
const PROG_CF_PURE_FLOAT: &str = r#"
fn compute(x: f64) -> f64 {
    let a: f64 = 1.5 + 2.5;
    let b: f64 = 3.0 * 4.0;
    let c: f64 = a * b - 1.0;
    return c + x;
}
print(compute(0.5));
"#;

// ============================================================================
// LICM-favoring programs (loops with hoistable invariants)
// ============================================================================

const PROG_LOOP_INVARIANT_BIG: &str = r#"
fn loop_inv(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        total = total + n * 10 + i;
        i = i + 1;
    }
    return total;
}
print(loop_inv(50000));
"#;

const PROG_LOOP_NESTED2_BIG: &str = r#"
fn nested2(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let mut j: i64 = 0;
        while j < n {
            total = total + i * j + n;
            j = j + 1;
        }
        i = i + 1;
    }
    return total;
}
print(nested2(200));
"#;

const PROG_LOOP_NESTED3_BIG: &str = r#"
fn nested3(n: i64) -> i64 {
    let mut t: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let mut j: i64 = 0;
        while j < n {
            let mut k: i64 = 0;
            while k < n {
                t = t + i + j + k;
                k = k + 1;
            }
            j = j + 1;
        }
        i = i + 1;
    }
    return t;
}
print(nested3(30));
"#;

const PROG_LOOP_NESTED4_BIG: &str = r#"
fn nested4(n: i64) -> i64 {
    let mut t: i64 = 0;
    let mut a: i64 = 0;
    while a < n {
        let mut b: i64 = 0;
        while b < n {
            let mut c: i64 = 0;
            while c < n {
                let mut d: i64 = 0;
                while d < n {
                    t = t + 1;
                    d = d + 1;
                }
                c = c + 1;
            }
            b = b + 1;
        }
        a = a + 1;
    }
    return t;
}
print(nested4(10));
"#;

/// Loop with multiple hoistable invariants. Each iteration recomputes
/// (k*k), (k+m), (m*m). LICM moves all three to the pre-header.
const PROG_LICM_MANY_INV: &str = r#"
fn many_inv(n: i64, k: i64, m: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        total = total + (k * k) + (k + m) + (m * m) + i;
        i = i + 1;
    }
    return total;
}
print(many_inv(20000, 7, 3));
"#;

/// LICM with conditional invariant hoisting opportunity.
const PROG_LICM_BRANCHY: &str = r#"
fn branchy(n: i64, k: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let v: i64 = if k > 0 { k * k + i } else { -k * k + i };
        total = total + v;
        i = i + 1;
    }
    return total;
}
print(branchy(15000, 5));
"#;

/// Long outer loop, very small inner body. LICM hoists the inner
/// constant expression out of the hot loop.
const PROG_LICM_LONG_OUTER: &str = r#"
fn long_outer(n: i64, k: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        total = total + k * 100;
        i = i + 1;
    }
    return total;
}
print(long_outer(100000, 13));
"#;

// ============================================================================
// CSE-favoring programs (repeated subexpressions)
// ============================================================================

const PROG_CSE_REPEAT: &str = r#"
fn repeats(x: i64, y: i64) -> i64 {
    let a: i64 = (x + y) * 2;
    let b: i64 = (x + y) * 3;
    let c: i64 = (x + y) * 5;
    let d: i64 = (x + y) * 7;
    return a + b + c + d;
}
print(repeats(3, 4));
"#;

const PROG_CSE_IN_LOOP_BIG: &str = r#"
fn cse_loop(n: i64, k: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let a: i64 = (k * k) + i;
        let b: i64 = (k * k) - i;
        total = total + a + b;
        i = i + 1;
    }
    return total;
}
print(cse_loop(20000, 7));
"#;

/// CSE with deeply repeated subexpression (10× repetition).
const PROG_CSE_HEAVY: &str = r#"
fn heavy(x: i64, y: i64, z: i64) -> i64 {
    let a: i64 = (x + y * z) * 2;
    let b: i64 = (x + y * z) * 3;
    let c: i64 = (x + y * z) * 4;
    let d: i64 = (x + y * z) * 5;
    let e: i64 = (x + y * z) * 6;
    let f: i64 = (x + y * z) * 7;
    let g: i64 = (x + y * z) * 8;
    let h: i64 = (x + y * z) * 9;
    let i: i64 = (x + y * z) * 10;
    let j: i64 = (x + y * z) * 11;
    return a + b + c + d + e + f + g + h + i + j;
}
print(heavy(2, 3, 4));
"#;

/// Repeated subexpression inside an outer + inner loop. CSE saves work
/// per inner-loop iteration; the savings compound.
const PROG_CSE_DOUBLE_LOOP: &str = r#"
fn double_cse(n: i64, k: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let mut j: i64 = 0;
        while j < n {
            let a: i64 = (k * k * k) + j;
            let b: i64 = (k * k * k) - j;
            total = total + a + b + i;
            j = j + 1;
        }
        i = i + 1;
    }
    return total;
}
print(double_cse(150, 4));
"#;

// ============================================================================
// SR-favoring programs (multiplications by powers of two)
// ============================================================================

const PROG_SR_POW2: &str = r#"
fn power_mul(n: i64) -> i64 {
    let a: i64 = n * 2;
    let b: i64 = n * 4;
    let c: i64 = n * 8;
    let d: i64 = n * 16;
    let e: i64 = n * 32;
    return a + b + c + d + e;
}
print(power_mul(13));
"#;

const PROG_SR_IN_LOOP_BIG: &str = r#"
fn sr_loop(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        total = total + i * 8;
        i = i + 1;
    }
    return total;
}
print(sr_loop(20000));
"#;

/// Many pow-2 multiplications inside a tight loop. Maximum SR signal.
const PROG_SR_HEAVY_LOOP: &str = r#"
fn sr_heavy(n: i64) -> i64 {
    let mut t: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let a: i64 = i * 2;
        let b: i64 = i * 4;
        let c: i64 = i * 8;
        let d: i64 = i * 16;
        t = t + a + b + c + d;
        i = i + 1;
    }
    return t;
}
print(sr_heavy(10000));
"#;

/// SR over division (i / 2, i / 4 etc. → shifts). Tests SR's coverage
/// of the divide side, not just multiply.
const PROG_SR_DIV_POW2: &str = r#"
fn sr_div(n: i64) -> i64 {
    let mut t: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let a: i64 = i / 2;
        let b: i64 = i / 4;
        t = t + a + b;
        i = i + 1;
    }
    return t;
}
print(sr_div(15000));
"#;

// ============================================================================
// DCE-favoring programs (computed-but-unused values)
// ============================================================================

const PROG_DCE_DEAD: &str = r#"
fn dead_lets(n: i64) -> i64 {
    let dead1: i64 = n * 100;
    let dead2: i64 = n + 5000;
    let dead3: i64 = n - 7;
    let dead4: i64 = n * 11 + 22;
    let dead5: i64 = n * 33;
    return n + 1;
}
print(dead_lets(9));
"#;

const PROG_DCE_BRANCHY: &str = r#"
fn branchy(n: i64) -> i64 {
    let unused: i64 = n * 999;
    let result: i64 = if n > 0 { n + 1 } else { n - 1 };
    let also_unused: i64 = result * 7;
    return result;
}
print(branchy(5));
"#;

/// Many dead lets in a row (~20). DCE has a giant linear cleanup.
const PROG_DCE_TWENTY_DEAD: &str = r#"
fn many_dead(n: i64) -> i64 {
    let d1: i64 = n * 2; let d2: i64 = n * 3; let d3: i64 = n * 4;
    let d4: i64 = n * 5; let d5: i64 = n * 6; let d6: i64 = n * 7;
    let d7: i64 = n * 8; let d8: i64 = n * 9; let d9: i64 = n * 10;
    let d10: i64 = n * 11;
    let d11: i64 = n + 1; let d12: i64 = n + 2; let d13: i64 = n + 3;
    let d14: i64 = n + 4; let d15: i64 = n + 5; let d16: i64 = n + 6;
    let d17: i64 = n + 7; let d18: i64 = n + 8; let d19: i64 = n + 9;
    let d20: i64 = n + 10;
    return n;
}
print(many_dead(7));
"#;

/// Dead code inside a hot loop. DCE saves work per iteration.
const PROG_DCE_IN_LOOP: &str = r#"
fn dce_loop(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let dead: i64 = i * i * i;
        total = total + i;
        i = i + 1;
    }
    return total;
}
print(dce_loop(15000));
"#;

// ============================================================================
// Branch-heavy programs (DCE benefits from unreachable elimination)
// ============================================================================

const PROG_BRANCH_LADDER: &str = r#"
fn classify(n: i64) -> i64 {
    let a: i64 = if n > 10 { 1 } else { 0 };
    let b: i64 = if n > 20 { 2 } else { 0 };
    let c: i64 = if n > 30 { 3 } else { 0 };
    let d: i64 = if n > 40 { 4 } else { 0 };
    let e: i64 = if n > 50 { 5 } else { 0 };
    let f: i64 = if n > 60 { 6 } else { 0 };
    return a + b + c + d + e + f;
}
print(classify(35));
"#;

const PROG_NESTED_IFS: &str = r#"
fn nested_ifs(n: i64) -> i64 {
    let r: i64 = if n > 0 {
        if n > 10 {
            if n > 100 { 100 } else { 10 }
        } else { 1 }
    } else {
        if n < -10 { -10 } else { -1 }
    };
    return r;
}
print(nested_ifs(55));
"#;

/// Many independent if/else chains, each computing into separate locals.
const PROG_BRANCH_HEAVY: &str = r#"
fn branchy(n: i64) -> i64 {
    let a: i64 = if n > 5 { n * 2 } else { n };
    let b: i64 = if n > 10 { n * 3 } else { n };
    let c: i64 = if n > 15 { n * 4 } else { n };
    let d: i64 = if n > 20 { n * 5 } else { n };
    let e: i64 = if n > 25 { n * 6 } else { n };
    return a + b + c + d + e;
}
print(branchy(18));
"#;

// ============================================================================
// Mixed / multi-pass programs (no single dominant pass)
// ============================================================================

const PROG_MANY_FN: &str = r#"
fn add1(x: i64) -> i64 { return x + 1; }
fn add2(x: i64) -> i64 { return x + 2; }
fn add3(x: i64) -> i64 { return x + 3; }
fn mul2(x: i64) -> i64 { return x * 2; }
fn driver() -> i64 {
    let mut r: i64 = 0;
    r = add1(r);
    r = add2(r);
    r = add3(r);
    r = mul2(r);
    return r;
}
print(driver());
"#;

const PROG_RECURSIVE: &str = r#"
fn factorial(n: i64) -> i64 {
    let result: i64 = if n <= 1 { 1 } else { n * factorial(n - 1) };
    return result;
}
print(factorial(10));
"#;

const PROG_FLOAT: &str = r#"
fn polynomial(x: f64) -> f64 {
    let a: f64 = 3.14;
    let b: f64 = 2.71;
    let c: f64 = 1.41;
    return a * x * x + b * x + c;
}
print(polynomial(1.5));
"#;

const PROG_MIXED_BIG: &str = r#"
fn classify(n: i64) -> i64 {
    let mut sum: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let inc: i64 = if i * 2 > n { i } else { 0 };
        sum = sum + inc;
        i = i + 1;
    }
    return sum;
}
print(classify(5000));
"#;

const PROG_LARGE: &str = r#"
fn count_evens(n: i64) -> i64 {
    let mut c: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        if i * 2 / 2 == i {
            c = c + 1;
        }
        i = i + 1;
    }
    return c;
}
fn count_squares(n: i64) -> i64 {
    let mut c: i64 = 0;
    let mut i: i64 = 0;
    while i * i < n {
        c = c + 1;
        i = i + 1;
    }
    return c;
}
fn sum_to(n: i64) -> i64 {
    let mut s: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        s = s + i;
        i = i + 1;
    }
    return s;
}
fn combined(n: i64) -> i64 {
    let a: i64 = count_evens(n);
    let b: i64 = count_squares(n);
    let c: i64 = sum_to(n);
    return a + b + c;
}
print(combined(2000));
"#;

// ============================================================================
// Recursion variations
// ============================================================================

const PROG_RECURSIVE_BIG: &str = r#"
fn factorial(n: i64) -> i64 {
    let r: i64 = if n <= 1 { 1 } else { n * factorial(n - 1) };
    return r;
}
print(factorial(12));
"#;

const PROG_FIB_REC: &str = r#"
fn fib(n: i64) -> i64 {
    let r: i64 = if n <= 1 { n } else { fib(n - 1) + fib(n - 2) };
    return r;
}
print(fib(15));
"#;

const PROG_MUTUAL_REC: &str = r#"
fn is_even(n: i64) -> i64 {
    let r: i64 = if n == 0 { 1 } else { is_odd(n - 1) };
    return r;
}
fn is_odd(n: i64) -> i64 {
    let r: i64 = if n == 0 { 0 } else { is_even(n - 1) };
    return r;
}
print(is_even(20));
"#;

// ============================================================================
// Float-heavy programs (StrictFold reductions present)
// ============================================================================

const PROG_FLOAT_LOOP: &str = r#"
fn float_sum(n: i64) -> f64 {
    let mut total: f64 = 0.0;
    let mut i: i64 = 0;
    while i < n {
        total = total + 1.5;
        i = i + 1;
    }
    return total;
}
print(float_sum(10000));
"#;

const PROG_FLOAT_POLY: &str = r#"
fn poly(x: f64, n: i64) -> f64 {
    let mut acc: f64 = 0.0;
    let mut i: i64 = 0;
    while i < n {
        acc = acc + x * x + 2.0 * x + 1.0;
        i = i + 1;
    }
    return acc;
}
print(poly(2.5, 5000));
"#;

const PROG_FLOAT_TRIG: &str = r#"
fn trig_chain(x: f64) -> f64 {
    let a: f64 = x + 1.0;
    let b: f64 = a * 2.0;
    let c: f64 = b - 0.5;
    let d: f64 = c * c;
    return d + a + b;
}
print(trig_chain(3.14));
"#;

// ============================================================================
// Straight-line size programs (vary expr_count without loops)
// ============================================================================

const PROG_STRAIGHT_SHORT: &str = r#"
fn sl(n: i64) -> i64 {
    let a: i64 = n + 1;
    let b: i64 = a + 2;
    let c: i64 = b + 3;
    return c;
}
print(sl(0));
"#;

const PROG_STRAIGHT_MED: &str = r#"
fn sl(n: i64) -> i64 {
    let a: i64 = n + 1;
    let b: i64 = a + 2;
    let c: i64 = b + 3;
    let d: i64 = c + 4;
    let e: i64 = d + 5;
    let f: i64 = e + 6;
    let g: i64 = f + 7;
    let h: i64 = g + 8;
    return h;
}
print(sl(0));
"#;

const PROG_STRAIGHT_LONG: &str = r#"
fn sl(n: i64) -> i64 {
    let a1: i64 = n + 1; let a2: i64 = a1 + 1; let a3: i64 = a2 + 1;
    let a4: i64 = a3 + 1; let a5: i64 = a4 + 1; let a6: i64 = a5 + 1;
    let a7: i64 = a6 + 1; let a8: i64 = a7 + 1; let a9: i64 = a8 + 1;
    let a10: i64 = a9 + 1;
    let b1: i64 = a10 * 2; let b2: i64 = b1 * 2; let b3: i64 = b2 * 2;
    let b4: i64 = b3 * 2; let b5: i64 = b4 * 2; let b6: i64 = b5 * 2;
    let b7: i64 = b6 * 2; let b8: i64 = b7 * 2; let b9: i64 = b8 * 2;
    let b10: i64 = b9 * 2;
    return b10;
}
print(sl(0));
"#;

// ============================================================================
// Empty / degenerate / corner-case programs
// ============================================================================

const PROG_EMPTY_RETURN: &str = r#"
fn nothing() -> i64 { return 0; }
print(nothing());
"#;

const PROG_SINGLE_LIT: &str = r#"
fn lit() -> i64 { return 42; }
print(lit());
"#;

const PROG_IDENTITY: &str = r#"
fn identity(x: i64) -> i64 { return x; }
print(identity(7));
"#;

// ============================================================================
// Hot inner loop, cold outer (where LICM compounds with CSE)
// ============================================================================

const PROG_HOT_INNER: &str = r#"
fn hot(n: i64, k: i64) -> i64 {
    let mut total: i64 = 0;
    let mut o: i64 = 0;
    while o < n {
        let inv: i64 = (k * k) + (k + 1);
        let mut i: i64 = 0;
        while i < n {
            total = total + inv + i;
            i = i + 1;
        }
        o = o + 1;
    }
    return total;
}
print(hot(200, 5));
"#;

const PROG_LICM_CSE_MIX: &str = r#"
fn mix(n: i64, k: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let a: i64 = (k * k * k) + i;
        let b: i64 = (k * k * k) - i;
        let c: i64 = (k + 1) * i;
        total = total + a + b + c;
        i = i + 1;
    }
    return total;
}
print(mix(15000, 4));
"#;

// ============================================================================
// Many-function call graphs (per-function ranking dispatch)
// ============================================================================

const PROG_CHAIN_OF_TEN: &str = r#"
fn f1(x: i64) -> i64 { return x + 1; }
fn f2(x: i64) -> i64 { return x + 2; }
fn f3(x: i64) -> i64 { return x + 3; }
fn f4(x: i64) -> i64 { return x + 4; }
fn f5(x: i64) -> i64 { return x + 5; }
fn f6(x: i64) -> i64 { return x + 6; }
fn f7(x: i64) -> i64 { return x + 7; }
fn f8(x: i64) -> i64 { return x + 8; }
fn f9(x: i64) -> i64 { return x + 9; }
fn driver() -> i64 {
    let mut r: i64 = 0;
    r = f1(r); r = f2(r); r = f3(r); r = f4(r); r = f5(r);
    r = f6(r); r = f7(r); r = f8(r); r = f9(r);
    return r;
}
print(driver());
"#;

const PROG_THREE_LOOP_FNS: &str = r#"
fn loop_a(n: i64) -> i64 {
    let mut t: i64 = 0;
    let mut i: i64 = 0;
    while i < n { t = t + i; i = i + 1; }
    return t;
}
fn loop_b(n: i64) -> i64 {
    let mut t: i64 = 0;
    let mut i: i64 = 0;
    while i < n { t = t + i * 2; i = i + 1; }
    return t;
}
fn loop_c(n: i64) -> i64 {
    let mut t: i64 = 0;
    let mut i: i64 = 0;
    while i < n { t = t + i + 10; i = i + 1; }
    return t;
}
fn driver(n: i64) -> i64 {
    return loop_a(n) + loop_b(n) + loop_c(n);
}
print(driver(5000));
"#;

// ============================================================================
// Mixed-arithmetic stress (large but no single dominant pass)
// ============================================================================

const PROG_MIXED_ARITH: &str = r#"
fn mixed(n: i64) -> i64 {
    let a: i64 = (n + 1) * 2;
    let b: i64 = a + (n + 1) * 3;
    let c: i64 = b - a + (n - 1);
    let d: i64 = c * 4 + a + b;
    let e: i64 = d + (n + 2) * 5;
    return e + a + b + c + d;
}
print(mixed(11));
"#;

const PROG_FUNCALL_IN_LOOP: &str = r#"
fn helper(x: i64) -> i64 { return x * 2 + 1; }
fn caller(n: i64) -> i64 {
    let mut t: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        t = t + helper(i);
        i = i + 1;
    }
    return t;
}
print(caller(5000));
"#;

// ============================================================================
// Big composite programs (max expr_count + multi-pass interactions)
// ============================================================================

const PROG_BIG_COMPOSITE_1: &str = r#"
fn process(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let a: i64 = (i + 1) * 2;
        let b: i64 = (i + 1) * 3;
        let dead: i64 = i * 999;
        let c: i64 = a + b;
        total = total + c;
        i = i + 1;
    }
    return total;
}
print(process(5000));
"#;

const PROG_BIG_COMPOSITE_2: &str = r#"
fn process(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let const_part: i64 = 100 * 200 + 5;
        let var_part: i64 = i * 8;
        let repeat: i64 = (i + 1) * (i + 1);
        let also_repeat: i64 = (i + 1) * (i + 1) + 10;
        total = total + const_part + var_part + repeat + also_repeat;
        i = i + 1;
    }
    return total;
}
print(process(3000));
"#;

const PROG_BIG_COMPOSITE_3: &str = r#"
fn process(n: i64, k: i64) -> i64 {
    let mut total: i64 = 0;
    let mut o: i64 = 0;
    while o < n {
        let mut i: i64 = 0;
        while i < n {
            let cse: i64 = (k * k) + 1;
            let dead: i64 = i * 7;
            let real: i64 = (i + 1) * 2;
            total = total + cse + real;
            i = i + 1;
        }
        o = o + 1;
    }
    return total;
}
print(process(120, 3));
"#;

// ============================================================================
// New 10-pack: programs with branches inside loops + many lets
// ============================================================================

const PROG_LOOP_WITH_BRANCH_1: &str = r#"
fn lwb(n: i64) -> i64 {
    let mut t: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let v: i64 = if i % 2 == 0 { i + 1 } else { i - 1 };
        t = t + v;
        i = i + 1;
    }
    return t;
}
print(lwb(8000));
"#;

const PROG_LOOP_WITH_BRANCH_2: &str = r#"
fn lwb2(n: i64) -> i64 {
    let mut t: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let v: i64 = if i > n / 2 { i * 2 } else { i };
        let w: i64 = if v > n { 0 } else { v };
        t = t + w;
        i = i + 1;
    }
    return t;
}
print(lwb2(5000));
"#;

const PROG_MANY_LETS_LOOP: &str = r#"
fn mll(n: i64) -> i64 {
    let mut t: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let a: i64 = i + 1;
        let b: i64 = a + 2;
        let c: i64 = b + 3;
        let d: i64 = c + 4;
        let e: i64 = d + 5;
        t = t + a + b + c + d + e;
        i = i + 1;
    }
    return t;
}
print(mll(5000));
"#;

const PROG_ACCUMULATOR: &str = r#"
fn acc(n: i64) -> i64 {
    let mut sum: i64 = 0;
    let mut prod: i64 = 1;
    let mut i: i64 = 1;
    while i < n {
        sum = sum + i;
        prod = prod + i * 2;
        i = i + 1;
    }
    return sum + prod;
}
print(acc(5000));
"#;

const PROG_RANGED_LOOP: &str = r#"
fn ranged(lo: i64, hi: i64) -> i64 {
    let mut t: i64 = 0;
    let mut i: i64 = lo;
    while i < hi {
        t = t + i;
        i = i + 1;
    }
    return t;
}
print(ranged(0, 10000));
"#;

const PROG_WHILE_DOUBLE: &str = r#"
fn whd(n: i64) -> i64 {
    let mut i: i64 = 1;
    let mut t: i64 = 0;
    while i < n {
        t = t + i;
        i = i * 2;
    }
    return t;
}
print(whd(1000000));
"#;

const PROG_COND_ACCUM: &str = r#"
fn cnd(n: i64) -> i64 {
    let mut pos: i64 = 0;
    let mut neg: i64 = 0;
    let mut i: i64 = -n;
    while i < n {
        let inc: i64 = if i > 0 { 1 } else { 0 };
        pos = pos + inc;
        let dec: i64 = if i < 0 { 1 } else { 0 };
        neg = neg + dec;
        i = i + 1;
    }
    return pos + neg;
}
print(cnd(5000));
"#;

const PROG_TWO_LOOPS_SEQUENTIAL: &str = r#"
fn two(n: i64) -> i64 {
    let mut a: i64 = 0;
    let mut i: i64 = 0;
    while i < n { a = a + i; i = i + 1; }
    let mut b: i64 = 0;
    let mut j: i64 = 0;
    while j < n { b = b + j * 2; j = j + 1; }
    return a + b;
}
print(two(5000));
"#;

const PROG_INNER_LOOP_BREAK_LIKE: &str = r#"
fn ilbl(n: i64) -> i64 {
    let mut t: i64 = 0;
    let mut o: i64 = 0;
    while o < n {
        let mut i: i64 = 0;
        let limit: i64 = if o > 5 { 5 } else { o };
        while i < limit { t = t + 1; i = i + 1; }
        o = o + 1;
    }
    return t;
}
print(ilbl(500));
"#;

const PROG_NESTED2_WITH_INV: &str = r#"
fn n2i(n: i64, k: i64) -> i64 {
    let mut t: i64 = 0;
    let mut o: i64 = 0;
    while o < n {
        let mut i: i64 = 0;
        while i < n {
            t = t + k * k + i + o;
            i = i + 1;
        }
        o = o + 1;
    }
    return t;
}
print(n2i(150, 4));
"#;

// ============================================================================
// CSE-eligible programs (explicit duplicate let-bindings)
// ============================================================================

/// Two lets with identical pure init — CSE replaces uses of `b` with `a`.
const PROG_CSE_LET_DUP_TINY: &str = r#"
fn dup(x: i64, y: i64) -> i64 {
    let a: i64 = x + y;
    let b: i64 = x + y;
    return a + b * 2;
}
print(dup(3, 4));
"#;

/// 5 duplicate let bindings → 4 replacements.
const PROG_CSE_LET_DUP_FIVE: &str = r#"
fn five_dup(x: i64, y: i64) -> i64 {
    let a: i64 = x * y + 1;
    let b: i64 = x * y + 1;
    let c: i64 = x * y + 1;
    let d: i64 = x * y + 1;
    let e: i64 = x * y + 1;
    return a + b + c + d + e;
}
print(five_dup(3, 4));
"#;

/// Many independent duplicate pairs.
const PROG_CSE_PAIRS: &str = r#"
fn pairs(x: i64, y: i64, z: i64) -> i64 {
    let p1a: i64 = x + y;
    let p1b: i64 = x + y;
    let p2a: i64 = y * z;
    let p2b: i64 = y * z;
    let p3a: i64 = x * z + 1;
    let p3b: i64 = x * z + 1;
    return p1a + p1b + p2a + p2b + p3a + p3b;
}
print(pairs(2, 3, 4));
"#;

/// CSE-eligible duplicates inside a loop body — CSE saves work per
/// iteration AND the original computation is still there for LICM to
/// hoist on a later pass.
const PROG_CSE_DUP_IN_LOOP: &str = r#"
fn dup_loop(n: i64, k: i64) -> i64 {
    let mut t: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let a: i64 = k * k + i;
        let b: i64 = k * k + i;
        t = t + a + b;
        i = i + 1;
    }
    return t;
}
print(dup_loop(5000, 7));
"#;

// ============================================================================
// LICM-eligible programs (explicit invariant let-bindings inside loops)
// ============================================================================

/// Single invariant let inside loop — LICM hoists one binding.
const PROG_LICM_ONE_LET: &str = r#"
fn one_inv(n: i64, k: i64) -> i64 {
    let mut t: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let inv: i64 = k * k * k;
        t = t + inv + i;
        i = i + 1;
    }
    return t;
}
print(one_inv(20000, 5));
"#;

/// Three invariant lets inside loop — LICM hoists all three.
const PROG_LICM_THREE_LETS: &str = r#"
fn three_inv(n: i64, k: i64, m: i64) -> i64 {
    let mut t: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let inv1: i64 = k * k;
        let inv2: i64 = k + m;
        let inv3: i64 = m * m;
        t = t + inv1 + inv2 + inv3 + i;
        i = i + 1;
    }
    return t;
}
print(three_inv(15000, 7, 3));
"#;

/// Doubly-nested loop with invariants at each level — LICM hoists
/// twice (inner invariant goes to between-loops, outer goes above outer).
const PROG_LICM_NESTED: &str = r#"
fn n_inv(n: i64, k: i64) -> i64 {
    let mut t: i64 = 0;
    let mut o: i64 = 0;
    while o < n {
        let inv_outer: i64 = k * k;
        let mut i: i64 = 0;
        while i < n {
            let inv_inner: i64 = k + 1;
            t = t + inv_outer + inv_inner + i;
            i = i + 1;
        }
        o = o + 1;
    }
    return t;
}
print(n_inv(200, 4));
"#;

/// 5 invariant lets in one loop. Maximum LICM signal.
const PROG_LICM_FIVE_LETS: &str = r#"
fn five_inv(n: i64, a: i64, b: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let inv1: i64 = a * b;
        let inv2: i64 = a + b;
        let inv3: i64 = a - b;
        let inv4: i64 = a * a;
        let inv5: i64 = b * b;
        total = total + inv1 + inv2 + inv3 + inv4 + inv5 + i;
        i = i + 1;
    }
    return total;
}
print(five_inv(10000, 5, 3));
"#;

// ============================================================================
// Mixed CSE+LICM (duplicate invariant lets — both passes can help)
// ============================================================================

const PROG_CSE_LICM_DUP_INV: &str = r#"
fn dup_inv(n: i64, k: i64) -> i64 {
    let mut t: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let a: i64 = k * k + 1;
        let b: i64 = k * k + 1;
        t = t + a + b + i;
        i = i + 1;
    }
    return t;
}
print(dup_inv(8000, 7));
"#;

// ============================================================================
// §3A.4 follow-up — Alloc-heavy programs (close `alloc_sites` blind dimension)
// ============================================================================
//
// The §3A.4 feature distribution audit found that the corpus had
// `alloc_sites = 0` across all 73 programs (max == 0, unique == 1),
// because every CF/SR/DCE/CSE/LICM-favoring program was pure integer
// arithmetic. The trained `w_alloc_sites` coefficient was mathematically
// unconstrained — OLS picked 0 because no data point varied along that
// axis.
//
// These programs exercise `MirExprKind::ArrayLit`, `TupleLit`, and
// `StringLit` — the three cheapest constructs that increment
// `MemoryProxy::alloc_sites`. Each program also exercises one of the
// trainable passes (CF, DCE, CSE) so that the alloc signal is paired
// with a non-trivial benefit measurement, not silenced by an all-zero
// label.

const PROG_ALLOC_ARRAY_FOLD: &str = r#"
fn arrays_and_arith(n: i64) -> i64 {
    let a: [i64] = [1, 2, 3];
    let b: [i64] = [4, 5, 6];
    let c: [i64] = [7, 8, 9];
    let d: [i64] = [n, n + 1, n + 2];
    let x: i64 = 1 + 2 + 3 + 4 + 5;
    let y: i64 = x * 10 - 20 + 30;
    let z: i64 = y + a[0] + b[1] + c[2] + d[0];
    return z;
}
print(arrays_and_arith(7));
"#;

const PROG_ALLOC_TUPLES_CHAIN: &str = r#"
fn tuples_chain(n: i64) -> i64 {
    let p1: (i64, i64) = (1, 2);
    let p2: (i64, i64) = (3, 4);
    let p3: (i64, i64) = (5, 6);
    let p4: (i64, i64) = (n, n + 1);
    let p5: (i64, i64) = (10, 20);
    let sum_const: i64 = 1 + 2 + 3 + 4 + 5;
    let mul_const: i64 = 10 * 20 - 100 + 50;
    return sum_const + mul_const + p1.0 + p2.1 + p3.0 + p4.1 + p5.0;
}
print(tuples_chain(11));
"#;

const PROG_ALLOC_STRINGS_LOOP: &str = r#"
fn string_count(n: i64) -> i64 {
    let s1: Str = "alpha";
    let s2: Str = "beta";
    let s3: Str = "gamma";
    let s4: Str = "delta";
    let s5: Str = "epsilon";
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let inv: i64 = 7 * 11 - 5;
        let dup: i64 = inv * 2;
        total = total + dup;
        i = i + 1;
    }
    return total + len(s1) + len(s2) + len(s3) + len(s4) + len(s5);
}
print(string_count(40));
"#;

// ============================================================================
// §3A.4 follow-up — High-branch programs (saturate `branch_count` distribution)
// ============================================================================
//
// The same audit found `branch_count` had only 5 unique values (0-4),
// with the corpus median at 0. Coefficient signs on this column were
// suspect — wrong sign for CF and DCE per the original v3 findings.
// These programs reach `branch_count >= 10` so the OLS fit has data
// points beyond the corpus's previous cluster-at-zero distribution.
//
// Each program is a chain of `if/else` decisions paired with one of
// the trainable passes (CF, DCE) so the branch signal is paired with
// a non-trivial benefit measurement.

const PROG_BRANCH_LADDER_12: &str = r#"
fn classify(n: i64) -> i64 {
    let mut r: i64 = 0;
    if n > 100 { r = 12; } else { r = 0; }
    if n > 90  { r = r + 11; } else { r = r + 0; }
    if n > 80  { r = r + 10; } else { r = r + 0; }
    if n > 70  { r = r + 9;  } else { r = r + 0; }
    if n > 60  { r = r + 8;  } else { r = r + 0; }
    if n > 50  { r = r + 7;  } else { r = r + 0; }
    if n > 40  { r = r + 6;  } else { r = r + 0; }
    if n > 30  { r = r + 5;  } else { r = r + 0; }
    if n > 20  { r = r + 4;  } else { r = r + 0; }
    if n > 10  { r = r + 3;  } else { r = r + 0; }
    if n > 5   { r = r + 2;  } else { r = r + 0; }
    if n > 0   { r = r + 1;  } else { r = r + 0; }
    return r;
}
print(classify(55));
"#;

const PROG_BRANCH_NESTED_15: &str = r#"
fn deep_branch(n: i64) -> i64 {
    let mut r: i64 = 0;
    if n > 50 {
        if n > 70 {
            if n > 90 { r = 9; } else { r = 7; }
        } else {
            if n > 60 { r = 6; } else { r = 5; }
        }
    } else {
        if n > 20 {
            if n > 40 { r = 4; } else { r = 3; }
        } else {
            if n > 10 { r = 2; } else { r = 1; }
        }
    }
    if r > 5 {
        if r > 7 { r = r + 100; } else { r = r + 50; }
    } else {
        if r > 2 { r = r + 25; } else { r = r + 10; }
    }
    return r + 1 + 2 + 3 + 4 + 5;
}
print(deep_branch(45));
"#;

const PROG_BRANCH_LOOP_MIX: &str = r#"
fn branchy_loop(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let mut step: i64 = 0;
        if i > 100 { step = step + 10; } else { step = step + 1; }
        if i > 80  { step = step + 9;  } else { step = step + 1; }
        if i > 60  { step = step + 8;  } else { step = step + 1; }
        if i > 40  { step = step + 7;  } else { step = step + 1; }
        if i > 20  { step = step + 6;  } else { step = step + 1; }
        if i > 10  { step = step + 5;  } else { step = step + 1; }
        if i > 5   { step = step + 4;  } else { step = step + 1; }
        if i > 0   { step = step + 3;  } else { step = step + 1; }
        let inv: i64 = 7 * 8 + 100;
        total = total + step + inv;
        i = i + 1;
    }
    return total;
}
print(branchy_loop(50));
"#;

// ============================================================================
// The corpus  (60 programs)
// ============================================================================

pub const PROGRAMS: &[Program] = &[
    // CF (6)
    Program { name: "arith_tiny",            source: PROG_ARITH_TINY,            expected_dominant_pass: "constant_fold" },
    Program { name: "arith_med",             source: PROG_ARITH_MED,             expected_dominant_pass: "constant_fold" },
    Program { name: "arith_heavy",           source: PROG_ARITH_HEAVY,           expected_dominant_pass: "constant_fold" },
    Program { name: "cf_forty",              source: PROG_CF_FORTY,              expected_dominant_pass: "constant_fold" },
    Program { name: "cf_in_hot_loop",        source: PROG_CF_IN_HOT_LOOP,        expected_dominant_pass: "constant_fold" },
    Program { name: "cf_pure_float",         source: PROG_CF_PURE_FLOAT,         expected_dominant_pass: "constant_fold" },
    // LICM (7)
    Program { name: "loop_invariant_big",    source: PROG_LOOP_INVARIANT_BIG,    expected_dominant_pass: "licm" },
    Program { name: "loop_nested2_big",      source: PROG_LOOP_NESTED2_BIG,      expected_dominant_pass: "licm" },
    Program { name: "loop_nested3_big",      source: PROG_LOOP_NESTED3_BIG,      expected_dominant_pass: "licm" },
    Program { name: "loop_nested4_big",      source: PROG_LOOP_NESTED4_BIG,      expected_dominant_pass: "licm" },
    Program { name: "licm_many_inv",         source: PROG_LICM_MANY_INV,         expected_dominant_pass: "licm" },
    Program { name: "licm_branchy",          source: PROG_LICM_BRANCHY,          expected_dominant_pass: "licm" },
    Program { name: "licm_long_outer",       source: PROG_LICM_LONG_OUTER,       expected_dominant_pass: "licm" },
    // CSE (4)
    Program { name: "cse_repeat",            source: PROG_CSE_REPEAT,            expected_dominant_pass: "cse" },
    Program { name: "cse_in_loop_big",       source: PROG_CSE_IN_LOOP_BIG,       expected_dominant_pass: "cse" },
    Program { name: "cse_heavy",             source: PROG_CSE_HEAVY,             expected_dominant_pass: "cse" },
    Program { name: "cse_double_loop",       source: PROG_CSE_DOUBLE_LOOP,       expected_dominant_pass: "cse" },
    // SR (4)
    Program { name: "sr_pow2",               source: PROG_SR_POW2,               expected_dominant_pass: "strength_reduce" },
    Program { name: "sr_in_loop_big",        source: PROG_SR_IN_LOOP_BIG,        expected_dominant_pass: "strength_reduce" },
    Program { name: "sr_heavy_loop",         source: PROG_SR_HEAVY_LOOP,         expected_dominant_pass: "strength_reduce" },
    Program { name: "sr_div_pow2",           source: PROG_SR_DIV_POW2,           expected_dominant_pass: "strength_reduce" },
    // DCE (4)
    Program { name: "dce_dead",              source: PROG_DCE_DEAD,              expected_dominant_pass: "dce" },
    Program { name: "dce_branchy",           source: PROG_DCE_BRANCHY,           expected_dominant_pass: "dce" },
    Program { name: "dce_twenty_dead",       source: PROG_DCE_TWENTY_DEAD,       expected_dominant_pass: "dce" },
    Program { name: "dce_in_loop",           source: PROG_DCE_IN_LOOP,           expected_dominant_pass: "dce" },
    // Branch-heavy (3)
    Program { name: "branch_ladder",         source: PROG_BRANCH_LADDER,         expected_dominant_pass: "dce" },
    Program { name: "nested_ifs",            source: PROG_NESTED_IFS,            expected_dominant_pass: "constant_fold" },
    Program { name: "branch_heavy",          source: PROG_BRANCH_HEAVY,          expected_dominant_pass: "dce" },
    // Mixed multi-fn (5)
    Program { name: "many_fn",               source: PROG_MANY_FN,               expected_dominant_pass: "constant_fold" },
    Program { name: "recursive",             source: PROG_RECURSIVE,             expected_dominant_pass: "constant_fold" },
    Program { name: "float",                 source: PROG_FLOAT,                 expected_dominant_pass: "constant_fold" },
    Program { name: "mixed_big",             source: PROG_MIXED_BIG,             expected_dominant_pass: "licm" },
    Program { name: "large",                 source: PROG_LARGE,                 expected_dominant_pass: "licm" },
    // Recursion (3)
    Program { name: "recursive_big",         source: PROG_RECURSIVE_BIG,         expected_dominant_pass: "constant_fold" },
    Program { name: "fib_rec",               source: PROG_FIB_REC,               expected_dominant_pass: "constant_fold" },
    Program { name: "mutual_rec",            source: PROG_MUTUAL_REC,            expected_dominant_pass: "constant_fold" },
    // Float (3)
    Program { name: "float_loop",            source: PROG_FLOAT_LOOP,            expected_dominant_pass: "licm" },
    Program { name: "float_poly",            source: PROG_FLOAT_POLY,            expected_dominant_pass: "licm" },
    Program { name: "float_trig",            source: PROG_FLOAT_TRIG,            expected_dominant_pass: "constant_fold" },
    // Straight-line (3)
    Program { name: "straight_short",        source: PROG_STRAIGHT_SHORT,        expected_dominant_pass: "constant_fold" },
    Program { name: "straight_med",          source: PROG_STRAIGHT_MED,          expected_dominant_pass: "constant_fold" },
    Program { name: "straight_long",         source: PROG_STRAIGHT_LONG,         expected_dominant_pass: "constant_fold" },
    // Degenerate (3)
    Program { name: "empty_return",          source: PROG_EMPTY_RETURN,          expected_dominant_pass: "constant_fold" },
    Program { name: "single_lit",            source: PROG_SINGLE_LIT,            expected_dominant_pass: "constant_fold" },
    Program { name: "identity",              source: PROG_IDENTITY,              expected_dominant_pass: "constant_fold" },
    // LICM × CSE mix (2)
    Program { name: "hot_inner",             source: PROG_HOT_INNER,             expected_dominant_pass: "licm" },
    Program { name: "licm_cse_mix",          source: PROG_LICM_CSE_MIX,          expected_dominant_pass: "licm" },
    // Many-fn (2)
    Program { name: "chain_of_ten",          source: PROG_CHAIN_OF_TEN,          expected_dominant_pass: "constant_fold" },
    Program { name: "three_loop_fns",        source: PROG_THREE_LOOP_FNS,        expected_dominant_pass: "licm" },
    // Mixed arith / funcall (2)
    Program { name: "mixed_arith",           source: PROG_MIXED_ARITH,           expected_dominant_pass: "constant_fold" },
    Program { name: "funcall_in_loop",       source: PROG_FUNCALL_IN_LOOP,       expected_dominant_pass: "licm" },
    // Big composites (3)
    Program { name: "big_composite_1",       source: PROG_BIG_COMPOSITE_1,       expected_dominant_pass: "dce" },
    Program { name: "big_composite_2",       source: PROG_BIG_COMPOSITE_2,       expected_dominant_pass: "cse" },
    Program { name: "big_composite_3",       source: PROG_BIG_COMPOSITE_3,       expected_dominant_pass: "licm" },
    // 10-pack additions (10)
    Program { name: "loop_with_branch_1",    source: PROG_LOOP_WITH_BRANCH_1,    expected_dominant_pass: "licm" },
    Program { name: "loop_with_branch_2",    source: PROG_LOOP_WITH_BRANCH_2,    expected_dominant_pass: "licm" },
    Program { name: "many_lets_loop",        source: PROG_MANY_LETS_LOOP,        expected_dominant_pass: "cse" },
    Program { name: "accumulator",           source: PROG_ACCUMULATOR,           expected_dominant_pass: "licm" },
    Program { name: "ranged_loop",           source: PROG_RANGED_LOOP,           expected_dominant_pass: "licm" },
    Program { name: "while_double",          source: PROG_WHILE_DOUBLE,          expected_dominant_pass: "licm" },
    Program { name: "cond_accum",            source: PROG_COND_ACCUM,            expected_dominant_pass: "licm" },
    Program { name: "two_loops_sequential",  source: PROG_TWO_LOOPS_SEQUENTIAL,  expected_dominant_pass: "licm" },
    Program { name: "inner_loop_break_like", source: PROG_INNER_LOOP_BREAK_LIKE, expected_dominant_pass: "licm" },
    Program { name: "nested2_with_inv",      source: PROG_NESTED2_WITH_INV,      expected_dominant_pass: "licm" },
    // CSE-eligible programs with explicit duplicate let bindings (4)
    Program { name: "cse_let_dup_tiny",       source: PROG_CSE_LET_DUP_TINY,      expected_dominant_pass: "cse" },
    Program { name: "cse_let_dup_five",       source: PROG_CSE_LET_DUP_FIVE,      expected_dominant_pass: "cse" },
    Program { name: "cse_pairs",              source: PROG_CSE_PAIRS,             expected_dominant_pass: "cse" },
    Program { name: "cse_dup_in_loop",        source: PROG_CSE_DUP_IN_LOOP,       expected_dominant_pass: "cse" },
    // LICM-eligible programs with explicit invariant let bindings (4)
    Program { name: "licm_one_let",           source: PROG_LICM_ONE_LET,          expected_dominant_pass: "licm" },
    Program { name: "licm_three_lets",        source: PROG_LICM_THREE_LETS,       expected_dominant_pass: "licm" },
    Program { name: "licm_nested",            source: PROG_LICM_NESTED,           expected_dominant_pass: "licm" },
    Program { name: "licm_five_lets",         source: PROG_LICM_FIVE_LETS,        expected_dominant_pass: "licm" },
    // Mixed CSE + LICM eligible (1)
    Program { name: "cse_licm_dup_inv",       source: PROG_CSE_LICM_DUP_INV,      expected_dominant_pass: "licm" },
    // §3A.4 follow-up — Alloc-heavy (3) — closes alloc_sites blind dimension
    Program { name: "alloc_array_fold",       source: PROG_ALLOC_ARRAY_FOLD,      expected_dominant_pass: "constant_fold" },
    Program { name: "alloc_tuples_chain",     source: PROG_ALLOC_TUPLES_CHAIN,    expected_dominant_pass: "constant_fold" },
    Program { name: "alloc_strings_loop",     source: PROG_ALLOC_STRINGS_LOOP,    expected_dominant_pass: "licm" },
    // §3A.4 follow-up — High-branch (3) — saturates branch_count tail
    Program { name: "branch_ladder_12",       source: PROG_BRANCH_LADDER_12,      expected_dominant_pass: "constant_fold" },
    Program { name: "branch_nested_15",       source: PROG_BRANCH_NESTED_15,      expected_dominant_pass: "constant_fold" },
    Program { name: "branch_loop_mix",        source: PROG_BRANCH_LOOP_MIX,       expected_dominant_pass: "licm" },
];
