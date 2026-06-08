//! Training corpus for the LinearCostModel.
//!
//! 18 CJC-Lang programs chosen to span the feature space of `CanaFeatures`:
//!
//!   - `expr_count`: from tiny (single literal) to huge (60+ exprs)
//!   - `max_loop_depth`: 0 (no loops), 1, 2, 3, 4 (deeply nested)
//!   - `branch_count`: 0 to ~10
//!   - `alloc_sites`: 0 to ~6
//!
//! Each program is intentionally designed to make ONE pass shine:
//!
//!   - "arith_*" → CF (constants everywhere)
//!   - "loop_*"  → LICM (loops with hoistable invariants)
//!   - "cse_*"   → CSE (repeated subexpressions)
//!   - "sr_*"    → SR (multiplications by powers of 2)
//!   - "dce_*"   → DCE (computed-but-unused values)
//!   - "mixed_*" → no single pass dominates
//!
//! Programs use single-return shape (let-bound if/else) throughout — the
//! `if cond { return x; } ... return y;` shape triggers the dominators OOB
//! that was task_9d7ae8b2 (now fixed but the convention is documented in
//! the original cana_pass_ordering bench).
//!
//! Each program is documented with its expected dominant pass to make the
//! resulting trained coefficients auditable: if the fitted w_loop_depth
//! coefficient for LICM is near zero after training, something went wrong.

pub struct Program {
    pub name: &'static str,
    pub source: &'static str,
    /// Pass we expect to deliver the biggest benefit on this program.
    /// Used to sanity-check the fitted coefficients downstream.
    #[allow(dead_code)]
    pub expected_dominant_pass: &'static str,
}

// ============================================================================
// CF-favoring programs (constant arithmetic)
// ============================================================================

/// Tiny: a single foldable expression. CF nukes nearly everything.
const PROG_ARITH_TINY: &str = r#"
fn compute() -> i64 {
    let a: i64 = 5 + 3;
    return a * 2;
}
print(compute());
"#;

/// Medium: chain of foldable arithmetic. CF saves the most.
const PROG_ARITH_MED: &str = r#"
fn compute(n: i64) -> i64 {
    let a: i64 = 10 * 5 + 2;
    let b: i64 = (a + 100) * 2;
    let c: i64 = b - 50 + n;
    return c + a + b;
}
print(compute(7));
"#;

/// Heavy: many constants chained, but with one runtime input. CF folds
/// most and DCE may eliminate a few unused intermediates.
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

// ============================================================================
// LICM-favoring programs (loops with hoistable invariants)
// ============================================================================

/// Simple loop with loop-invariant expression. LICM can hoist `n * 10`.
const PROG_LOOP_INVARIANT: &str = r#"
fn loop_inv(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        total = total + n * 10 + i;
        i = i + 1;
    }
    return total;
}
print(loop_inv(100));
"#;

/// Doubly-nested loop. LICM has TWO levels of hoisting opportunities.
const PROG_LOOP_NESTED2: &str = r#"
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
print(nested2(20));
"#;

/// Triply-nested loop. Loop depth 3 — strong LICM signal.
const PROG_LOOP_NESTED3: &str = r#"
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
print(nested3(8));
"#;

/// Quadruply-nested. Loop depth 4 — extreme.
const PROG_LOOP_NESTED4: &str = r#"
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
print(nested4(4));
"#;

// ============================================================================
// CSE-favoring programs (repeated subexpressions)
// ============================================================================

/// Same subexpression used many times. CSE replaces them with one binding.
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

/// Repeated subexpression inside a loop body. CSE saves a multiplication
/// per iteration (and LICM may move the result out).
const PROG_CSE_IN_LOOP: &str = r#"
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
print(cse_loop(50, 7));
"#;

// ============================================================================
// SR-favoring programs (multiplications by powers of two)
// ============================================================================

/// Many multiplications by 2/4/8. SR rewrites as shifts.
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

/// SR inside a loop body — bigger payoff per iteration.
const PROG_SR_IN_LOOP: &str = r#"
fn sr_loop(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        total = total + i * 8;
        i = i + 1;
    }
    return total;
}
print(sr_loop(200));
"#;

// ============================================================================
// DCE-favoring programs (computed-but-unused values)
// ============================================================================

/// Many lets computed but never read. DCE wipes them out.
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

/// Branches that compute but discard intermediates. DCE + CF win together.
const PROG_DCE_BRANCHY: &str = r#"
fn branchy(n: i64) -> i64 {
    let unused: i64 = n * 999;
    let result: i64 = if n > 0 { n + 1 } else { n - 1 };
    let also_unused: i64 = result * 7;
    return result;
}
print(branchy(5));
"#;

// ============================================================================
// Mixed / multi-pass programs
// ============================================================================

/// Multi-function with a call graph. Each function gets its own per-fn
/// PassPlan entry.
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

/// Recursive — distinct CFG shape (call graph rather than basic blocks).
const PROG_RECURSIVE: &str = r#"
fn factorial(n: i64) -> i64 {
    let result: i64 = if n <= 1 { 1 } else { n * factorial(n - 1) };
    return result;
}
print(factorial(10));
"#;

/// Floats: introduces StrictFold reductions, which the DefaultLegalityGate
/// refuses to reorder. Useful for ensuring the cost model learns "high
/// expr_count alone doesn't justify aggressive reordering on float fns."
const PROG_FLOAT: &str = r#"
fn polynomial(x: f64) -> f64 {
    let a: f64 = 3.14;
    let b: f64 = 2.71;
    let c: f64 = 1.41;
    return a * x * x + b * x + c;
}
print(polynomial(1.5));
"#;

/// Mixed: loop + branches + constants. No single pass dominates.
const PROG_MIXED: &str = r#"
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
print(classify(40));
"#;

/// Larger multi-function program — composite stress test.
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
print(combined(50));
"#;

// ============================================================================
// The corpus
// ============================================================================

pub const PROGRAMS: &[Program] = &[
    // CF
    Program { name: "arith_tiny",      source: PROG_ARITH_TINY,      expected_dominant_pass: "constant_fold" },
    Program { name: "arith_med",       source: PROG_ARITH_MED,       expected_dominant_pass: "constant_fold" },
    Program { name: "arith_heavy",     source: PROG_ARITH_HEAVY,     expected_dominant_pass: "constant_fold" },
    // LICM
    Program { name: "loop_invariant",  source: PROG_LOOP_INVARIANT,  expected_dominant_pass: "licm" },
    Program { name: "loop_nested2",    source: PROG_LOOP_NESTED2,    expected_dominant_pass: "licm" },
    Program { name: "loop_nested3",    source: PROG_LOOP_NESTED3,    expected_dominant_pass: "licm" },
    Program { name: "loop_nested4",    source: PROG_LOOP_NESTED4,    expected_dominant_pass: "licm" },
    // CSE
    Program { name: "cse_repeat",      source: PROG_CSE_REPEAT,      expected_dominant_pass: "cse" },
    Program { name: "cse_in_loop",     source: PROG_CSE_IN_LOOP,     expected_dominant_pass: "cse" },
    // SR
    Program { name: "sr_pow2",         source: PROG_SR_POW2,         expected_dominant_pass: "strength_reduce" },
    Program { name: "sr_in_loop",      source: PROG_SR_IN_LOOP,      expected_dominant_pass: "strength_reduce" },
    // DCE
    Program { name: "dce_dead",        source: PROG_DCE_DEAD,        expected_dominant_pass: "dce" },
    Program { name: "dce_branchy",     source: PROG_DCE_BRANCHY,     expected_dominant_pass: "dce" },
    // Mixed
    Program { name: "many_fn",         source: PROG_MANY_FN,         expected_dominant_pass: "constant_fold" },
    Program { name: "recursive",       source: PROG_RECURSIVE,       expected_dominant_pass: "constant_fold" },
    Program { name: "float",           source: PROG_FLOAT,           expected_dominant_pass: "constant_fold" },
    Program { name: "mixed",           source: PROG_MIXED,           expected_dominant_pass: "licm" },
    Program { name: "large",           source: PROG_LARGE,           expected_dominant_pass: "licm" },
];
