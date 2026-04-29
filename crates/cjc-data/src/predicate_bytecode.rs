//! Predicate bytecode for `TidyView::filter` — Adaptive Engine v2.1.
//!
//! Lowers a `DExpr` predicate to a flat stack-bytecode program once per
//! `filter()` call, then interprets it in a tight loop. Produces bit-
//! identical output to the legacy AST-walk path (`try_eval_predicate_columnar`)
//! on every shape it accepts; `lower` returns `None` for unsupported shapes
//! so the caller falls through to row-wise evaluation, exactly as before.
//!
//! Why bytecode rather than recursive AST descent?
//! - Lowering happens once. Interpretation is a flat loop over a `Vec<PredicateOp>`
//!   with no per-node recursion or `Option`-return error plumbing.
//! - The supported predicate language is closed: leaves are always
//!   `Col op Lit` and interior nodes are always `And`/`Or`. Three opcodes
//!   are enough — no symbol table, no scopes.
//! - Determinism is preserved by construction: every `Cmp` opcode delegates
//!   to the same `columnar_cmp_*` kernels the AST walk used. Floating-point
//!   semantics, NaN handling, and i64→f64 promotion all match bit-for-bit.

use crate::{columnar_cmp_f64, columnar_cmp_i64, nwords_for, BitMask, Column, DBinOp, DExpr, DataFrame};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpKind {
    Gt,
    Lt,
    Ge,
    Le,
    Eq,
    Ne,
}

impl CmpKind {
    fn from_dbinop(op: DBinOp) -> Option<Self> {
        match op {
            DBinOp::Gt => Some(Self::Gt),
            DBinOp::Lt => Some(Self::Lt),
            DBinOp::Ge => Some(Self::Ge),
            DBinOp::Le => Some(Self::Le),
            DBinOp::Eq => Some(Self::Eq),
            DBinOp::Ne => Some(Self::Ne),
            _ => None,
        }
    }

    fn to_dbinop(self) -> DBinOp {
        match self {
            Self::Gt => DBinOp::Gt,
            Self::Lt => DBinOp::Lt,
            Self::Ge => DBinOp::Ge,
            Self::Le => DBinOp::Le,
            Self::Eq => DBinOp::Eq,
            Self::Ne => DBinOp::Ne,
        }
    }

    /// Flip the relation when literal is on the LHS: `5 > col` becomes `col < 5`.
    /// `Eq`/`Ne` are symmetric.
    fn flip(self) -> Self {
        match self {
            Self::Gt => Self::Lt,
            Self::Lt => Self::Gt,
            Self::Ge => Self::Le,
            Self::Le => Self::Ge,
            Self::Eq => Self::Eq,
            Self::Ne => Self::Ne,
        }
    }
}

/// One leaf comparison, fully resolved at lowering time.
///
/// The four variants correspond to the four arms in
/// `try_eval_predicate_columnar`: a Float column compared to a float literal,
/// a Float column compared to an int literal (literal promoted to f64), an
/// Int column compared to an int literal, or an Int column compared to a float
/// literal (column promoted to f64 at run time).
#[derive(Clone, Copy, Debug)]
pub enum LeafKind {
    FloatColFloatLit { col_idx: usize, lit: f64 },
    FloatColIntLit { col_idx: usize, lit: i64 },
    IntColIntLit { col_idx: usize, lit: i64 },
    IntColFloatLit { col_idx: usize, lit: f64 },
}

#[derive(Clone, Debug)]
pub enum PredicateOp {
    /// Push the bitmask resulting from comparing a column against a literal.
    Cmp { kind: LeafKind, op: CmpKind },
    /// Pop two masks; push their bitwise AND.
    And,
    /// Pop two masks; push their bitwise OR.
    Or,
}

#[derive(Clone, Debug)]
pub struct PredicateBytecode {
    ops: Vec<PredicateOp>,
}

impl PredicateBytecode {
    /// Read-only view of the lowered op sequence. Useful for tests and for
    /// the optional optimizer pass landing in v2.2.
    pub fn ops(&self) -> &[PredicateOp] {
        &self.ops
    }

    /// Lower a `DExpr` predicate to bytecode. Returns `None` if the predicate
    /// shape is not supported by the columnar fast path (caller falls through
    /// to row-wise evaluation, exactly as the AST-walk path did).
    pub fn lower(predicate: &DExpr, base: &DataFrame) -> Option<Self> {
        let mut ops = Vec::new();
        lower_into(predicate, base, &mut ops)?;
        Some(PredicateBytecode { ops })
    }

    /// Sparse-aware interpretation (v2.2). Evaluates the predicate over only
    /// the rows listed in `existing_indices` (must be ascending and a subset
    /// of `0..nrows`). Result bits are set exclusively for indices in the
    /// input set, so no final AND with `existing_mask` is needed — the
    /// caller has effectively pre-applied it via index materialization.
    ///
    /// Bit-identical to `interpret(base, &existing_mask)` whenever
    /// `existing_indices == existing_mask.iter_set().collect()`. The win
    /// is amortizing per-`Cmp` cost from O(nrows) to O(|existing_indices|),
    /// which is meaningful when the parent selection has already narrowed
    /// the row set substantially (typical for chained `.filter(...).filter(...)`).
    ///
    /// Caller is responsible for the density decision (see
    /// `should_use_sparse_path`). Below that threshold, the random-access
    /// gather here beats the sequential column scan in `interpret`.
    pub fn interpret_sparse(
        &self,
        base: &DataFrame,
        existing_indices: &[usize],
        nrows: usize,
    ) -> BitMask {
        let nwords = nwords_for(nrows);
        let mut stack: Vec<Vec<u64>> = Vec::with_capacity(4);

        for op in &self.ops {
            match op {
                PredicateOp::Cmp { kind, op: cop } => {
                    let mut words = vec![0u64; nwords];
                    let dop = cop.to_dbinop();
                    match *kind {
                        LeafKind::FloatColFloatLit { col_idx, lit } => {
                            if let Column::Float(data) = &base.columns[col_idx].1 {
                                for &i in existing_indices {
                                    if scalar_cmp_f64(data[i], lit, dop) {
                                        words[i / 64] |= 1u64 << (i % 64);
                                    }
                                }
                            }
                        }
                        LeafKind::FloatColIntLit { col_idx, lit } => {
                            if let Column::Float(data) = &base.columns[col_idx].1 {
                                let lit_f = lit as f64;
                                for &i in existing_indices {
                                    if scalar_cmp_f64(data[i], lit_f, dop) {
                                        words[i / 64] |= 1u64 << (i % 64);
                                    }
                                }
                            }
                        }
                        LeafKind::IntColIntLit { col_idx, lit } => {
                            if let Column::Int(data) = &base.columns[col_idx].1 {
                                for &i in existing_indices {
                                    if scalar_cmp_i64(data[i], lit, dop) {
                                        words[i / 64] |= 1u64 << (i % 64);
                                    }
                                }
                            }
                        }
                        LeafKind::IntColFloatLit { col_idx, lit } => {
                            if let Column::Int(data) = &base.columns[col_idx].1 {
                                for &i in existing_indices {
                                    if scalar_cmp_f64(data[i] as f64, lit, dop) {
                                        words[i / 64] |= 1u64 << (i % 64);
                                    }
                                }
                            }
                        }
                    }
                    stack.push(words);
                }
                PredicateOp::And => {
                    let r = stack.pop().expect("predicate bytecode: And on empty stack");
                    let l = stack.pop().expect("predicate bytecode: And on empty stack");
                    let merged: Vec<u64> = l.iter().zip(r.iter()).map(|(a, b)| a & b).collect();
                    stack.push(merged);
                }
                PredicateOp::Or => {
                    let r = stack.pop().expect("predicate bytecode: Or on empty stack");
                    let l = stack.pop().expect("predicate bytecode: Or on empty stack");
                    let merged: Vec<u64> = l.iter().zip(r.iter()).map(|(a, b)| a | b).collect();
                    stack.push(merged);
                }
            }
        }

        let top = stack
            .pop()
            .expect("predicate bytecode: empty stack after interpretation");

        // No final AND with existing_mask needed: AND/OR are monotone, so
        // the result is already bounded by the input set, and the input set
        // is by precondition a subset of the existing mask.
        BitMask::from_words_for_test(top, nrows)
    }

    /// Interpret the bytecode against `base`, AND-merging the result with
    /// `existing_mask`. Result is bit-identical to AST-walk on every shape
    /// `lower` accepts.
    /// Evaluate the predicate **without** ANDing into an existing mask,
    /// classifying the result as an `AdaptiveSelection`. Used by the
    /// v3 Phase 3 production wiring path: when the existing selection is
    /// already a Hybrid (mid-band density, large nrows), we want the
    /// fresh predicate result to also be classified so the downstream
    /// `existing.intersect(fresh)` call can route through Phase 3's
    /// per-chunk dispatch instead of materialising-and-AND.
    ///
    /// Bit-equivalent to `interpret` followed by `AdaptiveSelection::
    /// from_predicate_result(...)`, but skips the per-call existing-mask
    /// AND so the classification reflects the predicate's own selectivity.
    pub fn evaluate_to_selection(
        &self,
        base: &DataFrame,
        nrows: usize,
    ) -> crate::adaptive_selection::AdaptiveSelection {
        let words = self.evaluate_words(base, nrows);
        crate::adaptive_selection::AdaptiveSelection::from_predicate_result(words, nrows)
    }

    /// Internal: evaluate the predicate and return the raw `Vec<u64>`
    /// bitmap words. Shared between `interpret` and
    /// `evaluate_to_selection`. Caller decides whether to AND with an
    /// existing mask, classify into AdaptiveSelection, or just wrap as
    /// BitMask.
    fn evaluate_words(&self, base: &DataFrame, nrows: usize) -> Vec<u64> {
        let nwords = nwords_for(nrows);
        let mut stack: Vec<Vec<u64>> = Vec::with_capacity(4);
        for op in &self.ops {
            match op {
                PredicateOp::Cmp { kind, op: cop } => {
                    let mut words = vec![0u64; nwords];
                    let dop = cop.to_dbinop();
                    match *kind {
                        LeafKind::FloatColFloatLit { col_idx, lit } => {
                            if let Column::Float(data) = &base.columns[col_idx].1 {
                                columnar_cmp_f64(data, lit, dop, &mut words);
                            }
                        }
                        LeafKind::FloatColIntLit { col_idx, lit } => {
                            if let Column::Float(data) = &base.columns[col_idx].1 {
                                columnar_cmp_f64(data, lit as f64, dop, &mut words);
                            }
                        }
                        LeafKind::IntColIntLit { col_idx, lit } => {
                            if let Column::Int(data) = &base.columns[col_idx].1 {
                                columnar_cmp_i64(data, lit, dop, &mut words);
                            }
                        }
                        LeafKind::IntColFloatLit { col_idx, lit } => {
                            if let Column::Int(data) = &base.columns[col_idx].1 {
                                let floats: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                                columnar_cmp_f64(&floats, lit, dop, &mut words);
                            }
                        }
                    }
                    stack.push(words);
                }
                PredicateOp::And => {
                    let r = stack.pop().expect("predicate bytecode: And on empty stack");
                    let l = stack.pop().expect("predicate bytecode: And on empty stack");
                    let merged: Vec<u64> = l.iter().zip(r.iter()).map(|(a, b)| a & b).collect();
                    stack.push(merged);
                }
                PredicateOp::Or => {
                    let r = stack.pop().expect("predicate bytecode: Or on empty stack");
                    let l = stack.pop().expect("predicate bytecode: Or on empty stack");
                    let merged: Vec<u64> = l.iter().zip(r.iter()).map(|(a, b)| a | b).collect();
                    stack.push(merged);
                }
            }
        }
        stack.pop().expect("predicate bytecode: empty stack after interpretation")
    }

    pub fn interpret(&self, base: &DataFrame, existing_mask: &BitMask) -> BitMask {
        let nrows = existing_mask.nrows();
        let mut top = self.evaluate_words(base, nrows);
        // Final AND with existing mask. Equivalent to AST-walk's per-leaf
        // AND because bitwise AND distributes over AND/OR trees:
        //   ((a | b) & c) & m == (a | b) & (c & m)  etc.
        for (w, ew) in top.iter_mut().zip(existing_mask.words_slice().iter()) {
            *w &= *ew;
        }
        BitMask::from_words_for_test(top, nrows)
    }
}

fn lower_into(predicate: &DExpr, base: &DataFrame, ops: &mut Vec<PredicateOp>) -> Option<()> {
    match predicate {
        DExpr::BinOp {
            op: DBinOp::And,
            left,
            right,
        } => {
            lower_into(left, base, ops)?;
            lower_into(right, base, ops)?;
            ops.push(PredicateOp::And);
            Some(())
        }
        DExpr::BinOp {
            op: DBinOp::Or,
            left,
            right,
        } => {
            lower_into(left, base, ops)?;
            lower_into(right, base, ops)?;
            ops.push(PredicateOp::Or);
            Some(())
        }
        DExpr::BinOp { op, left, right } => {
            let cmp = CmpKind::from_dbinop(*op)?;

            #[derive(Clone, Copy)]
            enum LitVal {
                F(f64),
                I(i64),
            }

            let (col_name, lit, reversed) = match (left.as_ref(), right.as_ref()) {
                (DExpr::Col(n), DExpr::LitFloat(v)) => (n.as_str(), LitVal::F(*v), false),
                (DExpr::LitFloat(v), DExpr::Col(n)) => (n.as_str(), LitVal::F(*v), true),
                (DExpr::Col(n), DExpr::LitInt(v)) => (n.as_str(), LitVal::I(*v), false),
                (DExpr::LitInt(v), DExpr::Col(n)) => (n.as_str(), LitVal::I(*v), true),
                _ => return None,
            };

            let effective = if reversed { cmp.flip() } else { cmp };

            let col_idx = base.columns.iter().position(|(n, _)| n == col_name)?;
            let kind = match (&base.columns[col_idx].1, lit) {
                (Column::Float(_), LitVal::F(v)) => LeafKind::FloatColFloatLit { col_idx, lit: v },
                (Column::Float(_), LitVal::I(v)) => LeafKind::FloatColIntLit { col_idx, lit: v },
                (Column::Int(_), LitVal::I(v)) => LeafKind::IntColIntLit { col_idx, lit: v },
                (Column::Int(_), LitVal::F(v)) => LeafKind::IntColFloatLit { col_idx, lit: v },
                _ => return None,
            };

            ops.push(PredicateOp::Cmp {
                kind,
                op: effective,
            });
            Some(())
        }
        _ => None,
    }
}

#[inline]
fn scalar_cmp_f64(val: f64, lit: f64, op: DBinOp) -> bool {
    match op {
        DBinOp::Gt => val > lit,
        DBinOp::Lt => val < lit,
        DBinOp::Ge => val >= lit,
        DBinOp::Le => val <= lit,
        DBinOp::Eq => val == lit,
        DBinOp::Ne => val != lit,
        _ => false,
    }
}

#[inline]
fn scalar_cmp_i64(val: i64, lit: i64, op: DBinOp) -> bool {
    match op {
        DBinOp::Gt => val > lit,
        DBinOp::Lt => val < lit,
        DBinOp::Ge => val >= lit,
        DBinOp::Le => val <= lit,
        DBinOp::Eq => val == lit,
        DBinOp::Ne => val != lit,
        _ => false,
    }
}

/// Density rule for picking `interpret_sparse` over `interpret`.
///
/// Below ~25% density the random-access gather over `existing_indices`
/// beats the sequential column scan: each `Cmp` opcode does ~`count`
/// scattered loads vs ~`nrows` sequential loads, and the sequential
/// streaming-prefetch advantage is roughly 4× per byte on modern x86.
/// Empirical break-even tracked the 25% boundary closely on the
/// `bench_density_crossover` workload, so we encode it directly as
/// `count * 4 < nrows` (integer-only — no float density boundary, per
/// the determinism contract).
#[inline]
pub fn should_use_sparse_path(count: usize, nrows: usize) -> bool {
    count.saturating_mul(4) < nrows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Column;

    fn df_with_int(name: &str, xs: Vec<i64>) -> DataFrame {
        DataFrame::from_columns(vec![(name.into(), Column::Int(xs))]).unwrap()
    }

    fn pred_lt(col: &str, v: i64) -> DExpr {
        DExpr::BinOp {
            op: DBinOp::Lt,
            left: Box::new(DExpr::Col(col.into())),
            right: Box::new(DExpr::LitInt(v)),
        }
    }

    #[test]
    fn lower_simple_lt() {
        let df = df_with_int("x", (0..10).collect());
        let bc = PredicateBytecode::lower(&pred_lt("x", 5), &df).unwrap();
        assert_eq!(bc.ops().len(), 1);
        match &bc.ops()[0] {
            PredicateOp::Cmp {
                kind: LeafKind::IntColIntLit { col_idx, lit },
                op,
            } => {
                assert_eq!(*col_idx, 0);
                assert_eq!(*lit, 5);
                assert_eq!(*op, CmpKind::Lt);
            }
            other => panic!("unexpected op: {:?}", other),
        }
    }

    #[test]
    fn lower_reversed_lit_op_col_flips() {
        // 5 > x  becomes  x < 5
        let df = df_with_int("x", (0..10).collect());
        let pred = DExpr::BinOp {
            op: DBinOp::Gt,
            left: Box::new(DExpr::LitInt(5)),
            right: Box::new(DExpr::Col("x".into())),
        };
        let bc = PredicateBytecode::lower(&pred, &df).unwrap();
        match &bc.ops()[0] {
            PredicateOp::Cmp { op, .. } => assert_eq!(*op, CmpKind::Lt),
            other => panic!("unexpected op: {:?}", other),
        }
    }

    #[test]
    fn lower_and_emits_postorder() {
        let df = df_with_int("x", (0..10).collect());
        let pred = DExpr::BinOp {
            op: DBinOp::And,
            left: Box::new(pred_lt("x", 5)),
            right: Box::new(pred_lt("x", 8)),
        };
        let bc = PredicateBytecode::lower(&pred, &df).unwrap();
        assert_eq!(bc.ops().len(), 3);
        assert!(matches!(bc.ops()[0], PredicateOp::Cmp { .. }));
        assert!(matches!(bc.ops()[1], PredicateOp::Cmp { .. }));
        assert!(matches!(bc.ops()[2], PredicateOp::And));
    }

    #[test]
    fn lower_unsupported_returns_none() {
        // Add op is not a comparison; not handled.
        let df = df_with_int("x", (0..10).collect());
        let pred = DExpr::BinOp {
            op: DBinOp::Add,
            left: Box::new(DExpr::Col("x".into())),
            right: Box::new(DExpr::LitInt(1)),
        };
        assert!(PredicateBytecode::lower(&pred, &df).is_none());
    }

    #[test]
    fn lower_unknown_column_returns_none() {
        let df = df_with_int("x", (0..10).collect());
        assert!(PredicateBytecode::lower(&pred_lt("ghost", 5), &df).is_none());
    }

    #[test]
    fn interpret_simple_lt_matches_count() {
        let df = df_with_int("x", (0..100).collect());
        let bc = PredicateBytecode::lower(&pred_lt("x", 30), &df).unwrap();
        let mask = bc.interpret(&df, &BitMask::all_true(100));
        assert_eq!(mask.count_ones(), 30);
        for i in 0..30 {
            assert!(mask.get(i));
        }
        for i in 30..100 {
            assert!(!mask.get(i));
        }
    }

    #[test]
    fn interpret_and_intersects() {
        let df = df_with_int("x", (0..100).collect());
        // x < 50 AND x < 30  ==  x < 30
        let pred = DExpr::BinOp {
            op: DBinOp::And,
            left: Box::new(pred_lt("x", 50)),
            right: Box::new(pred_lt("x", 30)),
        };
        let bc = PredicateBytecode::lower(&pred, &df).unwrap();
        let mask = bc.interpret(&df, &BitMask::all_true(100));
        assert_eq!(mask.count_ones(), 30);
    }

    #[test]
    fn interpret_or_unions() {
        let df = df_with_int("x", (0..100).collect());
        // x < 30 OR x < 70  ==  x < 70
        let pred = DExpr::BinOp {
            op: DBinOp::Or,
            left: Box::new(pred_lt("x", 30)),
            right: Box::new(pred_lt("x", 70)),
        };
        let bc = PredicateBytecode::lower(&pred, &df).unwrap();
        let mask = bc.interpret(&df, &BitMask::all_true(100));
        assert_eq!(mask.count_ones(), 70);
    }

    #[test]
    fn interpret_respects_existing_mask() {
        let df = df_with_int("x", (0..100).collect());
        // existing_mask = only odd indices
        let mut bools = vec![false; 100];
        for i in (1..100).step_by(2) {
            bools[i] = true;
        }
        let existing = BitMask::from_bools(&bools);

        let bc = PredicateBytecode::lower(&pred_lt("x", 50), &df).unwrap();
        let mask = bc.interpret(&df, &existing);

        // x < 50 has 50 hits; intersecting with odd-only gives indices 1,3,…,49 = 25 hits.
        assert_eq!(mask.count_ones(), 25);
    }

    #[test]
    fn interpret_int_col_float_lit_promotes() {
        // Verifies the IntColFloatLit arm reaches columnar_cmp_f64 with
        // i64→f64-promoted column data.
        let df = df_with_int("x", vec![1, 2, 3, 4, 5]);
        let pred = DExpr::BinOp {
            op: DBinOp::Lt,
            left: Box::new(DExpr::Col("x".into())),
            right: Box::new(DExpr::LitFloat(3.5)),
        };
        let bc = PredicateBytecode::lower(&pred, &df).unwrap();
        let mask = bc.interpret(&df, &BitMask::all_true(5));
        assert_eq!(mask.count_ones(), 3); // 1, 2, 3
    }

    #[test]
    fn interpret_nested_and_or() {
        let df = df_with_int("x", (0..100).collect());
        // (x < 10) OR (x > 90 AND x < 95)  ==  10 + 4 = 14
        let pred = DExpr::BinOp {
            op: DBinOp::Or,
            left: Box::new(pred_lt("x", 10)),
            right: Box::new(DExpr::BinOp {
                op: DBinOp::And,
                left: Box::new(DExpr::BinOp {
                    op: DBinOp::Gt,
                    left: Box::new(DExpr::Col("x".into())),
                    right: Box::new(DExpr::LitInt(90)),
                }),
                right: Box::new(pred_lt("x", 95)),
            }),
        };
        let bc = PredicateBytecode::lower(&pred, &df).unwrap();
        let mask = bc.interpret(&df, &BitMask::all_true(100));
        assert_eq!(mask.count_ones(), 14);
    }

    // ── v2.2 sparse-aware interpretation ───────────────────────────────

    #[test]
    fn sparse_path_density_threshold() {
        // 25% boundary, integer-only.
        assert!(should_use_sparse_path(0, 100));
        assert!(should_use_sparse_path(24, 100));
        assert!(!should_use_sparse_path(25, 100));
        assert!(!should_use_sparse_path(50, 100));
        // Edge: nrows == 0 (vacuously dense — no rows to gather).
        assert!(!should_use_sparse_path(0, 0));
    }

    #[test]
    fn interpret_sparse_simple_lt() {
        let df = df_with_int("x", (0..100).collect());
        let bc = PredicateBytecode::lower(&pred_lt("x", 30), &df).unwrap();
        // Existing set is every other row (sparse: 50/100 < threshold trivially false,
        // but use a sparse subset anyway to validate the gather)
        let existing: Vec<usize> = (0..100).step_by(4).collect(); // 0, 4, 8, ..., 96 → 25 indices
        let mask = bc.interpret_sparse(&df, &existing, 100);
        // Pass condition: x < 30, restricted to 0,4,8,...,96 → 0,4,8,...,28 = 8 hits.
        assert_eq!(mask.count_ones(), 8);
        for &i in &existing {
            assert_eq!(mask.get(i), i < 30);
        }
        // Bits outside existing must be zero.
        for i in 0..100 {
            if !existing.contains(&i) {
                assert!(!mask.get(i), "bit {} set but not in existing", i);
            }
        }
    }

    #[test]
    fn interpret_sparse_matches_dense_on_same_inputs() {
        // Parity oracle: for any predicate accepted by lower(), the sparse
        // path with `existing_indices = existing_mask.iter_set()` produces
        // bit-identical output to the dense path.
        let df = df_with_int("x", (-100..100).collect());

        let preds = [
            pred_lt("x", 0),
            pred_lt("x", 50),
            DExpr::BinOp {
                op: DBinOp::And,
                left: Box::new(pred_lt("x", 50)),
                right: Box::new(DExpr::BinOp {
                    op: DBinOp::Gt,
                    left: Box::new(DExpr::Col("x".into())),
                    right: Box::new(DExpr::LitInt(-50)),
                }),
            },
            DExpr::BinOp {
                op: DBinOp::Or,
                left: Box::new(pred_lt("x", -80)),
                right: Box::new(DExpr::BinOp {
                    op: DBinOp::Gt,
                    left: Box::new(DExpr::Col("x".into())),
                    right: Box::new(DExpr::LitInt(80)),
                }),
            },
        ];

        // Three masks: every other row, every fifth row, sparse 5%.
        let masks: Vec<Vec<bool>> = vec![
            (0..200).map(|i| i % 2 == 0).collect(),
            (0..200).map(|i| i % 5 == 0).collect(),
            (0..200).map(|i| i % 20 == 0).collect(),
        ];

        for pred in &preds {
            let bc = PredicateBytecode::lower(pred, &df).unwrap();
            for bools in &masks {
                let bm = BitMask::from_bools(bools);
                let dense = bc.interpret(&df, &bm);
                let indices: Vec<usize> = bm.iter_set().collect();
                let sparse = bc.interpret_sparse(&df, &indices, 200);
                assert_eq!(
                    dense.words_slice(),
                    sparse.words_slice(),
                    "sparse path diverged from dense path"
                );
            }
        }
    }

    #[test]
    fn interpret_sparse_or_monotone() {
        // Tests that OR over sparse inputs stays bounded to the input set —
        // i.e., a row that would pass OR in the full column scan but is not
        // in `existing_indices` must NOT be set in the sparse output.
        let df = df_with_int("x", (0..20).collect());
        // Predicate: x < 5 OR x > 15 — would normally pass at 0..5 and 16..20.
        let pred = DExpr::BinOp {
            op: DBinOp::Or,
            left: Box::new(pred_lt("x", 5)),
            right: Box::new(DExpr::BinOp {
                op: DBinOp::Gt,
                left: Box::new(DExpr::Col("x".into())),
                right: Box::new(DExpr::LitInt(15)),
            }),
        };
        let bc = PredicateBytecode::lower(&pred, &df).unwrap();
        // Existing: only even indices < 10. Of those, predicate passes at 0,2,4 (only).
        let existing: Vec<usize> = (0..10).step_by(2).collect(); // 0,2,4,6,8
        let mask = bc.interpret_sparse(&df, &existing, 20);
        assert_eq!(mask.count_ones(), 3); // 0, 2, 4
        for i in 16..20 {
            assert!(!mask.get(i), "OR-pass row {} leaked despite not being in existing", i);
        }
    }
}
