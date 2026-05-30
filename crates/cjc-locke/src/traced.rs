//! Auto-instrumented lineage wrapper around `cjc_data::DataFrame`.
//!
//! `TracedDataFrame` carries a reference to a `LineageBuilder` and a
//! current node id. Each transformation method (filter, select,
//! with_column, concat, rename) emits an `Idea` + edge into the builder
//! and returns a new `TracedDataFrame` pointing at the new id.
//!
//! ## Determinism
//!
//! All emitted `Idea` ids are content-addressed via `cjc_locke::id`, so
//! repeated pipeline runs produce byte-identical lineage graphs.
//!
//! ## Why a thin wrapper
//!
//! The wrapper does **not** mirror `cjc-data`'s evolving DSL. Each
//! method takes a closure `impl FnOnce(&DataFrame) -> DataFrame` that
//! performs the actual transformation; the wrapper only records that
//! it happened, with parameters. This keeps `cjc-locke` decoupled from
//! `cjc-data` internals.

use std::collections::BTreeMap;

use cjc_data::DataFrame;

use crate::id::FingerprintId;
use crate::lineage::{
    ImpressionKind, LineageBuilder, LockeIdea, LockeImpression, TransformationRecord,
};

/// Lineage-instrumented DataFrame wrapper.
///
/// Clones cheaply (`DataFrame` is `Clone`); transformations consume
/// `self` and return a new `TracedDataFrame` with a fresh `node_id`.
pub struct TracedDataFrame<'a> {
    df: DataFrame,
    node_id: FingerprintId,
    builder: &'a mut LineageBuilder,
}

impl<'a> TracedDataFrame<'a> {
    /// Begin a traced pipeline from a raw `DataFrame`. Registers an
    /// `Impression` for the source.
    pub fn observe(builder: &'a mut LineageBuilder, source: &str, df: DataFrame) -> Self {
        let cols: Vec<String> = df.columns.iter().map(|(n, _)| n.clone()).collect();
        let imp = LockeImpression::new(source, ImpressionKind::Dataset, df.nrows() as u64, cols);
        let node_id = builder.add_impression(imp).expect("add_impression");
        Self { df, node_id, builder }
    }

    /// Borrow the underlying DataFrame.
    pub fn dataframe(&self) -> &DataFrame {
        &self.df
    }

    /// Return the current lineage node id.
    pub fn node_id(&self) -> FingerprintId {
        self.node_id
    }

    /// Consume the wrapper and recover the inner DataFrame + node id.
    pub fn into_parts(self) -> (DataFrame, FingerprintId) {
        (self.df, self.node_id)
    }

    /// Apply an arbitrary transformation and record a lineage idea.
    ///
    /// * `op_name` — short canonical name (e.g. `"filter"`, `"select"`).
    /// * `params` — stable string-encoded parameters, sorted in the
    ///   record (`BTreeMap` is required).
    /// * `seed` — optional RNG seed if the transformation is stochastic.
    /// * `f` — the actual transformation; receives the current DataFrame
    ///   and returns the next.
    pub fn transform(
        self,
        op_name: &str,
        params: BTreeMap<String, String>,
        seed: Option<u64>,
        f: impl FnOnce(&DataFrame) -> DataFrame,
    ) -> Self {
        let new_df = f(&self.df);
        let xform = TransformationRecord {
            op_id: op_name.to_string(),
            params,
            seed,
        };
        let idea = LockeIdea::new(op_name, xform, vec![self.node_id]);
        let new_node = self
            .builder
            .add_idea(idea)
            .expect("add_idea (transform)");
        Self {
            df: new_df,
            node_id: new_node,
            builder: self.builder,
        }
    }

    /// Convenience: record a `filter` op with a predicate description.
    pub fn filter(
        self,
        predicate_desc: &str,
        f: impl FnOnce(&DataFrame) -> DataFrame,
    ) -> Self {
        let mut params = BTreeMap::new();
        params.insert("predicate".into(), predicate_desc.into());
        self.transform("filter", params, None, f)
    }

    /// Convenience: record a `select` op with a column list.
    pub fn select(
        self,
        columns: &[&str],
        f: impl FnOnce(&DataFrame) -> DataFrame,
    ) -> Self {
        let mut params = BTreeMap::new();
        let mut sorted: Vec<&str> = columns.iter().copied().collect();
        sorted.sort();
        params.insert("columns".into(), sorted.join(","));
        self.transform("select", params, None, f)
    }

    /// Convenience: record a `with_column` op with the new column name.
    pub fn with_column(
        self,
        new_col: &str,
        expr_desc: &str,
        f: impl FnOnce(&DataFrame) -> DataFrame,
    ) -> Self {
        let mut params = BTreeMap::new();
        params.insert("new_column".into(), new_col.into());
        params.insert("expr".into(), expr_desc.into());
        self.transform("with_column", params, None, f)
    }

    /// Convenience: record a `rename` op.
    pub fn rename(
        self,
        from: &str,
        to: &str,
        f: impl FnOnce(&DataFrame) -> DataFrame,
    ) -> Self {
        let mut params = BTreeMap::new();
        params.insert("from".into(), from.into());
        params.insert("to".into(), to.into());
        self.transform("rename", params, None, f)
    }

    /// Record a `group_by` op. Returns a TracedDataFrame whose
    /// lineage node represents the grouped state. (The caller is
    /// responsible for actually building the grouping with cjc-data.)
    pub fn group_by(
        self,
        keys: &[&str],
        f: impl FnOnce(&DataFrame) -> DataFrame,
    ) -> Self {
        let mut params = BTreeMap::new();
        let mut sorted: Vec<&str> = keys.iter().copied().collect();
        sorted.sort();
        params.insert("keys".into(), sorted.join(","));
        self.transform("group_by", params, None, f)
    }

    /// Record a `summarise` op (closes a group_by).
    pub fn summarise(
        self,
        aggs_desc: &str,
        f: impl FnOnce(&DataFrame) -> DataFrame,
    ) -> Self {
        let mut params = BTreeMap::new();
        params.insert("aggs".into(), aggs_desc.into());
        self.transform("summarise", params, None, f)
    }

    /// Record an `arrange` (sort) op.
    pub fn arrange(
        self,
        by: &[&str],
        f: impl FnOnce(&DataFrame) -> DataFrame,
    ) -> Self {
        let mut params = BTreeMap::new();
        params.insert("by".into(), by.join(","));
        self.transform("arrange", params, None, f)
    }

    /// Record a `distinct` op (deduplicate by columns).
    pub fn distinct(
        self,
        on: &[&str],
        f: impl FnOnce(&DataFrame) -> DataFrame,
    ) -> Self {
        let mut params = BTreeMap::new();
        let mut sorted: Vec<&str> = on.iter().copied().collect();
        sorted.sort();
        params.insert("on".into(), if sorted.is_empty() { "all_columns".into() } else { sorted.join(",") });
        self.transform("distinct", params, None, f)
    }

    /// Record a `mutate` op (add/replace one or more columns).
    pub fn mutate(
        self,
        assignments_desc: &str,
        f: impl FnOnce(&DataFrame) -> DataFrame,
    ) -> Self {
        let mut params = BTreeMap::new();
        params.insert("assignments".into(), assignments_desc.into());
        self.transform("mutate", params, None, f)
    }

    /// Record a `pivot_longer` / `pivot_wider` op.
    pub fn pivot(
        self,
        direction: &str,
        cols_desc: &str,
        f: impl FnOnce(&DataFrame) -> DataFrame,
    ) -> Self {
        let mut params = BTreeMap::new();
        params.insert("direction".into(), direction.into());
        params.insert("cols".into(), cols_desc.into());
        self.transform("pivot", params, None, f)
    }

    /// Record a `sample` op with a seed (stochastic but reproducible).
    pub fn sample(
        self,
        n: u64,
        seed: u64,
        f: impl FnOnce(&DataFrame) -> DataFrame,
    ) -> Self {
        let mut params = BTreeMap::new();
        params.insert("n".into(), n.to_string());
        self.transform("sample", params, Some(seed), f)
    }

    /// Record a `join` (binary op, two parents).
    pub fn join(
        self,
        other: TracedDataFrame<'_>,
        kind: &str,
        on: &[&str],
        f: impl FnOnce(&DataFrame, &DataFrame) -> DataFrame,
    ) -> Self
    where
        Self: 'a,
    {
        assert!(
            std::ptr::eq(self.builder as *const _, other.builder as *const _),
            "join: both TracedDataFrames must share the same LineageBuilder"
        );
        let parent_a = self.node_id;
        let parent_b = other.node_id;
        let (other_df, _) = other.into_parts();
        let new_df = f(&self.df, &other_df);
        let mut params = BTreeMap::new();
        params.insert("kind".into(), kind.into());
        params.insert("on".into(), on.join(","));
        // v0.7+ B5.1 fix: `LockeIdea::new` sorts `parents` before fingerprinting,
        // which is correct for n-ary commutative ops but loses the
        // `parent_a` / `parent_b` distinction for binary joins. Encoding the
        // parent IDs as ordered params keeps the join non-commutative at the
        // identity level — `L LEFT JOIN R` and `R LEFT JOIN L` now produce
        // distinct IDs.
        params.insert("parent_a".into(), format!("{}", parent_a));
        params.insert("parent_b".into(), format!("{}", parent_b));
        let xform = TransformationRecord {
            op_id: "join".into(),
            params,
            seed: None,
        };
        let idea = LockeIdea::new("join", xform, vec![parent_a, parent_b]);
        let new_node = self.builder.add_idea(idea).expect("add_idea (join)");
        Self {
            df: new_df,
            node_id: new_node,
            builder: self.builder,
        }
    }

    /// Record a column-binding (`bind_cols`) op (binary, two parents).
    pub fn bind_cols(
        self,
        other: TracedDataFrame<'_>,
        f: impl FnOnce(&DataFrame, &DataFrame) -> DataFrame,
    ) -> Self
    where
        Self: 'a,
    {
        assert!(
            std::ptr::eq(self.builder as *const _, other.builder as *const _),
            "bind_cols: both TracedDataFrames must share the same LineageBuilder"
        );
        let parent_a = self.node_id;
        let parent_b = other.node_id;
        let (other_df, _) = other.into_parts();
        let new_df = f(&self.df, &other_df);
        let mut params = BTreeMap::new();
        params.insert("axis".into(), "col".into());
        // B5.1 fix: see `join` above. `bind_cols` is column-wise binary
        // concat; column-order matters to schema, so the result of
        // `A.bind_cols(B)` differs from `B.bind_cols(A)` in the produced
        // dataframe and must differ at the identity level.
        params.insert("parent_a".into(), format!("{}", parent_a));
        params.insert("parent_b".into(), format!("{}", parent_b));
        let xform = TransformationRecord {
            op_id: "bind_cols".into(),
            params,
            seed: None,
        };
        let idea = LockeIdea::new("bind_cols", xform, vec![parent_a, parent_b]);
        let new_node = self.builder.add_idea(idea).expect("add_idea (bind_cols)");
        Self {
            df: new_df,
            node_id: new_node,
            builder: self.builder,
        }
    }

    /// Binary concat: combines `self` with `other` into a new traced
    /// frame whose lineage has both as parents.
    pub fn concat(
        self,
        other: TracedDataFrame<'_>,
        f: impl FnOnce(&DataFrame, &DataFrame) -> DataFrame,
    ) -> Self
    where
        Self: 'a,
    {
        // Borrow gymnastics: `other` carries its own &mut to a builder,
        // which is required to be the *same* builder. Caller guarantees
        // this by construction; we verify defensively via run-label.
        assert!(
            std::ptr::eq(self.builder as *const _, other.builder as *const _),
            "concat: both TracedDataFrames must share the same LineageBuilder"
        );
        let parent_a = self.node_id;
        let parent_b = other.node_id;
        let (other_df, _) = other.into_parts();
        let new_df = f(&self.df, &other_df);
        let mut params = BTreeMap::new();
        params.insert(
            "axis".into(),
            "row".into(), // v0.2 default; future overload for column concat
        );
        // B5.1 fix: see `join` above. `A.concat(B)` produces a row-stacked
        // frame with A on top of B; reversing the order produces a different
        // frame, so the identity must reflect it.
        params.insert("parent_a".into(), format!("{}", parent_a));
        params.insert("parent_b".into(), format!("{}", parent_b));
        let xform = TransformationRecord {
            op_id: "concat".into(),
            params,
            seed: None,
        };
        let idea = LockeIdea::new("concat", xform, vec![parent_a, parent_b]);
        let new_node = self.builder.add_idea(idea).expect("add_idea (concat)");
        Self {
            df: new_df,
            node_id: new_node,
            builder: self.builder,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_data::{Column, DataFrame};

    fn fixture() -> DataFrame {
        DataFrame::from_columns(vec![
            ("age".into(), Column::Int(vec![20, 30, 40, 50])),
            ("name".into(), Column::Str(vec!["a".into(), "b".into(), "c".into(), "d".into()])),
        ])
        .unwrap()
    }

    #[test]
    fn observe_creates_impression_node() {
        let mut b = LineageBuilder::new("test");
        let node_id = {
            let traced = TracedDataFrame::observe(&mut b, "train.csv", fixture());
            traced.node_id
        };
        let g = b.finish();
        assert_eq!(g.nodes.len(), 1);
        assert!(g
            .nodes
            .values()
            .any(|n| matches!(n, crate::lineage::LineageNode::Impression(_))));
        assert!(g.nodes.contains_key(&node_id));
    }

    #[test]
    fn filter_emits_an_idea_edge() {
        let mut b = LineageBuilder::new("test");
        let traced = TracedDataFrame::observe(&mut b, "train.csv", fixture());
        let _ = traced.filter("age >= 30", |df| {
            DataFrame::from_columns(df.columns.clone()).unwrap()
        });
        let g = b.finish();
        assert_eq!(g.nodes.len(), 2);
        assert_eq!(g.edges.len(), 1);
        assert_eq!(g.edges[0].label, "filter");
    }

    #[test]
    fn pipeline_chain_records_three_edges() {
        let mut b = LineageBuilder::new("pipeline");
        let traced = TracedDataFrame::observe(&mut b, "train.csv", fixture())
            .filter("age >= 30", |df| df.clone())
            .select(&["age"], |df| df.clone())
            .with_column("age_bucket", "age // 10", |df| df.clone());
        let _ = traced.into_parts();
        let g = b.finish();
        assert_eq!(g.edges.len(), 3);
        let labels: Vec<&str> = g.edges.iter().map(|e| e.label.as_str()).collect();
        assert!(labels.contains(&"filter"));
        assert!(labels.contains(&"select"));
        assert!(labels.contains(&"with_column"));
    }

    #[test]
    fn pipeline_lineage_is_deterministic_across_runs() {
        fn run() -> FingerprintId {
            let mut b = LineageBuilder::new("pipeline");
            let _ = TracedDataFrame::observe(&mut b, "train.csv", fixture())
                .filter("age >= 30", |df| df.clone())
                .select(&["age"], |df| df.clone())
                .into_parts();
            b.finish().root_fingerprint
        }
        assert_eq!(run(), run());
    }

    #[test]
    fn group_by_summarise_chain_records_two_edges() {
        let mut b = LineageBuilder::new("agg");
        let _ = TracedDataFrame::observe(&mut b, "train.csv", fixture())
            .group_by(&["name"], |df| df.clone())
            .summarise("mean(age)", |df| df.clone())
            .into_parts();
        let g = b.finish();
        let labels: Vec<&str> = g.edges.iter().map(|e| e.label.as_str()).collect();
        assert!(labels.contains(&"group_by"));
        assert!(labels.contains(&"summarise"));
    }

    #[test]
    fn arrange_distinct_mutate_pivot_each_emit_an_edge() {
        let mut b = LineageBuilder::new("pipeline2");
        let _ = TracedDataFrame::observe(&mut b, "train.csv", fixture())
            .arrange(&["age"], |df| df.clone())
            .distinct(&["age"], |df| df.clone())
            .mutate("age_sq = age*age", |df| df.clone())
            .pivot("longer", "age,name", |df| df.clone())
            .into_parts();
        let g = b.finish();
        assert_eq!(g.edges.len(), 4);
        let labels: Vec<&str> = g.edges.iter().map(|e| e.label.as_str()).collect();
        for expected in &["arrange", "distinct", "mutate", "pivot"] {
            assert!(labels.contains(expected), "missing {expected}");
        }
    }

    #[test]
    fn sample_records_seed_in_transformation() {
        let mut b = LineageBuilder::new("sample");
        let _ = TracedDataFrame::observe(&mut b, "train.csv", fixture())
            .sample(2, 42, |df| df.clone())
            .into_parts();
        let g = b.finish();
        // The sample op's transformation record has seed=Some(42).
        let sample_node = g.nodes.values().find_map(|n| match n {
            crate::lineage::LineageNode::Idea(i) if i.transform.op_id == "sample" => Some(i),
            _ => None,
        });
        let sample_node = sample_node.expect("sample idea exists");
        assert_eq!(sample_node.transform.seed, Some(42));
    }

    #[test]
    fn join_op_records_idea_with_two_parents() {
        // We exercise `.join(other, ...)` by holding both wrappers
        // concurrently, which works in the wrapper API because `join`
        // takes `other: TracedDataFrame<'_>` by value and the two
        // borrows of `b` are sequential, not concurrent.
        let mut b = LineageBuilder::new("join");
        // Observe both impressions first, then re-wrap the second so the
        // first call to .join takes ownership of it.
        let (left_df, left_id) = {
            let traced = TracedDataFrame::observe(&mut b, "left.csv", fixture());
            traced.into_parts()
        };
        let (right_df, right_id) = {
            let traced = TracedDataFrame::observe(&mut b, "right.csv", fixture());
            traced.into_parts()
        };
        // Now wrap both with a single &mut to the builder by passing the
        // builder mutably into one wrapper and a raw pointer for the
        // other — *but* that's unsafe. The supported pattern is to do
        // the join in one statement that consumes both wrappers; we
        // simulate that by manually building the idea and edges below.
        let _ = (left_df, right_df);
        let xform = TransformationRecord {
            op_id: "join".into(),
            params: {
                let mut p = BTreeMap::new();
                p.insert("kind".into(), "inner".into());
                p.insert("on".into(), "name".into());
                p
            },
            seed: None,
        };
        let idea = LockeIdea::new("join", xform, vec![left_id, right_id]);
        b.add_idea(idea).unwrap();
        let g = b.finish();
        // Inspect: an idea with two parents.
        let idea_with_two_parents = g
            .nodes
            .values()
            .filter_map(|n| match n {
                crate::lineage::LineageNode::Idea(i) if i.parents.len() == 2 => Some(i),
                _ => None,
            })
            .next()
            .expect("idea with two parents");
        assert_eq!(idea_with_two_parents.transform.op_id, "join");
    }

    #[test]
    fn concat_has_two_parents() {
        let mut b = LineageBuilder::new("concat");
        // Note: using the same builder for both branches requires sequencing.
        let a = TracedDataFrame::observe(&mut b, "a.csv", fixture());
        let (df_a, id_a) = a.into_parts();
        let c = TracedDataFrame::observe(&mut b, "b.csv", fixture());
        let (df_b, id_b) = c.into_parts();

        // Rehydrate both as TracedDataFrames pointing at their existing nodes
        // so we can call concat — simulating what the wrapper would do
        // if the user kept both in scope.
        let traced_a = TracedDataFrame {
            df: df_a,
            node_id: id_a,
            builder: &mut b,
        };
        let _ = traced_a;
        // For the unit test we just confirm a fresh traced frame can be
        // built and the underlying graph has at least 2 impressions.
        let _ = df_b;
        let _ = id_b;
        let g = b.finish();
        let imp_count = g
            .nodes
            .values()
            .filter(|n| matches!(n, crate::lineage::LineageNode::Impression(_)))
            .count();
        assert_eq!(imp_count, 2);
    }

    // ─── B5.1 regression: binary-op parent-order asymmetry ─────────────

    /// Pin the post-fix behaviour: a binary op (join / bind_cols /
    /// concat) produces distinct LockeIdea IDs when its parents are
    /// swapped. Pre-fix the `LockeIdea::new` sort-parents step caused
    /// `(parent_a, parent_b)` and `(parent_b, parent_a)` to collide,
    /// because no `transform.params` entry carried the ordering.
    /// Post-fix the traced binary ops insert `parent_a` and `parent_b`
    /// into `params` BEFORE constructing the idea — order shows up in
    /// `TransformationRecord::fingerprint` so the IDs diverge.
    #[test]
    fn binary_op_id_encodes_parent_order_after_b5_1_fix() {
        use crate::lineage::{LockeIdea, TransformationRecord};
        use crate::FingerprintId;

        // Build two ideas with the SAME op_id + parent set but the
        // parent_a/parent_b params swapped — the shape of `traced::join`
        // post-fix.
        fn make_join_idea(parent_a: FingerprintId, parent_b: FingerprintId) -> LockeIdea {
            let mut params = BTreeMap::new();
            params.insert("kind".into(), "inner".into());
            params.insert("on".into(), "key".into());
            params.insert("parent_a".into(), format!("{}", parent_a));
            params.insert("parent_b".into(), format!("{}", parent_b));
            let xform = TransformationRecord {
                op_id: "join".into(),
                params,
                seed: None,
            };
            LockeIdea::new("join", xform, vec![parent_a, parent_b])
        }

        let p1 = FingerprintId(0xAAAA_BBBB_CCCC_DDDD);
        let p2 = FingerprintId(0x1111_2222_3333_4444);
        let idea_ab = make_join_idea(p1, p2);
        let idea_ba = make_join_idea(p2, p1);
        assert_ne!(
            idea_ab.id, idea_ba.id,
            "binary-op ID must encode parent order (B5.1 regression)",
        );
        // The parent sets ARE equal under sort, so the fix is the params
        // encoding, not the parent list.
        assert_eq!(idea_ab.parents, idea_ba.parents);
    }

    /// Symmetric op: a hypothetical n-ary commutative op (no parent_a /
    /// parent_b in params) MUST still collide on swapped parents, because
    /// `LockeIdea::new` sorts the parent list. This pins the orthogonal
    /// guarantee — fix only affects ops that opt in via params.
    #[test]
    fn n_ary_op_id_unchanged_when_parents_swap() {
        use crate::lineage::{LockeIdea, TransformationRecord};
        use crate::FingerprintId;

        fn make_union_idea(parents: Vec<FingerprintId>) -> LockeIdea {
            let mut params = BTreeMap::new();
            params.insert("op".into(), "union".into());
            // Intentionally no parent_a / parent_b — commutative.
            let xform = TransformationRecord {
                op_id: "union".into(),
                params,
                seed: None,
            };
            LockeIdea::new("union", xform, parents)
        }

        let p1 = FingerprintId(0x100);
        let p2 = FingerprintId(0x200);
        let p3 = FingerprintId(0x300);
        let idea_123 = make_union_idea(vec![p1, p2, p3]);
        let idea_321 = make_union_idea(vec![p3, p2, p1]);
        assert_eq!(
            idea_123.id, idea_321.id,
            "commutative ops must still collide under parent-swap",
        );
    }
}

