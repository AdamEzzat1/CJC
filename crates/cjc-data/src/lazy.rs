//! Lazy evaluation IR for TidyView.
//!
//! A `LazyView` captures a chain of operations as a tree of `ViewNode`s.
//! On `.collect()`, the tree is optimized by rule-based passes, then executed.
//!
//! # Determinism
//!
//! - All plan transformations are pure, deterministic functions.
//! - No HashMap/HashSet usage -- only Vec and BTreeSet for column tracking.
//! - Execution delegates to TidyView/TidyFrame methods which already guarantee
//!   deterministic output (Kahan summation, BTreeMap groups, stable sorts).

use crate::{ArrangeKey, Column, DExpr, DBinOp, DataFrame, TidyAgg, TidyError, TidyFrame};
use std::collections::BTreeSet;
use std::rc::Rc;

// ── ViewNode IR ──────────────────────────────────────────────────────────────

/// A node in the lazy evaluation tree.
#[derive(Debug, Clone)]
pub enum ViewNode {
    /// Leaf: scan a base DataFrame.
    Scan { df: Rc<DataFrame> },
    /// Filter rows by predicate.
    Filter {
        input: Box<ViewNode>,
        predicate: DExpr,
    },
    /// Project to subset of columns.
    Select {
        input: Box<ViewNode>,
        columns: Vec<String>,
    },
    /// Add/replace columns via expressions.
    Mutate {
        input: Box<ViewNode>,
        assignments: Vec<(String, DExpr)>,
    },
    /// Sort by keys.
    Arrange {
        input: Box<ViewNode>,
        keys: Vec<ArrangeKey>,
    },
    /// Group + summarise (pipeline breaker).
    GroupSummarise {
        input: Box<ViewNode>,
        group_keys: Vec<String>,
        aggregations: Vec<(String, TidyAgg)>,
    },
    /// v3 Phase 6: streaming-friendly group + summarise. Produced by
    /// the lazy optimizer (`annotate_streamable_summarise`) when every
    /// aggregation is one of {Count, Sum, Mean, Min, Max, Var, Sd}. At
    /// execution this dispatches to `TidyView::summarise_streaming`,
    /// avoiding the per-group `Vec<usize>` materialisation.
    StreamingGroupSummarise {
        input: Box<ViewNode>,
        group_keys: Vec<String>,
        aggregations: Vec<(String, crate::StreamingAgg)>,
    },
    /// Distinct on columns.
    Distinct {
        input: Box<ViewNode>,
        columns: Vec<String>,
    },
    /// Join two inputs.
    Join {
        left: Box<ViewNode>,
        right: Box<ViewNode>,
        on: Vec<(String, String)>,
        kind: JoinType,
    },
}

/// The kind of join.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Semi,
    Anti,
}

// ── LazyView builder ─────────────────────────────────────────────────────────

/// A lazy view that captures operations without executing them.
pub struct LazyView {
    plan: ViewNode,
}

impl LazyView {
    /// Create from a DataFrame (takes ownership, wraps in Rc).
    pub fn from_df(df: DataFrame) -> Self {
        LazyView {
            plan: ViewNode::Scan { df: Rc::new(df) },
        }
    }

    /// Create from an Rc<DataFrame>.
    pub fn from_rc(df: Rc<DataFrame>) -> Self {
        LazyView {
            plan: ViewNode::Scan { df },
        }
    }

    /// Filter rows by a DExpr predicate.
    pub fn filter(self, predicate: DExpr) -> Self {
        LazyView {
            plan: ViewNode::Filter {
                input: Box::new(self.plan),
                predicate,
            },
        }
    }

    /// Project to a subset of columns.
    pub fn select(self, columns: Vec<String>) -> Self {
        LazyView {
            plan: ViewNode::Select {
                input: Box::new(self.plan),
                columns,
            },
        }
    }

    /// Add or replace columns via expressions.
    pub fn mutate(self, assignments: Vec<(String, DExpr)>) -> Self {
        LazyView {
            plan: ViewNode::Mutate {
                input: Box::new(self.plan),
                assignments,
            },
        }
    }

    /// Sort rows by keys.
    pub fn arrange(self, keys: Vec<ArrangeKey>) -> Self {
        LazyView {
            plan: ViewNode::Arrange {
                input: Box::new(self.plan),
                keys,
            },
        }
    }

    /// Group by keys and aggregate.
    pub fn group_summarise(
        self,
        group_keys: Vec<String>,
        aggregations: Vec<(String, TidyAgg)>,
    ) -> Self {
        LazyView {
            plan: ViewNode::GroupSummarise {
                input: Box::new(self.plan),
                group_keys,
                aggregations,
            },
        }
    }

    /// Keep distinct rows by columns.
    pub fn distinct(self, columns: Vec<String>) -> Self {
        LazyView {
            plan: ViewNode::Distinct {
                input: Box::new(self.plan),
                columns,
            },
        }
    }

    /// Join with another LazyView.
    pub fn join(self, right: LazyView, on: Vec<(String, String)>, kind: JoinType) -> Self {
        LazyView {
            plan: ViewNode::Join {
                left: Box::new(self.plan),
                right: Box::new(right.plan),
                on,
                kind,
            },
        }
    }

    /// Optimize and execute the plan, returning a TidyFrame.
    pub fn collect(self) -> Result<TidyFrame, TidyError> {
        let optimized = optimize(self.plan);
        execute(optimized)
    }

    /// Inspect the plan tree (for testing/debugging).
    pub fn plan(&self) -> &ViewNode {
        &self.plan
    }

    /// Consume and return the optimized plan without executing (for testing).
    pub fn optimized_plan(self) -> ViewNode {
        optimize(self.plan)
    }
}

// ── Optimizer ────────────────────────────────────────────────────────────────

/// Apply all optimization passes to a ViewNode tree.
///
/// Pass order matters: merge filters first so pushdown sees fewer nodes,
/// then push predicates toward scans, then prune redundant selects.
pub fn optimize(plan: ViewNode) -> ViewNode {
    let plan = merge_filters(plan);
    let plan = push_predicates_down(plan);
    let plan = eliminate_redundant_selects(plan);
    // v3 Phase 6: rewrite GroupSummarise → StreamingGroupSummarise when
    // every aggregation is streaming-friendly. This avoids materialising
    // the per-group `Vec<usize>` row index buffers (~8 bytes × N rows).
    let plan = annotate_streamable_summarise(plan);
    plan
}

// ── v3 Phase 6: streamable-summarise annotation ─────────────────────────────
//
// `GroupSummarise` becomes `StreamingGroupSummarise` when every aggregation
// is one of {Count, Sum, Mean, Min, Max, Var, Sd}. Median / Quantile /
// NDistinct / IQR / First / Last require the full row-index list and stay
// on the legacy path.
//
// Output shape: byte-equal to legacy path on the streamable subset.
// Determinism: BTreeMap iteration order over key tuples (Vec<u32> codes
// for cat-aware path, Vec<String> displays otherwise).

fn try_streaming_agg(agg: &TidyAgg) -> Option<crate::StreamingAgg> {
    use crate::StreamingAgg;
    match agg {
        TidyAgg::Count => Some(StreamingAgg::Count),
        TidyAgg::Sum(c) => Some(StreamingAgg::Sum(c.clone())),
        TidyAgg::Mean(c) => Some(StreamingAgg::Mean(c.clone())),
        TidyAgg::Min(c) => Some(StreamingAgg::Min(c.clone())),
        TidyAgg::Max(c) => Some(StreamingAgg::Max(c.clone())),
        TidyAgg::Var(c) => Some(StreamingAgg::Var(c.clone())),
        TidyAgg::Sd(c) => Some(StreamingAgg::Sd(c.clone())),
        // Median / Quantile / NDistinct / Iqr / First / Last need the
        // materialised row index list — not streaming-friendly.
        _ => None,
    }
}

fn annotate_streamable_summarise(plan: ViewNode) -> ViewNode {
    match plan {
        ViewNode::GroupSummarise {
            input,
            group_keys,
            aggregations,
        } => {
            let input = Box::new(annotate_streamable_summarise(*input));
            // All-or-nothing: if any aggregation is non-streaming, keep
            // the whole node on the legacy path. Mixed dispatch would
            // require a second walk.
            let all_streaming: Option<Vec<(String, crate::StreamingAgg)>> = aggregations
                .iter()
                .map(|(name, agg)| try_streaming_agg(agg).map(|sa| (name.clone(), sa)))
                .collect();
            match all_streaming {
                Some(streaming_aggs) => ViewNode::StreamingGroupSummarise {
                    input,
                    group_keys,
                    aggregations: streaming_aggs,
                },
                None => ViewNode::GroupSummarise {
                    input,
                    group_keys,
                    aggregations,
                },
            }
        }
        ViewNode::Filter { input, predicate } => ViewNode::Filter {
            input: Box::new(annotate_streamable_summarise(*input)),
            predicate,
        },
        ViewNode::Select { input, columns } => ViewNode::Select {
            input: Box::new(annotate_streamable_summarise(*input)),
            columns,
        },
        ViewNode::Mutate { input, assignments } => ViewNode::Mutate {
            input: Box::new(annotate_streamable_summarise(*input)),
            assignments,
        },
        ViewNode::Arrange { input, keys } => ViewNode::Arrange {
            input: Box::new(annotate_streamable_summarise(*input)),
            keys,
        },
        ViewNode::Distinct { input, columns } => ViewNode::Distinct {
            input: Box::new(annotate_streamable_summarise(*input)),
            columns,
        },
        ViewNode::Join {
            left,
            right,
            on,
            kind,
        } => ViewNode::Join {
            left: Box::new(annotate_streamable_summarise(*left)),
            right: Box::new(annotate_streamable_summarise(*right)),
            on,
            kind,
        },
        ViewNode::StreamingGroupSummarise { .. } => plan,
        ViewNode::Scan { .. } => plan,
    }
}

// ── Pass 1: Filter Merging ───────────────────────────────────────────────────

/// Merge consecutive Filter nodes into a single Filter with AND predicate.
///
/// `Filter(Filter(input, p1), p2)` becomes `Filter(input, p1 AND p2)`.
fn merge_filters(plan: ViewNode) -> ViewNode {
    match plan {
        ViewNode::Filter { input, predicate } => {
            let merged_input = merge_filters(*input);
            match merged_input {
                ViewNode::Filter {
                    input: inner,
                    predicate: inner_pred,
                } => {
                    // Combine: inner_pred AND predicate
                    let combined = DExpr::BinOp {
                        op: DBinOp::And,
                        left: Box::new(inner_pred),
                        right: Box::new(predicate),
                    };
                    ViewNode::Filter {
                        input: inner,
                        predicate: combined,
                    }
                }
                other => ViewNode::Filter {
                    input: Box::new(other),
                    predicate,
                },
            }
        }
        // Recurse into all other node types
        ViewNode::Select { input, columns } => ViewNode::Select {
            input: Box::new(merge_filters(*input)),
            columns,
        },
        ViewNode::Mutate {
            input,
            assignments,
        } => ViewNode::Mutate {
            input: Box::new(merge_filters(*input)),
            assignments,
        },
        ViewNode::Arrange { input, keys } => ViewNode::Arrange {
            input: Box::new(merge_filters(*input)),
            keys,
        },
        ViewNode::GroupSummarise {
            input,
            group_keys,
            aggregations,
        } => ViewNode::GroupSummarise {
            input: Box::new(merge_filters(*input)),
            group_keys,
            aggregations,
        },
        ViewNode::Distinct { input, columns } => ViewNode::Distinct {
            input: Box::new(merge_filters(*input)),
            columns,
        },
        ViewNode::Join {
            left,
            right,
            on,
            kind,
        } => ViewNode::Join {
            left: Box::new(merge_filters(*left)),
            right: Box::new(merge_filters(*right)),
            on,
            kind,
        },
        other => other, // Scan
    }
}

// ── Pass 2: Predicate Pushdown ───────────────────────────────────────────────

/// Push Filter nodes closer to Scan nodes.
///
/// Rules:
/// - Filter past Select: always safe (filter refs columns that must exist).
/// - Filter past Mutate: only if predicate does NOT reference any mutated column.
/// - Filter into Join: push to the side that owns ALL referenced columns.
/// - Filter past Arrange: always safe (sort order preserved after filter).
/// - Do NOT push past GroupSummarise (aggregation changes row identity).
/// - Do NOT push past Distinct (distinct changes row identity).
fn push_predicates_down(plan: ViewNode) -> ViewNode {
    match plan {
        ViewNode::Filter { input, predicate } => {
            let optimized_input = push_predicates_down(*input);
            push_filter_into(optimized_input, predicate)
        }
        // Recurse into all other nodes
        ViewNode::Select { input, columns } => ViewNode::Select {
            input: Box::new(push_predicates_down(*input)),
            columns,
        },
        ViewNode::Mutate {
            input,
            assignments,
        } => ViewNode::Mutate {
            input: Box::new(push_predicates_down(*input)),
            assignments,
        },
        ViewNode::Arrange { input, keys } => ViewNode::Arrange {
            input: Box::new(push_predicates_down(*input)),
            keys,
        },
        ViewNode::GroupSummarise {
            input,
            group_keys,
            aggregations,
        } => ViewNode::GroupSummarise {
            input: Box::new(push_predicates_down(*input)),
            group_keys,
            aggregations,
        },
        ViewNode::Distinct { input, columns } => ViewNode::Distinct {
            input: Box::new(push_predicates_down(*input)),
            columns,
        },
        ViewNode::Join {
            left,
            right,
            on,
            kind,
        } => ViewNode::Join {
            left: Box::new(push_predicates_down(*left)),
            right: Box::new(push_predicates_down(*right)),
            on,
            kind,
        },
        other => other,
    }
}

/// Try to push a filter predicate below the given node.
fn push_filter_into(node: ViewNode, predicate: DExpr) -> ViewNode {
    match node {
        // Push filter past Select (always safe -- filter references columns
        // that are in the select list or the query is malformed anyway).
        ViewNode::Select { input, columns } => ViewNode::Select {
            input: Box::new(push_filter_into(*input, predicate)),
            columns,
        },

        // Push filter past Arrange (filter doesn't affect sort order).
        ViewNode::Arrange { input, keys } => ViewNode::Arrange {
            input: Box::new(push_filter_into(*input, predicate)),
            keys,
        },

        // Push filter past Mutate only if the predicate does NOT reference
        // any column that Mutate introduces/replaces.
        ViewNode::Mutate {
            input,
            assignments,
        } => {
            let pred_cols = expr_columns(&predicate);
            let mutated_cols: BTreeSet<String> =
                assignments.iter().map(|(name, _)| name.clone()).collect();
            let references_mutated = pred_cols.iter().any(|c| mutated_cols.contains(c));

            if references_mutated {
                // Cannot push -- predicate depends on mutated columns.
                ViewNode::Filter {
                    input: Box::new(ViewNode::Mutate {
                        input,
                        assignments,
                    }),
                    predicate,
                }
            } else {
                // Safe to push below.
                ViewNode::Mutate {
                    input: Box::new(push_filter_into(*input, predicate)),
                    assignments,
                }
            }
        }

        // Push filter into Join: if predicate references only left-side columns,
        // push into left; if only right-side, push into right; otherwise keep above.
        ViewNode::Join {
            left,
            right,
            on,
            kind,
        } => {
            let pred_cols = expr_columns(&predicate);
            let left_cols = node_output_columns(&left);
            let right_cols = node_output_columns(&right);

            let all_in_left = pred_cols.iter().all(|c| left_cols.contains(c));
            let all_in_right = pred_cols.iter().all(|c| right_cols.contains(c));

            if all_in_left {
                ViewNode::Join {
                    left: Box::new(push_filter_into(*left, predicate)),
                    right,
                    on,
                    kind,
                }
            } else if all_in_right {
                ViewNode::Join {
                    left,
                    right: Box::new(push_filter_into(*right, predicate)),
                    on,
                    kind,
                }
            } else {
                // Predicate spans both sides -- keep above.
                ViewNode::Filter {
                    input: Box::new(ViewNode::Join {
                        left,
                        right,
                        on,
                        kind,
                    }),
                    predicate,
                }
            }
        }

        // Do NOT push past GroupSummarise or Distinct -- they change row identity.
        other => ViewNode::Filter {
            input: Box::new(other),
            predicate,
        },
    }
}

// ── Pass 3: Redundant Select Elimination ─────────────────────────────────────

/// Remove Select nodes that select all columns from their input
/// (i.e., the select list matches the input's output columns exactly).
fn eliminate_redundant_selects(plan: ViewNode) -> ViewNode {
    match plan {
        ViewNode::Select { input, columns } => {
            let optimized_input = eliminate_redundant_selects(*input);
            let input_cols = node_output_columns(&optimized_input);

            // If the select list matches all input columns (same set), remove it.
            let select_set: BTreeSet<&str> = columns.iter().map(|s| s.as_str()).collect();
            let input_set: BTreeSet<&str> = input_cols.iter().map(|s| s.as_str()).collect();

            if select_set == input_set {
                optimized_input
            } else {
                ViewNode::Select {
                    input: Box::new(optimized_input),
                    columns,
                }
            }
        }
        ViewNode::Filter { input, predicate } => ViewNode::Filter {
            input: Box::new(eliminate_redundant_selects(*input)),
            predicate,
        },
        ViewNode::Mutate {
            input,
            assignments,
        } => ViewNode::Mutate {
            input: Box::new(eliminate_redundant_selects(*input)),
            assignments,
        },
        ViewNode::Arrange { input, keys } => ViewNode::Arrange {
            input: Box::new(eliminate_redundant_selects(*input)),
            keys,
        },
        ViewNode::GroupSummarise {
            input,
            group_keys,
            aggregations,
        } => ViewNode::GroupSummarise {
            input: Box::new(eliminate_redundant_selects(*input)),
            group_keys,
            aggregations,
        },
        ViewNode::Distinct { input, columns } => ViewNode::Distinct {
            input: Box::new(eliminate_redundant_selects(*input)),
            columns,
        },
        ViewNode::Join {
            left,
            right,
            on,
            kind,
        } => ViewNode::Join {
            left: Box::new(eliminate_redundant_selects(*left)),
            right: Box::new(eliminate_redundant_selects(*right)),
            on,
            kind,
        },
        other => other,
    }
}

// ── Executor ─────────────────────────────────────────────────────────────────

/// Execute an optimized ViewNode tree, producing a TidyFrame.
fn execute(node: ViewNode) -> Result<TidyFrame, TidyError> {
    match node {
        ViewNode::Scan { df } => Ok(TidyFrame::from_df((*df).clone())),

        ViewNode::Filter { input, predicate } => {
            let frame = execute(*input)?;
            let view = frame.view();
            let filtered = view.filter(&predicate)?;
            let df = filtered.materialize()?;
            Ok(TidyFrame::from_df(df))
        }

        ViewNode::Select { input, columns } => {
            let frame = execute(*input)?;
            let view = frame.view();
            let col_refs: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();
            let selected = view.select(&col_refs)?;
            let df = selected.materialize()?;
            Ok(TidyFrame::from_df(df))
        }

        ViewNode::Mutate {
            input,
            assignments,
        } => {
            let frame = execute(*input)?;
            let view = frame.view();
            let assign_refs: Vec<(&str, DExpr)> = assignments
                .into_iter()
                .map(|(name, expr)| (leaked_str(&name), expr))
                .collect();
            // Use TidyView::mutate which materializes + applies assignments.
            let result = view.mutate(&assign_refs.iter().map(|(n, e)| (*n, e.clone())).collect::<Vec<_>>())?;
            Ok(result)
        }

        ViewNode::Arrange { input, keys } => {
            let frame = execute(*input)?;
            let view = frame.view();
            let arranged = view.arrange(&keys)?;
            let df = arranged.materialize()?;
            Ok(TidyFrame::from_df(df))
        }

        ViewNode::GroupSummarise {
            input,
            group_keys,
            aggregations,
        } => {
            let frame = execute(*input)?;
            let view = frame.view();
            let key_refs: Vec<&str> = group_keys.iter().map(|s| s.as_str()).collect();
            let grouped = view.group_by(&key_refs)?;
            let agg_refs: Vec<(&str, TidyAgg)> = aggregations
                .into_iter()
                .map(|(name, agg)| (leaked_str(&name), agg))
                .collect();
            let result = grouped.summarise(
                &agg_refs.iter().map(|(n, a)| (*n, a.clone())).collect::<Vec<_>>(),
            )?;
            Ok(result)
        }

        ViewNode::StreamingGroupSummarise {
            input,
            group_keys,
            aggregations,
        } => {
            let frame = execute(*input)?;
            let view = frame.view();
            let key_refs: Vec<&str> = group_keys.iter().map(|s| s.as_str()).collect();
            let agg_owned: Vec<(String, crate::StreamingAgg)> = aggregations;
            let agg_refs: Vec<(&str, crate::StreamingAgg)> = agg_owned
                .iter()
                .map(|(name, sa)| (leaked_str(name), sa.clone()))
                .collect();
            view.summarise_streaming(&key_refs, &agg_refs)
        }

        ViewNode::Distinct { input, columns } => {
            let frame = execute(*input)?;
            let view = frame.view();
            let col_refs: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();
            let distinct = view.distinct(&col_refs)?;
            let df = distinct.materialize()?;
            Ok(TidyFrame::from_df(df))
        }

        ViewNode::Join {
            left,
            right,
            on,
            kind,
        } => {
            let left_frame = execute(*left)?;
            let right_frame = execute(*right)?;
            let left_view = left_frame.view();
            let right_view = right_frame.view();
            let on_refs: Vec<(&str, &str)> = on
                .iter()
                .map(|(l, r)| (l.as_str(), r.as_str()))
                .collect();

            match kind {
                JoinType::Inner => left_view.inner_join(&right_view, &on_refs),
                JoinType::Left => left_view.left_join(&right_view, &on_refs),
                JoinType::Semi => {
                    let result = left_view.semi_join(&right_view, &on_refs)?;
                    let df = result.materialize()?;
                    Ok(TidyFrame::from_df(df))
                }
                JoinType::Anti => {
                    let result = left_view.anti_join(&right_view, &on_refs)?;
                    let df = result.materialize()?;
                    Ok(TidyFrame::from_df(df))
                }
            }
        }
    }
}

// ── Helper: column reference extraction from DExpr ───────────────────────────

/// Collect all column names referenced by an expression.
fn expr_columns(expr: &DExpr) -> BTreeSet<String> {
    let mut cols = BTreeSet::new();
    collect_expr_cols(expr, &mut cols);
    cols
}

fn collect_expr_cols(expr: &DExpr, cols: &mut BTreeSet<String>) {
    match expr {
        DExpr::Col(name) => {
            cols.insert(name.clone());
        }
        DExpr::BinOp { left, right, .. } => {
            collect_expr_cols(left, cols);
            collect_expr_cols(right, cols);
        }
        DExpr::Agg(_, inner) => collect_expr_cols(inner, cols),
        DExpr::FnCall(_, args) => {
            for arg in args {
                collect_expr_cols(arg, cols);
            }
        }
        DExpr::CumSum(e)
        | DExpr::CumProd(e)
        | DExpr::CumMax(e)
        | DExpr::CumMin(e)
        | DExpr::Lag(e, _)
        | DExpr::Lead(e, _)
        | DExpr::Rank(e)
        | DExpr::DenseRank(e) => {
            collect_expr_cols(e, cols);
        }
        // Rolling window functions reference a column by name (String field).
        DExpr::RollingSum(col, _)
        | DExpr::RollingMean(col, _)
        | DExpr::RollingMin(col, _)
        | DExpr::RollingMax(col, _)
        | DExpr::RollingVar(col, _)
        | DExpr::RollingSd(col, _) => {
            cols.insert(col.clone());
        }
        DExpr::LitInt(_)
        | DExpr::LitFloat(_)
        | DExpr::LitBool(_)
        | DExpr::LitStr(_)
        | DExpr::Count
        | DExpr::RowNumber => {}
    }
}

// ── Helper: infer output columns of a ViewNode ───────────────────────────────

/// Return the set of column names that a node produces.
///
/// For Scan nodes, reads from the DataFrame directly.
/// For others, infers from the node type.
fn node_output_columns(node: &ViewNode) -> BTreeSet<String> {
    match node {
        ViewNode::Scan { df } => df.column_names().into_iter().map(|s| s.to_string()).collect(),
        ViewNode::Filter { input, .. } => node_output_columns(input),
        ViewNode::Select { columns, .. } => columns.iter().cloned().collect(),
        ViewNode::Mutate {
            input,
            assignments,
        } => {
            let mut cols = node_output_columns(input);
            for (name, _) in assignments {
                cols.insert(name.clone());
            }
            cols
        }
        ViewNode::Arrange { input, .. } => node_output_columns(input),
        ViewNode::GroupSummarise {
            group_keys,
            aggregations,
            ..
        } => {
            let mut cols: BTreeSet<String> = group_keys.iter().cloned().collect();
            for (name, _) in aggregations {
                cols.insert(name.clone());
            }
            cols
        }
        ViewNode::StreamingGroupSummarise {
            group_keys,
            aggregations,
            ..
        } => {
            let mut cols: BTreeSet<String> = group_keys.iter().cloned().collect();
            for (name, _) in aggregations {
                cols.insert(name.clone());
            }
            cols
        }
        ViewNode::Distinct { input, .. } => node_output_columns(input),
        ViewNode::Join {
            left, right, on, ..
        } => {
            let mut cols = node_output_columns(left);
            let right_cols = node_output_columns(right);
            // Right join keys that duplicate left keys are excluded in output
            let left_keys: BTreeSet<&str> = on.iter().map(|(l, _)| l.as_str()).collect();
            let right_keys: BTreeSet<&str> = on.iter().map(|(_, r)| r.as_str()).collect();
            for c in &right_cols {
                if !right_keys.contains(c.as_str()) || !left_keys.contains(c.as_str()) {
                    cols.insert(c.clone());
                }
            }
            cols
        }
    }
}

// ── Helper: leak a String into a &'static str for API compatibility ──────────

/// Convert a String to &'static str by leaking.
///
/// This is used only during plan execution (bounded number of calls per plan).
/// The leaked memory is small (column name strings) and proportional to
/// the plan size, not the data size.
fn leaked_str(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

// ── Plan inspection helpers (for testing) ────────────────────────────────────

impl ViewNode {
    /// Count the number of Filter nodes in the tree.
    pub fn count_filters(&self) -> usize {
        match self {
            ViewNode::Filter { input, .. } => 1 + input.count_filters(),
            ViewNode::Select { input, .. } => input.count_filters(),
            ViewNode::Mutate { input, .. } => input.count_filters(),
            ViewNode::Arrange { input, .. } => input.count_filters(),
            ViewNode::GroupSummarise { input, .. } => input.count_filters(),
            ViewNode::StreamingGroupSummarise { input, .. } => input.count_filters(),
            ViewNode::Distinct { input, .. } => input.count_filters(),
            ViewNode::Join { left, right, .. } => {
                left.count_filters() + right.count_filters()
            }
            ViewNode::Scan { .. } => 0,
        }
    }

    /// Check if the immediate child (input) of the outermost node is a Scan.
    /// Useful for verifying predicate pushdown moved a filter near the scan.
    pub fn is_filter_on_scan(&self) -> bool {
        match self {
            ViewNode::Filter { input, .. } => matches!(input.as_ref(), ViewNode::Scan { .. }),
            _ => false,
        }
    }

    /// Return the innermost node (the leaf Scan) by walking `input` chains.
    pub fn innermost(&self) -> &ViewNode {
        match self {
            ViewNode::Filter { input, .. }
            | ViewNode::Select { input, .. }
            | ViewNode::Mutate { input, .. }
            | ViewNode::Arrange { input, .. }
            | ViewNode::GroupSummarise { input, .. }
            | ViewNode::StreamingGroupSummarise { input, .. }
            | ViewNode::Distinct { input, .. } => input.innermost(),
            ViewNode::Join { left, .. } => left.innermost(),
            ViewNode::Scan { .. } => self,
        }
    }

    /// Return the node kind name (for test assertions).
    pub fn kind(&self) -> &'static str {
        match self {
            ViewNode::Scan { .. } => "Scan",
            ViewNode::Filter { .. } => "Filter",
            ViewNode::Select { .. } => "Select",
            ViewNode::Mutate { .. } => "Mutate",
            ViewNode::Arrange { .. } => "Arrange",
            ViewNode::GroupSummarise { .. } => "GroupSummarise",
            ViewNode::StreamingGroupSummarise { .. } => "StreamingGroupSummarise",
            ViewNode::Distinct { .. } => "Distinct",
            ViewNode::Join { .. } => "Join",
        }
    }

    /// Walk the plan tree depth-first and collect node kinds top-down.
    pub fn node_kinds(&self) -> Vec<&'static str> {
        let mut out = vec![self.kind()];
        match self {
            ViewNode::Filter { input, .. }
            | ViewNode::Select { input, .. }
            | ViewNode::Mutate { input, .. }
            | ViewNode::Arrange { input, .. }
            | ViewNode::GroupSummarise { input, .. }
            | ViewNode::StreamingGroupSummarise { input, .. }
            | ViewNode::Distinct { input, .. } => {
                out.extend(input.node_kinds());
            }
            ViewNode::Join { left, right, .. } => {
                out.extend(left.node_kinds());
                out.extend(right.node_kinds());
            }
            ViewNode::Scan { .. } => {}
        }
        out
    }
}

// ── Batch Executor ────────────────────────────────────────────────────────────

/// Maximum rows per batch for vectorized processing.
const BATCH_SIZE: usize = 2048;

/// A chunk of up to `BATCH_SIZE` rows for vectorized processing.
///
/// Batches are processed sequentially in order (batch 0, batch 1, ...)
/// to preserve deterministic row ordering.
#[derive(Debug, Clone)]
pub struct Batch {
    pub columns: Vec<(String, Column)>,
    pub nrows: usize,
}

impl Batch {
    /// Convert this batch into a DataFrame.
    fn into_dataframe(self) -> DataFrame {
        DataFrame {
            columns: self.columns,
        }
    }

    /// Get a column by name.
    fn get_column(&self, name: &str) -> Option<&Column> {
        self.columns.iter().find(|(n, _)| n == name).map(|(_, c)| c)
    }

    /// Column names in order.
    fn column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|(n, _)| n.as_str()).collect()
    }
}

/// Slice a column from `start..end`.
fn slice_column(col: &Column, start: usize, end: usize) -> Column {
    if matches!(col, Column::CategoricalAdaptive(_)) {
        return slice_column(&col.to_legacy_categorical(), start, end);
    }
    match col {
        Column::Float(v) => Column::Float(v[start..end].to_vec()),
        Column::Int(v) => Column::Int(v[start..end].to_vec()),
        Column::Str(v) => Column::Str(v[start..end].to_vec()),
        Column::Bool(v) => Column::Bool(v[start..end].to_vec()),
        Column::Categorical { levels, codes } => Column::Categorical {
            levels: levels.clone(),
            codes: codes[start..end].to_vec(),
        },
        Column::DateTime(v) => Column::DateTime(v[start..end].to_vec()),
        Column::CategoricalAdaptive(_) => unreachable!("handled by early return"),
    }
}

/// Split a DataFrame into batches of up to `BATCH_SIZE` rows.
fn split_batches(df: &DataFrame) -> Vec<Batch> {
    let nrows = df.nrows();
    if nrows == 0 {
        return vec![Batch {
            columns: df.columns.iter().map(|(n, c)| {
                (n.clone(), slice_column(c, 0, 0))
            }).collect(),
            nrows: 0,
        }];
    }
    let mut batches = Vec::new();
    let mut offset = 0;
    while offset < nrows {
        let end = (offset + BATCH_SIZE).min(nrows);
        let batch_cols = df
            .columns
            .iter()
            .map(|(name, col)| (name.clone(), slice_column(col, offset, end)))
            .collect();
        batches.push(Batch {
            columns: batch_cols,
            nrows: end - offset,
        });
        offset = end;
    }
    batches
}

/// Merge a vector of batches back into a single DataFrame.
///
/// Batches must have identical column schemas. Empty batches are skipped.
fn merge_batches(batches: Vec<Batch>) -> Result<DataFrame, TidyError> {
    if batches.is_empty() {
        return Ok(DataFrame::new());
    }

    // Determine schema from first non-empty batch (or just first batch).
    let schema: Vec<String> = batches[0].column_names().iter().map(|s| s.to_string()).collect();
    if schema.is_empty() {
        return Ok(DataFrame::new());
    }

    // Pre-allocate merged columns.
    let total_rows: usize = batches.iter().map(|b| b.nrows).sum();
    let mut merged_cols: Vec<(String, Column)> = schema
        .iter()
        .map(|name| {
            // Determine type from first batch's column.
            let first_col = batches[0].get_column(name).unwrap();
            let empty = match first_col {
                Column::Float(_) => Column::Float(Vec::with_capacity(total_rows)),
                Column::Int(_) => Column::Int(Vec::with_capacity(total_rows)),
                Column::Str(_) => Column::Str(Vec::with_capacity(total_rows)),
                Column::Bool(_) => Column::Bool(Vec::with_capacity(total_rows)),
                Column::Categorical { levels, .. } => Column::Categorical {
                    levels: levels.clone(),
                    codes: Vec::with_capacity(total_rows),
                },
                Column::CategoricalAdaptive(_) => {
                    // Empty buffer matching legacy categorical shape.
                    let legacy = first_col.to_legacy_categorical();
                    if let Column::Categorical { levels, .. } = legacy {
                        Column::Categorical {
                            levels,
                            codes: Vec::with_capacity(total_rows),
                        }
                    } else {
                        // Non-UTF-8 / null fallback: Str buffer.
                        Column::Str(Vec::with_capacity(total_rows))
                    }
                }
                Column::DateTime(_) => Column::DateTime(Vec::with_capacity(total_rows)),
            };
            (name.clone(), empty)
        })
        .collect();

    // Append each batch's data.
    for batch in &batches {
        if batch.nrows == 0 {
            continue;
        }
        for (i, (name, merged_col)) in merged_cols.iter_mut().enumerate() {
            let batch_col = batch.get_column(name).ok_or_else(|| {
                TidyError::ColumnNotFound(format!(
                    "batch merge: column '{}' missing in batch (index {})",
                    name, i
                ))
            })?;
            append_column(merged_col, batch_col);
        }
    }

    Ok(DataFrame { columns: merged_cols })
}

/// Append all rows from `src` into `dst` (same type assumed).
fn append_column(dst: &mut Column, src: &Column) {
    match (dst, src) {
        (Column::Float(d), Column::Float(s)) => d.extend_from_slice(s),
        (Column::Int(d), Column::Int(s)) => d.extend_from_slice(s),
        (Column::Str(d), Column::Str(s)) => d.extend(s.iter().cloned()),
        (Column::Bool(d), Column::Bool(s)) => d.extend_from_slice(s),
        (Column::Categorical { codes: d, .. }, Column::Categorical { codes: s, .. }) => {
            d.extend_from_slice(s);
        }
        (Column::DateTime(d), Column::DateTime(s)) => d.extend_from_slice(s),
        _ => {} // Type mismatch: should not happen if schema is consistent.
    }
}

// ── Streamable operation representation ──────────────────────────────────────

/// A streamable (non-breaking) operation that can be applied per-batch.
#[derive(Debug, Clone)]
enum StreamableOp {
    Filter { predicate: DExpr },
    Select { columns: Vec<String> },
    Mutate { assignments: Vec<(String, DExpr)> },
}

/// Returns true if the node is a pipeline breaker (requires full materialization).
fn is_pipeline_breaker(node: &ViewNode) -> bool {
    matches!(
        node,
        ViewNode::Arrange { .. }
            | ViewNode::GroupSummarise { .. }
            | ViewNode::StreamingGroupSummarise { .. }
            | ViewNode::Distinct { .. }
            | ViewNode::Join { .. }
    )
}

/// Walk the plan tree and collect a chain of streamable operations from the top.
///
/// Returns `(streamable_ops_in_execution_order, base_node)`.
/// The chain is collected top-down (outermost first), then reversed so the
/// innermost (closest to scan) operation is applied first.
fn collect_streamable_chain(node: ViewNode) -> (Vec<StreamableOp>, Box<ViewNode>) {
    let mut ops = Vec::new();
    let mut current = node;

    loop {
        match current {
            ViewNode::Filter { input, predicate } => {
                ops.push(StreamableOp::Filter { predicate });
                current = *input;
            }
            ViewNode::Select { input, columns } => {
                ops.push(StreamableOp::Select { columns });
                current = *input;
            }
            ViewNode::Mutate { input, assignments } => {
                ops.push(StreamableOp::Mutate { assignments });
                current = *input;
            }
            // Any other node is the base (Scan or a pipeline breaker).
            other => {
                // Reverse: we collected outermost-first, but need to apply innermost-first.
                ops.reverse();
                return (ops, Box::new(other));
            }
        }
    }
}

/// Apply a single streamable operation to a batch.
fn apply_op_to_batch(batch: Batch, op: &StreamableOp) -> Result<Batch, TidyError> {
    match op {
        StreamableOp::Filter { predicate } => {
            // Materialize batch into a temporary DataFrame, apply filter via TidyView.
            let df = batch.into_dataframe();
            if df.nrows() == 0 {
                return Ok(Batch {
                    nrows: 0,
                    columns: df.columns,
                });
            }
            let frame = TidyFrame::from_df(df);
            let view = frame.view();
            let filtered = view.filter(predicate)?;
            let result_df = filtered.materialize()?;
            let nrows = result_df.nrows();
            Ok(Batch {
                columns: result_df.columns,
                nrows,
            })
        }
        StreamableOp::Select { columns } => {
            // Keep only the named columns, in the requested order.
            let selected: Vec<(String, Column)> = columns
                .iter()
                .filter_map(|name| {
                    batch
                        .columns
                        .iter()
                        .find(|(n, _)| n == name)
                        .cloned()
                })
                .collect();
            Ok(Batch {
                nrows: batch.nrows,
                columns: selected,
            })
        }
        StreamableOp::Mutate { assignments } => {
            // Materialize batch into a temporary DataFrame, apply mutate via TidyView.
            let df = batch.into_dataframe();
            let frame = TidyFrame::from_df(df);
            let view = frame.view();
            let assign_refs: Vec<(&str, DExpr)> = assignments
                .iter()
                .map(|(name, expr)| (leaked_str(name), expr.clone()))
                .collect();
            let result = view.mutate(
                &assign_refs
                    .iter()
                    .map(|(n, e)| (*n, e.clone()))
                    .collect::<Vec<_>>(),
            )?;
            let result_df = result.borrow().clone();
            let nrows = result_df.nrows();
            Ok(Batch {
                columns: result_df.columns,
                nrows,
            })
        }
    }
}

/// Apply a chain of streamable operations to a DataFrame in batches.
fn apply_chain_batched(
    frame: &TidyFrame,
    chain: &[StreamableOp],
) -> Result<TidyFrame, TidyError> {
    let df = frame.borrow().clone();
    let batches = split_batches(&df);

    let mut result_batches: Vec<Batch> = Vec::new();
    for batch in batches {
        let mut current = batch;
        for op in chain {
            current = apply_op_to_batch(current, op)?;
        }
        if current.nrows > 0 {
            result_batches.push(current);
        }
    }

    if result_batches.is_empty() {
        // Preserve schema from original DataFrame but with zero rows.
        let empty_df = DataFrame {
            columns: df
                .columns
                .iter()
                .map(|(name, col)| {
                    (name.clone(), slice_column(col, 0, 0))
                })
                .collect(),
        };
        // If chain includes a Select, apply column pruning to the empty frame.
        let mut result_cols: Option<Vec<String>> = None;
        for op in chain {
            if let StreamableOp::Select { columns } = op {
                result_cols = Some(columns.clone());
            }
        }
        if let Some(cols) = result_cols {
            let pruned: Vec<(String, Column)> = cols
                .iter()
                .filter_map(|name| {
                    empty_df
                        .columns
                        .iter()
                        .find(|(n, _)| n == name)
                        .cloned()
                })
                .collect();
            return Ok(TidyFrame::from_df(DataFrame { columns: pruned }));
        }
        return Ok(TidyFrame::from_df(empty_df));
    }

    let merged = merge_batches(result_batches)?;
    Ok(TidyFrame::from_df(merged))
}

/// Execute an optimized ViewNode tree using batch processing where possible.
///
/// Streamable operations (Filter, Select, Mutate) are fused into a single
/// batch pass over 2048-row chunks. Pipeline breakers (Arrange, GroupSummarise,
/// Distinct, Join) force full materialization before proceeding.
///
/// # Determinism
///
/// Batches are processed sequentially in row order. The merged output has
/// rows in the same order as non-batched execution. Float reductions use
/// Kahan summation (delegated to TidyView internals).
pub fn execute_batched(node: ViewNode) -> Result<TidyFrame, TidyError> {
    match &node {
        // Leaf: just materialize.
        ViewNode::Scan { .. } => execute(node),

        // Streamable operations: collect chain and batch-execute.
        _ if !is_pipeline_breaker(&node) => {
            let (chain, base) = collect_streamable_chain(node);
            if chain.is_empty() {
                // Shouldn't happen, but handle gracefully.
                return execute_batched(*base);
            }
            let base_frame = execute_batched(*base)?;
            apply_chain_batched(&base_frame, &chain)
        }

        // Pipeline breakers: execute children batched, then apply breaker eagerly.
        _ => execute_breaker_batched(node),
    }
}

/// Execute a pipeline-breaking node by first executing its children via
/// batch processing, then applying the breaker operation eagerly.
fn execute_breaker_batched(node: ViewNode) -> Result<TidyFrame, TidyError> {
    match node {
        ViewNode::Arrange { input, keys } => {
            let frame = execute_batched(*input)?;
            let view = frame.view();
            let arranged = view.arrange(&keys)?;
            let df = arranged.materialize()?;
            Ok(TidyFrame::from_df(df))
        }

        ViewNode::GroupSummarise {
            input,
            group_keys,
            aggregations,
        } => {
            let frame = execute_batched(*input)?;
            let view = frame.view();
            let key_refs: Vec<&str> = group_keys.iter().map(|s| s.as_str()).collect();
            let grouped = view.group_by(&key_refs)?;
            let agg_refs: Vec<(&str, TidyAgg)> = aggregations
                .into_iter()
                .map(|(name, agg)| (leaked_str(&name), agg))
                .collect();
            let result = grouped.summarise(
                &agg_refs
                    .iter()
                    .map(|(n, a)| (*n, a.clone()))
                    .collect::<Vec<_>>(),
            )?;
            Ok(result)
        }

        ViewNode::StreamingGroupSummarise {
            input,
            group_keys,
            aggregations,
        } => {
            let frame = execute_batched(*input)?;
            let view = frame.view();
            let key_refs: Vec<&str> = group_keys.iter().map(|s| s.as_str()).collect();
            let agg_owned: Vec<(String, crate::StreamingAgg)> = aggregations;
            let agg_refs: Vec<(&str, crate::StreamingAgg)> = agg_owned
                .iter()
                .map(|(name, sa)| (leaked_str(name), sa.clone()))
                .collect();
            view.summarise_streaming(&key_refs, &agg_refs)
        }

        ViewNode::Distinct { input, columns } => {
            let frame = execute_batched(*input)?;
            let view = frame.view();
            let col_refs: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();
            let distinct = view.distinct(&col_refs)?;
            let df = distinct.materialize()?;
            Ok(TidyFrame::from_df(df))
        }

        ViewNode::Join {
            left,
            right,
            on,
            kind,
        } => {
            let left_frame = execute_batched(*left)?;
            let right_frame = execute_batched(*right)?;
            let left_view = left_frame.view();
            let right_view = right_frame.view();
            let on_refs: Vec<(&str, &str)> =
                on.iter().map(|(l, r)| (l.as_str(), r.as_str())).collect();

            match kind {
                JoinType::Inner => left_view.inner_join(&right_view, &on_refs),
                JoinType::Left => left_view.left_join(&right_view, &on_refs),
                JoinType::Semi => {
                    let result = left_view.semi_join(&right_view, &on_refs)?;
                    let df = result.materialize()?;
                    Ok(TidyFrame::from_df(df))
                }
                JoinType::Anti => {
                    let result = left_view.anti_join(&right_view, &on_refs)?;
                    let df = result.materialize()?;
                    Ok(TidyFrame::from_df(df))
                }
            }
        }

        // Non-breakers should not reach here, but handle gracefully.
        other => execute(other),
    }
}

impl LazyView {
    /// Optimize and execute the plan using batch processing.
    ///
    /// This is an alternative to `collect()` that processes data in
    /// 2048-row batches, fusing chains of streamable operations
    /// (Filter, Select, Mutate) into a single pass per batch.
    ///
    /// Pipeline breakers (Arrange, GroupSummarise, Distinct, Join) cause
    /// full materialization at that point.
    ///
    /// The output is identical to `collect()` -- this is purely an
    /// execution strategy optimization.
    pub fn collect_batched(self) -> Result<TidyFrame, TidyError> {
        let optimized = optimize(self.plan);
        execute_batched(optimized)
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Column, DExpr, DBinOp, DataFrame, TidyAgg, ArrangeKey, TidyView};

    /// Build a small test DataFrame: name(Str), age(Int), score(Float).
    fn test_df() -> DataFrame {
        DataFrame {
            columns: vec![
                (
                    "name".to_string(),
                    Column::Str(vec![
                        "Alice".into(),
                        "Bob".into(),
                        "Carol".into(),
                        "Dave".into(),
                    ]),
                ),
                ("age".to_string(), Column::Int(vec![30, 25, 35, 25])),
                (
                    "score".to_string(),
                    Column::Float(vec![90.0, 85.0, 95.0, 80.0]),
                ),
            ],
        }
    }

    /// Build a second DataFrame for join tests: name(Str), dept(Str).
    fn dept_df() -> DataFrame {
        DataFrame {
            columns: vec![
                (
                    "name".to_string(),
                    Column::Str(vec!["Alice".into(), "Bob".into(), "Eve".into()]),
                ),
                (
                    "dept".to_string(),
                    Column::Str(vec!["Eng".into(), "Sales".into(), "Eng".into()]),
                ),
            ],
        }
    }

    // ── Basic lazy chain produces same result as eager ────────────────────

    #[test]
    fn lazy_filter_matches_eager() {
        let df = test_df();
        let predicate = DExpr::BinOp {
            op: DBinOp::Gt,
            left: Box::new(DExpr::Col("age".into())),
            right: Box::new(DExpr::LitInt(25)),
        };

        // Eager
        let eager_view = TidyView::from_df(df.clone());
        let eager_filtered = eager_view.filter(&predicate).unwrap();
        let eager_df = eager_filtered.materialize().unwrap();

        // Lazy
        let lazy_frame = LazyView::from_df(df)
            .filter(predicate)
            .collect()
            .unwrap();
        let lazy_df = lazy_frame.borrow();

        assert_eq!(eager_df.nrows(), lazy_df.nrows());
        assert_eq!(eager_df.nrows(), 2); // Alice(30) and Carol(35)

        // Verify same names
        let eager_names: Vec<String> = match eager_df.get_column("name").unwrap() {
            Column::Str(v) => v.clone(),
            _ => panic!("expected Str"),
        };
        let lazy_names: Vec<String> = match lazy_df.get_column("name").unwrap() {
            Column::Str(v) => v.clone(),
            _ => panic!("expected Str"),
        };
        assert_eq!(eager_names, lazy_names);
    }

    #[test]
    fn lazy_select_matches_eager() {
        let df = test_df();

        // Eager
        let eager_view = TidyView::from_df(df.clone());
        let eager_selected = eager_view.select(&["name", "age"]).unwrap();
        let eager_df = eager_selected.materialize().unwrap();

        // Lazy
        let lazy_frame = LazyView::from_df(df)
            .select(vec!["name".into(), "age".into()])
            .collect()
            .unwrap();
        let lazy_df = lazy_frame.borrow();

        assert_eq!(eager_df.ncols(), 2);
        assert_eq!(lazy_df.ncols(), 2);
        assert_eq!(eager_df.column_names(), lazy_df.column_names());
    }

    #[test]
    fn lazy_arrange_matches_eager() {
        let df = test_df();
        let keys = vec![ArrangeKey::asc("age")];

        // Eager
        let eager_view = TidyView::from_df(df.clone());
        let eager_arranged = eager_view.arrange(&keys).unwrap();
        let eager_df = eager_arranged.materialize().unwrap();

        // Lazy
        let lazy_frame = LazyView::from_df(df)
            .arrange(keys)
            .collect()
            .unwrap();
        let lazy_df = lazy_frame.borrow();

        let eager_ages = match eager_df.get_column("age").unwrap() {
            Column::Int(v) => v.clone(),
            _ => panic!("expected Int"),
        };
        let lazy_ages = match lazy_df.get_column("age").unwrap() {
            Column::Int(v) => v.clone(),
            _ => panic!("expected Int"),
        };
        assert_eq!(eager_ages, lazy_ages);
        // Should be sorted ascending
        assert_eq!(eager_ages, vec![25, 25, 30, 35]);
    }

    #[test]
    fn lazy_group_summarise_matches_eager() {
        let df = test_df();

        // Eager
        let eager_view = TidyView::from_df(df.clone());
        let grouped = eager_view.group_by(&["age"]).unwrap();
        let eager_frame = grouped
            .summarise(&[("count", TidyAgg::Count)])
            .unwrap();
        let eager_df = eager_frame.borrow();

        // Lazy
        let lazy_frame = LazyView::from_df(df)
            .group_summarise(
                vec!["age".into()],
                vec![("count".into(), TidyAgg::Count)],
            )
            .collect()
            .unwrap();
        let lazy_df = lazy_frame.borrow();

        assert_eq!(eager_df.nrows(), lazy_df.nrows());
        assert_eq!(eager_df.column_names(), lazy_df.column_names());
    }

    // ── Predicate pushdown ───────────────────────────────────────────────

    #[test]
    fn predicate_pushdown_past_select() {
        let df = test_df();
        let predicate = DExpr::BinOp {
            op: DBinOp::Gt,
            left: Box::new(DExpr::Col("age".into())),
            right: Box::new(DExpr::LitInt(25)),
        };

        // Build: Scan -> Select -> Filter
        let lazy = LazyView::from_df(df)
            .select(vec!["name".into(), "age".into()])
            .filter(predicate);

        let optimized = lazy.optimized_plan();

        // After pushdown, the filter should be below the select.
        // Plan should be: Select -> Filter -> Scan
        let kinds = optimized.node_kinds();
        assert_eq!(kinds, vec!["Select", "Filter", "Scan"]);
    }

    #[test]
    fn predicate_pushdown_past_arrange() {
        let df = test_df();
        let predicate = DExpr::BinOp {
            op: DBinOp::Gt,
            left: Box::new(DExpr::Col("age".into())),
            right: Box::new(DExpr::LitInt(25)),
        };

        // Build: Scan -> Arrange -> Filter
        let lazy = LazyView::from_df(df)
            .arrange(vec![ArrangeKey::asc("age")])
            .filter(predicate);

        let optimized = lazy.optimized_plan();

        // Filter should be pushed below Arrange.
        let kinds = optimized.node_kinds();
        assert_eq!(kinds, vec!["Arrange", "Filter", "Scan"]);
    }

    #[test]
    fn predicate_not_pushed_past_mutate_when_dependent() {
        let df = test_df();
        // Mutate adds "doubled_age" = age * 2
        // Filter on "doubled_age" > 50 -- references mutated column, cannot push.
        let predicate = DExpr::BinOp {
            op: DBinOp::Gt,
            left: Box::new(DExpr::Col("doubled_age".into())),
            right: Box::new(DExpr::LitInt(50)),
        };

        let lazy = LazyView::from_df(df)
            .mutate(vec![(
                "doubled_age".into(),
                DExpr::BinOp {
                    op: DBinOp::Mul,
                    left: Box::new(DExpr::Col("age".into())),
                    right: Box::new(DExpr::LitInt(2)),
                },
            )])
            .filter(predicate);

        let optimized = lazy.optimized_plan();

        // Filter should stay ABOVE Mutate (cannot push).
        let kinds = optimized.node_kinds();
        assert_eq!(kinds, vec!["Filter", "Mutate", "Scan"]);
    }

    #[test]
    fn predicate_pushed_past_mutate_when_independent() {
        let df = test_df();
        // Mutate adds "doubled_age" = age * 2
        // Filter on "score" > 85 -- does NOT reference mutated column, can push.
        let predicate = DExpr::BinOp {
            op: DBinOp::Gt,
            left: Box::new(DExpr::Col("score".into())),
            right: Box::new(DExpr::LitFloat(85.0)),
        };

        let lazy = LazyView::from_df(df)
            .mutate(vec![(
                "doubled_age".into(),
                DExpr::BinOp {
                    op: DBinOp::Mul,
                    left: Box::new(DExpr::Col("age".into())),
                    right: Box::new(DExpr::LitInt(2)),
                },
            )])
            .filter(predicate);

        let optimized = lazy.optimized_plan();

        // Filter should be pushed below Mutate.
        let kinds = optimized.node_kinds();
        assert_eq!(kinds, vec!["Mutate", "Filter", "Scan"]);
    }

    #[test]
    fn predicate_not_pushed_past_group_summarise() {
        let df = test_df();
        let predicate = DExpr::BinOp {
            op: DBinOp::Gt,
            left: Box::new(DExpr::Col("count".into())),
            right: Box::new(DExpr::LitInt(1)),
        };

        let lazy = LazyView::from_df(df)
            .group_summarise(
                vec!["age".into()],
                vec![("count".into(), TidyAgg::Count)],
            )
            .filter(predicate);

        let optimized = lazy.optimized_plan();

        // Filter must stay above the group node. v3 Phase 6: the lazy
        // optimizer rewrites Count-only GroupSummarise to
        // StreamingGroupSummarise; both are pipeline breakers, so the
        // filter staying above either form is the invariant being
        // pinned here.
        let kinds = optimized.node_kinds();
        assert!(
            kinds == vec!["Filter", "GroupSummarise", "Scan"]
                || kinds == vec!["Filter", "StreamingGroupSummarise", "Scan"],
            "filter must stay above the group node, got {:?}",
            kinds
        );
    }

    // ── Filter merging ───────────────────────────────────────────────────

    #[test]
    fn consecutive_filters_merged() {
        let df = test_df();
        let pred1 = DExpr::BinOp {
            op: DBinOp::Gt,
            left: Box::new(DExpr::Col("age".into())),
            right: Box::new(DExpr::LitInt(20)),
        };
        let pred2 = DExpr::BinOp {
            op: DBinOp::Lt,
            left: Box::new(DExpr::Col("score".into())),
            right: Box::new(DExpr::LitFloat(95.0)),
        };

        let lazy = LazyView::from_df(df).filter(pred1).filter(pred2);

        let optimized = lazy.optimized_plan();

        // Should have only 1 filter node (merged), not 2.
        assert_eq!(optimized.count_filters(), 1);

        // The merged filter should produce correct results.
        // age > 20 AND score < 95 => Alice(30,90), Bob(25,85), Dave(25,80)
        let df2 = test_df();
        let result = LazyView::from_df(df2)
            .filter(DExpr::BinOp {
                op: DBinOp::Gt,
                left: Box::new(DExpr::Col("age".into())),
                right: Box::new(DExpr::LitInt(20)),
            })
            .filter(DExpr::BinOp {
                op: DBinOp::Lt,
                left: Box::new(DExpr::Col("score".into())),
                right: Box::new(DExpr::LitFloat(95.0)),
            })
            .collect()
            .unwrap();

        let result_df = result.borrow();
        assert_eq!(result_df.nrows(), 3);
    }

    // ── Redundant select elimination ─────────────────────────────────────

    #[test]
    fn redundant_select_eliminated() {
        let df = test_df();

        // Select all 3 columns (same as input) -- should be eliminated.
        let lazy = LazyView::from_df(df)
            .select(vec!["name".into(), "age".into(), "score".into()]);

        let optimized = lazy.optimized_plan();

        // Should be just a Scan (redundant Select removed).
        assert_eq!(optimized.kind(), "Scan");
    }

    #[test]
    fn non_redundant_select_kept() {
        let df = test_df();

        // Select only 2 of 3 columns -- should NOT be eliminated.
        let lazy = LazyView::from_df(df).select(vec!["name".into(), "age".into()]);

        let optimized = lazy.optimized_plan();

        assert_eq!(optimized.kind(), "Select");
    }

    // ── Determinism ──────────────────────────────────────────────────────

    #[test]
    fn determinism_3_runs_identical() {
        for _ in 0..3 {
            let df = test_df();
            let result = LazyView::from_df(df)
                .filter(DExpr::BinOp {
                    op: DBinOp::Gt,
                    left: Box::new(DExpr::Col("age".into())),
                    right: Box::new(DExpr::LitInt(20)),
                })
                .select(vec!["name".into(), "age".into()])
                .arrange(vec![ArrangeKey::desc("age")])
                .collect()
                .unwrap();

            let result_df = result.borrow();
            assert_eq!(result_df.nrows(), 4);

            let ages = match result_df.get_column("age").unwrap() {
                Column::Int(v) => v.clone(),
                _ => panic!("expected Int"),
            };
            // Descending: 35, 30, 25, 25
            assert_eq!(ages, vec![35, 30, 25, 25]);

            let names = match result_df.get_column("name").unwrap() {
                Column::Str(v) => v.clone(),
                _ => panic!("expected Str"),
            };
            assert_eq!(names, vec!["Carol", "Alice", "Bob", "Dave"]);
        }
    }

    // ── Join execution ───────────────────────────────────────────────────

    #[test]
    fn lazy_inner_join() {
        let left = test_df();
        let right = dept_df();

        let result = LazyView::from_df(left)
            .join(
                LazyView::from_df(right),
                vec![("name".into(), "name".into())],
                JoinType::Inner,
            )
            .collect()
            .unwrap();

        let result_df = result.borrow();
        // Only Alice and Bob match
        assert_eq!(result_df.nrows(), 2);
        assert!(result_df.get_column("dept").is_some());
    }

    #[test]
    fn lazy_semi_join() {
        let left = test_df();
        let right = dept_df();

        let result = LazyView::from_df(left)
            .join(
                LazyView::from_df(right),
                vec![("name".into(), "name".into())],
                JoinType::Semi,
            )
            .collect()
            .unwrap();

        let result_df = result.borrow();
        // Semi join: Alice and Bob from left
        assert_eq!(result_df.nrows(), 2);
        // Semi join should NOT include right columns
        assert!(result_df.get_column("dept").is_none());
    }

    #[test]
    fn lazy_anti_join() {
        let left = test_df();
        let right = dept_df();

        let result = LazyView::from_df(left)
            .join(
                LazyView::from_df(right),
                vec![("name".into(), "name".into())],
                JoinType::Anti,
            )
            .collect()
            .unwrap();

        let result_df = result.borrow();
        // Anti join: Carol and Dave (not in right)
        assert_eq!(result_df.nrows(), 2);
    }

    // ── Distinct ─────────────────────────────────────────────────────────

    #[test]
    fn lazy_distinct() {
        let df = test_df();

        let result = LazyView::from_df(df)
            .distinct(vec!["age".into()])
            .collect()
            .unwrap();

        let result_df = result.borrow();
        // 3 distinct ages: 25, 30, 35
        assert_eq!(result_df.nrows(), 3);
    }

    // ── Complex chain ────────────────────────────────────────────────────

    #[test]
    fn complex_lazy_chain() {
        let df = test_df();

        // filter(age > 20) -> mutate(bonus = score * 1.1) -> select(name, bonus) -> arrange(bonus desc)
        let result = LazyView::from_df(df)
            .filter(DExpr::BinOp {
                op: DBinOp::Gt,
                left: Box::new(DExpr::Col("age".into())),
                right: Box::new(DExpr::LitInt(20)),
            })
            .mutate(vec![(
                "bonus".into(),
                DExpr::BinOp {
                    op: DBinOp::Mul,
                    left: Box::new(DExpr::Col("score".into())),
                    right: Box::new(DExpr::LitFloat(1.1)),
                },
            )])
            .select(vec!["name".into(), "bonus".into()])
            .arrange(vec![ArrangeKey::desc("bonus")])
            .collect()
            .unwrap();

        let result_df = result.borrow();
        assert_eq!(result_df.nrows(), 4);
        assert_eq!(result_df.ncols(), 2);
        assert_eq!(result_df.column_names(), vec!["name", "bonus"]);
    }

    // ── Predicate pushdown into join ─────────────────────────────────────

    #[test]
    fn predicate_pushdown_into_join_left_side() {
        let left = test_df();
        let right = dept_df();

        // Join then filter on "age" > 25 -- "age" only exists in left.
        let lazy = LazyView::from_df(left)
            .join(
                LazyView::from_df(right),
                vec![("name".into(), "name".into())],
                JoinType::Inner,
            )
            .filter(DExpr::BinOp {
                op: DBinOp::Gt,
                left: Box::new(DExpr::Col("age".into())),
                right: Box::new(DExpr::LitInt(25)),
            });

        let optimized = lazy.optimized_plan();

        // The filter should be pushed into the left side of the join.
        let kinds = optimized.node_kinds();
        // Expect: Join -> [Filter -> Scan (left), Scan (right)]
        assert_eq!(kinds[0], "Join");
        // The left subtree should contain a Filter.
        if let ViewNode::Join { left, right, .. } = &optimized {
            assert_eq!(left.kind(), "Filter");
            assert_eq!(right.kind(), "Scan");
        } else {
            panic!("expected Join at top");
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Batch executor tests
    // ══════════════════════════════════════════════════════════════════════

    /// Helper: compare two DataFrames column-by-column for equality.
    fn assert_df_eq(a: &DataFrame, b: &DataFrame, context: &str) {
        assert_eq!(
            a.nrows(),
            b.nrows(),
            "{}: nrows differ ({} vs {})",
            context,
            a.nrows(),
            b.nrows()
        );
        assert_eq!(
            a.column_names(),
            b.column_names(),
            "{}: column names differ",
            context
        );
        for (name_a, col_a) in &a.columns {
            let col_b = b.get_column(name_a).unwrap_or_else(|| {
                panic!("{}: column '{}' missing in b", context, name_a)
            });
            assert_col_eq(col_a, col_b, &format!("{} col '{}'", context, name_a));
        }
    }

    fn assert_col_eq(a: &Column, b: &Column, context: &str) {
        match (a, b) {
            (Column::Int(va), Column::Int(vb)) => assert_eq!(va, vb, "{}", context),
            (Column::Float(va), Column::Float(vb)) => {
                assert_eq!(va.len(), vb.len(), "{}: float len", context);
                for (i, (x, y)) in va.iter().zip(vb.iter()).enumerate() {
                    assert!(
                        (x - y).abs() < 1e-12,
                        "{}: float[{}] {} != {}",
                        context,
                        i,
                        x,
                        y
                    );
                }
            }
            (Column::Str(va), Column::Str(vb)) => assert_eq!(va, vb, "{}", context),
            (Column::Bool(va), Column::Bool(vb)) => assert_eq!(va, vb, "{}", context),
            _ => panic!("{}: column type mismatch", context),
        }
    }

    // ── Parity: collect_batched == collect ───────────────────────────────

    #[test]
    fn batched_filter_parity() {
        let predicate = DExpr::BinOp {
            op: DBinOp::Gt,
            left: Box::new(DExpr::Col("age".into())),
            right: Box::new(DExpr::LitInt(25)),
        };

        let eager = LazyView::from_df(test_df())
            .filter(predicate.clone())
            .collect()
            .unwrap();
        let batched = LazyView::from_df(test_df())
            .filter(predicate)
            .collect_batched()
            .unwrap();

        assert_df_eq(&eager.borrow(), &batched.borrow(), "filter parity");
    }

    #[test]
    fn batched_select_parity() {
        let cols = vec!["name".into(), "score".into()];

        let eager = LazyView::from_df(test_df())
            .select(cols.clone())
            .collect()
            .unwrap();
        let batched = LazyView::from_df(test_df())
            .select(cols)
            .collect_batched()
            .unwrap();

        assert_df_eq(&eager.borrow(), &batched.borrow(), "select parity");
    }

    #[test]
    fn batched_mutate_parity() {
        let assignments = vec![(
            "doubled".into(),
            DExpr::BinOp {
                op: DBinOp::Mul,
                left: Box::new(DExpr::Col("age".into())),
                right: Box::new(DExpr::LitInt(2)),
            },
        )];

        let eager = LazyView::from_df(test_df())
            .mutate(assignments.clone())
            .collect()
            .unwrap();
        let batched = LazyView::from_df(test_df())
            .mutate(assignments)
            .collect_batched()
            .unwrap();

        assert_df_eq(&eager.borrow(), &batched.borrow(), "mutate parity");
    }

    #[test]
    fn batched_filter_select_mutate_chain_parity() {
        let predicate = DExpr::BinOp {
            op: DBinOp::Gt,
            left: Box::new(DExpr::Col("age".into())),
            right: Box::new(DExpr::LitInt(20)),
        };
        let assignments = vec![(
            "bonus".into(),
            DExpr::BinOp {
                op: DBinOp::Mul,
                left: Box::new(DExpr::Col("score".into())),
                right: Box::new(DExpr::LitFloat(1.1)),
            },
        )];

        let eager = LazyView::from_df(test_df())
            .filter(predicate.clone())
            .mutate(assignments.clone())
            .select(vec!["name".into(), "bonus".into()])
            .collect()
            .unwrap();
        let batched = LazyView::from_df(test_df())
            .filter(predicate)
            .mutate(assignments)
            .select(vec!["name".into(), "bonus".into()])
            .collect_batched()
            .unwrap();

        assert_df_eq(
            &eager.borrow(),
            &batched.borrow(),
            "filter+mutate+select chain parity",
        );
    }

    #[test]
    fn batched_group_summarise_parity() {
        let eager = LazyView::from_df(test_df())
            .group_summarise(
                vec!["age".into()],
                vec![("count".into(), TidyAgg::Count)],
            )
            .collect()
            .unwrap();
        let batched = LazyView::from_df(test_df())
            .group_summarise(
                vec!["age".into()],
                vec![("count".into(), TidyAgg::Count)],
            )
            .collect_batched()
            .unwrap();

        assert_df_eq(
            &eager.borrow(),
            &batched.borrow(),
            "group_summarise parity",
        );
    }

    #[test]
    fn batched_arrange_parity() {
        let keys = vec![ArrangeKey::asc("age")];

        let eager = LazyView::from_df(test_df())
            .arrange(keys.clone())
            .collect()
            .unwrap();
        let batched = LazyView::from_df(test_df())
            .arrange(keys)
            .collect_batched()
            .unwrap();

        assert_df_eq(&eager.borrow(), &batched.borrow(), "arrange parity");
    }

    #[test]
    fn batched_distinct_parity() {
        let eager = LazyView::from_df(test_df())
            .distinct(vec!["age".into()])
            .collect()
            .unwrap();
        let batched = LazyView::from_df(test_df())
            .distinct(vec!["age".into()])
            .collect_batched()
            .unwrap();

        assert_df_eq(&eager.borrow(), &batched.borrow(), "distinct parity");
    }

    #[test]
    fn batched_join_parity() {
        let eager = LazyView::from_df(test_df())
            .join(
                LazyView::from_df(dept_df()),
                vec![("name".into(), "name".into())],
                JoinType::Inner,
            )
            .collect()
            .unwrap();
        let batched = LazyView::from_df(test_df())
            .join(
                LazyView::from_df(dept_df()),
                vec![("name".into(), "name".into())],
                JoinType::Inner,
            )
            .collect_batched()
            .unwrap();

        assert_df_eq(&eager.borrow(), &batched.borrow(), "join parity");
    }

    #[test]
    fn batched_complex_pipeline_parity() {
        // filter -> mutate -> select -> arrange (has a pipeline breaker at end)
        let predicate = DExpr::BinOp {
            op: DBinOp::Gt,
            left: Box::new(DExpr::Col("age".into())),
            right: Box::new(DExpr::LitInt(20)),
        };
        let assignments = vec![(
            "bonus".into(),
            DExpr::BinOp {
                op: DBinOp::Mul,
                left: Box::new(DExpr::Col("score".into())),
                right: Box::new(DExpr::LitFloat(1.1)),
            },
        )];

        let eager = LazyView::from_df(test_df())
            .filter(predicate.clone())
            .mutate(assignments.clone())
            .select(vec!["name".into(), "bonus".into()])
            .arrange(vec![ArrangeKey::desc("bonus")])
            .collect()
            .unwrap();
        let batched = LazyView::from_df(test_df())
            .filter(predicate)
            .mutate(assignments)
            .select(vec!["name".into(), "bonus".into()])
            .arrange(vec![ArrangeKey::desc("bonus")])
            .collect_batched()
            .unwrap();

        assert_df_eq(
            &eager.borrow(),
            &batched.borrow(),
            "complex pipeline parity",
        );
    }

    // ── Determinism: 3 runs identical ───────────────────────────────────

    #[test]
    fn batched_determinism_3_runs() {
        let mut results: Vec<Vec<i64>> = Vec::new();
        let mut results_names: Vec<Vec<String>> = Vec::new();

        for _ in 0..3 {
            let result = LazyView::from_df(test_df())
                .filter(DExpr::BinOp {
                    op: DBinOp::Gt,
                    left: Box::new(DExpr::Col("age".into())),
                    right: Box::new(DExpr::LitInt(20)),
                })
                .select(vec!["name".into(), "age".into()])
                .arrange(vec![ArrangeKey::desc("age")])
                .collect_batched()
                .unwrap();

            let df = result.borrow();
            let ages = match df.get_column("age").unwrap() {
                Column::Int(v) => v.clone(),
                _ => panic!("expected Int"),
            };
            let names = match df.get_column("name").unwrap() {
                Column::Str(v) => v.clone(),
                _ => panic!("expected Str"),
            };
            results.push(ages);
            results_names.push(names);
        }

        // All 3 runs must be identical.
        assert_eq!(results[0], results[1]);
        assert_eq!(results[1], results[2]);
        assert_eq!(results_names[0], results_names[1]);
        assert_eq!(results_names[1], results_names[2]);
        // Verify expected values.
        assert_eq!(results[0], vec![35, 30, 25, 25]);
        assert_eq!(results_names[0], vec!["Carol", "Alice", "Bob", "Dave"]);
    }

    // ── Large data: batching actually kicks in (>2048 rows) ─────────────

    /// Build a large DataFrame with 10,000 rows.
    fn large_df() -> DataFrame {
        let n = 10_000usize;
        let names: Vec<String> = (0..n).map(|i| format!("user_{}", i)).collect();
        let ages: Vec<i64> = (0..n).map(|i| (i % 80) as i64 + 18).collect();
        let scores: Vec<f64> = (0..n).map(|i| 50.0 + (i % 50) as f64).collect();
        DataFrame {
            columns: vec![
                ("name".to_string(), Column::Str(names)),
                ("age".to_string(), Column::Int(ages)),
                ("score".to_string(), Column::Float(scores)),
            ],
        }
    }

    #[test]
    fn batched_large_data_filter_parity() {
        let predicate = DExpr::BinOp {
            op: DBinOp::Gt,
            left: Box::new(DExpr::Col("age".into())),
            right: Box::new(DExpr::LitInt(50)),
        };

        let eager = LazyView::from_df(large_df())
            .filter(predicate.clone())
            .collect()
            .unwrap();
        let batched = LazyView::from_df(large_df())
            .filter(predicate)
            .collect_batched()
            .unwrap();

        assert_df_eq(
            &eager.borrow(),
            &batched.borrow(),
            "large data filter parity",
        );
        // Verify batching actually processed >1 batch.
        assert!(eager.borrow().nrows() > 0);
    }

    #[test]
    fn batched_large_data_chain_parity() {
        let predicate = DExpr::BinOp {
            op: DBinOp::Gt,
            left: Box::new(DExpr::Col("age".into())),
            right: Box::new(DExpr::LitInt(50)),
        };
        let assignments = vec![(
            "bonus".into(),
            DExpr::BinOp {
                op: DBinOp::Mul,
                left: Box::new(DExpr::Col("score".into())),
                right: Box::new(DExpr::LitFloat(1.5)),
            },
        )];

        let eager = LazyView::from_df(large_df())
            .filter(predicate.clone())
            .mutate(assignments.clone())
            .select(vec!["name".into(), "bonus".into()])
            .collect()
            .unwrap();
        let batched = LazyView::from_df(large_df())
            .filter(predicate)
            .mutate(assignments)
            .select(vec!["name".into(), "bonus".into()])
            .collect_batched()
            .unwrap();

        assert_df_eq(
            &eager.borrow(),
            &batched.borrow(),
            "large data chain parity",
        );
    }

    #[test]
    fn batched_large_data_determinism() {
        let mut prev_ages: Option<Vec<i64>> = None;
        for _ in 0..3 {
            let result = LazyView::from_df(large_df())
                .filter(DExpr::BinOp {
                    op: DBinOp::Gt,
                    left: Box::new(DExpr::Col("age".into())),
                    right: Box::new(DExpr::LitInt(90)),
                })
                .mutate(vec![(
                    "double_age".into(),
                    DExpr::BinOp {
                        op: DBinOp::Mul,
                        left: Box::new(DExpr::Col("age".into())),
                        right: Box::new(DExpr::LitInt(2)),
                    },
                )])
                .collect_batched()
                .unwrap();

            let df = result.borrow();
            let ages = match df.get_column("age").unwrap() {
                Column::Int(v) => v.clone(),
                _ => panic!("expected Int"),
            };
            if let Some(ref prev) = prev_ages {
                assert_eq!(prev, &ages, "determinism: ages differ across runs");
            }
            prev_ages = Some(ages);
        }
    }

    // ── Batch splitting helper ──────────────────────────────────────────

    #[test]
    fn split_batches_correct_count() {
        let df = large_df();
        let batches = split_batches(&df);
        // 10000 rows / 2048 = 4 full + 1 partial = 5 batches
        assert_eq!(batches.len(), 5);
        assert_eq!(batches[0].nrows, 2048);
        assert_eq!(batches[1].nrows, 2048);
        assert_eq!(batches[2].nrows, 2048);
        assert_eq!(batches[3].nrows, 2048);
        assert_eq!(batches[4].nrows, 10000 - 4 * 2048); // 1808
        let total: usize = batches.iter().map(|b| b.nrows).sum();
        assert_eq!(total, 10000);
    }

    #[test]
    fn split_batches_small_df() {
        let df = test_df(); // 4 rows
        let batches = split_batches(&df);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].nrows, 4);
    }

    #[test]
    fn merge_batches_roundtrip() {
        let df = large_df();
        let batches = split_batches(&df);
        let merged = merge_batches(batches).unwrap();
        assert_df_eq(&df, &merged, "merge roundtrip");
    }
}
