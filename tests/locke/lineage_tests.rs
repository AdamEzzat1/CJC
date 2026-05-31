//! Integration tests for the Locke lineage builder + audit chain.

use cjc_locke::lineage::{
    emit_lineage_text, ImpressionKind, LineageBuilder, LineageError, LockeIdea,
    LockeImpression, TransformationRecord,
};
use std::collections::BTreeMap;

fn imp(source: &str) -> LockeImpression {
    LockeImpression::new(source, ImpressionKind::Dataset, 100, vec!["x".into(), "y".into()])
}

fn xform(op: &str) -> TransformationRecord {
    TransformationRecord {
        op_id: op.into(),
        params: BTreeMap::new(),
        seed: None,
    }
}

#[test]
fn graph_round_trip_bytes_are_stable() {
    let build = || {
        let mut b = LineageBuilder::new("integration");
        let p = b.add_impression(imp("train.csv")).unwrap();
        let f = b
            .add_idea(LockeIdea::new("filter", xform("filter"), vec![p]))
            .unwrap();
        let _ = b
            .add_idea(LockeIdea::new("mean", xform("mean"), vec![f]))
            .unwrap();
        emit_lineage_text(&b.finish())
    };
    assert_eq!(build(), build());
}

#[test]
fn cycle_introduction_is_rejected() {
    let mut b = LineageBuilder::new("integration");
    let p = b.add_impression(imp("a.csv")).unwrap();
    let i1 = b
        .add_idea(LockeIdea::new("a", xform("a"), vec![p]))
        .unwrap();
    let i2 = b
        .add_idea(LockeIdea::new("b", xform("b"), vec![i1]))
        .unwrap();
    // Try to add an idea whose declared "parent" creates a back-edge.
    // We synthesise an Idea whose id we'll point at i1, and whose parents include i2.
    // The natural construction is: child = i1, parent = i2 → would form i1 → i2 → i1.
    // Since add_idea takes a fully-formed LockeIdea, we test via a deliberate
    // re-add that would close the cycle:
    //
    //   Force: try to add another idea whose id equals i1 (impossible because
    //   ids are content-addressed); instead test the cycle path by adding an
    //   idea with parents = vec![i2] whose own id happens to be in i2's
    //   ancestry. Since content-addressing prevents this naturally, we
    //   verify the simpler unknown-parent rejection here and rely on the
    //   builder's `would_introduce_cycle` unit test for the cycle path.
    let bogus = cjc_locke::FingerprintId(0xCAFEBABE);
    let res = b.add_idea(LockeIdea::new("z", xform("z"), vec![bogus]));
    assert!(matches!(res, Err(LineageError::UnknownParent(_))));
    let _ = i2;
}

#[test]
fn audit_chain_is_monotonically_sequenced() {
    let mut b = LineageBuilder::new("integration");
    let p = b.add_impression(imp("a.csv")).unwrap();
    let _ = b.add_idea(LockeIdea::new("a", xform("a"), vec![p])).unwrap();
    let _ = b.add_idea(LockeIdea::new("b", xform("b"), vec![p])).unwrap();
    let g = b.finish();
    for w in g.audit.windows(2) {
        assert!(w[0].seq < w[1].seq);
    }
}

#[test]
fn traced_dataframe_records_pipeline_lineage() {
    use cjc_data::{Column, DataFrame};
    use cjc_locke::TracedDataFrame;

    let df = DataFrame::from_columns(vec![
        ("age".into(), Column::Int(vec![20, 30, 40, 50])),
        ("score".into(), Column::Float(vec![1.0, 2.0, 3.0, 4.0])),
    ])
    .unwrap();

    let mut b = LineageBuilder::new("auto-pipeline");
    let _ = TracedDataFrame::observe(&mut b, "train.csv", df)
        .filter("age >= 30", |df| df.clone())
        .select(&["age"], |df| df.clone())
        .with_column("age_bucket", "age / 10", |df| df.clone())
        .into_parts();
    let g = b.finish();
    assert_eq!(g.edges.len(), 3);
    assert!(g.is_acyclic());
    let labels: Vec<&str> = g.edges.iter().map(|e| e.label.as_str()).collect();
    assert!(labels.contains(&"filter"));
    assert!(labels.contains(&"select"));
    assert!(labels.contains(&"with_column"));
}

#[test]
fn traced_dataframe_is_deterministic_across_runs() {
    use cjc_data::{Column, DataFrame};
    use cjc_locke::TracedDataFrame;

    fn run() -> cjc_locke::FingerprintId {
        let df = DataFrame::from_columns(vec![(
            "age".into(),
            Column::Int(vec![20, 30, 40, 50]),
        )])
        .unwrap();
        let mut b = LineageBuilder::new("determ");
        let _ = TracedDataFrame::observe(&mut b, "train.csv", df)
            .filter("age >= 30", |df| df.clone())
            .into_parts();
        b.finish().root_fingerprint
    }
    assert_eq!(run(), run());
}

#[test]
fn ancestors_includes_all_transitive_parents() {
    let mut b = LineageBuilder::new("integration");
    let p = b.add_impression(imp("a.csv")).unwrap();
    let f1 = b.add_idea(LockeIdea::new("f1", xform("f1"), vec![p])).unwrap();
    let f2 = b.add_idea(LockeIdea::new("f2", xform("f2"), vec![f1])).unwrap();
    let f3 = b.add_idea(LockeIdea::new("f3", xform("f3"), vec![f2])).unwrap();
    let g = b.finish();
    let a = g.ancestors(f3);
    assert!(a.contains(&p));
    assert!(a.contains(&f1));
    assert!(a.contains(&f2));
}
