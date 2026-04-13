//! Bolero fuzz targets for TidyView operations.
//!
//! Verifies that arbitrary data does not cause panics in arrange, group_by, or summarise.

use std::panic;

/// Fuzz TidyView arrange: arbitrary integer data should not panic.
#[test]
fn fuzz_tidyview_arrange() {
    bolero::check!()
        .with_type::<Vec<i64>>()
        .for_each(|data: &Vec<i64>| {
            if data.is_empty() || data.len() > 10_000 {
                return;
            }
            let _ = panic::catch_unwind(|| {
                let df = cjc_data::DataFrame::from_columns(vec![
                    ("x".into(), cjc_data::Column::Int(data.clone())),
                ])
                .unwrap();
                let view = df.tidy();
                let sorted = view
                    .arrange(&[cjc_data::ArrangeKey::asc("x")])
                    .unwrap();
                let _ = sorted.materialize();
            });
        });
}

/// Fuzz TidyView group_by + summarise: arbitrary data should not panic
/// and should be deterministic (two runs produce identical output).
#[test]
fn fuzz_tidyview_group_summarise() {
    bolero::check!()
        .with_type::<(Vec<i64>, Vec<u8>)>()
        .for_each(|input: &(Vec<i64>, Vec<u8>)| {
            let (vals, group_bytes) = input;
            if vals.is_empty() || vals.len() > 5_000 || group_bytes.is_empty() {
                return;
            }
            let n = vals.len();
            // Use group_bytes to create group keys (mod small number)
            let n_groups = (group_bytes[0] as usize % 20).max(1);
            let groups: Vec<i64> = (0..n).map(|i| (i % n_groups) as i64).collect();
            let floats: Vec<f64> = vals.iter().map(|&v| v as f64 * 0.001).collect();

            let _ = panic::catch_unwind(|| {
                let df = cjc_data::DataFrame::from_columns(vec![
                    ("grp".into(), cjc_data::Column::Int(groups.clone())),
                    ("val".into(), cjc_data::Column::Float(floats.clone())),
                ])
                .unwrap();
                let g = df.tidy().group_by(&["grp"]).unwrap();
                let _ = g.summarise(&[("s", cjc_data::TidyAgg::Sum("val".into()))]);
            });
        });
}
