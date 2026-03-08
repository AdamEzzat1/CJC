//! Faceting — multi-panel grid layouts split by a grouping variable.
//!
//! Supports facet_wrap (auto-arranged panels) and facet_grid (row × col).
//! All layout operations are deterministic.

use crate::spec::{PlotData, DataColumn};

/// Faceting specification.
#[derive(Debug, Clone)]
pub enum FacetSpec {
    /// No faceting — single panel.
    None,
    /// Wrap panels in rows with specified number of columns.
    Wrap {
        column: String,
        ncol: usize,
    },
    /// Full grid layout: row variable × column variable.
    Grid {
        row: String,
        col: String,
    },
}

impl Default for FacetSpec {
    fn default() -> Self {
        FacetSpec::None
    }
}

/// Layout result for faceted plot.
#[derive(Debug, Clone)]
pub struct FacetLayout {
    pub panels: Vec<FacetPanel>,
    pub n_rows: usize,
    pub n_cols: usize,
}

/// A single facet panel.
#[derive(Debug, Clone)]
pub struct FacetPanel {
    pub row_idx: usize,
    pub col_idx: usize,
    pub label: String,
    pub data_indices: Vec<usize>,
    /// Panel bounding box in pixels.
    pub plot_x: f64,
    pub plot_y: f64,
    pub plot_w: f64,
    pub plot_h: f64,
}

/// Compute facet layout from data and spec.
pub fn compute_facet_layout(
    data: &PlotData,
    facet: &FacetSpec,
    total_w: f64,
    total_h: f64,
    margin_left: f64,
    margin_top: f64,
    margin_right: f64,
    margin_bottom: f64,
) -> FacetLayout {
    match facet {
        FacetSpec::None => {
            FacetLayout {
                panels: vec![FacetPanel {
                    row_idx: 0,
                    col_idx: 0,
                    label: String::new(),
                    data_indices: (0..data.nrows()).collect(),
                    plot_x: margin_left,
                    plot_y: margin_top,
                    plot_w: total_w - margin_left - margin_right,
                    plot_h: total_h - margin_top - margin_bottom,
                }],
                n_rows: 1,
                n_cols: 1,
            }
        }
        FacetSpec::Wrap { column, ncol } => {
            compute_wrap_layout(data, column, *ncol, total_w, total_h,
                                margin_left, margin_top, margin_right, margin_bottom)
        }
        FacetSpec::Grid { row, col } => {
            compute_grid_layout(data, row, col, total_w, total_h,
                                margin_left, margin_top, margin_right, margin_bottom)
        }
    }
}

/// Subset data to only include rows at given indices.
pub fn subset_data(data: &PlotData, indices: &[usize]) -> PlotData {
    let columns = data.columns.iter().map(|(name, col)| {
        let new_col = match col {
            DataColumn::Float(v) => {
                DataColumn::Float(indices.iter().filter_map(|&i| v.get(i).copied()).collect())
            }
            DataColumn::Int(v) => {
                DataColumn::Int(indices.iter().filter_map(|&i| v.get(i).copied()).collect())
            }
            DataColumn::Str(v) => {
                DataColumn::Str(indices.iter().filter_map(|&i| v.get(i).cloned()).collect())
            }
        };
        (name.clone(), new_col)
    }).collect();
    PlotData { columns }
}

fn compute_wrap_layout(
    data: &PlotData,
    column: &str,
    ncol: usize,
    total_w: f64,
    total_h: f64,
    margin_left: f64,
    margin_top: f64,
    margin_right: f64,
    margin_bottom: f64,
) -> FacetLayout {
    let (groups, indices) = group_column(data, column);
    let n = groups.len();
    let ncol = ncol.max(1).min(n);
    let nrow = (n + ncol - 1) / ncol;

    let usable_w = total_w - margin_left - margin_right;
    let usable_h = total_h - margin_top - margin_bottom;
    let panel_gap = 8.0;
    let label_height = 16.0;

    let panel_w = (usable_w - (ncol as f64 - 1.0) * panel_gap) / ncol as f64;
    let panel_h = (usable_h - (nrow as f64 - 1.0) * panel_gap - nrow as f64 * label_height) / nrow as f64;

    let mut panels = Vec::with_capacity(n);
    for (i, (label, idxs)) in groups.into_iter().zip(indices.into_iter()).enumerate() {
        let r = i / ncol;
        let c = i % ncol;
        let px = margin_left + c as f64 * (panel_w + panel_gap);
        let py = margin_top + r as f64 * (panel_h + panel_gap + label_height) + label_height;
        panels.push(FacetPanel {
            row_idx: r,
            col_idx: c,
            label,
            data_indices: idxs,
            plot_x: px,
            plot_y: py,
            plot_w: panel_w,
            plot_h: panel_h,
        });
    }

    FacetLayout { panels, n_rows: nrow, n_cols: ncol }
}

fn compute_grid_layout(
    data: &PlotData,
    row_col: &str,
    col_col: &str,
    total_w: f64,
    total_h: f64,
    margin_left: f64,
    margin_top: f64,
    margin_right: f64,
    margin_bottom: f64,
) -> FacetLayout {
    let (row_groups, row_indices) = group_column(data, row_col);
    let (col_groups, _) = group_column(data, col_col);
    let nrow = row_groups.len();
    let ncol = col_groups.len();

    let usable_w = total_w - margin_left - margin_right;
    let usable_h = total_h - margin_top - margin_bottom;
    let panel_gap = 8.0;
    let label_height = 16.0;

    let panel_w = (usable_w - (ncol as f64 - 1.0) * panel_gap) / ncol as f64;
    let panel_h = (usable_h - (nrow as f64 - 1.0) * panel_gap - nrow as f64 * label_height) / nrow as f64;

    // Get the col-column values to partition
    let col_vals: Vec<String> = data.get(col_col)
        .map(|c| c.labels())
        .unwrap_or_default();

    let mut panels = Vec::new();
    for (ri, (row_label, row_idxs)) in row_groups.iter().zip(row_indices.iter()).enumerate() {
        for (ci, col_label) in col_groups.iter().enumerate() {
            // Intersection: indices that are in this row group AND match this col group
            let combined: Vec<usize> = row_idxs.iter()
                .filter(|&&idx| idx < col_vals.len() && col_vals[idx] == *col_label)
                .copied()
                .collect();

            let px = margin_left + ci as f64 * (panel_w + panel_gap);
            let py = margin_top + ri as f64 * (panel_h + panel_gap + label_height) + label_height;

            panels.push(FacetPanel {
                row_idx: ri,
                col_idx: ci,
                label: format!("{} | {}", row_label, col_label),
                data_indices: combined,
                plot_x: px,
                plot_y: py,
                plot_w: panel_w,
                plot_h: panel_h,
            });
        }
    }

    FacetLayout { panels, n_rows: nrow, n_cols: ncol }
}

/// Group by a column's string values, returning (unique_labels, indices_per_group).
/// Uses BTreeMap for deterministic ordering.
fn group_column(data: &PlotData, col_name: &str) -> (Vec<String>, Vec<Vec<usize>>) {
    use std::collections::BTreeMap;

    let labels = data.get(col_name)
        .map(|c| c.labels())
        .unwrap_or_default();

    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, label) in labels.into_iter().enumerate() {
        groups.entry(label).or_default().push(i);
    }

    let keys: Vec<String> = groups.keys().cloned().collect();
    let vals: Vec<Vec<usize>> = keys.iter().map(|k| groups[k].clone()).collect();
    (keys, vals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{PlotData, DataColumn};

    fn sample_data() -> PlotData {
        PlotData {
            columns: vec![
                ("x".to_string(), DataColumn::Float(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
                ("y".to_string(), DataColumn::Float(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0])),
                ("group".to_string(), DataColumn::Str(vec![
                    "A".into(), "A".into(), "A".into(),
                    "B".into(), "B".into(), "B".into(),
                ])),
            ],
        }
    }

    #[test]
    fn test_facet_none() {
        let data = sample_data();
        let layout = compute_facet_layout(&data, &FacetSpec::None, 800.0, 600.0, 60.0, 40.0, 20.0, 60.0);
        assert_eq!(layout.panels.len(), 1);
        assert_eq!(layout.panels[0].data_indices.len(), 6);
    }

    #[test]
    fn test_facet_wrap() {
        let data = sample_data();
        let facet = FacetSpec::Wrap { column: "group".to_string(), ncol: 2 };
        let layout = compute_facet_layout(&data, &facet, 800.0, 600.0, 60.0, 40.0, 20.0, 60.0);
        assert_eq!(layout.panels.len(), 2);
        assert_eq!(layout.panels[0].label, "A");
        assert_eq!(layout.panels[1].label, "B");
        assert_eq!(layout.panels[0].data_indices.len(), 3);
        assert_eq!(layout.panels[1].data_indices.len(), 3);
    }

    #[test]
    fn test_subset_data() {
        let data = sample_data();
        let sub = subset_data(&data, &[0, 2, 4]);
        assert_eq!(sub.nrows(), 3);
        if let Some(DataColumn::Float(v)) = sub.get("x") {
            assert_eq!(v, &[1.0, 3.0, 5.0]);
        } else {
            panic!("Expected Float column");
        }
    }

    #[test]
    fn test_group_column() {
        let data = sample_data();
        let (groups, indices) = group_column(&data, "group");
        assert_eq!(groups, vec!["A", "B"]);
        assert_eq!(indices[0], vec![0, 1, 2]);
        assert_eq!(indices[1], vec![3, 4, 5]);
    }
}
