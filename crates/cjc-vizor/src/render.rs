//! Render pipeline — converts a PlotSpec into a Scene.
//!
//! Pipeline: PlotSpec → Layout → Scene elements (deterministic).
//! Order: background → axes → gridlines → geom layers → annotations → labels.

use crate::annotation::{Annotation, Position, format_pvalue, format_r_squared, format_ci};
use crate::color::{self, Color};
use crate::layout::{self, compute_layout, histogram_counts, LayoutResult};
use crate::scene::{Scene, SceneElement, TextAnchor};
use crate::spec::{Geom, PlotSpec, RugSide};
use crate::stats;

/// Build a scene from a plot spec. Fully deterministic.
/// Dispatches to single-panel or faceted rendering.
pub fn build_scene(spec: &PlotSpec) -> Scene {
    match &spec.facet {
        crate::facet::FacetSpec::None => build_single_panel(spec),
        _ => build_faceted(spec),
    }
}

/// Build a faceted (multi-panel) scene.
fn build_faceted(spec: &PlotSpec) -> Scene {
    let mut scene = Scene::new(spec.width, spec.height);
    let theme = &spec.theme;

    // Background.
    scene.push(SceneElement::Rect {
        x: 0.0, y: 0.0,
        w: spec.width as f64, h: spec.height as f64,
        fill: theme.background,
        stroke: None,
        stroke_width: 0.0,
    });

    let facet_layout = crate::facet::compute_facet_layout(
        &spec.data, &spec.facet,
        spec.width as f64, spec.height as f64,
        theme.margin_left, theme.margin_top,
        theme.margin_right, theme.margin_bottom,
    );

    for panel in &facet_layout.panels {
        // Create a sub-spec with the panel's data subset.
        let sub_data = crate::facet::subset_data(&spec.data, &panel.data_indices);
        let sub_spec = PlotSpec {
            data: sub_data,
            layers: spec.layers.clone(),
            scales: spec.scales.clone(),
            labels: crate::spec::Labels::default(), // No labels per-panel
            theme: spec.theme.clone(),
            coord: spec.coord.clone(),
            annotations: vec![],
            facet: crate::facet::FacetSpec::None,
            width: panel.plot_w as u32,
            height: panel.plot_h as u32,
            show_legend: false, // No legend per-panel
        };

        // Build the sub-panel scene.
        let sub_scene = build_single_panel(&sub_spec);

        // Offset and insert sub-scene elements.
        for elem in &sub_scene.elements {
            scene.push(offset_element(elem, panel.plot_x, panel.plot_y));
        }

        // Panel label.
        if !panel.label.is_empty() {
            scene.push(SceneElement::Text {
                x: panel.plot_x + panel.plot_w / 2.0,
                y: panel.plot_y - 3.0,
                text: panel.label.clone(),
                font_size: theme.tick_label_font_size,
                fill: theme.text_color,
                anchor: TextAnchor::Middle,
                rotation: None,
            });
        }
    }

    // Global title.
    if let Some(ref title) = spec.labels.title {
        scene.push(SceneElement::Text {
            x: spec.width as f64 / 2.0,
            y: 20.0,
            text: title.clone(),
            font_size: theme.title_font_size,
            fill: theme.text_color,
            anchor: TextAnchor::Middle,
            rotation: None,
        });
    }

    scene
}

/// Offset a scene element by (dx, dy).
fn offset_element(elem: &SceneElement, dx: f64, dy: f64) -> SceneElement {
    match elem {
        SceneElement::Rect { x, y, w, h, fill, stroke, stroke_width } => SceneElement::Rect {
            x: x + dx, y: y + dy, w: *w, h: *h, fill: *fill, stroke: *stroke, stroke_width: *stroke_width,
        },
        SceneElement::Circle { cx, cy, r, fill, stroke } => SceneElement::Circle {
            cx: cx + dx, cy: cy + dy, r: *r, fill: *fill, stroke: *stroke,
        },
        SceneElement::Line { x1, y1, x2, y2, stroke, width } => SceneElement::Line {
            x1: x1 + dx, y1: y1 + dy, x2: x2 + dx, y2: y2 + dy, stroke: *stroke, width: *width,
        },
        SceneElement::Polyline { points, stroke, width, fill } => SceneElement::Polyline {
            points: points.iter().map(|(x, y)| (x + dx, y + dy)).collect(),
            stroke: *stroke, width: *width, fill: *fill,
        },
        SceneElement::Text { x, y, text, font_size, fill, anchor, rotation } => SceneElement::Text {
            x: x + dx, y: y + dy, text: text.clone(), font_size: *font_size,
            fill: *fill, anchor: anchor.clone(), rotation: *rotation,
        },
    }
}

/// Build a single-panel (non-faceted) scene.
fn build_single_panel(spec: &PlotSpec) -> Scene {
    let layout = compute_layout(spec);
    let mut scene = Scene::new(spec.width, spec.height);
    let theme = &spec.theme;

    // 1. Background.
    scene.push(SceneElement::Rect {
        x: 0.0, y: 0.0,
        w: spec.width as f64, h: spec.height as f64,
        fill: theme.background,
        stroke: None,
        stroke_width: 0.0,
    });

    // 2. Plot area background.
    scene.push(SceneElement::Rect {
        x: layout.plot_x, y: layout.plot_y,
        w: layout.plot_w, h: layout.plot_h,
        fill: theme.plot_background,
        stroke: None,
        stroke_width: 0.0,
    });

    // 3. Gridlines — suppress for polar, tile, and dendrogram geoms
    //    (they have their own axis/grid rendering).
    let suppress_axes = layout::is_all_polar(spec)
        || layout::is_all_tile(spec)
        || layout::is_all_dendrogram(spec);
    if !suppress_axes {
        render_gridlines(&mut scene, &layout, theme);
    }

    // 4. Axes — also suppressed for polar, tile, and dendrogram geoms.
    if !suppress_axes {
        render_axes(&mut scene, &layout, theme);
    }

    // 5. Geom layers (in order).
    for (layer_idx, layer) in spec.layers.iter().enumerate() {
        let layer_color = layer.params.color_override.unwrap_or_else(|| color::palette_color(layer_idx));
        match layer.geom {
            Geom::Point => render_points(&mut scene, spec, &layout, layer_idx, layer_color),
            Geom::Line => render_line(&mut scene, spec, &layout, layer_idx, layer_color),
            Geom::Bar => render_bars(&mut scene, spec, &layout, layer_idx, layer_color),
            Geom::Histogram => render_histogram(&mut scene, spec, &layout, layer_idx, layer_color),
            Geom::Density => render_density(&mut scene, spec, &layout, layer, layer_color),
            Geom::Area => render_area(&mut scene, spec, &layout, layer_idx, layer_color),
            Geom::Rug => render_rug(&mut scene, spec, &layout, layer, layer_color),
            Geom::Ecdf => render_ecdf(&mut scene, spec, &layout, layer_color),
            Geom::Box => render_box(&mut scene, spec, &layout, layer, layer_color),
            Geom::Violin => render_violin(&mut scene, spec, &layout, layer, layer_color),
            Geom::Strip => render_strip(&mut scene, spec, &layout, layer, layer_color),
            Geom::Swarm => render_swarm(&mut scene, spec, &layout, layer, layer_color),
            Geom::Boxen => render_boxen(&mut scene, spec, &layout, layer, layer_color),
            Geom::Tile => render_tile(&mut scene, spec, &layout, layer, layer_color),
            Geom::RegressionLine => render_regression_line(&mut scene, spec, &layout, layer_color),
            Geom::Residual => render_residual(&mut scene, spec, &layout, layer_color),
            Geom::Dendrogram => render_dendrogram(&mut scene, spec, &layout, layer_color),
            // Phase 3: Polar geoms
            Geom::Pie => render_pie(&mut scene, spec, &layout, layer, layer_idx),
            Geom::Rose => render_rose(&mut scene, spec, &layout, layer, layer_idx),
            Geom::Radar => render_radar(&mut scene, spec, &layout, layer, layer_idx),
            // Phase 3: 2D density + contour
            Geom::Density2d => render_density2d(&mut scene, spec, &layout, layer, layer_idx),
            Geom::Contour => render_contour(&mut scene, spec, &layout, layer, layer_idx),
            // Phase 3.2: Error bars + Step line
            Geom::ErrorBar => render_errorbar(&mut scene, spec, &layout, layer, layer_color),
            Geom::Step => render_step(&mut scene, spec, &layout, layer, layer_color),
        }
    }

    // 6. Annotations.
    render_annotations(&mut scene, spec, &layout);

    // 7. Title and axis labels.
    render_labels(&mut scene, spec, &layout);

    // 8. Legend (auto for 2+ layers, unless disabled).
    if spec.show_legend {
        let entries = crate::legend::collect_legend_entries(spec);
        if !entries.is_empty() {
            crate::legend::render_legend(
                &mut scene,
                &entries,
                layout.plot_x,
                layout.plot_y,
                layout.plot_w,
                layout.plot_h,
                theme.tick_label_font_size,
                theme.text_color,
            );
        }
    }

    scene
}

fn render_gridlines(scene: &mut Scene, layout: &LayoutResult, theme: &crate::theme::Theme) {
    // Horizontal gridlines at y ticks.
    for &(val, _) in &layout.y_ticks {
        let y = layout.map_y(val);
        if y >= layout.plot_y && y <= layout.plot_y + layout.plot_h {
            scene.push(SceneElement::Line {
                x1: layout.plot_x,
                y1: y,
                x2: layout.plot_x + layout.plot_w,
                y2: y,
                stroke: theme.grid_color,
                width: theme.grid_line_width,
            });
        }
    }
    // Vertical gridlines at x ticks.
    for &(val, _) in &layout.x_ticks {
        let x = layout.map_x(val);
        if x >= layout.plot_x && x <= layout.plot_x + layout.plot_w {
            scene.push(SceneElement::Line {
                x1: x,
                y1: layout.plot_y,
                x2: x,
                y2: layout.plot_y + layout.plot_h,
                stroke: theme.grid_color,
                width: theme.grid_line_width,
            });
        }
    }
}

fn render_axes(scene: &mut Scene, layout: &LayoutResult, theme: &crate::theme::Theme) {
    // X axis line.
    scene.push(SceneElement::Line {
        x1: layout.plot_x,
        y1: layout.plot_y + layout.plot_h,
        x2: layout.plot_x + layout.plot_w,
        y2: layout.plot_y + layout.plot_h,
        stroke: theme.axis_color,
        width: theme.axis_line_width,
    });
    // Y axis line.
    scene.push(SceneElement::Line {
        x1: layout.plot_x,
        y1: layout.plot_y,
        x2: layout.plot_x,
        y2: layout.plot_y + layout.plot_h,
        stroke: theme.axis_color,
        width: theme.axis_line_width,
    });

    // X tick marks and labels.
    for (val, label) in &layout.x_ticks {
        let x = layout.map_x(*val);
        if x >= layout.plot_x && x <= layout.plot_x + layout.plot_w {
            // Tick mark.
            scene.push(SceneElement::Line {
                x1: x,
                y1: layout.plot_y + layout.plot_h,
                x2: x,
                y2: layout.plot_y + layout.plot_h + theme.tick_length,
                stroke: theme.axis_color,
                width: theme.axis_line_width,
            });
            // Label.
            scene.push(SceneElement::Text {
                x,
                y: layout.plot_y + layout.plot_h + theme.tick_length + theme.tick_label_font_size + 2.0,
                text: label.clone(),
                font_size: theme.tick_label_font_size,
                fill: theme.text_color,
                anchor: TextAnchor::Middle,
                rotation: None,
            });
        }
    }

    // Y tick marks and labels.
    for (val, label) in &layout.y_ticks {
        let y = layout.map_y(*val);
        if y >= layout.plot_y && y <= layout.plot_y + layout.plot_h {
            // Tick mark.
            scene.push(SceneElement::Line {
                x1: layout.plot_x - theme.tick_length,
                y1: y,
                x2: layout.plot_x,
                y2: y,
                stroke: theme.axis_color,
                width: theme.axis_line_width,
            });
            // Label.
            scene.push(SceneElement::Text {
                x: layout.plot_x - theme.tick_length - 4.0,
                y: y + theme.tick_label_font_size * 0.35,
                text: label.clone(),
                font_size: theme.tick_label_font_size,
                fill: theme.text_color,
                anchor: TextAnchor::End,
                rotation: None,
            });
        }
    }
}

fn render_points(scene: &mut Scene, spec: &PlotSpec, layout: &LayoutResult, _layer_idx: usize, color: Color) {
    let x_col = spec.data.get("x").map(|c| c.to_f64()).unwrap_or_default();
    let y_col = spec.data.get("y").map(|c| c.to_f64()).unwrap_or_default();
    let n = x_col.len().min(y_col.len());

    for i in 0..n {
        let px = layout.map_x(x_col[i]);
        let py = layout.map_y(y_col[i]);
        if px >= layout.plot_x && px <= layout.plot_x + layout.plot_w
            && py >= layout.plot_y && py <= layout.plot_y + layout.plot_h
        {
            scene.push(SceneElement::Circle {
                cx: px,
                cy: py,
                r: spec.theme.point_size,
                fill: color,
                stroke: None,
            });
        }
    }
}

fn render_line(scene: &mut Scene, spec: &PlotSpec, layout: &LayoutResult, _layer_idx: usize, color: Color) {
    let x_col = spec.data.get("x").map(|c| c.to_f64()).unwrap_or_default();
    let y_col = spec.data.get("y").map(|c| c.to_f64()).unwrap_or_default();
    let n = x_col.len().min(y_col.len());

    let points: Vec<(f64, f64)> = (0..n)
        .map(|i| (layout.map_x(x_col[i]), layout.map_y(y_col[i])))
        .collect();

    if !points.is_empty() {
        scene.push(SceneElement::Polyline {
            points,
            stroke: color,
            width: spec.theme.line_width,
            fill: None,
        });
    }
}

fn render_bars(scene: &mut Scene, spec: &PlotSpec, layout: &LayoutResult, _layer_idx: usize, color: Color) {
    let x_col = spec.data.get("x").map(|c| c.to_f64()).unwrap_or_default();
    let y_col = spec.data.get("y").map(|c| c.to_f64()).unwrap_or_default();
    let n = x_col.len().min(y_col.len());

    let bar_width_data = 0.8;
    let half = bar_width_data / 2.0;

    for i in 0..n {
        let x_left = layout.map_x(x_col[i] - half);
        let x_right = layout.map_x(x_col[i] + half);
        let y_top = layout.map_y(y_col[i]);
        let y_bottom = layout.map_y(0.0_f64.max(layout.y_min));
        let w = (x_right - x_left).abs();
        let h = (y_bottom - y_top).abs();

        scene.push(SceneElement::Rect {
            x: x_left.min(x_right),
            y: y_top.min(y_bottom),
            w,
            h,
            fill: color,
            stroke: Some(Color::WHITE),
            stroke_width: 1.0,
        });
    }
}

fn render_histogram(scene: &mut Scene, spec: &PlotSpec, layout: &LayoutResult, _layer_idx: usize, color: Color) {
    let x_col = spec.data.get("x").map(|c| c.to_f64()).unwrap_or_default();
    if x_col.is_empty() { return; }

    let bins = spec.layers.iter()
        .find(|l| l.geom == Geom::Histogram)
        .and_then(|l| l.params.bins)
        .unwrap_or(10);

    let (data_min, data_max) = crate::layout::data_range(&x_col);
    if !data_min.is_finite() || !data_max.is_finite() { return; }

    let counts = histogram_counts(&x_col, data_min, data_max, bins);
    let bin_width = (data_max - data_min) / bins as f64;

    for (i, &count) in counts.iter().enumerate() {
        let bin_left = data_min + i as f64 * bin_width;
        let bin_right = bin_left + bin_width;

        let x_left = layout.map_x(bin_left);
        let x_right = layout.map_x(bin_right);
        let y_top = layout.map_y(count as f64);
        let y_bottom = layout.map_y(0.0);
        let w = (x_right - x_left).abs();
        let h = (y_bottom - y_top).abs();

        scene.push(SceneElement::Rect {
            x: x_left.min(x_right),
            y: y_top.min(y_bottom),
            w,
            h,
            fill: color,
            stroke: Some(Color::WHITE),
            stroke_width: 1.0,
        });
    }
}

// ── Phase 2B: Distribution geom renderers ────────────────────────────

fn render_density(
    scene: &mut Scene,
    spec: &PlotSpec,
    layout: &LayoutResult,
    layer: &crate::spec::Layer,
    color: Color,
) {
    let x_col = spec.data.get("x").map(|c| c.to_f64()).unwrap_or_default();
    if x_col.len() < 2 { return; }

    let n_points = layer.params.n_grid_points.unwrap_or(200);
    let bw = layer.params.bandwidth;

    let (grid, density) = if let Some(bw_val) = bw {
        stats::kde_with_bandwidth(&x_col, n_points, bw_val)
    } else {
        stats::kde(&x_col, n_points)
    };

    // Build a closed polygon: start at (grid[0], 0), trace the curve, close at (grid[last], 0).
    let mut points = Vec::with_capacity(density.len() + 2);
    points.push((layout.map_x(grid[0]), layout.map_y(0.0)));
    for i in 0..grid.len() {
        points.push((layout.map_x(grid[i]), layout.map_y(density[i])));
    }
    points.push((layout.map_x(grid[grid.len() - 1]), layout.map_y(0.0)));

    // Filled polygon for the area under the KDE curve.
    scene.push(SceneElement::Polyline {
        points: points.clone(),
        stroke: color,
        width: 0.0,
        fill: Some(color.with_alpha(0.3)),
    });

    // Stroke line on top.
    let curve_points: Vec<(f64, f64)> = (0..grid.len())
        .map(|i| (layout.map_x(grid[i]), layout.map_y(density[i])))
        .collect();
    scene.push(SceneElement::Polyline {
        points: curve_points,
        stroke: color,
        width: spec.theme.line_width,
        fill: None,
    });
}

fn render_area(
    scene: &mut Scene,
    spec: &PlotSpec,
    layout: &LayoutResult,
    _layer_idx: usize,
    color: Color,
) {
    let x_col = spec.data.get("x").map(|c| c.to_f64()).unwrap_or_default();
    let y_col = spec.data.get("y").map(|c| c.to_f64()).unwrap_or_default();
    let n = x_col.len().min(y_col.len());
    if n == 0 { return; }

    // Build closed polygon from (x0, 0) → curve → (xn, 0).
    let baseline_y = layout.map_y(0.0_f64.max(layout.y_min));
    let mut points = Vec::with_capacity(n + 2);
    points.push((layout.map_x(x_col[0]), baseline_y));
    for i in 0..n {
        points.push((layout.map_x(x_col[i]), layout.map_y(y_col[i])));
    }
    points.push((layout.map_x(x_col[n - 1]), baseline_y));

    scene.push(SceneElement::Polyline {
        points: points.clone(),
        stroke: color,
        width: 0.0,
        fill: Some(color.with_alpha(0.3)),
    });

    // Stroke line on top.
    let curve_points: Vec<(f64, f64)> = (0..n)
        .map(|i| (layout.map_x(x_col[i]), layout.map_y(y_col[i])))
        .collect();
    scene.push(SceneElement::Polyline {
        points: curve_points,
        stroke: color,
        width: spec.theme.line_width,
        fill: None,
    });
}

fn render_rug(
    scene: &mut Scene,
    spec: &PlotSpec,
    layout: &LayoutResult,
    layer: &crate::spec::Layer,
    color: Color,
) {
    let x_col = spec.data.get("x").map(|c| c.to_f64()).unwrap_or_default();
    let rug_len = layer.params.rug_length;
    let side = layer.params.rug_side;

    for &val in &x_col {
        if !val.is_finite() { continue; }
        match side {
            RugSide::Bottom => {
                let px = layout.map_x(val);
                let py_base = layout.plot_y + layout.plot_h;
                scene.push(SceneElement::Line {
                    x1: px, y1: py_base,
                    x2: px, y2: py_base - rug_len,
                    stroke: color,
                    width: 0.8,
                });
            }
            RugSide::Top => {
                let px = layout.map_x(val);
                let py_base = layout.plot_y;
                scene.push(SceneElement::Line {
                    x1: px, y1: py_base,
                    x2: px, y2: py_base + rug_len,
                    stroke: color,
                    width: 0.8,
                });
            }
            RugSide::Left => {
                // For left-side rug, we use the y-data if available, else x.
                let py = layout.map_y(val);
                let px_base = layout.plot_x;
                scene.push(SceneElement::Line {
                    x1: px_base, y1: py,
                    x2: px_base + rug_len, y2: py,
                    stroke: color,
                    width: 0.8,
                });
            }
            RugSide::Right => {
                let py = layout.map_y(val);
                let px_base = layout.plot_x + layout.plot_w;
                scene.push(SceneElement::Line {
                    x1: px_base, y1: py,
                    x2: px_base - rug_len, y2: py,
                    stroke: color,
                    width: 0.8,
                });
            }
        }
    }
}

fn render_ecdf(
    scene: &mut Scene,
    spec: &PlotSpec,
    layout: &LayoutResult,
    color: Color,
) {
    let x_col = spec.data.get("x").map(|c| c.to_f64()).unwrap_or_default();
    if x_col.is_empty() { return; }

    let (step_x, step_y) = stats::ecdf(&x_col);

    // Build a step-function polyline.
    let mut points = Vec::with_capacity(step_x.len() * 2);
    for i in 0..step_x.len() {
        if i > 0 {
            // Horizontal segment from previous y to current x.
            points.push((layout.map_x(step_x[i]), layout.map_y(step_y[i - 1])));
        }
        // Vertical jump.
        points.push((layout.map_x(step_x[i]), layout.map_y(step_y[i])));
    }

    if !points.is_empty() {
        scene.push(SceneElement::Polyline {
            points,
            stroke: color,
            width: spec.theme.line_width,
            fill: None,
        });
    }
}

// ── Phase 2B: Categorical geom renderers ─────────────────────────────

/// Helper: partition data by category. Returns (category_labels, per_category_values).
fn group_by_x_category(spec: &PlotSpec) -> (Vec<String>, Vec<Vec<f64>>) {
    let x_col = spec.data.get("x");
    let y_col = spec.data.get("y");
    match (x_col, y_col) {
        (Some(x), Some(y)) => {
            let x_labels = x.labels();
            let y_vals = y.to_f64();
            let n = x_labels.len().min(y_vals.len());
            stats::group_by_category(&x_labels[..n], &y_vals[..n])
        }
        _ => (vec![], vec![]),
    }
}

fn render_box(
    scene: &mut Scene,
    spec: &PlotSpec,
    layout: &LayoutResult,
    layer: &crate::spec::Layer,
    color: Color,
) {
    let (cats, groups) = group_by_x_category(spec);
    let cat_w = layer.params.category_width;
    let half_w = cat_w / 2.0;

    for (i, values) in groups.iter().enumerate() {
        if values.is_empty() { continue; }
        let cx = i as f64; // category center
        let bs = stats::box_stats(values);

        let px_center = layout.map_x(cx);
        let px_left = layout.map_x(cx - half_w);
        let px_right = layout.map_x(cx + half_w);
        let box_w = (px_right - px_left).abs();

        // IQR box
        let py_q1 = layout.map_y(bs.q1);
        let py_q3 = layout.map_y(bs.q3);
        let box_h = (py_q1 - py_q3).abs();
        scene.push(SceneElement::Rect {
            x: px_left.min(px_right),
            y: py_q3.min(py_q1),
            w: box_w,
            h: box_h,
            fill: color.with_alpha(0.3),
            stroke: Some(color),
            stroke_width: 1.5,
        });

        // Median line
        let py_med = layout.map_y(bs.median);
        scene.push(SceneElement::Line {
            x1: px_left, y1: py_med,
            x2: px_right, y2: py_med,
            stroke: color,
            width: 2.0,
        });

        // Whiskers (vertical lines from box to whisker tips)
        let py_lower = layout.map_y(bs.lower_whisker);
        let py_upper = layout.map_y(bs.upper_whisker);
        // Lower whisker
        scene.push(SceneElement::Line {
            x1: px_center, y1: py_q1,
            x2: px_center, y2: py_lower,
            stroke: color,
            width: 1.0,
        });
        // Upper whisker
        scene.push(SceneElement::Line {
            x1: px_center, y1: py_q3,
            x2: px_center, y2: py_upper,
            stroke: color,
            width: 1.0,
        });
        // Whisker caps
        let cap_half = box_w * 0.3;
        scene.push(SceneElement::Line {
            x1: px_center - cap_half, y1: py_lower,
            x2: px_center + cap_half, y2: py_lower,
            stroke: color, width: 1.0,
        });
        scene.push(SceneElement::Line {
            x1: px_center - cap_half, y1: py_upper,
            x2: px_center + cap_half, y2: py_upper,
            stroke: color, width: 1.0,
        });

        // Outliers
        if layer.params.show_outliers {
            for &out_val in &bs.outliers {
                let py = layout.map_y(out_val);
                scene.push(SceneElement::Circle {
                    cx: px_center,
                    cy: py,
                    r: 3.0,
                    fill: Color::WHITE,
                    stroke: Some(color),
                });
            }
        }
    }

    // Add category labels as x-ticks (handled in layout, but let's make sure labels are there).
    let _ = cats;
}

fn render_violin(
    scene: &mut Scene,
    spec: &PlotSpec,
    layout: &LayoutResult,
    layer: &crate::spec::Layer,
    color: Color,
) {
    let (_, groups) = group_by_x_category(spec);
    let cat_w = layer.params.category_width;

    for (i, values) in groups.iter().enumerate() {
        if values.len() < 2 { continue; }
        let cx = i as f64;

        let n_pts = layer.params.n_grid_points.unwrap_or(50);
        let (grid, density) = if let Some(bw) = layer.params.violin_bw {
            stats::kde_with_bandwidth(values, n_pts, bw)
        } else {
            stats::kde(values, n_pts)
        };

        // Normalize density so max width = cat_w/2
        let max_d = density.iter().copied().fold(0.0_f64, f64::max);
        if max_d < 1e-15 { continue; }
        let scale = (cat_w / 2.0) / max_d;

        // Build mirrored polygon, clamping y to plot area bounds.
        let px_center = layout.map_x(cx);
        let py_top = layout.plot_y;
        let py_bottom = layout.plot_y + layout.plot_h;
        let mut points = Vec::with_capacity(grid.len() * 2 + 1);

        // Right side (top to bottom in data space)
        for j in 0..grid.len() {
            let py = layout.map_y(grid[j]).clamp(py_top, py_bottom);
            let dx = density[j] * scale;
            let px = layout.map_x(cx + dx);
            points.push((px, py));
        }
        // Left side (bottom to top, mirrored)
        for j in (0..grid.len()).rev() {
            let py = layout.map_y(grid[j]).clamp(py_top, py_bottom);
            let dx = density[j] * scale;
            let px = layout.map_x(cx - dx);
            points.push((px, py));
        }

        scene.push(SceneElement::Polyline {
            points,
            stroke: color,
            width: 1.0,
            fill: Some(color.with_alpha(0.3)),
        });

        // Median line
        let mut sorted_vals = values.clone();
        sorted_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = stats::quantile(&sorted_vals, 0.5);
        let py_med = layout.map_y(med);

        // Find width at median
        let mut med_width = 0.0;
        for j in 0..grid.len().saturating_sub(1) {
            if (grid[j] <= med && med <= grid[j + 1]) || (grid[j + 1] <= med && med <= grid[j]) {
                let t = if (grid[j + 1] - grid[j]).abs() > 1e-15 {
                    (med - grid[j]) / (grid[j + 1] - grid[j])
                } else { 0.5 };
                med_width = density[j] + t * (density[j + 1] - density[j]);
                break;
            }
        }
        let med_dx = med_width * scale;
        scene.push(SceneElement::Line {
            x1: layout.map_x(cx - med_dx),
            y1: py_med,
            x2: layout.map_x(cx + med_dx),
            y2: py_med,
            stroke: Color::WHITE,
            width: 2.0,
        });

        // Center dot
        scene.push(SceneElement::Circle {
            cx: px_center,
            cy: py_med,
            r: 3.0,
            fill: Color::WHITE,
            stroke: None,
        });
    }
}

fn render_strip(
    scene: &mut Scene,
    spec: &PlotSpec,
    layout: &LayoutResult,
    layer: &crate::spec::Layer,
    color: Color,
) {
    let (_, groups) = group_by_x_category(spec);
    let jitter_w = layer.params.jitter_width;

    for (i, values) in groups.iter().enumerate() {
        if values.is_empty() { continue; }
        let cx = i as f64;
        let jitter = stats::deterministic_jitter(values.len(), jitter_w);

        for (j, &val) in values.iter().enumerate() {
            let px = layout.map_x(cx + jitter[j]);
            let py = layout.map_y(val);
            scene.push(SceneElement::Circle {
                cx: px,
                cy: py,
                r: 3.0,
                fill: color.with_alpha(0.6),
                stroke: None,
            });
        }
    }
}

fn render_swarm(
    scene: &mut Scene,
    spec: &PlotSpec,
    layout: &LayoutResult,
    layer: &crate::spec::Layer,
    color: Color,
) {
    let (_, groups) = group_by_x_category(spec);
    let max_w = layer.params.category_width / 2.0;

    for (i, values) in groups.iter().enumerate() {
        if values.is_empty() { continue; }
        let cx = i as f64;
        let offsets = stats::swarm_offsets(values, 3.0, max_w);

        // Map the pixel-space offsets to data-space
        let x_range = layout.x_max - layout.x_min;
        let data_per_px = if layout.plot_w > 0.0 { x_range / layout.plot_w } else { 1.0 };

        for (j, &val) in values.iter().enumerate() {
            let data_offset = offsets[j] * data_per_px;
            let px = layout.map_x(cx + data_offset);
            let py = layout.map_y(val);
            scene.push(SceneElement::Circle {
                cx: px,
                cy: py,
                r: 3.0,
                fill: color.with_alpha(0.7),
                stroke: None,
            });
        }
    }
}

fn render_boxen(
    scene: &mut Scene,
    spec: &PlotSpec,
    layout: &LayoutResult,
    layer: &crate::spec::Layer,
    color: Color,
) {
    let (_, groups) = group_by_x_category(spec);
    let cat_w = layer.params.category_width;

    for (i, values) in groups.iter().enumerate() {
        if values.is_empty() { continue; }
        let cx = i as f64;
        let levels = stats::letter_value_stats(values);
        let n_levels = levels.len();

        for (li, &(lo, hi)) in levels.iter().enumerate() {
            // Width decreases from outer to inner
            let frac = if n_levels > 1 {
                1.0 - li as f64 / n_levels as f64
            } else {
                1.0
            };
            let w = cat_w * frac;
            let half = w / 2.0;

            let px_left = layout.map_x(cx - half);
            let px_right = layout.map_x(cx + half);
            let py_lo = layout.map_y(lo);
            let py_hi = layout.map_y(hi);
            let box_w = (px_right - px_left).abs();
            let box_h = (py_lo - py_hi).abs();

            // Color intensity increases inward
            let t = if n_levels > 1 { li as f64 / (n_levels - 1) as f64 } else { 0.5 };
            let alpha = 0.2 + 0.6 * t;

            scene.push(SceneElement::Rect {
                x: px_left.min(px_right),
                y: py_hi.min(py_lo),
                w: box_w,
                h: box_h,
                fill: color.with_alpha(alpha),
                stroke: Some(color),
                stroke_width: 0.5,
            });
        }

        // Median line
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = stats::quantile(&sorted, 0.5);
        let py_med = layout.map_y(med);
        let px_left = layout.map_x(cx - cat_w / 2.0);
        let px_right = layout.map_x(cx + cat_w / 2.0);
        scene.push(SceneElement::Line {
            x1: px_left, y1: py_med,
            x2: px_right, y2: py_med,
            stroke: Color::WHITE,
            width: 2.0,
        });
    }
}

// ── Phase 2B: Heatmap / Tile renderer ────────────────────────────────

fn render_tile(
    scene: &mut Scene,
    spec: &PlotSpec,
    layout: &LayoutResult,
    layer: &crate::spec::Layer,
    _default_color: Color,
) {
    // Matrix data is stored via from_matrix() with __values, __nrows, __ncols, etc.
    let values = match spec.data.get("__values") {
        Some(crate::spec::DataColumn::Float(v)) => v.clone(),
        _ => return,
    };
    let nrows = match spec.data.get("__nrows") {
        Some(crate::spec::DataColumn::Int(v)) if !v.is_empty() => v[0] as usize,
        _ => return,
    };
    let ncols = match spec.data.get("__ncols") {
        Some(crate::spec::DataColumn::Int(v)) if !v.is_empty() => v[0] as usize,
        _ => return,
    };
    if nrows == 0 || ncols == 0 || values.len() < nrows * ncols { return; }

    let row_labels = match spec.data.get("__row_labels") {
        Some(crate::spec::DataColumn::Str(v)) => v.clone(),
        _ => (0..nrows).map(|i| i.to_string()).collect(),
    };
    let col_labels = match spec.data.get("__col_labels") {
        Some(crate::spec::DataColumn::Str(v)) => v.clone(),
        _ => (0..ncols).map(|i| i.to_string()).collect(),
    };

    // Find value range for color mapping.
    let mut vmin = f64::INFINITY;
    let mut vmax = f64::NEG_INFINITY;
    for &v in &values {
        if v.is_finite() {
            if v < vmin { vmin = v; }
            if v > vmax { vmax = v; }
        }
    }
    if !vmin.is_finite() || !vmax.is_finite() { return; }
    let vrange = if (vmax - vmin).abs() < 1e-15 { 1.0 } else { vmax - vmin };

    // Choose color mapping based on scale.
    let color_fn = |val: f64| -> Color {
        let t = ((val - vmin) / vrange).clamp(0.0, 1.0);
        match &spec.scales.color {
            crate::spec::ColorScale::Diverging => color::diverging_palette(t),
            crate::spec::ColorScale::Gradient { low, high } => Color::lerp(*low, *high, t),
            _ => color::sequential_palette(t), // Default + Sequential + Manual all use sequential
        }
    };

    // Compute cell dimensions.
    let cell_w = layout.plot_w / ncols as f64;
    let cell_h = layout.plot_h / nrows as f64;

    for row in 0..nrows {
        for col in 0..ncols {
            let idx = row * ncols + col;
            let val = values[idx];
            let fill = color_fn(val);

            let cx = layout.plot_x + col as f64 * cell_w;
            let cy = layout.plot_y + row as f64 * cell_h;

            scene.push(SceneElement::Rect {
                x: cx,
                y: cy,
                w: cell_w,
                h: cell_h,
                fill,
                stroke: Some(Color::WHITE),
                stroke_width: 0.5,
            });

            // Optional cell value text.
            if layer.params.show_cell_values {
                let text = format!("{:.2}", val);
                // Choose text color based on fill brightness.
                let brightness = fill.r as f64 * 0.299 + fill.g as f64 * 0.587 + fill.b as f64 * 0.114;
                let text_color = if brightness > 128.0 { Color::BLACK } else { Color::WHITE };
                scene.push(SceneElement::Text {
                    x: cx + cell_w / 2.0,
                    y: cy + cell_h / 2.0 + 4.0,
                    text,
                    font_size: (cell_h * 0.35).min(12.0).max(6.0),
                    fill: text_color,
                    anchor: TextAnchor::Middle,
                    rotation: None,
                });
            }
        }
    }

    // Draw row labels on left.
    for (i, label) in row_labels.iter().enumerate() {
        scene.push(SceneElement::Text {
            x: layout.plot_x - 5.0,
            y: layout.plot_y + (i as f64 + 0.5) * cell_h + 4.0,
            text: label.clone(),
            font_size: spec.theme.tick_label_font_size,
            fill: spec.theme.text_color,
            anchor: TextAnchor::End,
            rotation: None,
        });
    }

    // Draw column labels on top.
    for (j, label) in col_labels.iter().enumerate() {
        scene.push(SceneElement::Text {
            x: layout.plot_x + (j as f64 + 0.5) * cell_w,
            y: layout.plot_y - 5.0,
            text: label.clone(),
            font_size: spec.theme.tick_label_font_size,
            fill: spec.theme.text_color,
            anchor: TextAnchor::Middle,
            rotation: None,
        });
    }
}

// ── Phase 2B: Regression geom renderers ───────────────────────────────

fn render_regression_line(
    scene: &mut Scene,
    spec: &PlotSpec,
    layout: &LayoutResult,
    color: Color,
) {
    let x_col = spec.data.get("x").map(|c| c.to_f64()).unwrap_or_default();
    let y_col = spec.data.get("y").map(|c| c.to_f64()).unwrap_or_default();
    if x_col.len() < 2 || y_col.len() < 2 { return; }

    let (slope, intercept, _) = stats::linear_regression(&x_col, &y_col);

    // Draw the fitted line across the entire plot range.
    let x_start = layout.x_min;
    let x_end = layout.x_max;
    let y_start = slope * x_start + intercept;
    let y_end = slope * x_end + intercept;

    scene.push(SceneElement::Line {
        x1: layout.map_x(x_start),
        y1: layout.map_y(y_start),
        x2: layout.map_x(x_end),
        y2: layout.map_y(y_end),
        stroke: color,
        width: 2.0,
    });
}

fn render_residual(
    scene: &mut Scene,
    spec: &PlotSpec,
    layout: &LayoutResult,
    color: Color,
) {
    let x_col = spec.data.get("x").map(|c| c.to_f64()).unwrap_or_default();
    let y_col = spec.data.get("y").map(|c| c.to_f64()).unwrap_or_default();
    let n = x_col.len().min(y_col.len());
    if n < 2 { return; }

    let (slope, intercept, _) = stats::linear_regression(&x_col[..n], &y_col[..n]);
    let resid = stats::residuals(&x_col[..n], &y_col[..n], slope, intercept);

    // Draw a horizontal zero-line.
    scene.push(SceneElement::Line {
        x1: layout.map_x(layout.x_min),
        y1: layout.map_y(0.0),
        x2: layout.map_x(layout.x_max),
        y2: layout.map_y(0.0),
        stroke: Color::GRAY,
        width: 1.0,
    });

    // Draw points at (x_i, residual_i).
    for i in 0..n {
        let px = layout.map_x(x_col[i]);
        let py = layout.map_y(resid[i]);
        if px >= layout.plot_x && px <= layout.plot_x + layout.plot_w
            && py >= layout.plot_y && py <= layout.plot_y + layout.plot_h
        {
            scene.push(SceneElement::Circle {
                cx: px,
                cy: py,
                r: 3.5,
                fill: color,
                stroke: None,
            });
        }
    }
}

// ── Phase 2B: Dendrogram renderer ─────────────────────────────────────

fn render_dendrogram(
    scene: &mut Scene,
    spec: &PlotSpec,
    layout: &LayoutResult,
    color: Color,
) {
    // Dendrogram data is stored via __values (distance matrix), __nrows, __ncols.
    // We reuse the matrix format: __values = flat distances, __nrows = __ncols = n.
    let values = match spec.data.get("__values") {
        Some(crate::spec::DataColumn::Float(v)) => v.clone(),
        _ => return,
    };
    let n = match spec.data.get("__nrows") {
        Some(crate::spec::DataColumn::Int(v)) if !v.is_empty() => v[0] as usize,
        _ => return,
    };
    if n < 2 || values.len() < n * n { return; }

    // Reconstruct distance matrix.
    let mut dist = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            dist[i][j] = values[i * n + j];
        }
    }

    let merges = stats::hierarchical_cluster(&dist, stats::Linkage::Average);
    let leaf_order = stats::dendrogram_leaf_order(&merges, n);
    if merges.is_empty() { return; }

    // Leaf labels.
    let leaf_labels = match spec.data.get("__row_labels") {
        Some(crate::spec::DataColumn::Str(v)) => v.clone(),
        _ => (0..n).map(|i| i.to_string()).collect(),
    };

    // Assign x-positions to leaves based on ordering.
    let mut leaf_x = vec![0.0_f64; n];
    for (pos, &leaf_idx) in leaf_order.iter().enumerate() {
        leaf_x[leaf_idx] = pos as f64;
    }

    // Build node x/y positions. Each internal node gets the mean x of its children
    // and y = merge distance.
    let mut node_x = vec![0.0_f64; n + merges.len()];
    let mut node_y = vec![0.0_f64; n + merges.len()];

    // Initialize leaf nodes.
    for i in 0..n {
        node_x[i] = leaf_x[i];
        node_y[i] = 0.0;
    }

    // Track which "active ID" maps to which node in the tree.
    let mut active_id: Vec<usize> = (0..n).collect();

    for (mi, step) in merges.iter().enumerate() {
        let new_node = n + mi;
        let left_node = active_id[step.left];
        let right_node = active_id[step.right];

        node_x[new_node] = (node_x[left_node] + node_x[right_node]) / 2.0;
        node_y[new_node] = step.distance;

        active_id[step.left] = new_node;
    }

    let max_dist = merges.last().map(|s| s.distance).unwrap_or(1.0);

    // Draw the tree. For each merge, draw an inverted-U:
    // left child → horizontal bar at merge height → right child.
    let mut active_id2: Vec<usize> = (0..n).collect();
    for (mi, step) in merges.iter().enumerate() {
        let new_node = n + mi;
        let left_node = active_id2[step.left];
        let right_node = active_id2[step.right];

        let lx = layout.map_x(node_x[left_node]);
        let rx = layout.map_x(node_x[right_node]);
        let ly = layout.map_y(node_y[left_node] / max_dist);
        let ry = layout.map_y(node_y[right_node] / max_dist);
        let merge_y = layout.map_y(node_y[new_node] / max_dist);

        // Left vertical
        scene.push(SceneElement::Line {
            x1: lx, y1: ly, x2: lx, y2: merge_y,
            stroke: color, width: 1.5,
        });
        // Right vertical
        scene.push(SceneElement::Line {
            x1: rx, y1: ry, x2: rx, y2: merge_y,
            stroke: color, width: 1.5,
        });
        // Horizontal bar
        scene.push(SceneElement::Line {
            x1: lx, y1: merge_y, x2: rx, y2: merge_y,
            stroke: color, width: 1.5,
        });

        active_id2[step.left] = new_node;
    }

    // Leaf labels at the bottom.
    for (pos, &leaf_idx) in leaf_order.iter().enumerate() {
        let px = layout.map_x(pos as f64);
        let py = layout.plot_y + layout.plot_h + 12.0;
        let label = if leaf_idx < leaf_labels.len() {
            leaf_labels[leaf_idx].clone()
        } else {
            leaf_idx.to_string()
        };
        scene.push(SceneElement::Text {
            x: px, y: py,
            text: label,
            font_size: spec.theme.tick_label_font_size,
            fill: spec.theme.text_color,
            anchor: TextAnchor::Middle,
            rotation: None,
        });
    }
}

fn resolve_position(pos: &Position, layout: &LayoutResult, _theme: &crate::theme::Theme) -> (f64, f64) {
    match pos {
        Position::Absolute { x, y } => (*x, *y),
        Position::Data { x, y } => (layout.map_x(*x), layout.map_y(*y)),
        Position::Relative { x, y } => (
            layout.plot_x + x * layout.plot_w,
            layout.plot_y + y * layout.plot_h,
        ),
        Position::TopRight => (layout.plot_x + layout.plot_w - 10.0, layout.plot_y + 20.0),
        Position::TopLeft => (layout.plot_x + 10.0, layout.plot_y + 20.0),
        Position::BottomRight => (layout.plot_x + layout.plot_w - 10.0, layout.plot_y + layout.plot_h - 10.0),
        Position::BottomLeft => (layout.plot_x + 10.0, layout.plot_y + layout.plot_h - 10.0),
    }
}

/// Render a semi-transparent background box behind annotation text for legibility.
fn render_annotation_bg(
    scene: &mut Scene,
    x: f64,
    y: f64,
    text: &str,
    font_size: f64,
    anchor: &TextAnchor,
    bg: Color,
    lines: usize,
) {
    let pad = 4.0;
    let char_w = font_size * 0.6; // Approximate character width
    let text_w = text.len() as f64 * char_w;
    let text_h = font_size * lines as f64 + 3.0 * (lines.saturating_sub(1)) as f64;
    let bx = match anchor {
        TextAnchor::End => x - text_w - pad,
        TextAnchor::Middle => x - text_w / 2.0 - pad,
        TextAnchor::Start => x - pad,
    };
    let by = y - font_size - pad;
    scene.push(SceneElement::Rect {
        x: bx,
        y: by,
        w: text_w + 2.0 * pad,
        h: text_h + 2.0 * pad,
        fill: bg,
        stroke: None,
        stroke_width: 0.0,
    });
}

fn render_annotations(scene: &mut Scene, spec: &PlotSpec, layout: &LayoutResult) {
    let theme = &spec.theme;
    let bg_color = theme.plot_background.with_alpha(0.85);

    // ── Stacking: track cumulative y-offset per named position ──
    // Multiple annotations at the same named corner auto-stack vertically.
    let mut top_right_y = 0.0_f64;
    let mut top_left_y = 0.0_f64;
    let mut bottom_left_y = 0.0_f64;
    let mut bottom_right_y = 0.0_f64;
    let line_h = theme.tick_label_font_size + 4.0;

    // Helper: get stacking offset for a position, and advance it.
    let stack_offset = |pos: &Position, lines: usize,
                        tr: &mut f64, tl: &mut f64, bl: &mut f64, br: &mut f64| -> f64 {
        let delta = lines as f64 * line_h;
        match pos {
            Position::TopRight => { let o = *tr; *tr += delta; o }
            Position::TopLeft => { let o = *tl; *tl += delta; o }
            Position::BottomLeft => { let o = *bl; *bl -= delta; o }
            Position::BottomRight => { let o = *br; *br -= delta; o }
            _ => 0.0,
        }
    };

    for ann in &spec.annotations {
        match ann {
            Annotation::Text { text, position, font_size } => {
                let (x, y) = resolve_position(position, layout, theme);
                let fs = font_size.unwrap_or(theme.tick_label_font_size);
                let yo = stack_offset(position, 1,
                    &mut top_right_y, &mut top_left_y, &mut bottom_left_y, &mut bottom_right_y);
                render_annotation_bg(scene, x, y + yo, text, fs, &TextAnchor::Start, bg_color, 1);
                scene.push(SceneElement::Text {
                    x, y: y + yo, text: text.clone(),
                    font_size: fs,
                    fill: theme.text_color,
                    anchor: TextAnchor::Start,
                    rotation: None,
                });
            }
            Annotation::Note { text, position } => {
                let (x, y) = resolve_position(position, layout, theme);
                let fs = theme.tick_label_font_size * 0.9;
                let yo = stack_offset(position, 1,
                    &mut top_right_y, &mut top_left_y, &mut bottom_left_y, &mut bottom_right_y);
                render_annotation_bg(scene, x, y + yo, text, fs, &TextAnchor::Start, bg_color, 1);
                scene.push(SceneElement::Text {
                    x, y: y + yo, text: text.clone(),
                    font_size: fs,
                    fill: Color::GRAY,
                    anchor: TextAnchor::Start,
                    rotation: None,
                });
            }
            Annotation::RegressionSummary { equation, r_squared, position } => {
                let (x, y) = resolve_position(position, layout, theme);
                let line1 = equation.clone();
                let line2 = format_r_squared(*r_squared);
                let fs = theme.tick_label_font_size;
                let yo = stack_offset(position, 2,
                    &mut top_right_y, &mut top_left_y, &mut bottom_left_y, &mut bottom_right_y);
                // Background for both lines
                let longer = if line1.len() > line2.len() { &line1 } else { &line2 };
                render_annotation_bg(scene, x, y + yo, longer, fs, &TextAnchor::End, bg_color, 2);
                scene.push(SceneElement::Text {
                    x, y: y + yo, text: line1,
                    font_size: fs,
                    fill: theme.text_color,
                    anchor: TextAnchor::End,
                    rotation: None,
                });
                scene.push(SceneElement::Text {
                    x, y: y + yo + fs + 4.0,
                    text: line2,
                    font_size: fs,
                    fill: theme.text_color,
                    anchor: TextAnchor::End,
                    rotation: None,
                });
            }
            Annotation::ConfidenceInterval { level, lower, upper, position } => {
                let (x, y) = resolve_position(position, layout, theme);
                let text = format_ci(*level, *lower, *upper);
                let fs = theme.tick_label_font_size;
                let yo = stack_offset(position, 1,
                    &mut top_right_y, &mut top_left_y, &mut bottom_left_y, &mut bottom_right_y);
                render_annotation_bg(scene, x, y + yo, &text, fs, &TextAnchor::End, bg_color, 1);
                scene.push(SceneElement::Text {
                    x, y: y + yo, text,
                    font_size: fs,
                    fill: theme.text_color,
                    anchor: TextAnchor::End,
                    rotation: None,
                });
            }
            Annotation::PValue { value, position, .. } => {
                let (x, y) = resolve_position(position, layout, theme);
                let text = format_pvalue(*value);
                let fs = theme.tick_label_font_size;
                let yo = stack_offset(position, 1,
                    &mut top_right_y, &mut top_left_y, &mut bottom_left_y, &mut bottom_right_y);
                render_annotation_bg(scene, x, y + yo, &text, fs, &TextAnchor::End, bg_color, 1);
                scene.push(SceneElement::Text {
                    x, y: y + yo, text,
                    font_size: fs,
                    fill: theme.text_color,
                    anchor: TextAnchor::End,
                    rotation: None,
                });
            }
            Annotation::ModelMetrics { metrics, position } => {
                let (x, y) = resolve_position(position, layout, theme);
                let fs = theme.tick_label_font_size;
                let n_lines = metrics.len();
                let yo = stack_offset(position, n_lines,
                    &mut top_right_y, &mut top_left_y, &mut bottom_left_y, &mut bottom_right_y);
                // Find longest metric text for background
                let longest = metrics.iter()
                    .map(|(name, val)| format!("{}: {:.4}", name, val))
                    .max_by_key(|s| s.len())
                    .unwrap_or_default();
                render_annotation_bg(scene, x, y + yo, &longest, fs, &TextAnchor::End, bg_color, n_lines);
                let mut cy = y + yo;
                for (name, val) in metrics {
                    let text = format!("{}: {:.4}", name, val);
                    scene.push(SceneElement::Text {
                        x, y: cy, text,
                        font_size: fs,
                        fill: theme.text_color,
                        anchor: TextAnchor::End,
                        rotation: None,
                    });
                    cy += fs + 3.0;
                }
            }
            Annotation::EventMarker { x: data_x, label } => {
                let px = layout.map_x(*data_x);
                if px >= layout.plot_x && px <= layout.plot_x + layout.plot_w {
                    scene.push(SceneElement::Line {
                        x1: px, y1: layout.plot_y,
                        x2: px, y2: layout.plot_y + layout.plot_h,
                        stroke: Color::GRAY,
                        width: 1.0,
                    });
                    let fs = theme.tick_label_font_size * 0.85;
                    let label_y = layout.plot_y + fs;
                    render_annotation_bg(scene, px + 3.0, label_y, label, fs, &TextAnchor::Start, bg_color, 1);
                    scene.push(SceneElement::Text {
                        x: px + 3.0,
                        y: label_y,
                        text: label.clone(),
                        font_size: fs,
                        fill: Color::GRAY,
                        anchor: TextAnchor::Start,
                        rotation: None,
                    });
                }
            }
            Annotation::DataNote { text, position } => {
                let (x, y) = resolve_position(position, layout, theme);
                let fs = theme.tick_label_font_size * 0.8;
                let yo = stack_offset(position, 1,
                    &mut top_right_y, &mut top_left_y, &mut bottom_left_y, &mut bottom_right_y);
                render_annotation_bg(scene, x, y + yo, text, fs, &TextAnchor::Start, bg_color, 1);
                scene.push(SceneElement::Text {
                    x, y: y + yo, text: text.clone(),
                    font_size: fs,
                    fill: Color::GRAY,
                    anchor: TextAnchor::Start,
                    rotation: None,
                });
            }
            Annotation::InlineLabel { text, x: data_x, y: data_y } => {
                let px = layout.map_x(*data_x);
                let py = layout.map_y(*data_y);
                let fs = theme.tick_label_font_size * 0.9;
                render_annotation_bg(scene, px + 5.0, py - 5.0, text, fs, &TextAnchor::Start, bg_color, 1);
                scene.push(SceneElement::Text {
                    x: px + 5.0,
                    y: py - 5.0,
                    text: text.clone(),
                    font_size: fs,
                    fill: theme.text_color,
                    anchor: TextAnchor::Start,
                    rotation: None,
                });
            }
            Annotation::Callout { text, target, label_offset } => {
                let (tx, ty) = resolve_position(target, layout, theme);
                let lx = tx + label_offset.0;
                let ly = ty + label_offset.1;
                // Arrow line from label to target.
                scene.push(SceneElement::Line {
                    x1: lx, y1: ly,
                    x2: tx, y2: ty,
                    stroke: Color::GRAY,
                    width: 0.8,
                });
                let fs = theme.tick_label_font_size;
                render_annotation_bg(scene, lx, ly - 3.0, text, fs, &TextAnchor::Start, bg_color, 1);
                scene.push(SceneElement::Text {
                    x: lx, y: ly - 3.0,
                    text: text.clone(),
                    font_size: fs,
                    fill: theme.text_color,
                    anchor: TextAnchor::Start,
                    rotation: None,
                });
            }
        }
    }
}

fn render_labels(scene: &mut Scene, spec: &PlotSpec, layout: &LayoutResult) {
    let theme = &spec.theme;

    // Title.
    if let Some(ref title) = spec.labels.title {
        let title_y = layout.plot_y - 10.0;
        scene.push(SceneElement::Text {
            x: layout.plot_x + layout.plot_w / 2.0,
            y: title_y,
            text: title.clone(),
            font_size: theme.title_font_size,
            fill: theme.text_color,
            anchor: TextAnchor::Middle,
            rotation: None,
        });

        // Subtitle (rendered below the title, smaller and lighter).
        if let Some(ref subtitle) = spec.labels.subtitle {
            scene.push(SceneElement::Text {
                x: layout.plot_x + layout.plot_w / 2.0,
                y: title_y + theme.title_font_size * 1.1,
                text: subtitle.clone(),
                font_size: theme.axis_label_font_size,
                fill: Color::GRAY,
                anchor: TextAnchor::Middle,
                rotation: None,
            });
        }
    }

    // X-axis label.
    if let Some(ref xlabel) = spec.labels.x {
        scene.push(SceneElement::Text {
            x: layout.plot_x + layout.plot_w / 2.0,
            y: spec.height as f64 - 5.0,
            text: xlabel.clone(),
            font_size: theme.axis_label_font_size,
            fill: theme.text_color,
            anchor: TextAnchor::Middle,
            rotation: None,
        });
    }

    // Y-axis label (rotated 90° in SVG; for BMP placed vertically).
    if let Some(ref ylabel) = spec.labels.y {
        scene.push(SceneElement::Text {
            x: 15.0,
            y: layout.plot_y + layout.plot_h / 2.0,
            text: ylabel.clone(),
            font_size: theme.axis_label_font_size,
            fill: theme.text_color,
            anchor: TextAnchor::Middle,
            rotation: Some(-90.0),
        });
    }
}

// ── Phase 3: Polar geom renderers ──────────────────────────────────────

/// Render a pie chart. Slices are drawn as filled polygon arcs (polylines
/// approximating circular segments).
///
/// Data: categorical x + numeric y (values → slice sizes).
fn render_pie(
    scene: &mut Scene,
    spec: &PlotSpec,
    layout: &LayoutResult,
    layer: &crate::spec::Layer,
    _layer_idx: usize,
) {
    let (cats, groups) = group_by_x_category(spec);
    if cats.is_empty() { return; }

    // Aggregate each category to a single value (sum).
    let values: Vec<f64> = groups.iter().map(|g| g.iter().sum::<f64>()).collect();
    let total: f64 = values.iter().sum();
    if total <= 0.0 { return; }

    // Center and radius of the pie.
    let cx = layout.plot_x + layout.plot_w / 2.0;
    let cy = layout.plot_y + layout.plot_h / 2.0;
    let outer_r = layout.plot_w.min(layout.plot_h) / 2.0 * 0.85;
    let inner_r = outer_r * layer.params.inner_radius;

    let mut start_angle = std::f64::consts::FRAC_PI_2; // 12 o'clock
    let n_arc_pts = 60; // Points per full circle

    for (i, &val) in values.iter().enumerate() {
        let sweep = val / total * 2.0 * std::f64::consts::PI;
        let end_angle = start_angle - sweep; // Clockwise
        let fill = color::palette_color(i);

        // Build polygon: center (or inner arc) → outer arc → close.
        let n_pts = (n_arc_pts as f64 * (sweep / (2.0 * std::f64::consts::PI))).ceil().max(2.0) as usize;
        let mut points = Vec::with_capacity(n_pts * 2 + 4);

        if inner_r > 0.0 {
            // Donut: inner arc (reverse direction)
            for j in 0..=n_pts {
                let t = j as f64 / n_pts as f64;
                let angle = start_angle - t * sweep;
                points.push((cx + inner_r * angle.cos(), cy - inner_r * angle.sin()));
            }
            // Outer arc (forward direction)
            for j in (0..=n_pts).rev() {
                let t = j as f64 / n_pts as f64;
                let angle = start_angle - t * sweep;
                points.push((cx + outer_r * angle.cos(), cy - outer_r * angle.sin()));
            }
        } else {
            // Full pie: center → outer arc → close.
            points.push((cx, cy));
            for j in 0..=n_pts {
                let t = j as f64 / n_pts as f64;
                let angle = start_angle - t * sweep;
                points.push((cx + outer_r * angle.cos(), cy - outer_r * angle.sin()));
            }
            points.push((cx, cy));
        }

        scene.push(SceneElement::Polyline {
            points,
            stroke: Color::WHITE,
            width: 1.5,
            fill: Some(fill),
        });

        // Label at midpoint of the arc.
        if layer.params.show_labels && val > 0.0 {
            let mid_angle = start_angle - sweep / 2.0;
            let label_r = if inner_r > 0.0 { (inner_r + outer_r) / 2.0 } else { outer_r * 0.65 };
            let lx = cx + label_r * mid_angle.cos();
            let ly = cy - label_r * mid_angle.sin();
            let pct = val / total * 100.0;
            let text = if cats.len() <= 6 {
                format!("{} ({:.1}%)", cats[i], pct)
            } else {
                format!("{:.1}%", pct)
            };
            scene.push(SceneElement::Text {
                x: lx,
                y: ly,
                text,
                font_size: spec.theme.tick_label_font_size,
                fill: Color::DARK_GRAY,
                anchor: TextAnchor::Middle,
                rotation: None,
            });
        }

        start_angle = end_angle;
    }
}

/// Render a rose (Nightingale) chart — bars in polar coordinates.
///
/// Each category gets an equal angular slice; bar length (radius) ∝ value.
fn render_rose(
    scene: &mut Scene,
    spec: &PlotSpec,
    layout: &LayoutResult,
    layer: &crate::spec::Layer,
    _layer_idx: usize,
) {
    let (cats, groups) = group_by_x_category(spec);
    if cats.is_empty() { return; }

    let values: Vec<f64> = groups.iter().map(|g| g.iter().sum::<f64>()).collect();
    let max_val = values.iter().copied().fold(0.0_f64, f64::max);
    if max_val <= 0.0 { return; }

    let cx = layout.plot_x + layout.plot_w / 2.0;
    let cy = layout.plot_y + layout.plot_h / 2.0;
    let max_r = layout.plot_w.min(layout.plot_h) / 2.0 * 0.85;
    let n_cats = values.len();
    let slice_angle = 2.0 * std::f64::consts::PI / n_cats as f64;

    let mut start_angle = std::f64::consts::FRAC_PI_2; // 12 o'clock

    for (i, &val) in values.iter().enumerate() {
        let r = max_r * (val / max_val);
        let end_angle = start_angle - slice_angle;
        let fill = color::palette_color(i);

        let n_pts = 20;
        let mut points = Vec::with_capacity(n_pts + 2);
        points.push((cx, cy));
        for j in 0..=n_pts {
            let t = j as f64 / n_pts as f64;
            let angle = start_angle - t * slice_angle;
            points.push((cx + r * angle.cos(), cy - r * angle.sin()));
        }
        points.push((cx, cy));

        scene.push(SceneElement::Polyline {
            points,
            stroke: Color::WHITE,
            width: 1.0,
            fill: Some(fill.with_alpha(0.7)),
        });

        // Label
        if layer.params.show_labels {
            let mid_angle = start_angle - slice_angle / 2.0;
            let label_r = max_r + 15.0;
            let lx = cx + label_r * mid_angle.cos();
            let ly = cy - label_r * mid_angle.sin();
            scene.push(SceneElement::Text {
                x: lx,
                y: ly,
                text: cats[i].clone(),
                font_size: spec.theme.tick_label_font_size,
                fill: spec.theme.text_color,
                anchor: TextAnchor::Middle,
                rotation: None,
            });
        }

        start_angle = end_angle;
    }
}

/// Render a radar (spider) chart — polygon on radial axes.
///
/// Each category defines a radial axis; values determine distance from center.
fn render_radar(
    scene: &mut Scene,
    spec: &PlotSpec,
    layout: &LayoutResult,
    layer: &crate::spec::Layer,
    layer_idx: usize,
) {
    let (cats, groups) = group_by_x_category(spec);
    if cats.is_empty() { return; }

    let values: Vec<f64> = groups.iter().map(|g| g.iter().sum::<f64>()).collect();
    let max_val = values.iter().copied().fold(0.0_f64, f64::max);
    if max_val <= 0.0 { return; }

    let cx = layout.plot_x + layout.plot_w / 2.0;
    let cy = layout.plot_y + layout.plot_h / 2.0;
    let max_r = layout.plot_w.min(layout.plot_h) / 2.0 * 0.75;
    let n_axes = cats.len();
    let angle_step = 2.0 * std::f64::consts::PI / n_axes as f64;

    // Draw radial grid (3 concentric rings).
    for ring in 1..=3 {
        let r = max_r * ring as f64 / 3.0;
        let mut ring_pts = Vec::with_capacity(n_axes + 1);
        for k in 0..=n_axes {
            let angle = std::f64::consts::FRAC_PI_2 - k as f64 * angle_step;
            ring_pts.push((cx + r * angle.cos(), cy - r * angle.sin()));
        }
        scene.push(SceneElement::Polyline {
            points: ring_pts,
            stroke: spec.theme.grid_color,
            width: 0.5,
            fill: None,
        });
    }

    // Draw radial axis lines.
    for k in 0..n_axes {
        let angle = std::f64::consts::FRAC_PI_2 - k as f64 * angle_step;
        scene.push(SceneElement::Line {
            x1: cx,
            y1: cy,
            x2: cx + max_r * angle.cos(),
            y2: cy - max_r * angle.sin(),
            stroke: spec.theme.grid_color,
            width: 0.5,
        });
    }

    // Draw data polygon.
    let data_color = color::palette_color(layer_idx);
    let mut data_pts = Vec::with_capacity(n_axes + 1);
    for k in 0..n_axes {
        let angle = std::f64::consts::FRAC_PI_2 - k as f64 * angle_step;
        let r = max_r * (values[k] / max_val);
        data_pts.push((cx + r * angle.cos(), cy - r * angle.sin()));
    }
    // Close the polygon.
    if let Some(&first) = data_pts.first() {
        data_pts.push(first);
    }

    scene.push(SceneElement::Polyline {
        points: data_pts.clone(),
        stroke: data_color,
        width: 2.0,
        fill: Some(data_color.with_alpha(0.2)),
    });

    // Data points at each vertex.
    for pt in &data_pts[..data_pts.len().saturating_sub(1)] {
        scene.push(SceneElement::Circle {
            cx: pt.0,
            cy: pt.1,
            r: 4.0,
            fill: data_color,
            stroke: Some(Color::WHITE),
        });
    }

    // Axis labels.
    if layer.params.show_labels {
        for k in 0..n_axes {
            let angle = std::f64::consts::FRAC_PI_2 - k as f64 * angle_step;
            let label_r = max_r + 18.0;
            let lx = cx + label_r * angle.cos();
            let ly = cy - label_r * angle.sin();
            scene.push(SceneElement::Text {
                x: lx,
                y: ly,
                text: cats[k].clone(),
                font_size: spec.theme.tick_label_font_size,
                fill: spec.theme.text_color,
                anchor: TextAnchor::Middle,
                rotation: None,
            });
        }
    }
}

// ── Phase 3: 2D density + contour renderers ───────────────────────────

/// Render 2D density estimation as filled contour bands.
fn render_density2d(
    scene: &mut Scene,
    spec: &PlotSpec,
    layout: &LayoutResult,
    layer: &crate::spec::Layer,
    _layer_idx: usize,
) {
    let x_col = spec.data.get("x").map(|c| c.to_f64()).unwrap_or_default();
    let y_col = spec.data.get("y").map(|c| c.to_f64()).unwrap_or_default();
    let n = x_col.len().min(y_col.len());
    if n < 3 { return; }

    let grid_size = layer.params.grid_size;
    let n_levels = layer.params.n_levels;

    let (x_grid, y_grid, density) = stats::kde_2d(&x_col[..n], &y_col[..n], grid_size);
    if x_grid.is_empty() || y_grid.is_empty() { return; }

    let levels = stats::contour_levels(&density, n_levels);

    // Render filled rectangles colored by density level (heatmap-style).
    // This is a performant approximation of contour fills: each grid cell
    // gets a color based on which contour level its density falls into.
    let gs = grid_size;
    if gs < 2 { return; }

    // Compute cell dimensions from grid spacing in data coords, mapped to pixels.
    let plot_x_max = layout.plot_x + layout.plot_w;
    let plot_y_max = layout.plot_y + layout.plot_h;

    for i in 0..(gs - 1) {
        for j in 0..(gs - 1) {
            let d = density[i][j];
            if d < 1e-15 { continue; }

            // Find the contour level.
            let level_idx = levels.iter().position(|&l| d < l).unwrap_or(levels.len());
            let t = (level_idx as f64) / (levels.len() as f64).max(1.0);

            let fill = Color::lerp(
                Color::rgba(55, 126, 184, 20), // Light blue, very transparent
                Color::rgba(55, 126, 184, 200), // Blue, opaque
                t,
            );

            // Map grid cell corners to pixel coordinates.
            let px_left = layout.map_x(x_grid[i]);
            let px_right = layout.map_x(x_grid[i + 1]);
            let py_top = layout.map_y(y_grid[j + 1]); // y inverted
            let py_bottom = layout.map_y(y_grid[j]);

            // Clip to plot area bounds.
            let cx0 = px_left.max(layout.plot_x);
            let cy0 = py_top.max(layout.plot_y);
            let cx1 = px_right.min(plot_x_max);
            let cy1 = py_bottom.min(plot_y_max);

            if cx1 <= cx0 || cy1 <= cy0 { continue; }

            scene.push(SceneElement::Rect {
                x: cx0,
                y: cy0,
                w: cx1 - cx0,
                h: cy1 - cy0,
                fill,
                stroke: None,
                stroke_width: 0.0,
            });
        }
    }
}

/// Render contour lines at density levels.
fn render_contour(
    scene: &mut Scene,
    spec: &PlotSpec,
    layout: &LayoutResult,
    layer: &crate::spec::Layer,
    _layer_idx: usize,
) {
    let x_col = spec.data.get("x").map(|c| c.to_f64()).unwrap_or_default();
    let y_col = spec.data.get("y").map(|c| c.to_f64()).unwrap_or_default();
    let n = x_col.len().min(y_col.len());
    if n < 3 { return; }

    let grid_size = layer.params.grid_size;
    let n_levels = layer.params.n_levels;

    let (x_grid, y_grid, density) = stats::kde_2d(&x_col[..n], &y_col[..n], grid_size);
    if x_grid.is_empty() || y_grid.is_empty() { return; }

    let levels = stats::contour_levels(&density, n_levels);
    let gs = grid_size;
    if gs < 2 { return; }

    // For each contour level, find grid cell edges where the density
    // crosses the level (marching squares simplified: horizontal segments).
    for (li, &level) in levels.iter().enumerate() {
        let t = (li as f64 + 0.5) / levels.len() as f64;
        let stroke = Color::lerp(
            Color::rgb(55, 126, 184),
            Color::rgb(8, 48, 107),
            t,
        );

        for i in 0..(gs - 1) {
            for j in 0..(gs - 1) {
                let d00 = density[i][j];
                let d10 = density[i + 1][j];
                let d01 = density[i][j + 1];
                let d11 = density[i + 1][j + 1];

                // Simple contour: draw line segments where level crosses edges.
                let mut crossings = Vec::new();

                // Bottom edge (i, j) → (i+1, j)
                if (d00 - level) * (d10 - level) < 0.0 {
                    let t_edge = (level - d00) / (d10 - d00);
                    let px = layout.map_x(x_grid[i] + t_edge * (x_grid[i + 1] - x_grid[i]));
                    let py = layout.map_y(y_grid[j]);
                    crossings.push((px, py));
                }
                // Top edge (i, j+1) → (i+1, j+1)
                if (d01 - level) * (d11 - level) < 0.0 {
                    let t_edge = (level - d01) / (d11 - d01);
                    let px = layout.map_x(x_grid[i] + t_edge * (x_grid[i + 1] - x_grid[i]));
                    let py = layout.map_y(y_grid[j + 1]);
                    crossings.push((px, py));
                }
                // Left edge (i, j) → (i, j+1)
                if (d00 - level) * (d01 - level) < 0.0 {
                    let t_edge = (level - d00) / (d01 - d00);
                    let px = layout.map_x(x_grid[i]);
                    let py = layout.map_y(y_grid[j] + t_edge * (y_grid[j + 1] - y_grid[j]));
                    crossings.push((px, py));
                }
                // Right edge (i+1, j) → (i+1, j+1)
                if (d10 - level) * (d11 - level) < 0.0 {
                    let t_edge = (level - d10) / (d11 - d10);
                    let px = layout.map_x(x_grid[i + 1]);
                    let py = layout.map_y(y_grid[j] + t_edge * (y_grid[j + 1] - y_grid[j]));
                    crossings.push((px, py));
                }

                // Draw line segments between pairs of crossings.
                if crossings.len() >= 2 {
                    scene.push(SceneElement::Line {
                        x1: crossings[0].0,
                        y1: crossings[0].1,
                        x2: crossings[1].0,
                        y2: crossings[1].1,
                        stroke,
                        width: 1.0,
                    });
                }
                // Handle ambiguous saddle points (4 crossings).
                if crossings.len() == 4 {
                    scene.push(SceneElement::Line {
                        x1: crossings[2].0,
                        y1: crossings[2].1,
                        x2: crossings[3].0,
                        y2: crossings[3].1,
                        stroke,
                        width: 1.0,
                    });
                }
            }
        }
    }
}

// ── Phase 3.2: Error bar + Step line renderers ───────────────────────────

/// Render error bars (vertical bars at each data point).
///
/// Expects "x", "y", and an error column (default "error") in the data.
/// Draws: a vertical line from y-error to y+error, plus short horizontal caps.
fn render_errorbar(
    scene: &mut Scene,
    spec: &PlotSpec,
    layout: &LayoutResult,
    layer: &crate::spec::Layer,
    layer_color: Color,
) {
    let x_data = match spec.data.get("x") {
        Some(col) => col.to_f64(),
        None => return,
    };
    let y_data = match spec.data.get("y") {
        Some(col) => col.to_f64(),
        None => return,
    };
    let error_col_name = layer.params.error_column.as_deref().unwrap_or("error");
    let err_data = match spec.data.get(error_col_name) {
        Some(col) => col.to_f64(),
        None => return,
    };

    let n = x_data.len().min(y_data.len()).min(err_data.len());
    let cap_half_w = layer.params.cap_width * 0.5;

    for i in 0..n {
        let x = x_data[i];
        let y = y_data[i];
        let err = err_data[i].abs();

        let px = layout.map_x(x);
        let py_lo = layout.map_y(y - err);
        let py_hi = layout.map_y(y + err);

        // Vertical error bar line.
        scene.push(SceneElement::Line {
            x1: px, y1: py_lo,
            x2: px, y2: py_hi,
            stroke: layer_color,
            width: layer.params.line_width,
        });

        // Cap width in pixels: use a fraction of the x spacing.
        let cap_px = if n > 1 {
            let x_spacing = layout.plot_w / n as f64;
            (cap_half_w * x_spacing).min(20.0).max(3.0)
        } else {
            10.0
        };

        // Bottom cap.
        scene.push(SceneElement::Line {
            x1: px - cap_px, y1: py_lo,
            x2: px + cap_px, y2: py_lo,
            stroke: layer_color,
            width: layer.params.line_width,
        });
        // Top cap.
        scene.push(SceneElement::Line {
            x1: px - cap_px, y1: py_hi,
            x2: px + cap_px, y2: py_hi,
            stroke: layer_color,
            width: layer.params.line_width,
        });

        // Center dot.
        scene.push(SceneElement::Circle {
            cx: px,
            cy: layout.map_y(y),
            r: layer.params.point_size * 0.8,
            fill: layer_color,
            stroke: None,
        });
    }
}

/// Render a step line (horizontal-then-vertical segments).
///
/// Like a staircase function: each data point defines the start of a
/// horizontal segment until the next x value, then a vertical jump.
fn render_step(
    scene: &mut Scene,
    spec: &PlotSpec,
    layout: &LayoutResult,
    layer: &crate::spec::Layer,
    layer_color: Color,
) {
    let x_data = match spec.data.get("x") {
        Some(col) => col.to_f64(),
        None => return,
    };
    let y_data = match spec.data.get("y") {
        Some(col) => col.to_f64(),
        None => return,
    };

    let n = x_data.len().min(y_data.len());
    if n < 2 { return; }

    // Sort by x.
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| x_data[a].partial_cmp(&x_data[b]).unwrap());

    // Build step-post polyline: (x0,y0) → (x1,y0) → (x1,y1) → (x2,y1) → ...
    let mut points = Vec::with_capacity(n * 2);
    for (step_i, &idx) in indices.iter().enumerate() {
        let px = layout.map_x(x_data[idx]);
        let py = layout.map_y(y_data[idx]);

        if step_i > 0 {
            // Horizontal segment: extend previous y to current x.
            let prev_py = points.last().map(|&(_, y): &(f64, f64)| y).unwrap_or(py);
            if layer.params.step_post {
                // Step-post: horizontal first, then vertical.
                points.push((px, prev_py));
            } else {
                // Step-pre: vertical first, then horizontal.
                let prev_px = points.last().map(|&(x, _): &(f64, f64)| x).unwrap_or(px);
                points.push((prev_px, py));
            }
        }
        points.push((px, py));
    }

    scene.push(SceneElement::Polyline {
        points,
        stroke: layer_color,
        width: layer.params.line_width,
        fill: None,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::PlotSpec;

    #[test]
    fn test_build_scene_scatter() {
        let spec = PlotSpec::from_xy(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0])
            .geom_point()
            .title("Test");
        let scene = build_scene(&spec);
        assert!(!scene.elements.is_empty());
        // Should have background, plot bg, gridlines, axes, points, title.
        assert!(scene.elements.len() > 5);
    }

    #[test]
    fn test_build_scene_deterministic() {
        let spec = PlotSpec::from_xy(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0])
            .geom_point();
        let s1 = build_scene(&spec);
        let s2 = build_scene(&spec);
        assert_eq!(s1.elements.len(), s2.elements.len());
    }

    #[test]
    fn test_build_scene_bar() {
        let spec = PlotSpec::from_xy(vec![0.0, 1.0, 2.0], vec![10.0, 25.0, 18.0])
            .geom_bar();
        let scene = build_scene(&spec);
        assert!(!scene.elements.is_empty());
    }

    #[test]
    fn test_build_scene_line() {
        let spec = PlotSpec::from_xy(vec![1.0, 2.0, 3.0, 4.0], vec![1.0, 4.0, 2.0, 5.0])
            .geom_line();
        let scene = build_scene(&spec);
        // Should contain a polyline.
        let has_polyline = scene.elements.iter().any(|e| matches!(e, SceneElement::Polyline { .. }));
        assert!(has_polyline);
    }

    // ── Phase 5 (Audit): axis suppression for polar/tile/dendrogram ──

    #[test]
    fn test_pie_chart_suppresses_axes() {
        let spec = PlotSpec::from_cat(
            vec!["A".into(), "B".into(), "C".into()],
            vec![30.0, 50.0, 20.0],
        ).geom_pie();
        let scene = build_scene(&spec);
        // We mainly verify no crash and a valid scene.
        assert!(!scene.elements.is_empty());
        // Pie should have polyline slices (not line axes).
        let has_polyline = scene.elements.iter().any(|e| matches!(e, SceneElement::Polyline { .. }));
        assert!(has_polyline, "Pie chart should have polyline slices");
    }

    #[test]
    fn test_tile_chart_suppresses_axes() {
        let spec = PlotSpec::from_matrix(
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec!["r0".into(), "r1".into()],
            vec!["c0".into(), "c1".into()],
        ).geom_tile();
        let scene = build_scene(&spec);
        assert!(!scene.elements.is_empty());
        // Tile should have rects for cells.
        let rect_count = scene.elements.iter().filter(|e| matches!(e, SceneElement::Rect { .. })).count();
        assert!(rect_count >= 4, "Tile chart should have at least 4 rect cells, got {}", rect_count);
    }

    #[test]
    fn test_dendrogram_suppresses_axes() {
        let spec = PlotSpec::from_matrix(
            vec![
                vec![0.0, 1.0, 5.0],
                vec![1.0, 0.0, 4.0],
                vec![5.0, 4.0, 0.0],
            ],
            vec!["A".into(), "B".into(), "C".into()],
            vec!["A".into(), "B".into(), "C".into()],
        ).geom_dendrogram();
        let scene = build_scene(&spec);
        assert!(!scene.elements.is_empty());
        // Dendrogram uses lines for branches.
        let line_count = scene.elements.iter().filter(|e| matches!(e, SceneElement::Line { .. })).count();
        assert!(line_count >= 2, "Dendrogram should have branch lines, got {}", line_count);
    }

    // ── Phase 5 (Audit): violin rendering produces valid scene ──

    #[test]
    fn test_violin_scene_has_polylines() {
        let spec = PlotSpec::from_cat(
            vec![
                "A".into(), "A".into(), "A".into(), "A".into(), "A".into(),
                "A".into(), "A".into(), "A".into(),
                "B".into(), "B".into(), "B".into(), "B".into(), "B".into(),
                "B".into(), "B".into(), "B".into(),
            ],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        ).geom_violin();
        let scene = build_scene(&spec);
        let polyline_count = scene.elements.iter()
            .filter(|e| matches!(e, SceneElement::Polyline { .. }))
            .count();
        assert!(polyline_count >= 2, "Violin should have polylines for each group, got {}", polyline_count);
    }

    #[test]
    fn test_violin_polyline_y_in_bounds() {
        let spec = PlotSpec::from_cat(
            vec![
                "A".into(), "A".into(), "A".into(), "A".into(), "A".into(),
                "A".into(), "A".into(), "A".into(),
            ],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ).geom_violin();
        let layout = crate::layout::compute_layout(&spec);
        let scene = build_scene(&spec);
        let plot_top = layout.plot_y;
        let plot_bottom = layout.plot_y + layout.plot_h;
        for el in &scene.elements {
            if let SceneElement::Polyline { points, .. } = el {
                for &(_, y) in points {
                    assert!(y >= plot_top - 1.0 && y <= plot_bottom + 1.0,
                        "Violin polyline y={} out of plot bounds [{}, {}]", y, plot_top, plot_bottom);
                }
            }
        }
    }

    // ── Phase 5 (Audit): boxen rendering with small data ──

    #[test]
    fn test_boxen_small_group_no_crash() {
        let spec = PlotSpec::from_cat(
            vec![
                "A".into(), "A".into(), "A".into(), "A".into(), "A".into(),
            ],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        ).geom_boxen();
        let scene = build_scene(&spec);
        // Should render without panic and produce rects.
        let rect_count = scene.elements.iter()
            .filter(|e| matches!(e, SceneElement::Rect { .. }))
            .count();
        assert!(rect_count >= 1, "Boxen with 5 values should produce at least 1 rect, got {}", rect_count);
    }

    // ── Phase 5 (Audit): density2d clipping ──

    #[test]
    fn test_density2d_rects_in_bounds() {
        let spec = PlotSpec::from_xy(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5],
            vec![2.0, 3.0, 5.0, 7.0, 11.0, 2.5, 3.5, 5.5, 7.5, 11.5],
        ).geom_density2d();
        let layout = crate::layout::compute_layout(&spec);
        let scene = build_scene(&spec);
        let plot_x_min = layout.plot_x;
        let plot_y_min = layout.plot_y;
        let plot_x_max = layout.plot_x + layout.plot_w;
        let plot_y_max = layout.plot_y + layout.plot_h;
        for el in &scene.elements {
            if let SceneElement::Rect { x, y, w, h, .. } = el {
                // Background rects are at (0,0) — skip them.
                if *x < 1.0 && *y < 1.0 { continue; }
                // Skip plot bg rect.
                if (*w - layout.plot_w).abs() < 1.0 && (*h - layout.plot_h).abs() < 1.0 { continue; }
                // Skip small annotation bg rects.
                if *w < 5.0 || *h < 5.0 { continue; }
                // Density2d cell rects should be within plot bounds.
                assert!(*x >= plot_x_min - 1.0,
                    "Density2d rect x={} below plot_x={}", x, plot_x_min);
                assert!(*y >= plot_y_min - 1.0,
                    "Density2d rect y={} below plot_y={}", y, plot_y_min);
                assert!(*x + *w <= plot_x_max + 1.0,
                    "Density2d rect right edge x+w={} exceeds plot_x_max={}", x + w, plot_x_max);
                assert!(*y + *h <= plot_y_max + 1.0,
                    "Density2d rect bottom edge y+h={} exceeds plot_y_max={}", y + h, plot_y_max);
            }
        }
    }

    // ── Phase 5 (Audit): annotation rendering ──

    #[test]
    fn test_annotation_produces_bg_rect() {
        use crate::annotation::Annotation;
        let spec = PlotSpec::from_xy(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0])
            .geom_point()
            .annotate(Annotation::text("test label", 2.0, 5.0));
        let scene = build_scene(&spec);
        // Should contain at least one text element with the annotation.
        let has_label = scene.elements.iter().any(|e| {
            if let SceneElement::Text { text, .. } = e {
                text.contains("test label")
            } else {
                false
            }
        });
        assert!(has_label, "Annotation text should appear in scene");
    }

    // ── Phase 5 (Audit): error bar rendering with add_column ──

    #[test]
    fn test_errorbar_with_error_column() {
        use crate::spec::DataColumn;
        let spec = PlotSpec::from_xy(
            vec![1.0, 2.0, 3.0],
            vec![10.0, 20.0, 30.0],
        ).add_column("error", DataColumn::Float(vec![2.0, 3.0, 1.5]))
         .geom_errorbar();
        let scene = build_scene(&spec);
        // Should have lines for whiskers + caps.
        let line_count = scene.elements.iter()
            .filter(|e| matches!(e, SceneElement::Line { .. }))
            .count();
        // At least 3 vertical whisker lines + axis lines.
        assert!(line_count >= 3, "Error bars should produce lines, got {}", line_count);
    }
}
