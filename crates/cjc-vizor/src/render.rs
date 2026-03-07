//! Render pipeline — converts a PlotSpec into a Scene.
//!
//! Pipeline: PlotSpec → Layout → Scene elements (deterministic).
//! Order: background → axes → gridlines → geom layers → annotations → labels.

use crate::annotation::{Annotation, Position, format_pvalue, format_r_squared, format_ci};
use crate::color::{self, Color};
use crate::layout::{compute_layout, histogram_counts, LayoutResult};
use crate::scene::{Scene, SceneElement, TextAnchor};
use crate::spec::{Geom, PlotSpec};

/// Build a scene from a plot spec. Fully deterministic.
pub fn build_scene(spec: &PlotSpec) -> Scene {
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

    // 3. Gridlines.
    render_gridlines(&mut scene, &layout, theme);

    // 4. Axes.
    render_axes(&mut scene, &layout, theme);

    // 5. Geom layers (in order).
    for (layer_idx, layer) in spec.layers.iter().enumerate() {
        let layer_color = layer.params.color_override.unwrap_or_else(|| color::palette_color(layer_idx));
        match layer.geom {
            Geom::Point => render_points(&mut scene, spec, &layout, layer_idx, layer_color),
            Geom::Line => render_line(&mut scene, spec, &layout, layer_idx, layer_color),
            Geom::Bar => render_bars(&mut scene, spec, &layout, layer_idx, layer_color),
            Geom::Histogram => render_histogram(&mut scene, spec, &layout, layer_idx, layer_color),
        }
    }

    // 6. Annotations.
    render_annotations(&mut scene, spec, &layout);

    // 7. Title and axis labels.
    render_labels(&mut scene, spec, &layout);

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

fn render_annotations(scene: &mut Scene, spec: &PlotSpec, layout: &LayoutResult) {
    let theme = &spec.theme;

    for ann in &spec.annotations {
        match ann {
            Annotation::Text { text, position, font_size } => {
                let (x, y) = resolve_position(position, layout, theme);
                scene.push(SceneElement::Text {
                    x, y, text: text.clone(),
                    font_size: font_size.unwrap_or(theme.tick_label_font_size),
                    fill: theme.text_color,
                    anchor: TextAnchor::Start,
                    rotation: None,
                });
            }
            Annotation::Note { text, position } => {
                let (x, y) = resolve_position(position, layout, theme);
                scene.push(SceneElement::Text {
                    x, y, text: text.clone(),
                    font_size: theme.tick_label_font_size * 0.9,
                    fill: Color::GRAY,
                    anchor: TextAnchor::Start,
                    rotation: None,
                });
            }
            Annotation::RegressionSummary { equation, r_squared, position } => {
                let (x, y) = resolve_position(position, layout, theme);
                let line1 = equation.clone();
                let line2 = format_r_squared(*r_squared);
                scene.push(SceneElement::Text {
                    x, y, text: line1,
                    font_size: theme.tick_label_font_size,
                    fill: theme.text_color,
                    anchor: TextAnchor::End,
                    rotation: None,
                });
                scene.push(SceneElement::Text {
                    x, y: y + theme.tick_label_font_size + 4.0,
                    text: line2,
                    font_size: theme.tick_label_font_size,
                    fill: theme.text_color,
                    anchor: TextAnchor::End,
                    rotation: None,
                });
            }
            Annotation::ConfidenceInterval { level, lower, upper, position } => {
                let (x, y) = resolve_position(position, layout, theme);
                let text = format_ci(*level, *lower, *upper);
                scene.push(SceneElement::Text {
                    x, y, text,
                    font_size: theme.tick_label_font_size,
                    fill: theme.text_color,
                    anchor: TextAnchor::End,
                    rotation: None,
                });
            }
            Annotation::PValue { value, position, .. } => {
                let (x, y) = resolve_position(position, layout, theme);
                let text = format_pvalue(*value);
                scene.push(SceneElement::Text {
                    x, y, text,
                    font_size: theme.tick_label_font_size,
                    fill: theme.text_color,
                    anchor: TextAnchor::End,
                    rotation: None,
                });
            }
            Annotation::ModelMetrics { metrics, position } => {
                let (x, mut y) = resolve_position(position, layout, theme);
                for (name, val) in metrics {
                    let text = format!("{}: {:.4}", name, val);
                    scene.push(SceneElement::Text {
                        x, y, text,
                        font_size: theme.tick_label_font_size,
                        fill: theme.text_color,
                        anchor: TextAnchor::End,
                        rotation: None,
                    });
                    y += theme.tick_label_font_size + 3.0;
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
                    scene.push(SceneElement::Text {
                        x: px + 3.0,
                        y: layout.plot_y + theme.tick_label_font_size,
                        text: label.clone(),
                        font_size: theme.tick_label_font_size * 0.85,
                        fill: Color::GRAY,
                        anchor: TextAnchor::Start,
                        rotation: None,
                    });
                }
            }
            Annotation::DataNote { text, position } => {
                let (x, y) = resolve_position(position, layout, theme);
                scene.push(SceneElement::Text {
                    x, y, text: text.clone(),
                    font_size: theme.tick_label_font_size * 0.8,
                    fill: Color::GRAY,
                    anchor: TextAnchor::Start,
                    rotation: None,
                });
            }
            Annotation::InlineLabel { text, x: data_x, y: data_y } => {
                let px = layout.map_x(*data_x);
                let py = layout.map_y(*data_y);
                scene.push(SceneElement::Text {
                    x: px + 5.0,
                    y: py - 5.0,
                    text: text.clone(),
                    font_size: theme.tick_label_font_size * 0.9,
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
                scene.push(SceneElement::Text {
                    x: lx, y: ly - 3.0,
                    text: text.clone(),
                    font_size: theme.tick_label_font_size,
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
        scene.push(SceneElement::Text {
            x: layout.plot_x + layout.plot_w / 2.0,
            y: layout.plot_y - 10.0,
            text: title.clone(),
            font_size: theme.title_font_size,
            fill: theme.text_color,
            anchor: TextAnchor::Middle,
            rotation: None,
        });
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
}
