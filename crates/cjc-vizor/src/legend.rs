//! Legend rendering — generates legend entries for multi-layer and categorical plots.
//!
//! The legend is placed inside the plot area (top-right by default) so it
//! doesn't require additional margin negotiation. Pure scene-element output.

use crate::color::{self, Color};
use crate::scene::{Scene, SceneElement, TextAnchor};
use crate::spec::PlotSpec;

/// A single legend entry.
#[derive(Debug, Clone)]
pub struct LegendEntry {
    pub label: String,
    pub color: Color,
    pub kind: LegendSymbol,
}

/// Symbol shape drawn beside the legend text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegendSymbol {
    /// Filled circle (for point / scatter layers).
    Circle,
    /// Short horizontal line (for line / density layers).
    Line,
    /// Small filled rectangle (for bar / histogram / area layers).
    Rect,
}

/// Collect legend entries from a plot spec.
///
/// Returns entries only when there are 2+ layers (single-layer plots skip legend).
pub fn collect_legend_entries(spec: &PlotSpec) -> Vec<LegendEntry> {
    if spec.layers.len() < 2 {
        return vec![];
    }

    let mut entries = Vec::new();
    for (i, layer) in spec.layers.iter().enumerate() {
        let color = layer
            .params
            .color_override
            .unwrap_or_else(|| color::palette_color(i));

        let label = legend_label_for_geom(&layer.geom, i);

        let kind = match layer.geom {
            crate::spec::Geom::Point
            | crate::spec::Geom::Strip
            | crate::spec::Geom::Swarm => LegendSymbol::Circle,

            crate::spec::Geom::Line
            | crate::spec::Geom::Density
            | crate::spec::Geom::Ecdf
            | crate::spec::Geom::RegressionLine
            | crate::spec::Geom::Contour => LegendSymbol::Line,

            _ => LegendSymbol::Rect,
        };

        entries.push(LegendEntry { label, color, kind });
    }
    entries
}

/// Generate a human-readable label for a geom layer.
fn legend_label_for_geom(geom: &crate::spec::Geom, _index: usize) -> String {
    match geom {
        crate::spec::Geom::Point => "Points".to_string(),
        crate::spec::Geom::Line => "Line".to_string(),
        crate::spec::Geom::Bar => "Bar".to_string(),
        crate::spec::Geom::Histogram => "Histogram".to_string(),
        crate::spec::Geom::Density => "Density".to_string(),
        crate::spec::Geom::Area => "Area".to_string(),
        crate::spec::Geom::Rug => "Rug".to_string(),
        crate::spec::Geom::Ecdf => "ECDF".to_string(),
        crate::spec::Geom::Box => "Box".to_string(),
        crate::spec::Geom::Violin => "Violin".to_string(),
        crate::spec::Geom::Strip => "Strip".to_string(),
        crate::spec::Geom::Swarm => "Swarm".to_string(),
        crate::spec::Geom::Boxen => "Boxen".to_string(),
        crate::spec::Geom::Tile => "Tile".to_string(),
        crate::spec::Geom::RegressionLine => "Regression".to_string(),
        crate::spec::Geom::Residual => "Residuals".to_string(),
        crate::spec::Geom::Dendrogram => "Dendrogram".to_string(),
        crate::spec::Geom::Pie => "Pie".to_string(),
        crate::spec::Geom::Rose => "Rose".to_string(),
        crate::spec::Geom::Radar => "Radar".to_string(),
        crate::spec::Geom::Density2d => "Density 2D".to_string(),
        crate::spec::Geom::Contour => "Contour".to_string(),
        crate::spec::Geom::ErrorBar => "Error Bars".to_string(),
        crate::spec::Geom::Step => "Step".to_string(),
    }
}

/// Render legend entries into the scene.
///
/// Places the legend in the top-right corner of the plot area with a
/// semi-transparent background box, symbol, and label text for each entry.
pub fn render_legend(
    scene: &mut Scene,
    entries: &[LegendEntry],
    plot_x: f64,
    plot_y: f64,
    plot_w: f64,
    _plot_h: f64,
    font_size: f64,
    text_color: Color,
) {
    if entries.is_empty() {
        return;
    }

    let row_height = font_size * 1.6;
    let symbol_size = font_size * 0.8;
    let padding = 8.0;
    let symbol_text_gap = 6.0;

    // Estimate legend box width from longest label.
    let max_label_len = entries.iter().map(|e| e.label.len()).max().unwrap_or(5);
    let legend_w = padding * 2.0 + symbol_size + symbol_text_gap + (max_label_len as f64 * font_size * 0.6);
    let legend_h = padding * 2.0 + entries.len() as f64 * row_height;

    // Position: top-right inside plot area, with a small margin.
    let lx = plot_x + plot_w - legend_w - 10.0;
    let ly = plot_y + 10.0;

    // Background box.
    scene.push(SceneElement::Rect {
        x: lx,
        y: ly,
        w: legend_w,
        h: legend_h,
        fill: Color::rgba(255, 255, 255, 230),
        stroke: Some(Color::LIGHT_GRAY),
        stroke_width: 0.5,
    });

    for (i, entry) in entries.iter().enumerate() {
        let row_y = ly + padding + i as f64 * row_height + row_height / 2.0;
        let sym_x = lx + padding;
        let text_x = sym_x + symbol_size + symbol_text_gap;

        // Draw symbol.
        match entry.kind {
            LegendSymbol::Circle => {
                scene.push(SceneElement::Circle {
                    cx: sym_x + symbol_size / 2.0,
                    cy: row_y,
                    r: symbol_size / 2.0,
                    fill: entry.color,
                    stroke: None,
                });
            }
            LegendSymbol::Line => {
                scene.push(SceneElement::Line {
                    x1: sym_x,
                    y1: row_y,
                    x2: sym_x + symbol_size,
                    y2: row_y,
                    stroke: entry.color,
                    width: 2.0,
                });
            }
            LegendSymbol::Rect => {
                scene.push(SceneElement::Rect {
                    x: sym_x,
                    y: row_y - symbol_size / 2.0,
                    w: symbol_size,
                    h: symbol_size,
                    fill: entry.color,
                    stroke: None,
                    stroke_width: 0.0,
                });
            }
        }

        // Label text.
        scene.push(SceneElement::Text {
            x: text_x,
            y: row_y + font_size * 0.35,
            text: entry.label.clone(),
            font_size,
            fill: text_color,
            anchor: TextAnchor::Start,
            rotation: None,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::PlotSpec;

    #[test]
    fn test_collect_empty_for_single_layer() {
        let spec = PlotSpec::from_xy(vec![1.0], vec![2.0]).geom_point();
        let entries = collect_legend_entries(&spec);
        assert!(entries.is_empty());
    }

    #[test]
    fn test_collect_entries_for_multi_layer() {
        let spec = PlotSpec::from_xy(vec![1.0, 2.0], vec![3.0, 4.0])
            .geom_point()
            .geom_line();
        let entries = collect_legend_entries(&spec);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].label, "Points");
        assert_eq!(entries[0].kind, LegendSymbol::Circle);
        assert_eq!(entries[1].label, "Line");
        assert_eq!(entries[1].kind, LegendSymbol::Line);
    }

    #[test]
    fn test_render_legend_no_panic() {
        let mut scene = Scene::new(800, 600);
        let entries = vec![
            LegendEntry { label: "A".into(), color: Color::rgb(255, 0, 0), kind: LegendSymbol::Circle },
            LegendEntry { label: "B".into(), color: Color::rgb(0, 0, 255), kind: LegendSymbol::Line },
        ];
        render_legend(&mut scene, &entries, 70.0, 40.0, 710.0, 500.0, 10.0, Color::BLACK);
        // Background + 2 symbols + 2 texts = 5 elements.
        assert_eq!(scene.elements.len(), 5);
    }
}
