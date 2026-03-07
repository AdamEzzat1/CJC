//! SVG serializer — converts a Scene to SVG string.
//!
//! Pure string building, zero external dependencies.
//! Uses fixed float precision ({:.2}) for deterministic output.

use crate::scene::{Scene, SceneElement};

/// Render a scene as an SVG string.
pub fn render_svg(scene: &Scene) -> String {
    let mut buf = String::with_capacity(8192);

    buf.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">\n",
        scene.width, scene.height, scene.width, scene.height
    ));

    for elem in &scene.elements {
        match elem {
            SceneElement::Rect { x, y, w, h, fill, stroke, stroke_width } => {
                buf.push_str(&format!(
                    "  <rect x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{:.2}\" fill=\"{}\"",
                    x, y, w, h, fill.to_svg()
                ));
                if let Some(s) = stroke {
                    buf.push_str(&format!(" stroke=\"{}\" stroke-width=\"{:.2}\"", s.to_svg(), stroke_width));
                }
                buf.push_str("/>\n");
            }
            SceneElement::Circle { cx, cy, r, fill, stroke } => {
                buf.push_str(&format!(
                    "  <circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"{:.2}\" fill=\"{}\"",
                    cx, cy, r, fill.to_svg()
                ));
                if let Some(s) = stroke {
                    buf.push_str(&format!(" stroke=\"{}\"", s.to_svg()));
                }
                buf.push_str("/>\n");
            }
            SceneElement::Line { x1, y1, x2, y2, stroke, width } => {
                buf.push_str(&format!(
                    "  <line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"{}\" stroke-width=\"{:.2}\"/>\n",
                    x1, y1, x2, y2, stroke.to_svg(), width
                ));
            }
            SceneElement::Polyline { points, stroke, width, fill } => {
                buf.push_str("  <polyline points=\"");
                for (i, (px, py)) in points.iter().enumerate() {
                    if i > 0 { buf.push(' '); }
                    buf.push_str(&format!("{:.2},{:.2}", px, py));
                }
                buf.push_str(&format!(
                    "\" stroke=\"{}\" stroke-width=\"{:.2}\" fill=\"{}\"",
                    stroke.to_svg(),
                    width,
                    fill.map(|c| c.to_svg()).unwrap_or_else(|| "none".to_string()),
                ));
                buf.push_str("/>\n");
            }
            SceneElement::Text { x, y, text, font_size, fill, anchor, rotation } => {
                buf.push_str(&format!(
                    "  <text x=\"{:.2}\" y=\"{:.2}\" font-size=\"{:.1}\" fill=\"{}\" text-anchor=\"{}\" font-family=\"sans-serif\"",
                    x, y, font_size, fill.to_svg(), anchor.as_svg()
                ));
                if let Some(angle) = rotation {
                    buf.push_str(&format!(" transform=\"rotate({:.1},{:.2},{:.2})\"", angle, x, y));
                }
                buf.push('>');
                // Escape XML special characters.
                for ch in text.chars() {
                    match ch {
                        '<' => buf.push_str("&lt;"),
                        '>' => buf.push_str("&gt;"),
                        '&' => buf.push_str("&amp;"),
                        '"' => buf.push_str("&quot;"),
                        _ => buf.push(ch),
                    }
                }
                buf.push_str("</text>\n");
            }
        }
    }

    buf.push_str("</svg>\n");
    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Color;
    use crate::scene::{Scene, SceneElement, TextAnchor};

    #[test]
    fn test_render_svg_basic() {
        let mut scene = Scene::new(200, 100);
        scene.push(SceneElement::Rect {
            x: 0.0, y: 0.0, w: 200.0, h: 100.0,
            fill: Color::WHITE,
            stroke: None,
            stroke_width: 0.0,
        });
        scene.push(SceneElement::Circle {
            cx: 50.0, cy: 50.0, r: 10.0,
            fill: Color::rgb(255, 0, 0),
            stroke: None,
        });
        let svg = render_svg(&scene);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("<rect"));
        assert!(svg.contains("<circle"));
    }

    #[test]
    fn test_render_svg_deterministic() {
        let mut scene = Scene::new(100, 100);
        scene.push(SceneElement::Line {
            x1: 0.0, y1: 0.0, x2: 100.0, y2: 100.0,
            stroke: Color::BLACK,
            width: 1.0,
        });
        let s1 = render_svg(&scene);
        let s2 = render_svg(&scene);
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_render_svg_text_escaping() {
        let mut scene = Scene::new(100, 100);
        scene.push(SceneElement::Text {
            x: 10.0, y: 10.0,
            text: "a < b & c > d".to_string(),
            font_size: 12.0,
            fill: Color::BLACK,
            anchor: TextAnchor::Start,
            rotation: None,
        });
        let svg = render_svg(&scene);
        assert!(svg.contains("&lt;"));
        assert!(svg.contains("&amp;"));
        assert!(svg.contains("&gt;"));
    }

    #[test]
    fn test_render_svg_polyline() {
        let mut scene = Scene::new(100, 100);
        scene.push(SceneElement::Polyline {
            points: vec![(0.0, 0.0), (50.0, 50.0), (100.0, 0.0)],
            stroke: Color::rgb(0, 0, 255),
            width: 2.0,
            fill: None,
        });
        let svg = render_svg(&scene);
        assert!(svg.contains("<polyline"));
        assert!(svg.contains("fill=\"none\""));
    }
}
