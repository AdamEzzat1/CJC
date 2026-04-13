//! Scene graph — flat list of positioned, styled primitives.
//!
//! The scene graph is the intermediate representation between the plot spec
//! and the renderer. It contains only positioned visual elements.

use crate::color::Color;

/// A scene: a flat list of positioned primitives at a given resolution.
#[derive(Debug, Clone)]
pub struct Scene {
    /// Canvas width in pixels.
    pub width: u32,
    /// Canvas height in pixels.
    pub height: u32,
    /// Ordered list of visual elements (painter's algorithm: back to front).
    pub elements: Vec<SceneElement>,
}

impl Scene {
    /// Create a new empty scene with the given pixel dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Scene {
            width,
            height,
            elements: Vec::new(),
        }
    }

    /// Append a visual element to the scene.
    pub fn push(&mut self, elem: SceneElement) {
        self.elements.push(elem);
    }
}

/// A positioned visual element in the scene.
#[derive(Debug, Clone)]
pub enum SceneElement {
    /// An axis-aligned rectangle with optional stroke.
    Rect {
        x: f64,
        y: f64,
        w: f64,
        h: f64,
        fill: Color,
        stroke: Option<Color>,
        stroke_width: f64,
    },
    /// A circle defined by center and radius.
    Circle {
        cx: f64,
        cy: f64,
        r: f64,
        fill: Color,
        stroke: Option<Color>,
    },
    /// A straight line segment between two endpoints.
    Line {
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        stroke: Color,
        width: f64,
    },
    /// A connected sequence of line segments with optional fill.
    Polyline {
        points: Vec<(f64, f64)>,
        stroke: Color,
        width: f64,
        fill: Option<Color>,
    },
    /// A text label placed at a given position with optional rotation.
    Text {
        x: f64,
        y: f64,
        text: String,
        font_size: f64,
        fill: Color,
        anchor: TextAnchor,
        rotation: Option<f64>,
    },
}

/// Text anchor for horizontal alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextAnchor {
    /// Align text so the string begins at the anchor point.
    Start,
    /// Align text so the string is centered on the anchor point.
    Middle,
    /// Align text so the string ends at the anchor point.
    End,
}

impl TextAnchor {
    /// Return the SVG `text-anchor` attribute value.
    pub fn as_svg(&self) -> &'static str {
        match self {
            TextAnchor::Start => "start",
            TextAnchor::Middle => "middle",
            TextAnchor::End => "end",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scene_push() {
        let mut scene = Scene::new(800, 600);
        scene.push(SceneElement::Rect {
            x: 0.0, y: 0.0, w: 100.0, h: 100.0,
            fill: Color::WHITE,
            stroke: None,
            stroke_width: 0.0,
        });
        assert_eq!(scene.elements.len(), 1);
    }
}
