//! Vizor — grammar-of-graphics data visualization library for CJC.
//!
//! Vizor is the first CJC library, validating the library architecture.
//! It provides a declarative, grammar-of-graphics-style API for creating
//! plots with deterministic rendering to SVG and BMP (and optionally PNG).
//!
//! # Architecture
//!
//! ```text
//! PlotSpec → Layout → Scene → SVG / BMP / PNG
//!            (coord    (flat     (serializer)
//!             mapping)  primitives)
//! ```
//!
//! # Usage from CJC
//!
//! ```cjc
//! import vizor
//!
//! let p = vizor_plot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
//!     .geom_point()
//!     .title("My Plot")
//!     .xlab("X").ylab("Y")
//!
//! p.save("plot.svg")
//! ```

pub mod annotation;
pub mod bmp;
pub mod color;
pub mod dispatch;
pub mod docs;
pub mod facet;
pub mod layout;
pub mod legend;
pub mod png_export;
pub mod raster;
pub mod render;
pub mod scene;
pub mod spec;
pub mod stats;
pub mod svg;
pub mod text;
pub mod theme;

// Re-export key types for convenience.
pub use annotation::Annotation;
pub use color::Color;
pub use render::build_scene;
pub use scene::Scene;
pub use spec::PlotSpec;
pub use svg::render_svg;
pub use bmp::render_bmp;
pub use theme::Theme;

use crate::docs::{VIZOR_BUILTIN_NAMES, VIZOR_METHOD_NAMES};

/// CjcLibrary implementation for Vizor.
pub struct VizorLibrary;

impl cjc_runtime::lib_registry::CjcLibrary for VizorLibrary {
    fn name(&self) -> &'static str { "vizor" }
    fn version(&self) -> &'static str { "0.1.0" }
    fn builtin_names(&self) -> &[&'static str] { VIZOR_BUILTIN_NAMES }
    fn method_names(&self) -> &[&'static str] { VIZOR_METHOD_NAMES }
    fn value_type_names(&self) -> &[&'static str] { &["VizorPlot"] }
}
