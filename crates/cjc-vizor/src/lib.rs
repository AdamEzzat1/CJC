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

/// Semantic annotations (text, regression, CI, p-value, etc.).
pub mod annotation;
/// BMP raster export (uncompressed, zero-dependency).
pub mod bmp;
/// RGBA color types, hex parsing, and categorical palette.
pub mod color;
/// CJC language dispatch: maps builtin/method calls to Vizor API.
pub mod dispatch;
/// IDE/LSP documentation metadata for builtins and methods.
pub mod docs;
/// Faceting: multi-panel grid layouts split by a grouping variable.
pub mod facet;
/// Layout engine: coordinate mapping, tick generation, axis computation.
pub mod layout;
/// Legend rendering for multi-layer plots.
pub mod legend;
/// PNG export (behind the `png` feature flag).
pub mod png_export;
/// Shared rasterizer: scene to pixel buffer (used by BMP and PNG).
pub mod raster;
/// Render pipeline: PlotSpec to Scene conversion.
pub mod render;
/// Scene graph: flat list of positioned visual primitives.
pub mod scene;
/// Plot specification: data, layers, scales, labels, and themes.
pub mod spec;
/// Statistical computations: KDE, regression, clustering, quantiles.
pub mod stats;
/// SVG serializer: Scene to SVG string.
pub mod svg;
/// Text measurement and tick label formatting.
pub mod text;
/// Theme definitions: margins, colors, font sizes, line widths.
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
