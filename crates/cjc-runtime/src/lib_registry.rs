//! Library Registry — informational trait for CJC library self-description.
//!
//! CJC libraries implement this trait to declare their builtins, methods,
//! and value types. This enables:
//! - `:libs` REPL command for library discovery
//! - LSP hover info for library-provided symbols
//! - Future automated import validation
//!
//! Actual dispatch remains in library-specific dispatch modules
//! (the proven match-arm pattern). This trait is for metadata only.

/// Trait that CJC library crates implement for self-description.
///
/// # Example
///
/// ```ignore
/// pub struct VizorLibrary;
///
/// impl CjcLibrary for VizorLibrary {
///     fn name(&self) -> &'static str { "vizor" }
///     fn version(&self) -> &'static str { "0.1.0" }
///     fn builtin_names(&self) -> &[&'static str] { &["vizor_plot"] }
///     fn method_names(&self) -> &[&'static str] { &["geom_point", "to_svg", "save"] }
///     fn value_type_names(&self) -> &[&'static str] { &["VizorPlot"] }
/// }
/// ```
pub trait CjcLibrary {
    /// Library name (e.g., "vizor", "data").
    fn name(&self) -> &'static str;

    /// Library version string.
    fn version(&self) -> &'static str;

    /// List of free-function builtin names this library provides.
    fn builtin_names(&self) -> &[&'static str];

    /// List of method names this library handles on its types.
    fn method_names(&self) -> &[&'static str];

    /// List of Value variant type names this library introduces.
    fn value_type_names(&self) -> &[&'static str];
}
