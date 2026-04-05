//! Symbol index — collects all known symbols for completion and hover.
//!
//! The [`SymbolIndex`] is the central registry of every symbol the analyzer
//! knows about. It is backed by a [`BTreeMap`] so iteration order is
//! deterministic (sorted by name), which is important for reproducible
//! completion lists and tests.
//!
//! # Symbol sources
//!
//! 1. **Builtin functions** from `cjc-runtime` — always loaded via
//!    [`SymbolIndex::populate_builtins`].
//! 2. **Library symbols** (e.g. Vizor) — loaded on demand when an `import`
//!    statement is detected, via [`SymbolIndex::populate_vizor`].
//! 3. **User-defined symbols** extracted from the current AST — added with
//!    [`SymbolIndex::add_user_symbol`].

use std::collections::BTreeMap;

/// Metadata for a single symbol known to the analyzer.
///
/// Each `SymbolInfo` carries enough information to power both completion
/// items (label, kind, detail) and hover documentation (signature,
/// description, library attribution).
#[derive(Debug, Clone)]
pub struct SymbolInfo {
    /// The symbol name as it appears in source code (e.g. `"sqrt"`).
    pub name: String,
    /// The category of the symbol (function, method, variable, etc.).
    pub kind: SymbolKind,
    /// An optional human-readable signature (e.g. `"fn sqrt(x: f64) -> f64"`).
    pub signature: Option<String>,
    /// A short plain-text description of what the symbol does.
    pub description: String,
    /// The library this symbol originates from, or `None` for builtins and
    /// user-defined symbols.
    pub library: Option<String>,
}

/// The category of a symbol in the index.
///
/// Maps directly to LSP `CompletionItemKind` values when generating
/// completion responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    /// A free function (e.g. `sqrt`, `print`).
    Function,
    /// A method invoked with dot syntax (e.g. `.geom_point()`).
    Method,
    /// A local or global variable binding.
    Variable,
    /// A type name (struct, enum, alias).
    Type,
    /// A named constant.
    Constant,
    /// A module namespace.
    Module,
}

/// The master symbol index, sorted by name for deterministic iteration.
///
/// Create with [`SymbolIndex::new`], then call [`populate_builtins`](SymbolIndex::populate_builtins)
/// and optionally [`populate_vizor`](SymbolIndex::populate_vizor) to seed it.
/// User-defined symbols are added incrementally via [`add_user_symbol`](SymbolIndex::add_user_symbol).
#[derive(Debug, Default)]
pub struct SymbolIndex {
    /// All known symbols keyed by name. A `BTreeMap` guarantees
    /// deterministic ordering.
    symbols: BTreeMap<String, SymbolInfo>,
}

impl SymbolIndex {
    /// Create an empty symbol index.
    ///
    /// The returned index contains no symbols. Call
    /// [`populate_builtins`](Self::populate_builtins) to seed it with the
    /// core CJC builtins.
    ///
    /// # Examples
    ///
    /// ```
    /// use cjc_analyzer::symbol_index::SymbolIndex;
    /// let mut idx = SymbolIndex::new();
    /// assert!(idx.is_empty());
    /// idx.populate_builtins();
    /// assert!(!idx.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            symbols: BTreeMap::new(),
        }
    }

    /// Populate the index with all core CJC builtin functions.
    ///
    /// This registers symbols such as `print`, `sqrt`, `len`, `array_push`,
    /// and other functions that are always available without an `import`.
    /// Calling this multiple times is idempotent — duplicate names are
    /// overwritten with the same data.
    pub fn populate_builtins(&mut self) {
        // Core builtins that are always available
        let core_builtins = &[
            ("print", "fn print(value: Any)", "Print a value to stdout"),
            ("len", "fn len(collection: Any) -> i64", "Get collection length"),
            ("assert", "fn assert(condition: bool)", "Assert condition is true"),
            ("assert_eq", "fn assert_eq(a: Any, b: Any)", "Assert two values are equal"),
            ("sqrt", "fn sqrt(x: f64) -> f64", "Square root"),
            ("abs", "fn abs(x: f64) -> f64", "Absolute value"),
            ("floor", "fn floor(x: f64) -> f64", "Floor to integer"),
            ("ceil", "fn ceil(x: f64) -> f64", "Ceiling to integer"),
            ("round", "fn round(x: f64) -> f64", "Round to nearest integer"),
            ("int", "fn int(x: f64) -> i64", "Convert to integer"),
            ("float", "fn float(x: i64) -> f64", "Convert to float"),
            ("log", "fn log(x: f64) -> f64", "Natural logarithm"),
            ("exp", "fn exp(x: f64) -> f64", "Exponential function"),
            ("sin", "fn sin(x: f64) -> f64", "Sine"),
            ("cos", "fn cos(x: f64) -> f64", "Cosine"),
            ("tan", "fn tan(x: f64) -> f64", "Tangent"),
            ("min", "fn min(a: f64, b: f64) -> f64", "Minimum of two values"),
            ("max", "fn max(a: f64, b: f64) -> f64", "Maximum of two values"),
            ("to_string", "fn to_string(value: Any) -> string", "Convert to string"),
            ("array_push", "fn array_push(arr: [Any], val: Any) -> [Any]", "Push element to array (returns new array)"),
            ("array_len", "fn array_len(arr: [Any]) -> i64", "Get array length"),
        ];

        for &(name, sig, desc) in core_builtins {
            self.symbols.insert(
                name.to_string(),
                SymbolInfo {
                    name: name.to_string(),
                    kind: SymbolKind::Function,
                    signature: Some(sig.to_string()),
                    description: desc.to_string(),
                    library: None,
                },
            );
        }
    }

    /// Register all Vizor grammar-of-graphics library symbols.
    ///
    /// Call this only when `import vizor` is detected in the source file.
    /// Symbols are fetched from [`cjc_vizor::docs::vizor_docs`] and include
    /// both free functions (e.g. `vizor_plot`) and chainable methods
    /// (e.g. `.geom_point()`). Each symbol is tagged with
    /// `library = Some("vizor")` so it can be filtered later.
    pub fn populate_vizor(&mut self) {
        let vizor_docs = cjc_vizor::docs::vizor_docs();
        for entry in vizor_docs {
            let kind = match entry.kind {
                cjc_vizor::docs::DocKind::Function => SymbolKind::Function,
                cjc_vizor::docs::DocKind::Method => SymbolKind::Method,
            };
            self.symbols.insert(
                entry.name.to_string(),
                SymbolInfo {
                    name: entry.name.to_string(),
                    kind,
                    signature: Some(entry.signature.to_string()),
                    description: entry.description.to_string(),
                    library: Some("vizor".to_string()),
                },
            );
        }
    }

    /// Insert a user-defined symbol discovered during AST analysis.
    ///
    /// Use this for functions, variables, types, and constants declared in the
    /// user's source file. The symbol's `library` field is set to `None`.
    ///
    /// # Arguments
    ///
    /// * `name` - The identifier as it appears in source code.
    /// * `kind` - The symbol category ([`SymbolKind`]).
    /// * `signature` - An optional human-readable signature string.
    /// * `description` - A short description for hover documentation.
    pub fn add_user_symbol(&mut self, name: String, kind: SymbolKind, signature: Option<String>, description: String) {
        self.symbols.insert(
            name.clone(),
            SymbolInfo {
                name,
                kind,
                signature,
                description,
                library: None,
            },
        );
    }

    /// Look up a symbol by its exact name.
    ///
    /// # Arguments
    ///
    /// * `name` - The symbol name to search for (case-sensitive).
    ///
    /// # Returns
    ///
    /// `Some(&SymbolInfo)` if the symbol exists, `None` otherwise.
    pub fn lookup(&self, name: &str) -> Option<&SymbolInfo> {
        self.symbols.get(name)
    }

    /// Return all symbols whose name starts with `prefix`.
    ///
    /// Leverages the `BTreeMap` range query so only the relevant slice of
    /// the sorted index is scanned. An empty prefix returns every symbol.
    ///
    /// # Arguments
    ///
    /// * `prefix` - The partial identifier typed so far (e.g. `"sq"` matches `"sqrt"`).
    ///
    /// # Returns
    ///
    /// A `Vec` of references to matching [`SymbolInfo`] entries, in sorted
    /// name order.
    pub fn completions(&self, prefix: &str) -> Vec<&SymbolInfo> {
        self.symbols
            .range(prefix.to_string()..)
            .take_while(|(k, _)| k.starts_with(prefix))
            .map(|(_, v)| v)
            .collect()
    }

    /// Get all symbols from a specific library.
    pub fn library_symbols(&self, lib: &str) -> Vec<&SymbolInfo> {
        self.symbols
            .values()
            .filter(|s| s.library.as_deref() == Some(lib))
            .collect()
    }

    /// Total count of indexed symbols.
    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_population() {
        let mut idx = SymbolIndex::new();
        idx.populate_builtins();
        assert!(idx.len() > 10);
        assert!(idx.lookup("print").is_some());
        assert!(idx.lookup("sqrt").is_some());
    }

    #[test]
    fn test_vizor_population() {
        let mut idx = SymbolIndex::new();
        idx.populate_vizor();
        assert!(idx.len() > 0);
        assert!(idx.lookup("vizor_plot").is_some());
    }

    #[test]
    fn test_completions() {
        let mut idx = SymbolIndex::new();
        idx.populate_builtins();
        let matches = idx.completions("sq");
        assert!(matches.iter().any(|s| s.name == "sqrt"));
    }

    #[test]
    fn test_library_filter() {
        let mut idx = SymbolIndex::new();
        idx.populate_builtins();
        idx.populate_vizor();
        let vizor_syms = idx.library_symbols("vizor");
        assert!(vizor_syms.len() > 0);
        assert!(vizor_syms.iter().all(|s| s.library.as_deref() == Some("vizor")));
    }
}
