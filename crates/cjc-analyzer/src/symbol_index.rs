//! Symbol index — collects all known symbols for completion & hover.
//!
//! Sources:
//! 1. Builtin functions from cjc-runtime
//! 2. Library docs (vizor, etc.) via CjcLibrary trait
//! 3. User-defined symbols from the current AST

use std::collections::BTreeMap;

/// A symbol known to the analyzer.
#[derive(Debug, Clone)]
pub struct SymbolInfo {
    pub name: String,
    pub kind: SymbolKind,
    pub signature: Option<String>,
    pub description: String,
    pub library: Option<String>,
}

/// What kind of symbol this is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    Function,
    Method,
    Variable,
    Type,
    Constant,
    Module,
}

/// The master symbol index — sorted by name for determinism.
#[derive(Debug, Default)]
pub struct SymbolIndex {
    symbols: BTreeMap<String, SymbolInfo>,
}

impl SymbolIndex {
    pub fn new() -> Self {
        Self {
            symbols: BTreeMap::new(),
        }
    }

    /// Populate with all built-in symbols.
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

    /// Add all Vizor library symbols (only when `import vizor` is detected).
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

    /// Add a user-defined symbol from AST analysis.
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

    /// Look up a symbol by name.
    pub fn lookup(&self, name: &str) -> Option<&SymbolInfo> {
        self.symbols.get(name)
    }

    /// Get all symbols matching a prefix (for completion).
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
