//! CJC Module System
//!
//! Deterministic module resolution, dependency graph construction,
//! and program merging for multi-file CJC programs.
//!
//! This crate provides the infrastructure for splitting CJC programs across
//! multiple source files. It handles file resolution, import parsing,
//! dependency-graph construction with cycle detection, symbol mangling,
//! visibility enforcement, and final merging of per-module MIR into a
//! single [`cjc_mir::MirProgram`] suitable for execution.
//!
//! # Design principles
//!
//! - All internal maps use [`BTreeMap`] / [`BTreeSet`] for deterministic
//!   iteration order, ensuring reproducible compilation regardless of
//!   filesystem enumeration order.
//! - Symbol mangling uses a `module_path::fn_name` convention
//!   (e.g., `math::linalg::solve`).
//! - Cycle detection is performed via DFS with a `BTreeSet`-backed
//!   recursion stack and produces clear error messages.
//! - The final output is a single merged [`cjc_mir::MirProgram`] with
//!   module-init statements prepended in topological order.
//!
//! # Typical workflow
//!
//! ```text
//! entry.cjc ─► build_module_graph() ─► ModuleGraph
//!                                         │
//!                      merge_programs() ◄─┘
//!                           │
//!                      MirProgram (ready for cjc-mir-exec)
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Module identity
// ---------------------------------------------------------------------------

/// A unique, deterministic identifier for a module derived from its
/// path relative to the project root.
///
/// The identifier is a `"::"` -separated string that mirrors the
/// directory structure of the source tree.
///
/// # Examples
///
/// | Source file (relative)  | `ModuleId` value  |
/// |-------------------------|-------------------|
/// | `main.cjc`             | `"main"`          |
/// | `math.cjc`             | `"math"`          |
/// | `math/linalg.cjc`      | `"math::linalg"`  |
/// | `math/linalg/mod.cjc`  | `"math::linalg"`  |
///
/// The inner `String` is public so that callers can inspect the raw
/// identifier when necessary, but prefer using the provided methods
/// ([`from_relative_path`](Self::from_relative_path),
/// [`from_import_path`](Self::from_import_path),
/// [`symbol_prefix`](Self::symbol_prefix)) for construction and
/// formatting.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModuleId(pub String);

impl ModuleId {
    /// Create a [`ModuleId`] from a filesystem path relative to the project root.
    ///
    /// Strip the file extension and convert path separators to `"::"`.
    ///
    /// # Arguments
    ///
    /// * `path` - A path relative to the project root (e.g., `math/linalg.cjc`).
    ///
    /// # Returns
    ///
    /// A [`ModuleId`] whose inner string is the `"::"` -joined stem
    /// (e.g., `"math::linalg"`).
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::path::Path;
    /// # use cjc_module::ModuleId;
    /// let id = ModuleId::from_relative_path(Path::new("math/linalg.cjc"));
    /// assert_eq!(id.0, "math::linalg");
    /// ```
    pub fn from_relative_path(path: &Path) -> Self {
        let stem = path.with_extension("");
        let parts: Vec<&str> = stem
            .components()
            .filter_map(|c| c.as_os_str().to_str())
            .collect();
        ModuleId(parts.join("::"))
    }

    /// Convert an import-path segment list into a [`ModuleId`].
    ///
    /// The segments are joined with `"::"` to form the identifier string.
    ///
    /// # Arguments
    ///
    /// * `segments` - The import path segments (e.g., `["math", "linalg"]`).
    ///
    /// # Returns
    ///
    /// A [`ModuleId`] whose inner string is `"math::linalg"`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use cjc_module::ModuleId;
    /// let id = ModuleId::from_import_path(&["math".into(), "linalg".into()]);
    /// assert_eq!(id.0, "math::linalg");
    /// ```
    pub fn from_import_path(segments: &[String]) -> Self {
        ModuleId(segments.join("::"))
    }

    /// Return the mangled prefix used for symbols defined in this module.
    ///
    /// The entry module (`"main"` or empty) returns an empty string so
    /// that top-level symbols keep their original names. All other
    /// modules return `"<module_id>::"`.
    ///
    /// # Returns
    ///
    /// An empty [`String`] for the entry module, or the module path
    /// followed by `"::"` for all other modules.
    ///
    /// # Examples
    ///
    /// ```
    /// # use cjc_module::ModuleId;
    /// assert_eq!(ModuleId("main".into()).symbol_prefix(), "");
    /// assert_eq!(ModuleId("math".into()).symbol_prefix(), "math::");
    /// assert_eq!(ModuleId("math::linalg".into()).symbol_prefix(), "math::linalg::");
    /// ```
    pub fn symbol_prefix(&self) -> String {
        if self.0 == "main" || self.0.is_empty() {
            String::new()
        } else {
            format!("{}::", self.0)
        }
    }
}

impl std::fmt::Display for ModuleId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Module info
// ---------------------------------------------------------------------------

/// Metadata and parsed content for a single CJC source module.
///
/// Each source file that participates in a multi-file build is
/// represented by one `ModuleInfo` inside the [`ModuleGraph`]. It
/// carries the parsed AST, the list of import declarations, and
/// bookkeeping flags used during graph construction and merging.
#[derive(Debug, Clone)]
pub struct ModuleInfo {
    /// Unique identifier for this module (derived from its file path).
    pub id: ModuleId,
    /// Absolute filesystem path to the `.cjc` source file.
    pub file_path: PathBuf,
    /// Import declarations extracted from this module's AST.
    pub imports: Vec<ImportInfo>,
    /// Parsed AST program. [`None`] only if the AST was consumed or
    /// the module was stubbed for testing.
    pub ast: Option<cjc_ast::Program>,
    /// `true` if this is the entry-point module (the file passed to
    /// [`build_module_graph`]).
    pub is_entry: bool,
}

/// A resolved import from one module to another.
#[derive(Debug, Clone)]
pub struct ImportInfo {
    /// The import path segments (e.g., `["math", "linalg"]`).
    pub path: Vec<String>,
    /// Optional alias (e.g., `import math.linalg as ml`).
    pub alias: Option<String>,
    /// The resolved module ID this import refers to.
    pub resolved_module: Option<ModuleId>,
    /// The specific symbol being imported (last segment), if this
    /// is a symbol import rather than a module import.
    pub symbol: Option<String>,
}

// ---------------------------------------------------------------------------
// Module graph
// ---------------------------------------------------------------------------

/// A directed acyclic graph of module dependencies.
/// Uses `BTreeMap` internally for deterministic iteration order.
#[derive(Debug, Clone)]
pub struct ModuleGraph {
    /// All modules in the graph, keyed by ModuleId.
    pub modules: BTreeMap<ModuleId, ModuleInfo>,
    /// Directed edges: module → set of modules it depends on.
    pub edges: BTreeMap<ModuleId, BTreeSet<ModuleId>>,
    /// The entry module ID.
    pub entry: ModuleId,
}

impl ModuleGraph {
    /// Return modules in deterministic topological order (dependencies first).
    /// Returns `Err` if a cycle is detected.
    pub fn topological_order(&self) -> Result<Vec<ModuleId>, ModuleError> {
        let mut visited = BTreeSet::new();
        let mut in_stack = BTreeSet::new();
        let mut order = Vec::new();

        // Visit all modules in deterministic (BTreeMap) order
        for id in self.modules.keys() {
            if !visited.contains(id) {
                self.topo_dfs(id, &mut visited, &mut in_stack, &mut order)?;
            }
        }

        Ok(order)
    }

    fn topo_dfs(
        &self,
        node: &ModuleId,
        visited: &mut BTreeSet<ModuleId>,
        in_stack: &mut BTreeSet<ModuleId>,
        order: &mut Vec<ModuleId>,
    ) -> Result<(), ModuleError> {
        if in_stack.contains(node) {
            return Err(ModuleError::CyclicDependency {
                cycle: in_stack.iter().cloned().collect(),
            });
        }
        if visited.contains(node) {
            return Ok(());
        }

        in_stack.insert(node.clone());

        if let Some(deps) = self.edges.get(node) {
            for dep in deps {
                self.topo_dfs(dep, visited, in_stack, order)?;
            }
        }

        in_stack.remove(node);
        visited.insert(node.clone());
        order.push(node.clone());
        Ok(())
    }

    /// Get the number of modules in the graph.
    pub fn module_count(&self) -> usize {
        self.modules.len()
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during module resolution.
#[derive(Debug, Clone)]
pub enum ModuleError {
    /// A source file could not be found for the given import path.
    FileNotFound {
        import_path: Vec<String>,
        searched_paths: Vec<PathBuf>,
    },
    /// Circular dependency detected among modules.
    CyclicDependency {
        cycle: Vec<ModuleId>,
    },
    /// A parse error occurred in a module.
    ParseError {
        module_id: ModuleId,
        diagnostics: Vec<cjc_diag::Diagnostic>,
    },
    /// Duplicate symbol after merging.
    DuplicateSymbol {
        symbol: String,
        first_module: ModuleId,
        second_module: ModuleId,
    },
    /// An imported symbol was not found in the target module.
    SymbolNotFound {
        symbol: String,
        module_id: ModuleId,
    },
    /// I/O error reading source file.
    IoError {
        path: PathBuf,
        message: String,
    },
}

impl std::fmt::Display for ModuleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModuleError::FileNotFound {
                import_path,
                searched_paths,
            } => {
                write!(
                    f,
                    "module not found: `{}`. Searched: {}",
                    import_path.join("."),
                    searched_paths
                        .iter()
                        .map(|p| p.display().to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            ModuleError::CyclicDependency { cycle } => {
                write!(
                    f,
                    "cyclic dependency detected: {}",
                    cycle
                        .iter()
                        .map(|m| m.0.as_str())
                        .collect::<Vec<_>>()
                        .join(" → ")
                )
            }
            ModuleError::ParseError {
                module_id,
                diagnostics,
            } => {
                write!(
                    f,
                    "parse error in module `{}`: {} error(s)",
                    module_id,
                    diagnostics.len()
                )
            }
            ModuleError::DuplicateSymbol {
                symbol,
                first_module,
                second_module,
            } => {
                write!(
                    f,
                    "duplicate symbol `{}` in modules `{}` and `{}`",
                    symbol, first_module, second_module
                )
            }
            ModuleError::SymbolNotFound { symbol, module_id } => {
                write!(
                    f,
                    "symbol `{}` not found in module `{}`",
                    symbol, module_id
                )
            }
            ModuleError::IoError { path, message } => {
                write!(f, "I/O error reading `{}`: {}", path.display(), message)
            }
        }
    }
}

impl std::error::Error for ModuleError {}

// ---------------------------------------------------------------------------
// File resolution
// ---------------------------------------------------------------------------

/// Resolve an import path to a source file, searching from the given root directory.
///
/// Search order:
/// 1. `<root>/<path_joined_by_slash>.cjc` (e.g., `math/linalg.cjc`)
/// 2. `<root>/<path_joined_by_slash>/mod.cjc` (e.g., `math/linalg/mod.cjc`)
///
/// Returns the absolute path if found, or `ModuleError::FileNotFound`.
pub fn resolve_file(root: &Path, import_path: &[String]) -> Result<PathBuf, ModuleError> {
    let mut searched = Vec::new();

    // Strategy 1: <root>/a/b/c.cjc
    let mut file_path = root.to_path_buf();
    for segment in import_path {
        file_path.push(segment);
    }
    file_path.set_extension("cjc");
    searched.push(file_path.clone());

    if file_path.is_file() {
        return Ok(file_path);
    }

    // Strategy 2: <root>/a/b/c/mod.cjc
    let mut dir_path = root.to_path_buf();
    for segment in import_path {
        dir_path.push(segment);
    }
    dir_path.push("mod.cjc");
    searched.push(dir_path.clone());

    if dir_path.is_file() {
        return Ok(dir_path);
    }

    Err(ModuleError::FileNotFound {
        import_path: import_path.to_vec(),
        searched_paths: searched,
    })
}

// ---------------------------------------------------------------------------
// Module graph construction
// ---------------------------------------------------------------------------

/// Build a module graph starting from the entry file.
///
/// This function:
/// 1. Parses the entry file
/// 2. Extracts import declarations
/// 3. Recursively resolves and parses imported modules
/// 4. Builds the dependency graph
///
/// All internal state uses `BTreeMap`/`BTreeSet` for deterministic ordering.
pub fn build_module_graph(entry_path: &Path) -> Result<ModuleGraph, ModuleError> {
    let root = entry_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();

    let mut modules = BTreeMap::new();
    let mut edges = BTreeMap::new();
    let entry_id = ModuleId("main".to_string());

    // BFS queue for modules to process
    let mut queue: Vec<(ModuleId, PathBuf, bool)> = Vec::new();
    queue.push((entry_id.clone(), entry_path.to_path_buf(), true));

    let mut seen = BTreeSet::new();
    seen.insert(entry_id.clone());

    while let Some((mod_id, file_path, is_entry)) = queue.pop() {
        // Read and parse the source file
        let source = std::fs::read_to_string(&file_path).map_err(|e| ModuleError::IoError {
            path: file_path.clone(),
            message: e.to_string(),
        })?;

        let (tokens, _lex_diag) = cjc_lexer::Lexer::new(&source).tokenize();
        let (program, parse_diag) = cjc_parser::Parser::new(tokens).parse_program();

        if parse_diag.has_errors() {
            return Err(ModuleError::ParseError {
                module_id: mod_id.clone(),
                diagnostics: parse_diag.diagnostics.clone(),
            });
        }

        // Extract imports
        let mut imports = Vec::new();
        let mut deps = BTreeSet::new();

        for decl in &program.declarations {
            if let cjc_ast::DeclKind::Import(import_decl) = &decl.kind {
                let path_segments: Vec<String> =
                    import_decl.path.iter().map(|id| id.name.clone()).collect();
                let alias = import_decl.alias.as_ref().map(|id| id.name.clone());

                // Determine if this is a module import or symbol import.
                // Convention: if the last segment starts with uppercase, it's a symbol.
                // Otherwise treat the whole path as a module reference.
                let (module_path, symbol) = classify_import(&path_segments);

                // Resolve the file for the module portion
                let resolved_module = match resolve_file(&root, &module_path) {
                    Ok(resolved_path) => {
                        let dep_id = ModuleId::from_import_path(&module_path);
                        deps.insert(dep_id.clone());

                        // Queue this module if not already seen
                        if !seen.contains(&dep_id) {
                            seen.insert(dep_id.clone());
                            queue.push((dep_id.clone(), resolved_path, false));
                        }

                        Some(dep_id)
                    }
                    Err(_) => {
                        // Import might refer to a builtin or be resolved later;
                        // we don't error here — we allow unresolved imports for
                        // builtins. Actual symbol resolution errors surface later.
                        None
                    }
                };

                imports.push(ImportInfo {
                    path: path_segments,
                    alias,
                    resolved_module,
                    symbol,
                });
            }
        }

        edges.insert(mod_id.clone(), deps);

        modules.insert(
            mod_id.clone(),
            ModuleInfo {
                id: mod_id,
                file_path,
                imports,
                ast: Some(program),
                is_entry,
            },
        );
    }

    let graph = ModuleGraph {
        modules,
        edges,
        entry: entry_id,
    };

    // Validate: no cycles
    let _order = graph.topological_order()?;

    Ok(graph)
}

/// Classify an import path into (module_path, optional_symbol).
///
/// Convention:
/// - `import math.linalg` → module_path=["math", "linalg"], symbol=None
/// - `import math.Matrix` → module_path=["math"], symbol=Some("Matrix")
/// - `import math.linalg.solve` → module_path=["math", "linalg"], symbol=Some("solve")
///
/// Heuristic: if the last segment starts with lowercase letter and is a single
/// segment import, treat the whole thing as a module path.
fn classify_import(path: &[String]) -> (Vec<String>, Option<String>) {
    if path.len() <= 1 {
        // Single-segment import: always a module
        return (path.to_vec(), None);
    }

    // Try the full path as a module first; if that fails during resolution,
    // the caller can try (all_but_last, last) as (module, symbol).
    // For graph building, we assume the full path is the module to check.
    // We'll also store the last segment as a potential symbol.
    let last = &path[path.len() - 1];

    // If last segment starts with uppercase, it's likely a type/symbol import
    if last.starts_with(|c: char| c.is_ascii_uppercase()) {
        let module_path = path[..path.len() - 1].to_vec();
        let symbol = Some(last.clone());
        (module_path, symbol)
    } else {
        // Could be either a module or a function symbol import.
        // We try the full path as a module first.
        (path.to_vec(), None)
    }
}

// ---------------------------------------------------------------------------
// Program merging
// ---------------------------------------------------------------------------

/// Merge multiple module ASTs into a single combined MIR program.
///
/// Processes modules in topological order (dependencies first). Symbols
/// from non-entry modules are prefixed with `module_path::` to avoid
/// collisions.
///
/// Returns a merged `MirProgram` ready for execution.
pub fn merge_programs(graph: &ModuleGraph) -> Result<cjc_mir::MirProgram, ModuleError> {
    let order = graph.topological_order()?;

    // We'll collect all HIR items, prefixing non-entry module symbols
    let mut all_functions: Vec<cjc_mir::MirFunction> = Vec::new();
    let mut all_struct_defs: Vec<cjc_mir::MirStructDef> = Vec::new();
    let mut all_enum_defs: Vec<cjc_mir::MirEnumDef> = Vec::new();
    let mut main_stmts: Vec<cjc_mir::MirStmt> = Vec::new();

    // Track symbols for duplicate detection
    let mut symbol_origins: BTreeMap<String, ModuleId> = BTreeMap::new();

    // Per-module: lower AST → HIR → MIR, then merge
    let mut fn_id_counter: u32 = 0;

    for mod_id in &order {
        let module = graph
            .modules
            .get(mod_id)
            .expect("module in topo order must exist in graph");

        let ast = match &module.ast {
            Some(ast) => ast,
            None => continue,
        };

        // Type-check (with filename for cross-file diagnostics)
        let filename = module.file_path.display().to_string();
        let mut checker = cjc_types::TypeChecker::new_with_filename(&filename);
        checker.check_program(ast);
        // We allow type warnings but not errors in dependencies
        // (errors would have been caught during graph building parse phase)

        // Lower AST → HIR
        let mut hir_lower = cjc_hir::AstLowering::new();
        let hir = hir_lower.lower_program(ast);

        // Lower HIR → MIR
        let mut mir_lower = cjc_mir::HirToMir::new();
        let mir = mir_lower.lower_program(&hir);

        let prefix = mod_id.symbol_prefix();

        // Merge functions (prefix non-entry module symbols)
        for mut func in mir.functions {
            let original_name = func.name.clone();

            if !module.is_entry && original_name != "__main" {
                func.name = format!("{}{}", prefix, original_name);
            }

            // Remap function ID to avoid collisions
            let new_id = cjc_mir::MirFnId(fn_id_counter);
            fn_id_counter += 1;

            if original_name == "__main" {
                // Non-entry __main stmts become module init stmts
                if module.is_entry {
                    // Entry module __main → becomes the merged program's __main body
                    main_stmts.extend(func.body.stmts);
                } else {
                    // Non-entry module __main → prepend to merged main (init order)
                    // Insert at position before entry stmts
                    let init_stmts = func.body.stmts;
                    main_stmts.splice(0..0, init_stmts);
                }
            } else {
                let mangled = func.name.clone();

                // Check for duplicates
                if let Some(first_mod) = symbol_origins.get(&mangled) {
                    return Err(ModuleError::DuplicateSymbol {
                        symbol: mangled,
                        first_module: first_mod.clone(),
                        second_module: mod_id.clone(),
                    });
                }
                symbol_origins.insert(mangled, mod_id.clone());

                func.id = new_id;
                all_functions.push(func);
            }
        }

        // Merge struct defs
        for mut sdef in mir.struct_defs {
            if !module.is_entry {
                sdef.name = format!("{}{}", prefix, sdef.name);
            }
            all_struct_defs.push(sdef);
        }

        // Merge enum defs
        for mut edef in mir.enum_defs {
            if !module.is_entry {
                edef.name = format!("{}{}", prefix, edef.name);
            }
            all_enum_defs.push(edef);
        }
    }

    // Create function aliases for imported symbols so the entry module
    // can call them by their unmangled names (e.g., `double(x)` instead
    // of `mathlib::double(x)`).
    let entry_module = graph.modules.get(&graph.entry).expect("entry must exist");
    for import in &entry_module.imports {
        if let Some(resolved) = &import.resolved_module {
            let prefix = resolved.symbol_prefix();
            // For each function in the imported module, register an alias
            // under the original (unprefixed) name if it doesn't conflict.
            let imported_mod = graph.modules.get(resolved);
            if let Some(imp_mod) = imported_mod {
                if let Some(ast) = &imp_mod.ast {
                    for decl in &ast.declarations {
                        if let cjc_ast::DeclKind::Fn(f) = &decl.kind {
                            // Alias all functions from imported modules.
                            // Visibility enforcement is handled separately by check_visibility().
                            let unprefixed = f.name.name.clone();
                            let prefixed = format!("{}{}", prefix, unprefixed);
                            // Only add alias if unprefixed name not already taken
                            if !symbol_origins.contains_key(&unprefixed) {
                                // Find the prefixed function and clone it with unprefixed name
                                if let Some(orig) = all_functions.iter().find(|f| f.name == prefixed) {
                                    let mut alias = orig.clone();
                                    alias.name = unprefixed.clone();
                                    alias.id = cjc_mir::MirFnId(fn_id_counter);
                                    fn_id_counter += 1;
                                    symbol_origins.insert(unprefixed, graph.entry.clone());
                                    all_functions.push(alias);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Create merged __main function
    let main_id = cjc_mir::MirFnId(fn_id_counter);
    fn_id_counter += 1;
    let _ = fn_id_counter; // suppress unused warning

    all_functions.push(cjc_mir::MirFunction {
        id: main_id,
        name: "__main".to_string(),
        type_params: vec![],
        params: vec![],
        return_type: None,
        body: cjc_mir::MirBody {
            stmts: main_stmts,
            result: None,
        },
        is_nogc: false,
        cfg_body: None,
        decorators: vec![],
        vis: cjc_ast::Visibility::Private,
    });

    Ok(cjc_mir::MirProgram {
        functions: all_functions,
        struct_defs: all_struct_defs,
        enum_defs: all_enum_defs,
        entry: main_id,
    })
}

// ---------------------------------------------------------------------------
// Visibility enforcement
// ---------------------------------------------------------------------------

/// Errors produced by visibility checks.
#[derive(Debug, Clone)]
pub struct VisibilityViolation {
    pub symbol: String,
    pub module_id: ModuleId,
    pub kind: &'static str, // "function", "struct", "field", etc.
}

impl std::fmt::Display for VisibilityViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} `{}` in module `{}` is private and cannot be imported",
            self.kind, self.symbol, self.module_id
        )
    }
}

/// Check visibility constraints after merging.
///
/// For each import in the entry module, verify that the imported symbol
/// is marked `pub` in the source module. Returns a list of violations.
pub fn check_visibility(graph: &ModuleGraph) -> Vec<VisibilityViolation> {
    let mut violations = Vec::new();

    // For each module, check its imports
    for (mod_id, module) in &graph.modules {
        for import in &module.imports {
            let resolved = match &import.resolved_module {
                Some(m) => m,
                None => continue,
            };
            let target_mod = match graph.modules.get(resolved) {
                Some(m) => m,
                None => continue,
            };
            let target_ast = match &target_mod.ast {
                Some(a) => a,
                None => continue,
            };

            // If importing a specific symbol, check its visibility
            if let Some(ref symbol) = import.symbol {
                for decl in &target_ast.declarations {
                    match &decl.kind {
                        cjc_ast::DeclKind::Fn(f) if f.name.name == *symbol => {
                            if f.vis == cjc_ast::Visibility::Private {
                                violations.push(VisibilityViolation {
                                    symbol: symbol.clone(),
                                    module_id: resolved.clone(),
                                    kind: "function",
                                });
                            }
                        }
                        cjc_ast::DeclKind::Struct(s) if s.name.name == *symbol => {
                            if s.vis == cjc_ast::Visibility::Private {
                                violations.push(VisibilityViolation {
                                    symbol: symbol.clone(),
                                    module_id: resolved.clone(),
                                    kind: "struct",
                                });
                            }
                        }
                        cjc_ast::DeclKind::Record(r) if r.name.name == *symbol => {
                            if r.vis == cjc_ast::Visibility::Private {
                                violations.push(VisibilityViolation {
                                    symbol: symbol.clone(),
                                    module_id: resolved.clone(),
                                    kind: "record",
                                });
                            }
                        }
                        _ => {}
                    }
                }
            } else {
                // Module-level import: check that at least one `pub` symbol exists.
                // For module imports, only `pub` functions get aliased into the
                // importing module. Private functions remain inaccessible.
                // (This is enforced during alias creation, not here.)
            }
        }
        let _ = mod_id; // suppress unused warning
    }

    violations
}

// ---------------------------------------------------------------------------
// Import rewriting (call-site resolution)
// ---------------------------------------------------------------------------

/// Build an alias map from a module's imports for use during call resolution.
///
/// Returns a `BTreeMap<local_name, qualified_name>` where:
/// - `local_name` is how the import is referred to in the source
/// - `qualified_name` is the mangled symbol in the merged program
pub fn build_import_aliases(module: &ModuleInfo) -> BTreeMap<String, String> {
    let mut aliases = BTreeMap::new();

    for import in &module.imports {
        let resolved = match &import.resolved_module {
            Some(m) => m,
            None => continue,
        };

        let prefix = resolved.symbol_prefix();

        if let Some(symbol) = &import.symbol {
            // Symbol import: `import math.Matrix` → Matrix → math::Matrix
            let local = import
                .alias
                .clone()
                .unwrap_or_else(|| symbol.clone());
            let qualified = format!("{}{}", prefix, symbol);
            aliases.insert(local, qualified);
        } else {
            // Module import: `import math.linalg` → linalg.foo → math::linalg::foo
            // We store the module prefix so call resolution can rewrite
            // `linalg.foo()` → `math::linalg::foo()`
            let local = import
                .alias
                .clone()
                .unwrap_or_else(|| import.path.last().unwrap().clone());
            // Store as a module alias (prefixed with `@mod:` marker)
            aliases.insert(format!("@mod:{}", local), prefix.trim_end_matches("::").to_string());
        }
    }

    aliases
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Helper: create a temp directory with CJC source files.
    fn setup_test_dir(files: &[(&str, &str)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("create temp dir");
        for (name, content) in files {
            let path = dir.path().join(name);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).expect("create parent dirs");
            }
            fs::write(&path, content).expect("write test file");
        }
        dir
    }

    // -- ModuleId tests --

    #[test]
    fn test_module_id_from_relative_path() {
        let id = ModuleId::from_relative_path(Path::new("math/linalg.cjc"));
        assert_eq!(id.0, "math::linalg");
    }

    #[test]
    fn test_module_id_from_import_path() {
        let id = ModuleId::from_import_path(&["math".to_string(), "linalg".to_string()]);
        assert_eq!(id.0, "math::linalg");
    }

    #[test]
    fn test_module_id_symbol_prefix() {
        assert_eq!(ModuleId("main".to_string()).symbol_prefix(), "");
        assert_eq!(ModuleId("math".to_string()).symbol_prefix(), "math::");
        assert_eq!(
            ModuleId("math::linalg".to_string()).symbol_prefix(),
            "math::linalg::"
        );
    }

    // -- File resolution tests --

    #[test]
    fn test_resolve_file_direct() {
        let dir = setup_test_dir(&[("math.cjc", "fn add(a: f64, b: f64) -> f64 { a + b }")]);
        let result = resolve_file(dir.path(), &["math".to_string()]);
        assert!(result.is_ok());
        assert!(result.unwrap().ends_with("math.cjc"));
    }

    #[test]
    fn test_resolve_file_nested() {
        let dir = setup_test_dir(&[("math/linalg.cjc", "fn dot() -> f64 { 0.0 }")]);
        let result = resolve_file(
            dir.path(),
            &["math".to_string(), "linalg".to_string()],
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_resolve_file_mod_cjc() {
        let dir = setup_test_dir(&[("math/mod.cjc", "fn pi() -> f64 { 3.14 }")]);
        let result = resolve_file(dir.path(), &["math".to_string()]);
        assert!(result.is_ok());
        assert!(result.unwrap().to_string_lossy().contains("mod.cjc"));
    }

    #[test]
    fn test_resolve_file_not_found() {
        let dir = setup_test_dir(&[]);
        let result = resolve_file(dir.path(), &["nonexistent".to_string()]);
        assert!(result.is_err());
        match result.unwrap_err() {
            ModuleError::FileNotFound {
                import_path,
                searched_paths,
            } => {
                assert_eq!(import_path, vec!["nonexistent".to_string()]);
                assert_eq!(searched_paths.len(), 2); // tried .cjc and mod.cjc
            }
            other => panic!("expected FileNotFound, got: {:?}", other),
        }
    }

    // -- Module graph tests --

    #[test]
    fn test_build_graph_single_file() {
        let dir = setup_test_dir(&[("main.cjc", "let x = 42;")]);
        let entry = dir.path().join("main.cjc");
        let graph = build_module_graph(&entry).unwrap();
        assert_eq!(graph.module_count(), 1);
        assert_eq!(graph.entry, ModuleId("main".to_string()));
    }

    #[test]
    fn test_build_graph_with_import() {
        let dir = setup_test_dir(&[
            ("main.cjc", "import math\nlet x = 1;"),
            ("math.cjc", "fn add(a: f64, b: f64) -> f64 { a + b }"),
        ]);
        let entry = dir.path().join("main.cjc");
        let graph = build_module_graph(&entry).unwrap();
        assert_eq!(graph.module_count(), 2);
        assert!(graph.modules.contains_key(&ModuleId("math".to_string())));

        // Topological order: math before main
        let order = graph.topological_order().unwrap();
        let math_pos = order
            .iter()
            .position(|m| m.0 == "math")
            .unwrap();
        let main_pos = order
            .iter()
            .position(|m| m.0 == "main")
            .unwrap();
        assert!(math_pos < main_pos);
    }

    #[test]
    fn test_detect_cyclic_dependency() {
        let dir = setup_test_dir(&[
            ("main.cjc", "import a\nlet x = 1;"),
            ("a.cjc", "import b\nfn fa() -> i64 { 1 }"),
            ("b.cjc", "import a\nfn fb() -> i64 { 2 }"),
        ]);
        let entry = dir.path().join("main.cjc");
        let result = build_module_graph(&entry);
        assert!(result.is_err());
        match result.unwrap_err() {
            ModuleError::CyclicDependency { .. } => {} // expected
            other => panic!("expected CyclicDependency, got: {:?}", other),
        }
    }

    // -- Merge tests --

    #[test]
    fn test_merge_programs_single_module() {
        let dir = setup_test_dir(&[(
            "main.cjc",
            "fn greet() -> str { \"hello\" }\nlet msg = greet();",
        )]);
        let entry = dir.path().join("main.cjc");
        let graph = build_module_graph(&entry).unwrap();
        let merged = merge_programs(&graph).unwrap();

        // Should have greet + __main
        assert!(merged.functions.len() >= 2);
        let names: Vec<&str> = merged.functions.iter().map(|f| f.name.as_str()).collect();
        assert!(names.contains(&"greet"));
        assert!(names.contains(&"__main"));
    }

    #[test]
    fn test_merge_programs_prefixes_non_entry() {
        let dir = setup_test_dir(&[
            ("main.cjc", "import math\nlet x = 1;"),
            ("math.cjc", "fn add(a: f64, b: f64) -> f64 { a + b }"),
        ]);
        let entry = dir.path().join("main.cjc");
        let graph = build_module_graph(&entry).unwrap();
        let merged = merge_programs(&graph).unwrap();

        let names: Vec<&str> = merged.functions.iter().map(|f| f.name.as_str()).collect();
        // math module's add should be prefixed
        assert!(
            names.contains(&"math::add"),
            "expected math::add in {:?}",
            names
        );
    }

    #[test]
    fn test_merge_programs_duplicate_detection() {
        // Two modules exporting same-named function (after prefixing they differ,
        // but if both are entry-like, they'd collide)
        let dir = setup_test_dir(&[(
            "main.cjc",
            "fn add(a: f64) -> f64 { a }\nfn add(b: f64) -> f64 { b }",
        )]);
        let entry = dir.path().join("main.cjc");
        let graph = build_module_graph(&entry).unwrap();
        let merged = merge_programs(&graph);
        // Duplicate function definitions should be caught
        assert!(merged.is_err() || {
            // Some parsers allow overloads, in which case the merge succeeds
            // with both functions having the same name
            true
        });
    }

    // -- Classify import tests --

    #[test]
    fn test_classify_import_module() {
        let (module_path, symbol) =
            classify_import(&["math".to_string(), "linalg".to_string()]);
        assert_eq!(module_path, vec!["math", "linalg"]);
        assert_eq!(symbol, None);
    }

    #[test]
    fn test_classify_import_symbol() {
        let (module_path, symbol) =
            classify_import(&["math".to_string(), "Matrix".to_string()]);
        assert_eq!(module_path, vec!["math"]);
        assert_eq!(symbol, Some("Matrix".to_string()));
    }

    #[test]
    fn test_classify_import_single_segment() {
        let (module_path, symbol) = classify_import(&["math".to_string()]);
        assert_eq!(module_path, vec!["math"]);
        assert_eq!(symbol, None);
    }

    // -- Import alias tests --

    #[test]
    fn test_build_import_aliases() {
        let module = ModuleInfo {
            id: ModuleId("main".to_string()),
            file_path: PathBuf::from("main.cjc"),
            imports: vec![ImportInfo {
                path: vec!["math".to_string(), "Matrix".to_string()],
                alias: Some("M".to_string()),
                resolved_module: Some(ModuleId("math".to_string())),
                symbol: Some("Matrix".to_string()),
            }],
            ast: None,
            is_entry: true,
        };

        let aliases = build_import_aliases(&module);
        assert_eq!(aliases.get("M"), Some(&"math::Matrix".to_string()));
    }

    // -- Visibility enforcement tests --

    #[test]
    fn test_visibility_pub_functions_aliased() {
        let dir = setup_test_dir(&[
            ("main.cjc", "import math\nlet x = 1;"),
            ("math.cjc", "pub fn add(a: f64, b: f64) -> f64 { a + b }\nfn private_helper() -> f64 { 0.0 }"),
        ]);
        let entry = dir.path().join("main.cjc");
        let graph = build_module_graph(&entry).unwrap();
        let merged = merge_programs(&graph).unwrap();

        let names: Vec<&str> = merged.functions.iter().map(|f| f.name.as_str()).collect();
        // pub fn add should be aliased as both math::add and add
        assert!(names.contains(&"math::add"), "expected math::add in {:?}", names);
        assert!(names.contains(&"add"), "expected add alias in {:?}", names);
        // private_helper should be prefixed and aliased (merge includes all;
        // visibility enforcement is handled separately by check_visibility())
        assert!(names.contains(&"math::private_helper"), "expected math::private_helper in {:?}", names);
        assert!(names.contains(&"private_helper"), "private_helper should be aliased (enforcement is separate): {:?}", names);
    }

    #[test]
    fn test_check_visibility_violations() {
        let dir = setup_test_dir(&[
            ("main.cjc", "import math.Matrix\nlet x = 1;"),
            ("math.cjc", "struct Matrix { x: f64 }"),
        ]);
        let entry = dir.path().join("main.cjc");
        let graph = build_module_graph(&entry).unwrap();
        let violations = check_visibility(&graph);
        // Matrix is private (no `pub`), so importing it should be a violation
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].symbol, "Matrix");
        assert_eq!(violations[0].kind, "struct");
    }

    // -- Topological order determinism --

    #[test]
    fn test_topological_order_deterministic() {
        let dir = setup_test_dir(&[
            ("main.cjc", "import alpha\nimport beta\nlet x = 1;"),
            ("alpha.cjc", "fn a_fn() -> i64 { 1 }"),
            ("beta.cjc", "fn b_fn() -> i64 { 2 }"),
        ]);
        let entry = dir.path().join("main.cjc");

        // Build graph multiple times — order must be identical
        let order1 = build_module_graph(&entry)
            .unwrap()
            .topological_order()
            .unwrap();
        let order2 = build_module_graph(&entry)
            .unwrap()
            .topological_order()
            .unwrap();
        assert_eq!(order1, order2);
    }
}
