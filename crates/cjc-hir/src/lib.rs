//! CJC HIR (High-level Intermediate Representation)
//!
//! HIR is a desugared AST: it removes syntactic sugar (pipes, multi-index)
//! but preserves high-level types and structure. Every HIR node carries a
//! `HirId` for diagnostics/debug tracing.
//!
//! Key differences from AST:
//! - Pipe `a |> f(b)` is lowered to `Call(f, [a, b])`
//! - MultiIndex `t[i, j]` is lowered to nested Index or method calls
//! - If/While/Return are expressions (not statements)
//! - All names are resolved strings (no Ident wrappers)

use cjc_ast::{BinOp, UnaryOp, Visibility};

// ---------------------------------------------------------------------------
// HIR Node IDs
// ---------------------------------------------------------------------------

/// Unique ID for every HIR node (monotonically increasing per lowering session).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HirId(pub u32);

// ---------------------------------------------------------------------------
// Top-level Program
// ---------------------------------------------------------------------------

/// A complete HIR program after AST lowering.
///
/// Contains all top-level items (functions, structs, enums, etc.) in their
/// desugared HIR form, ready for downstream MIR lowering.
#[derive(Debug, Clone)]
pub struct HirProgram {
    /// Top-level items in declaration order.
    pub items: Vec<HirItem>,
}

/// Top-level item after desugaring.
///
/// Each variant corresponds to a distinct kind of declaration that can appear
/// at the top level of a CJC program.
#[derive(Debug, Clone)]
pub enum HirItem {
    /// A function declaration (including decorated functions).
    Fn(HirFn),
    /// A struct type definition with named fields.
    Struct(HirStructDef),
    /// A class type definition with named fields.
    Class(HirClassDef),
    /// A record (immutable value type) definition.
    Record(HirRecordDef),
    /// A trait definition with method signatures.
    Trait(HirTraitDef),
    /// An `impl` block providing method implementations for a type.
    Impl(HirImplDef),
    /// An enum type definition with variants.
    Enum(HirEnumDef),
    /// A top-level `let` binding.
    Let(HirLetDecl),
    /// A bare statement at the top level (e.g., expression statements, imports lowered to void).
    Stmt(HirStmt),
}

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------

/// A function definition in HIR.
///
/// Represents both plain functions and decorated functions after lowering.
/// Parameters, return type, and body have all been desugared from AST form.
#[derive(Debug, Clone)]
pub struct HirFn {
    /// The function name.
    pub name: String,
    /// Generic type parameters as `(param_name, trait_bounds)` pairs.
    pub type_params: Vec<(String, Vec<String>)>,
    /// Positional parameters (may include defaults and variadics).
    pub params: Vec<HirParam>,
    /// Optional return type annotation as a string.
    pub return_type: Option<String>,
    /// The function body block.
    pub body: HirBlock,
    /// Whether this function is annotated `nogc` (no garbage-collected allocations allowed).
    pub is_nogc: bool,
    /// Unique HIR node identifier.
    pub hir_id: HirId,
    /// Decorator names applied to this function (e.g., `@memoize`, `@trace`).
    pub decorators: Vec<String>,
    /// Visibility of this function (`pub` or private).
    pub vis: Visibility,
}

/// A function parameter in HIR.
///
/// Carries the parameter name, type annotation, optional default value,
/// and whether it is variadic.
#[derive(Debug, Clone)]
pub struct HirParam {
    /// The parameter name.
    pub name: String,
    /// Type annotation as a string (e.g., `"f64"`, `"Tensor"`).
    pub ty_name: String,
    /// Optional default value expression for this parameter.
    pub default: Option<HirExpr>,
    /// Variadic parameter: collects remaining args into an array.
    pub is_variadic: bool,
    /// Unique HIR node identifier.
    pub hir_id: HirId,
}

// ---------------------------------------------------------------------------
// Struct / Class / Trait / Impl
// ---------------------------------------------------------------------------

/// A struct type definition in HIR.
///
/// Structs are mutable product types with named fields.
#[derive(Debug, Clone)]
pub struct HirStructDef {
    /// The struct name.
    pub name: String,
    /// Fields as `(field_name, type_name)` pairs.
    pub fields: Vec<(String, String)>,
    /// Unique HIR node identifier.
    pub hir_id: HirId,
    /// Visibility of this struct (`pub` or private).
    pub vis: Visibility,
}

/// A class type definition in HIR.
///
/// Classes are similar to structs but follow class semantics with
/// reference identity.
#[derive(Debug, Clone)]
pub struct HirClassDef {
    /// The class name.
    pub name: String,
    /// Fields as `(field_name, type_name)` pairs.
    pub fields: Vec<(String, String)>,
    /// Unique HIR node identifier.
    pub hir_id: HirId,
    /// Visibility of this class (`pub` or private).
    pub vis: Visibility,
}

/// Record: immutable value type.
#[derive(Debug, Clone)]
pub struct HirRecordDef {
    pub name: String,
    pub fields: Vec<(String, String)>,
    pub hir_id: HirId,
    pub vis: Visibility,
}

/// A trait definition in HIR.
///
/// Defines a set of method signatures that implementors must provide.
#[derive(Debug, Clone)]
pub struct HirTraitDef {
    /// The trait name.
    pub name: String,
    /// Method signatures declared in this trait.
    pub methods: Vec<HirFnSig>,
    /// Unique HIR node identifier.
    pub hir_id: HirId,
}

/// A function signature (without body) in HIR.
///
/// Used inside trait definitions to declare method contracts.
#[derive(Debug, Clone)]
pub struct HirFnSig {
    /// The method name.
    pub name: String,
    /// Parameters of this method signature.
    pub params: Vec<HirParam>,
    /// Optional return type annotation as a string.
    pub return_type: Option<String>,
}

/// An `impl` block in HIR.
///
/// Provides method implementations for a target type, optionally satisfying
/// a trait reference.
#[derive(Debug, Clone)]
pub struct HirImplDef {
    /// The type being implemented (e.g., `"Matrix"`).
    pub target: String,
    /// Optional trait being implemented (e.g., `"Display"`).
    pub trait_ref: Option<String>,
    /// Method implementations.
    pub methods: Vec<HirFn>,
    /// Unique HIR node identifier.
    pub hir_id: HirId,
}

/// An enum type definition in HIR.
///
/// Enums are sum types with named variants, optionally parameterized by
/// generic type parameters.
#[derive(Debug, Clone)]
pub struct HirEnumDef {
    /// The enum name.
    pub name: String,
    /// Generic type parameter names (e.g., `["T"]` for `Option<T>`).
    pub type_params: Vec<String>,
    /// Variant definitions for this enum.
    pub variants: Vec<HirVariantDef>,
    /// Unique HIR node identifier.
    pub hir_id: HirId,
}

/// A single variant of an enum definition in HIR.
#[derive(Debug, Clone)]
pub struct HirVariantDef {
    /// The variant name.
    pub name: String,
    /// Positional field types as strings (empty for unit variants).
    pub fields: Vec<String>,
}

// ---------------------------------------------------------------------------
// Let declaration (top-level)
// ---------------------------------------------------------------------------

/// A top-level `let` binding in HIR.
///
/// Represents a variable declaration with an initializer expression.
/// May be mutable or immutable, and may carry an explicit type annotation.
#[derive(Debug, Clone)]
pub struct HirLetDecl {
    /// The variable name.
    pub name: String,
    /// Whether this binding is mutable (`let mut`).
    pub mutable: bool,
    /// Optional explicit type annotation as a string.
    pub ty_name: Option<String>,
    /// The initializer expression.
    pub init: HirExpr,
    /// Unique HIR node identifier.
    pub hir_id: HirId,
}

// ---------------------------------------------------------------------------
// Blocks
// ---------------------------------------------------------------------------

/// A block of statements in HIR, optionally terminated by a tail expression.
///
/// The tail expression (if present) determines the value produced by the block.
/// This is the HIR equivalent of `{ stmt; stmt; expr }`.
#[derive(Debug, Clone)]
pub struct HirBlock {
    /// Statements in this block, in order.
    pub stmts: Vec<HirStmt>,
    /// Optional tail expression whose value becomes the block's value.
    pub expr: Option<Box<HirExpr>>,
    /// Unique HIR node identifier.
    pub hir_id: HirId,
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

/// A statement in HIR.
///
/// Wraps a [`HirStmtKind`] with a unique [`HirId`] for diagnostics and tracing.
#[derive(Debug, Clone)]
pub struct HirStmt {
    /// The kind of statement.
    pub kind: HirStmtKind,
    /// Unique HIR node identifier.
    pub hir_id: HirId,
}

/// The kind of a HIR statement.
///
/// Statements do not produce values (unlike expressions). Control flow
/// constructs like `if` and `while` appear here when used in statement
/// position.
#[derive(Debug, Clone)]
pub enum HirStmtKind {
    /// A local variable binding: `let [mut] name [: ty] = init;`.
    Let {
        /// The variable name.
        name: String,
        /// Whether this binding is mutable.
        mutable: bool,
        /// Optional type annotation as a string.
        ty_name: Option<String>,
        /// The initializer expression.
        init: HirExpr,
    },
    /// A bare expression statement.
    Expr(HirExpr),
    /// An `if`/`else if`/`else` chain in statement position.
    If(HirIfExpr),
    /// A `while` loop with condition and body.
    While {
        /// Loop condition expression.
        cond: HirExpr,
        /// Loop body block.
        body: HirBlock,
    },
    /// A `return` statement with an optional value.
    Return(Option<HirExpr>),
    /// A `break` statement (exits the innermost loop).
    Break,
    /// A `continue` statement (skips to the next loop iteration).
    Continue,
    /// A `nogc { ... }` block that disallows GC allocations inside.
    NoGcBlock(HirBlock),
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

/// An expression node in HIR.
///
/// Wraps a [`HirExprKind`] with a unique [`HirId`] for diagnostics and tracing.
#[derive(Debug, Clone)]
pub struct HirExpr {
    /// The kind of expression.
    pub kind: HirExprKind,
    /// Unique HIR node identifier.
    pub hir_id: HirId,
}

/// The kind of a HIR expression.
///
/// After lowering, syntactic sugar has been removed: pipes become calls,
/// `for` loops become `while` loops, `as` casts become builtin calls,
/// compound assignments become plain assignments, and f-strings become
/// string concatenation chains.
#[derive(Debug, Clone)]
pub enum HirExprKind {
    // -- Literals --
    /// An integer literal (e.g., `42`).
    IntLit(i64),
    /// A floating-point literal (e.g., `3.14`).
    FloatLit(f64),
    /// A boolean literal (`true` or `false`).
    BoolLit(bool),
    /// A string literal (e.g., `"hello"`).
    StringLit(String),
    /// A byte string literal (e.g., `b"hello"`).
    ByteStringLit(Vec<u8>),
    /// A byte character literal (e.g., `b'x'`).
    ByteCharLit(u8),
    /// A raw string literal (e.g., `r"no\escapes"`).
    RawStringLit(String),
    /// A raw byte string literal (e.g., `rb"raw\bytes"`).
    RawByteStringLit(Vec<u8>),
    /// A regex literal with pattern and flags (e.g., `/pattern/gi`).
    RegexLit {
        /// The regex pattern string.
        pattern: String,
        /// Regex flags (e.g., `"gi"`).
        flags: String,
    },
    /// A tensor literal with rows of expressions (e.g., `[[1, 2], [3, 4]]`).
    TensorLit {
        /// Rows of element expressions.
        rows: Vec<Vec<HirExpr>>,
    },
    /// The `NA` (not available) literal for missing values.
    NaLit,

    // -- Variable reference --
    /// A variable reference by name.
    Var(String),

    // -- Operations --
    /// A binary operation (e.g., `a + b`, `x < y`).
    Binary {
        /// The binary operator.
        op: BinOp,
        /// The left-hand operand.
        left: Box<HirExpr>,
        /// The right-hand operand.
        right: Box<HirExpr>,
    },
    /// A unary operation (e.g., `-x`, `!flag`).
    Unary {
        /// The unary operator.
        op: UnaryOp,
        /// The operand expression.
        operand: Box<HirExpr>,
    },

    // -- Call (unified: pipe lowered into this) --
    /// A function call. Pipe expressions (`a |> f(b)`) are desugared into
    /// calls with the left-hand side prepended as the first argument.
    Call {
        /// The callee expression (typically a [`Var`](HirExprKind::Var)).
        callee: Box<HirExpr>,
        /// Positional arguments.
        args: Vec<HirExpr>,
    },

    // -- Field access --
    /// Field access on an object (e.g., `point.x`).
    Field {
        /// The object being accessed.
        object: Box<HirExpr>,
        /// The field name.
        name: String,
    },

    // -- Index --
    /// Single-index access (e.g., `arr[i]`).
    Index {
        /// The object being indexed.
        object: Box<HirExpr>,
        /// The index expression.
        index: Box<HirExpr>,
    },

    // -- Multi-index (preserved for tensor indexing) --
    /// Multi-index access for tensors (e.g., `t[i, j]`).
    MultiIndex {
        /// The tensor or object being indexed.
        object: Box<HirExpr>,
        /// Index expressions for each dimension.
        indices: Vec<HirExpr>,
    },

    // -- Assign --
    /// Assignment expression (e.g., `x = 42`). Compound assignments like
    /// `x += 1` are desugared to `x = x + 1` during lowering.
    Assign {
        /// The assignment target (variable, field, or index).
        target: Box<HirExpr>,
        /// The value being assigned.
        value: Box<HirExpr>,
    },

    // -- Block --
    /// A block expression that evaluates to the value of its tail expression.
    Block(HirBlock),

    // -- Struct literal --
    /// A struct literal (e.g., `Point { x: 1, y: 2 }`).
    StructLit {
        /// The struct type name.
        name: String,
        /// Field initializers as `(field_name, value)` pairs.
        fields: Vec<(String, HirExpr)>,
    },

    // -- Array literal --
    /// An array literal (e.g., `[1, 2, 3]`).
    ArrayLit(Vec<HirExpr>),

    // -- Col DSL --
    /// A column reference in the data DSL (e.g., `$col_name`).
    Col(String),

    // -- Lambda (no captures -- kept for backward compat) --
    /// A lambda expression with no captured variables.
    ///
    /// Lambdas that capture outer variables are represented as
    /// [`Closure`](HirExprKind::Closure) instead.
    Lambda {
        /// Lambda parameters.
        params: Vec<HirParam>,
        /// Lambda body expression.
        body: Box<HirExpr>,
    },

    // -- Closure (lambda with analyzed captures) --
    /// A closure: a lambda that captures variables from enclosing scopes.
    ///
    /// Capture analysis determines which outer variables are referenced and
    /// whether they are captured by reference or by clone (inside `nogc` blocks).
    Closure {
        /// Closure parameters.
        params: Vec<HirParam>,
        /// Closure body expression.
        body: Box<HirExpr>,
        /// Variables captured from enclosing scopes.
        captures: Vec<HirCapture>,
    },

    // -- Match expression --
    /// A `match` expression with a scrutinee and pattern-guarded arms.
    Match {
        /// The expression being matched against.
        scrutinee: Box<HirExpr>,
        /// Match arms, each with a pattern and body.
        arms: Vec<HirMatchArm>,
    },

    // -- Tuple literal --
    /// A tuple literal (e.g., `(1, "hello", true)`).
    TupleLit(Vec<HirExpr>),

    // -- Enum variant literal --
    /// An enum variant constructor (e.g., `Some(42)`, `None`).
    VariantLit {
        /// The enum type name (e.g., `"Option"`).
        enum_name: String,
        /// The variant name (e.g., `"Some"`).
        variant: String,
        /// Positional field values (empty for unit variants).
        fields: Vec<HirExpr>,
    },

    // -- If expression (produces a value) --
    /// An `if` expression that produces a value from the taken branch.
    ///
    /// Used when `if` appears in expression position (e.g.,
    /// `let x = if cond { a } else { b };`).
    If {
        /// The condition expression.
        cond: Box<HirExpr>,
        /// The `then` branch block.
        then_block: HirBlock,
        /// Optional `else`/`else if` branch.
        else_branch: Option<HirElseBranch>,
    },

    // -- Void (for empty else branches, etc.) --
    /// A void expression representing no value (used for empty branches
    /// and lowered import statements).
    Void,
}

// ---------------------------------------------------------------------------
// Match / Pattern types
// ---------------------------------------------------------------------------

/// A single arm of a `match` expression in HIR.
///
/// Each arm consists of a pattern to match against the scrutinee and a body
/// expression to evaluate if the pattern matches.
#[derive(Debug, Clone)]
pub struct HirMatchArm {
    /// The pattern to match against.
    pub pattern: HirPattern,
    /// The body expression evaluated when this arm matches.
    pub body: HirExpr,
    /// Unique HIR node identifier.
    pub hir_id: HirId,
}

/// A pattern in HIR for use in match arms and destructuring.
///
/// Wraps a [`HirPatternKind`] with a unique [`HirId`].
#[derive(Debug, Clone)]
pub struct HirPattern {
    /// The kind of pattern.
    pub kind: HirPatternKind,
    /// Unique HIR node identifier.
    pub hir_id: HirId,
}

/// The kind of a HIR pattern, used in match arms and destructuring.
#[derive(Debug, Clone)]
pub enum HirPatternKind {
    /// Wildcard pattern `_`: matches any value without binding.
    Wildcard,
    /// Binding pattern `x`: matches any value and binds it to the given name.
    Binding(String),
    /// Integer literal pattern (e.g., `42`).
    LitInt(i64),
    /// Float literal pattern (e.g., `3.14`).
    LitFloat(f64),
    /// Boolean literal pattern (`true` or `false`).
    LitBool(bool),
    /// String literal pattern (e.g., `"hello"`).
    LitString(String),
    /// Tuple destructuring pattern (e.g., `(a, b, c)`).
    Tuple(Vec<HirPattern>),
    /// Struct destructuring pattern (e.g., `Point { x, y }`).
    Struct {
        /// The struct type name.
        name: String,
        /// Field patterns for destructuring.
        fields: Vec<HirPatternField>,
    },
    /// Enum variant pattern (e.g., `Some(x)`, `None`, `Ok(v)`, `Err(e)`).
    Variant {
        /// The enum type name (e.g., `"Option"`, `"Result"`).
        enum_name: String,
        /// The variant name (e.g., `"Some"`, `"None"`).
        variant: String,
        /// Sub-patterns for the variant's fields (empty for unit variants).
        fields: Vec<HirPattern>,
    },
}

/// A field inside a struct destructuring pattern in HIR.
///
/// Supports both explicit patterns (`Point { x: px }`) and shorthand
/// (`Point { x }`, which desugars to `Point { x: x }`).
#[derive(Debug, Clone)]
pub struct HirPatternField {
    /// The field name being matched.
    pub name: String,
    /// The sub-pattern to match against this field's value.
    pub pattern: HirPattern,
    /// Unique HIR node identifier.
    pub hir_id: HirId,
}

// ---------------------------------------------------------------------------
// Capture analysis types
// ---------------------------------------------------------------------------

/// How a variable is captured by a closure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureMode {
    /// Immutable reference to the outer variable (default).
    Ref,
    /// Deep copy of the value (forced inside `nogc` blocks).
    Clone,
}

/// A single captured variable with its name and capture mode.
///
/// Produced during closure capture analysis in [`AstLowering`].
#[derive(Debug, Clone)]
pub struct HirCapture {
    /// The name of the captured variable.
    pub name: String,
    /// How this variable is captured (by reference or by clone).
    pub mode: CaptureMode,
    /// Unique HIR node identifier.
    pub hir_id: HirId,
}

// ---------------------------------------------------------------------------
// If expression
// ---------------------------------------------------------------------------

/// An `if`/`else if`/`else` chain in statement position.
///
/// Used by [`HirStmtKind::If`]. For `if` in expression position, see
/// [`HirExprKind::If`].
#[derive(Debug, Clone)]
pub struct HirIfExpr {
    /// The condition expression.
    pub cond: Box<HirExpr>,
    /// The `then` branch block.
    pub then_block: HirBlock,
    /// Optional `else` or `else if` continuation.
    pub else_branch: Option<HirElseBranch>,
    /// Unique HIR node identifier.
    pub hir_id: HirId,
}

/// The else branch of an `if` statement or expression.
#[derive(Debug, Clone)]
pub enum HirElseBranch {
    /// An `else if` continuation, chaining another conditional.
    ElseIf(Box<HirIfExpr>),
    /// A terminal `else` block.
    Else(HirBlock),
}

// ===========================================================================
// AST -> HIR Lowering
// ===========================================================================

use std::collections::BTreeSet;

use cjc_ast;

/// Information about a variable in scope.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ScopeVar {
    /// Whether the variable was declared `mut`.
    mutable: bool,
}

/// Lowers an AST `Program` into a HIR `HirProgram`.
///
/// Tracks lexical scopes for capture analysis and `nogc` context
/// for determining capture modes.
pub struct AstLowering {
    next_id: u32,
    /// Stack of scopes. Each scope maps variable names to their info.
    scopes: Vec<BTreeMap<String, ScopeVar>>,
    /// Set of known function names (direct calls, not captures).
    known_functions: BTreeSet<String>,
    /// Whether we are currently inside a `nogc` context.
    in_nogc: bool,
    /// Maps variant name → enum name for variant resolution.
    variant_names: BTreeMap<String, String>,
}

use std::collections::BTreeMap;

impl AstLowering {
    /// Creates a new [`AstLowering`] instance with prelude variant names
    /// (`Some`, `None`, `Ok`, `Err`) pre-registered for enum variant resolution.
    pub fn new() -> Self {
        let mut variant_names = BTreeMap::new();
        // Register prelude variant names
        variant_names.insert("Some".into(), "Option".into());
        variant_names.insert("None".into(), "Option".into());
        variant_names.insert("Ok".into(), "Result".into());
        variant_names.insert("Err".into(), "Result".into());

        Self {
            next_id: 0,
            scopes: vec![BTreeMap::new()],
            known_functions: BTreeSet::new(),
            in_nogc: false,
            variant_names,
        }
    }

    fn fresh_id(&mut self) -> HirId {
        let id = HirId(self.next_id);
        self.next_id += 1;
        id
    }

    // -- Scope tracking for capture analysis --

    fn push_scope(&mut self) {
        self.scopes.push(BTreeMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn define_var(&mut self, name: &str, mutable: bool) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), ScopeVar { mutable });
        }
    }

    /// Check if a variable is defined in any enclosing scope.
    /// (Used by future mutable-capture error checking.)
    #[allow(dead_code)]
    fn is_defined(&self, name: &str) -> bool {
        self.scopes.iter().rev().any(|s| s.contains_key(name))
    }

    /// Check if a variable is mutable in its defining scope.
    /// (Used by future mutable-capture error checking.)
    #[allow(dead_code)]
    fn is_mutable(&self, name: &str) -> bool {
        for scope in self.scopes.iter().rev() {
            if let Some(var) = scope.get(name) {
                return var.mutable;
            }
        }
        false
    }

    /// Builtin names that should not be treated as captures.
    fn is_builtin(name: &str) -> bool {
        matches!(
            name,
            "print"
                | "Tensor"
                | "matmul"
                | "Buffer"
                | "len"
                | "push"
                | "assert"
                | "assert_eq"
                | "clock"
                | "gc_alloc"
                | "gc_collect"
                | "gc_live_count"
                | "true"
                | "false"
                | "Some"
                | "None"
                | "Ok"
                | "Err"
                | "bf16_to_f32"
                | "f32_to_bf16"
        )
    }

    /// Lowers a complete AST [`Program`](cjc_ast::Program) into a [`HirProgram`].
    ///
    /// Performs a pre-scan to register all top-level function names and enum
    /// variant names, then lowers each declaration in order. This ensures that
    /// function names are not mistakenly treated as captured variables inside
    /// closures, and that variant names resolve correctly.
    ///
    /// # Arguments
    ///
    /// * `program` - The AST program to lower.
    ///
    /// # Returns
    ///
    /// A fully lowered [`HirProgram`] with all syntactic sugar removed.
    pub fn lower_program(&mut self, program: &cjc_ast::Program) -> HirProgram {
        // Pre-scan: register all top-level function names so they aren't
        // treated as captured variables inside closures.
        // Also register enum variant names for resolution.
        for decl in &program.declarations {
            match &decl.kind {
                cjc_ast::DeclKind::Fn(f) => {
                    self.known_functions.insert(f.name.name.clone());
                }
                cjc_ast::DeclKind::Impl(i) => {
                    for m in &i.methods {
                        let target = match &i.target.kind {
                            cjc_ast::TypeExprKind::Named { name, .. } => name.name.clone(),
                            _ => continue,
                        };
                        self.known_functions
                            .insert(format!("{}.{}", target, m.name.name));
                    }
                }
                cjc_ast::DeclKind::Enum(e) => {
                    for v in &e.variants {
                        self.variant_names
                            .insert(v.name.name.clone(), e.name.name.clone());
                    }
                }
                _ => {}
            }
        }

        let items = program
            .declarations
            .iter()
            .map(|d| self.lower_decl(d))
            .collect();
        HirProgram { items }
    }

    fn lower_decl(&mut self, decl: &cjc_ast::Decl) -> HirItem {
        match &decl.kind {
            cjc_ast::DeclKind::Fn(f) => HirItem::Fn(self.lower_fn_decl(f)),
            cjc_ast::DeclKind::Struct(s) => HirItem::Struct(self.lower_struct(s)),
            cjc_ast::DeclKind::Class(c) => HirItem::Class(self.lower_class(c)),
            cjc_ast::DeclKind::Record(r) => HirItem::Record(self.lower_record(r)),
            cjc_ast::DeclKind::Trait(t) => HirItem::Trait(self.lower_trait(t)),
            cjc_ast::DeclKind::Impl(i) => HirItem::Impl(self.lower_impl(i)),
            cjc_ast::DeclKind::Let(l) => HirItem::Let(self.lower_let_decl(l)),
            cjc_ast::DeclKind::Import(_) => {
                // Imports are no-ops in the interpreter
                let id = self.fresh_id();
                HirItem::Stmt(HirStmt {
                    kind: HirStmtKind::Expr(HirExpr {
                        kind: HirExprKind::Void,
                        hir_id: id,
                    }),
                    hir_id: id,
                })
            }
            cjc_ast::DeclKind::Stmt(s) => HirItem::Stmt(self.lower_stmt(s)),
            cjc_ast::DeclKind::Enum(e) => HirItem::Enum(self.lower_enum(e)),
            // P2-3: Const declarations are lowered as immutable let bindings.
            cjc_ast::DeclKind::Const(c) => {
                let let_stmt = cjc_ast::LetStmt {
                    name: c.name.clone(),
                    mutable: false,
                    ty: Some(c.ty.clone()),
                    init: c.value.clone(),
                };
                HirItem::Let(self.lower_let_decl(&let_stmt))
            }
        }
    }

    /// Lowers an AST function declaration into a [`HirFn`].
    ///
    /// Pushes a new scope for the function body, registers parameters as
    /// local variables, lowers the body block, and restores the `nogc` context
    /// afterward.
    ///
    /// # Arguments
    ///
    /// * `f` - The AST function declaration to lower.
    ///
    /// # Returns
    ///
    /// A lowered [`HirFn`] with desugared body, parameters, and decorators.
    pub fn lower_fn_decl(&mut self, f: &cjc_ast::FnDecl) -> HirFn {
        let hir_id = self.fresh_id();
        let old_nogc = self.in_nogc;
        if f.is_nogc {
            self.in_nogc = true;
        }
        self.push_scope();
        // Register params in scope
        for p in &f.params {
            self.define_var(&p.name.name, false);
        }
        let type_params: Vec<(String, Vec<String>)> = f
            .type_params
            .iter()
            .map(|tp| {
                (
                    tp.name.name.clone(),
                    tp.bounds.iter().map(|b| self.type_expr_to_string(b)).collect(),
                )
            })
            .collect();
        let params = f.params.iter().map(|p| self.lower_param(p)).collect();
        let return_type = f.return_type.as_ref().map(|t| self.type_expr_to_string(t));
        let body = self.lower_block(&f.body);
        self.pop_scope();
        self.in_nogc = old_nogc;
        HirFn {
            name: f.name.name.clone(),
            type_params,
            params,
            return_type,
            body,
            is_nogc: f.is_nogc,
            hir_id,
            decorators: f.decorators.iter().map(|d| d.name.name.clone()).collect(),
            vis: f.vis,
        }
    }

    fn lower_enum(&mut self, e: &cjc_ast::EnumDecl) -> HirEnumDef {
        let hir_id = self.fresh_id();
        let type_params = e.type_params.iter().map(|p| p.name.name.clone()).collect();
        let variants = e
            .variants
            .iter()
            .map(|v| HirVariantDef {
                name: v.name.name.clone(),
                fields: v.fields.iter().map(|f| self.type_expr_to_string(f)).collect(),
            })
            .collect();
        HirEnumDef {
            name: e.name.name.clone(),
            type_params,
            variants,
            hir_id,
        }
    }

    fn lower_param(&mut self, p: &cjc_ast::Param) -> HirParam {
        let hir_id = self.fresh_id();
        let default = p.default.as_ref().map(|d| self.lower_expr(d));
        HirParam {
            name: p.name.name.clone(),
            ty_name: self.type_expr_to_string(&p.ty),
            default,
            is_variadic: p.is_variadic,
            hir_id,
        }
    }

    fn lower_struct(&mut self, s: &cjc_ast::StructDecl) -> HirStructDef {
        let hir_id = self.fresh_id();
        let fields = s
            .fields
            .iter()
            .map(|f| (f.name.name.clone(), self.type_expr_to_string(&f.ty)))
            .collect();
        HirStructDef {
            name: s.name.name.clone(),
            fields,
            hir_id,
            vis: s.vis,
        }
    }

    fn lower_class(&mut self, c: &cjc_ast::ClassDecl) -> HirClassDef {
        let hir_id = self.fresh_id();
        let fields = c
            .fields
            .iter()
            .map(|f| (f.name.name.clone(), self.type_expr_to_string(&f.ty)))
            .collect();
        HirClassDef {
            name: c.name.name.clone(),
            fields,
            hir_id,
            vis: c.vis,
        }
    }

    fn lower_record(&mut self, r: &cjc_ast::RecordDecl) -> HirRecordDef {
        let hir_id = self.fresh_id();
        let fields = r
            .fields
            .iter()
            .map(|f| (f.name.name.clone(), self.type_expr_to_string(&f.ty)))
            .collect();
        HirRecordDef {
            name: r.name.name.clone(),
            fields,
            hir_id,
            vis: r.vis,
        }
    }

    fn lower_trait(&mut self, t: &cjc_ast::TraitDecl) -> HirTraitDef {
        let hir_id = self.fresh_id();
        let methods = t
            .methods
            .iter()
            .map(|m| HirFnSig {
                name: m.name.name.clone(),
                params: m.params.iter().map(|p| self.lower_param(p)).collect(),
                return_type: m.return_type.as_ref().map(|t| self.type_expr_to_string(t)),
            })
            .collect();
        HirTraitDef {
            name: t.name.name.clone(),
            methods,
            hir_id,
        }
    }

    fn lower_impl(&mut self, i: &cjc_ast::ImplDecl) -> HirImplDef {
        let hir_id = self.fresh_id();
        let target = self.type_expr_to_string(&i.target);
        let trait_ref = i.trait_ref.as_ref().map(|t| self.type_expr_to_string(t));
        let methods = i.methods.iter().map(|m| self.lower_fn_decl(m)).collect();
        HirImplDef {
            target,
            trait_ref,
            methods,
            hir_id,
        }
    }

    fn lower_let_decl(&mut self, l: &cjc_ast::LetStmt) -> HirLetDecl {
        let hir_id = self.fresh_id();
        let ty_name = l.ty.as_ref().map(|t| self.type_expr_to_string(t));
        let init = self.lower_expr(&l.init);
        // Register the variable in the current scope
        self.define_var(&l.name.name, l.mutable);
        HirLetDecl {
            name: l.name.name.clone(),
            mutable: l.mutable,
            ty_name,
            init,
            hir_id,
        }
    }

    fn lower_block(&mut self, block: &cjc_ast::Block) -> HirBlock {
        let hir_id = self.fresh_id();
        let stmts = block.stmts.iter().map(|s| self.lower_stmt(s)).collect();
        let expr = block.expr.as_ref().map(|e| Box::new(self.lower_expr(e)));
        HirBlock {
            stmts,
            expr,
            hir_id,
        }
    }

    /// Lowers an AST statement into a [`HirStmt`].
    ///
    /// Handles all statement kinds including `let`, `if`, `while`, `for`,
    /// `return`, `break`, `continue`, and `nogc` blocks. For-loops are
    /// desugared into while-loops via [`desugar_for`](Self::desugar_for).
    ///
    /// # Arguments
    ///
    /// * `stmt` - The AST statement to lower.
    ///
    /// # Returns
    ///
    /// A lowered [`HirStmt`] with all syntactic sugar removed.
    pub fn lower_stmt(&mut self, stmt: &cjc_ast::Stmt) -> HirStmt {
        let hir_id = self.fresh_id();
        let kind = match &stmt.kind {
            cjc_ast::StmtKind::Let(l) => {
                let init = self.lower_expr(&l.init);
                // Register the variable in the current scope
                self.define_var(&l.name.name, l.mutable);
                HirStmtKind::Let {
                    name: l.name.name.clone(),
                    mutable: l.mutable,
                    ty_name: l.ty.as_ref().map(|t| self.type_expr_to_string(t)),
                    init,
                }
            }
            cjc_ast::StmtKind::Expr(e) => HirStmtKind::Expr(self.lower_expr(e)),
            cjc_ast::StmtKind::Return(e) => {
                HirStmtKind::Return(e.as_ref().map(|ex| self.lower_expr(ex)))
            }
            cjc_ast::StmtKind::Break => HirStmtKind::Break,
            cjc_ast::StmtKind::Continue => HirStmtKind::Continue,
            cjc_ast::StmtKind::If(if_stmt) => HirStmtKind::If(self.lower_if(if_stmt)),
            cjc_ast::StmtKind::While(w) => HirStmtKind::While {
                cond: self.lower_expr(&w.condition),
                body: self.lower_block(&w.body),
            },
            cjc_ast::StmtKind::For(f) => {
                return self.desugar_for(f);
            }
            cjc_ast::StmtKind::NoGcBlock(block) => {
                let old_nogc = self.in_nogc;
                self.in_nogc = true;
                let result = HirStmtKind::NoGcBlock(self.lower_block(block));
                self.in_nogc = old_nogc;
                result
            }
        };
        HirStmt { kind, hir_id }
    }

    /// Counter for generating unique hygienic names that cannot collide with
    /// user identifiers (double-underscore prefix + monotonic index).
    fn gensym(&mut self, prefix: &str) -> String {
        let id = self.next_id;
        // next_id is bumped by fresh_id, but we need a stable counter
        // separate from HirId for naming; reuse next_id as it's always unique.
        format!("__for_{}_{}", prefix, id)
    }

    /// Desugar a `for` statement into HIR statements wrapped in a block.
    ///
    /// Range form: `for i in start..end { body }` →
    /// ```text
    /// {
    ///     let mut __for_end_N = end;
    ///     let mut __for_idx_N = start;
    ///     while __for_idx_N < __for_end_N {
    ///         let i = __for_idx_N;
    ///         <body>
    ///         __for_idx_N = __for_idx_N + 1;
    ///     }
    /// }
    /// ```
    ///
    /// Expr form: `for x in arr { body }` →
    /// ```text
    /// {
    ///     let __for_arr_N = arr;
    ///     let __for_len_N = len(__for_arr_N);
    ///     let mut __for_idx_N = 0;
    ///     while __for_idx_N < __for_len_N {
    ///         let x = __for_arr_N[__for_idx_N];
    ///         <body>
    ///         __for_idx_N = __for_idx_N + 1;
    ///     }
    /// }
    /// ```
    fn desugar_for(&mut self, f: &cjc_ast::ForStmt) -> HirStmt {
        let outer_id = self.fresh_id();
        let loop_var_name = f.ident.name.clone();

        match &f.iter {
            cjc_ast::ForIter::Range { start, end } => {
                self.desugar_for_range(&loop_var_name, start, end, &f.body, outer_id)
            }
            cjc_ast::ForIter::Expr(expr) => {
                self.desugar_for_expr(&loop_var_name, expr, &f.body, outer_id)
            }
        }
    }

    fn desugar_for_range(
        &mut self,
        loop_var: &str,
        start: &cjc_ast::Expr,
        end: &cjc_ast::Expr,
        body: &cjc_ast::Block,
        outer_id: HirId,
    ) -> HirStmt {
        let end_name = self.gensym("end");
        let idx_name = self.gensym("idx");

        self.push_scope();

        // let __for_end_N = end;
        let end_init = self.lower_expr(end);
        let let_end_id = self.fresh_id();
        self.define_var(&end_name, false);
        let let_end = HirStmt {
            kind: HirStmtKind::Let {
                name: end_name.clone(),
                mutable: false,
                ty_name: None,
                init: end_init,
            },
            hir_id: let_end_id,
        };

        // let mut __for_idx_N = start;
        let start_init = self.lower_expr(start);
        let let_idx_id = self.fresh_id();
        self.define_var(&idx_name, true);
        let let_idx = HirStmt {
            kind: HirStmtKind::Let {
                name: idx_name.clone(),
                mutable: true,
                ty_name: None,
                init: start_init,
            },
            hir_id: let_idx_id,
        };

        // while __for_idx_N < __for_end_N { let i = __for_idx_N; body; __for_idx_N = __for_idx_N + 1; }
        let while_stmt = self.build_while_with_counter(
            loop_var,
            &idx_name,
            &end_name,
            body,
            None, // no array to index into
        );

        let block = HirBlock {
            stmts: vec![let_end, let_idx, while_stmt],
            expr: None,
            hir_id: self.fresh_id(),
        };

        self.pop_scope();

        // Wrap the block as an expression statement
        HirStmt {
            kind: HirStmtKind::Expr(HirExpr {
                kind: HirExprKind::Block(block),
                hir_id: self.fresh_id(),
            }),
            hir_id: outer_id,
        }
    }

    fn desugar_for_expr(
        &mut self,
        loop_var: &str,
        expr: &cjc_ast::Expr,
        body: &cjc_ast::Block,
        outer_id: HirId,
    ) -> HirStmt {
        let arr_name = self.gensym("arr");
        let len_name = self.gensym("len");
        let idx_name = self.gensym("idx");

        self.push_scope();

        // let __for_arr_N = expr;
        let arr_init = self.lower_expr(expr);
        let let_arr_id = self.fresh_id();
        self.define_var(&arr_name, false);
        let let_arr = HirStmt {
            kind: HirStmtKind::Let {
                name: arr_name.clone(),
                mutable: false,
                ty_name: None,
                init: arr_init,
            },
            hir_id: let_arr_id,
        };

        // let __for_len_N = len(__for_arr_N);
        let len_call = HirExpr {
            kind: HirExprKind::Call {
                callee: Box::new(HirExpr {
                    kind: HirExprKind::Var("len".to_string()),
                    hir_id: self.fresh_id(),
                }),
                args: vec![HirExpr {
                    kind: HirExprKind::Var(arr_name.clone()),
                    hir_id: self.fresh_id(),
                }],
            },
            hir_id: self.fresh_id(),
        };
        let let_len_id = self.fresh_id();
        self.define_var(&len_name, false);
        let let_len = HirStmt {
            kind: HirStmtKind::Let {
                name: len_name.clone(),
                mutable: false,
                ty_name: None,
                init: len_call,
            },
            hir_id: let_len_id,
        };

        // let mut __for_idx_N = 0;
        let let_idx_id = self.fresh_id();
        self.define_var(&idx_name, true);
        let let_idx = HirStmt {
            kind: HirStmtKind::Let {
                name: idx_name.clone(),
                mutable: true,
                ty_name: None,
                init: HirExpr {
                    kind: HirExprKind::IntLit(0),
                    hir_id: self.fresh_id(),
                },
            },
            hir_id: let_idx_id,
        };

        // while __for_idx_N < __for_len_N { let x = __for_arr_N[__for_idx_N]; body; __for_idx_N = __for_idx_N + 1; }
        let while_stmt = self.build_while_with_counter(
            loop_var,
            &idx_name,
            &len_name,
            body,
            Some(&arr_name),
        );

        let block = HirBlock {
            stmts: vec![let_arr, let_len, let_idx, while_stmt],
            expr: None,
            hir_id: self.fresh_id(),
        };

        self.pop_scope();

        HirStmt {
            kind: HirStmtKind::Expr(HirExpr {
                kind: HirExprKind::Block(block),
                hir_id: self.fresh_id(),
            }),
            hir_id: outer_id,
        }
    }

    /// Build the inner while loop shared by both range and expr iteration.
    ///
    /// - `loop_var`: the user-visible variable name (e.g., `i` or `x`)
    /// - `idx_name`: the hygienic counter variable
    /// - `bound_name`: the hygienic upper bound variable
    /// - `body`: the user's loop body (AST)
    /// - `arr_name`: if `Some`, the loop variable is bound via indexing
    ///   (`let x = arr[idx]`); if `None`, the loop variable is bound
    ///   directly from the counter (`let i = idx`).
    fn build_while_with_counter(
        &mut self,
        loop_var: &str,
        idx_name: &str,
        bound_name: &str,
        body: &cjc_ast::Block,
        arr_name: Option<&str>,
    ) -> HirStmt {
        let while_id = self.fresh_id();

        // Condition: __for_idx_N < bound_name
        let cond = HirExpr {
            kind: HirExprKind::Binary {
                op: cjc_ast::BinOp::Lt,
                left: Box::new(HirExpr {
                    kind: HirExprKind::Var(idx_name.to_string()),
                    hir_id: self.fresh_id(),
                }),
                right: Box::new(HirExpr {
                    kind: HirExprKind::Var(bound_name.to_string()),
                    hir_id: self.fresh_id(),
                }),
            },
            hir_id: self.fresh_id(),
        };

        // Build the while body
        self.push_scope();

        // let loop_var = <value>;
        let loop_var_init = if let Some(arr) = arr_name {
            // let x = __for_arr_N[__for_idx_N];
            HirExpr {
                kind: HirExprKind::Index {
                    object: Box::new(HirExpr {
                        kind: HirExprKind::Var(arr.to_string()),
                        hir_id: self.fresh_id(),
                    }),
                    index: Box::new(HirExpr {
                        kind: HirExprKind::Var(idx_name.to_string()),
                        hir_id: self.fresh_id(),
                    }),
                },
                hir_id: self.fresh_id(),
            }
        } else {
            // let i = __for_idx_N;
            HirExpr {
                kind: HirExprKind::Var(idx_name.to_string()),
                hir_id: self.fresh_id(),
            }
        };

        self.define_var(loop_var, false);
        let let_loop_var_id = self.fresh_id();
        let let_loop_var = HirStmt {
            kind: HirStmtKind::Let {
                name: loop_var.to_string(),
                mutable: false,
                ty_name: None,
                init: loop_var_init,
            },
            hir_id: let_loop_var_id,
        };

        // Lower the user's body statements
        // Increment: __for_idx_N = __for_idx_N + 1;
        // Placed BEFORE the user body so that `continue` in user code cannot
        // skip the increment (which would cause an infinite loop in desugared
        // for-loops).  The loop variable `i` already captured its value via
        // `let i = __for_idx_N`, so pre-incrementing is correct.
        let increment = HirStmt {
            kind: HirStmtKind::Expr(HirExpr {
                kind: HirExprKind::Assign {
                    target: Box::new(HirExpr {
                        kind: HirExprKind::Var(idx_name.to_string()),
                        hir_id: self.fresh_id(),
                    }),
                    value: Box::new(HirExpr {
                        kind: HirExprKind::Binary {
                            op: cjc_ast::BinOp::Add,
                            left: Box::new(HirExpr {
                                kind: HirExprKind::Var(idx_name.to_string()),
                                hir_id: self.fresh_id(),
                            }),
                            right: Box::new(HirExpr {
                                kind: HirExprKind::IntLit(1),
                                hir_id: self.fresh_id(),
                            }),
                        },
                        hir_id: self.fresh_id(),
                    }),
                },
                hir_id: self.fresh_id(),
            }),
            hir_id: self.fresh_id(),
        };

        let mut while_body_stmts = vec![let_loop_var, increment];
        for stmt in &body.stmts {
            while_body_stmts.push(self.lower_stmt(stmt));
        }

        // Lower the tail expression if any
        let tail_expr = body.expr.as_ref().map(|e| Box::new(self.lower_expr(e)));

        self.pop_scope();

        let while_body = HirBlock {
            stmts: while_body_stmts,
            expr: tail_expr,
            hir_id: self.fresh_id(),
        };

        HirStmt {
            kind: HirStmtKind::While {
                cond,
                body: while_body,
            },
            hir_id: while_id,
        }
    }

    /// Lowers an AST `if` statement into a [`HirIfExpr`].
    ///
    /// Recursively lowers `else if` chains and terminal `else` blocks.
    ///
    /// # Arguments
    ///
    /// * `if_stmt` - The AST if-statement to lower.
    ///
    /// # Returns
    ///
    /// A lowered [`HirIfExpr`] with condition, then-block, and optional else branch.
    pub fn lower_if(&mut self, if_stmt: &cjc_ast::IfStmt) -> HirIfExpr {
        let hir_id = self.fresh_id();
        let cond = Box::new(self.lower_expr(&if_stmt.condition));
        let then_block = self.lower_block(&if_stmt.then_block);
        let else_branch = if_stmt.else_branch.as_ref().map(|eb| match eb {
            cjc_ast::ElseBranch::ElseIf(elif) => {
                HirElseBranch::ElseIf(Box::new(self.lower_if(elif)))
            }
            cjc_ast::ElseBranch::Else(block) => HirElseBranch::Else(self.lower_block(block)),
        });
        HirIfExpr {
            cond,
            then_block,
            else_branch,
            hir_id,
        }
    }

    /// Lowers an AST expression into a [`HirExpr`].
    ///
    /// Performs all expression-level desugaring:
    /// - Pipe `a |> f(b)` becomes `Call(f, [a, b])`
    /// - Compound assignment `x += 1` becomes `x = x + 1`
    /// - Cast `expr as f64` becomes a builtin function call
    /// - F-string `f"...{expr}..."` becomes string concatenation
    /// - `expr?` (try) becomes a match on `Ok`/`Err`
    /// - Lambda expressions undergo capture analysis to produce
    ///   [`Lambda`](HirExprKind::Lambda) or [`Closure`](HirExprKind::Closure)
    /// - Identifiers matching known enum variants resolve to
    ///   [`VariantLit`](HirExprKind::VariantLit)
    ///
    /// # Arguments
    ///
    /// * `expr` - The AST expression to lower.
    ///
    /// # Returns
    ///
    /// A lowered [`HirExpr`] with all syntactic sugar removed.
    pub fn lower_expr(&mut self, expr: &cjc_ast::Expr) -> HirExpr {
        let hir_id = self.fresh_id();
        let kind = match &expr.kind {
            cjc_ast::ExprKind::IntLit(v) => HirExprKind::IntLit(*v),
            cjc_ast::ExprKind::FloatLit(v) => HirExprKind::FloatLit(*v),
            cjc_ast::ExprKind::StringLit(s) => HirExprKind::StringLit(s.clone()),
            cjc_ast::ExprKind::ByteStringLit(bytes) => HirExprKind::ByteStringLit(bytes.clone()),
            cjc_ast::ExprKind::ByteCharLit(b) => HirExprKind::ByteCharLit(*b),
            cjc_ast::ExprKind::RawStringLit(s) => HirExprKind::RawStringLit(s.clone()),
            cjc_ast::ExprKind::RawByteStringLit(bytes) => HirExprKind::RawByteStringLit(bytes.clone()),
            cjc_ast::ExprKind::FStringLit(segments) => {
                // P2-5: Desugar f"...{expr}..." into string concatenation.
                // Build a chain of BinOp::Add(Str, Str) expressions.
                let mut parts: Vec<HirExpr> = Vec::new();
                for (lit, interp) in segments {
                    if !lit.is_empty() {
                        parts.push(HirExpr {
                            kind: HirExprKind::StringLit(lit.clone()),
                            hir_id: self.fresh_id(),
                        });
                    }
                    if let Some(e) = interp {
                        // Wrap the interpolated expr in a to_string() call
                        let inner = self.lower_expr(e);
                        parts.push(HirExpr {
                            hir_id: self.fresh_id(),
                            kind: HirExprKind::Call {
                                callee: Box::new(HirExpr {
                                    hir_id: self.fresh_id(),
                                    kind: HirExprKind::Var("to_string".to_string()),
                                }),
                                args: vec![inner],
                            },
                        });
                    }
                }
                if parts.is_empty() {
                    HirExprKind::StringLit(String::new())
                } else {
                    let mut iter = parts.into_iter();
                    let first = iter.next().unwrap();
                    let folded = iter.fold(first, |acc, part| HirExpr {
                        hir_id: self.fresh_id(),
                        kind: HirExprKind::Binary {
                            op: cjc_ast::BinOp::Add,
                            left: Box::new(acc),
                            right: Box::new(part),
                        },
                    });
                    folded.kind
                }
            }
            cjc_ast::ExprKind::RegexLit { pattern, flags } => HirExprKind::RegexLit { pattern: pattern.clone(), flags: flags.clone() },
            cjc_ast::ExprKind::TensorLit { rows } => {
                let hir_rows = rows.iter().map(|row| {
                    row.iter().map(|e| self.lower_expr(e)).collect()
                }).collect();
                HirExprKind::TensorLit { rows: hir_rows }
            }
            cjc_ast::ExprKind::BoolLit(b) => HirExprKind::BoolLit(*b),
            cjc_ast::ExprKind::NaLit => HirExprKind::NaLit,
            cjc_ast::ExprKind::Ident(id) => {
                // Check if this identifier is a unit variant (like None)
                if let Some(enum_name) = self.variant_names.get(&id.name).cloned() {
                    // Only resolve as unit variant if it's not also a local variable
                    if !self.is_defined(&id.name) {
                        HirExprKind::VariantLit {
                            enum_name,
                            variant: id.name.clone(),
                            fields: vec![],
                        }
                    } else {
                        HirExprKind::Var(id.name.clone())
                    }
                } else {
                    HirExprKind::Var(id.name.clone())
                }
            }

            cjc_ast::ExprKind::Binary { op, left, right } => HirExprKind::Binary {
                op: *op,
                left: Box::new(self.lower_expr(left)),
                right: Box::new(self.lower_expr(right)),
            },

            cjc_ast::ExprKind::Unary { op, operand } => HirExprKind::Unary {
                op: *op,
                operand: Box::new(self.lower_expr(operand)),
            },

            cjc_ast::ExprKind::Call { callee, args } => {
                // Check if callee is an identifier that matches a known variant name
                if let cjc_ast::ExprKind::Ident(id) = &callee.kind {
                    if let Some(enum_name) = self.variant_names.get(&id.name).cloned() {
                        // Resolve to VariantLit
                        let fields = args.iter().map(|a| self.lower_expr(&a.value)).collect();
                        HirExprKind::VariantLit {
                            enum_name,
                            variant: id.name.clone(),
                            fields,
                        }
                    } else {
                        let hir_callee = Box::new(self.lower_expr(callee));
                        let hir_args = args.iter().map(|a| self.lower_expr(&a.value)).collect();
                        HirExprKind::Call {
                            callee: hir_callee,
                            args: hir_args,
                        }
                    }
                } else {
                    let hir_callee = Box::new(self.lower_expr(callee));
                    let hir_args = args.iter().map(|a| self.lower_expr(&a.value)).collect();
                    HirExprKind::Call {
                        callee: hir_callee,
                        args: hir_args,
                    }
                }
            }

            cjc_ast::ExprKind::Field { object, name } => HirExprKind::Field {
                object: Box::new(self.lower_expr(object)),
                name: name.name.clone(),
            },

            cjc_ast::ExprKind::Index { object, index } => HirExprKind::Index {
                object: Box::new(self.lower_expr(object)),
                index: Box::new(self.lower_expr(index)),
            },

            cjc_ast::ExprKind::MultiIndex { object, indices } => HirExprKind::MultiIndex {
                object: Box::new(self.lower_expr(object)),
                indices: indices.iter().map(|i| self.lower_expr(i)).collect(),
            },

            cjc_ast::ExprKind::Assign { target, value } => HirExprKind::Assign {
                target: Box::new(self.lower_expr(target)),
                value: Box::new(self.lower_expr(value)),
            },

            // KEY DESUGARING: Pipe `a |> f(b)` -> `Call(f, [a, b])`
            cjc_ast::ExprKind::Pipe { left, right } => {
                let lhs = self.lower_expr(left);
                // The right side should be a Call; we prepend the left as first arg
                match &right.kind {
                    cjc_ast::ExprKind::Call { callee, args } => {
                        let hir_callee = Box::new(self.lower_expr(callee));
                        let mut hir_args = vec![lhs];
                        for a in args {
                            hir_args.push(self.lower_expr(&a.value));
                        }
                        HirExprKind::Call {
                            callee: hir_callee,
                            args: hir_args,
                        }
                    }
                    // If right is just an identifier `a |> f`, treat as `f(a)`
                    cjc_ast::ExprKind::Ident(_) => {
                        let hir_callee = Box::new(self.lower_expr(right));
                        HirExprKind::Call {
                            callee: hir_callee,
                            args: vec![lhs],
                        }
                    }
                    _ => {
                        // Fallback: just call the right with left as arg
                        let hir_callee = Box::new(self.lower_expr(right));
                        HirExprKind::Call {
                            callee: hir_callee,
                            args: vec![lhs],
                        }
                    }
                }
            }

            cjc_ast::ExprKind::Block(block) => {
                HirExprKind::Block(self.lower_block(block))
            }

            cjc_ast::ExprKind::StructLit { name, fields } => HirExprKind::StructLit {
                name: name.name.clone(),
                fields: fields
                    .iter()
                    .map(|f| (f.name.name.clone(), self.lower_expr(&f.value)))
                    .collect(),
            },

            cjc_ast::ExprKind::ArrayLit(elems) => {
                HirExprKind::ArrayLit(elems.iter().map(|e| self.lower_expr(e)).collect())
            }

            cjc_ast::ExprKind::Col(name) => HirExprKind::Col(name.clone()),

            cjc_ast::ExprKind::Match { scrutinee, arms } => {
                let hir_scrutinee = Box::new(self.lower_expr(scrutinee));
                let hir_arms = arms
                    .iter()
                    .map(|arm| {
                        let arm_id = self.fresh_id();
                        let pattern = self.lower_pattern(&arm.pattern);
                        // Introduce pattern bindings into scope for the arm body
                        self.push_scope();
                        self.define_pattern_bindings(&pattern);
                        let body = self.lower_expr(&arm.body);
                        self.pop_scope();
                        HirMatchArm {
                            pattern,
                            body,
                            hir_id: arm_id,
                        }
                    })
                    .collect();
                HirExprKind::Match {
                    scrutinee: hir_scrutinee,
                    arms: hir_arms,
                }
            }

            cjc_ast::ExprKind::TupleLit(elems) => {
                HirExprKind::TupleLit(elems.iter().map(|e| self.lower_expr(e)).collect())
            }

            cjc_ast::ExprKind::Lambda { params, body } => {
                // 1. Collect all Var references in the body
                let mut var_refs = Vec::new();
                Self::collect_var_refs(body, &mut var_refs);

                // 2. Build set of lambda params (these are local, not captured)
                let param_names: BTreeSet<&str> =
                    params.iter().map(|p| p.name.name.as_str()).collect();

                // 3. Collect all locals defined inside the body
                let mut body_locals = BTreeSet::new();
                Self::collect_body_locals(body, &mut body_locals);

                // 4. Determine which variables are captured
                let mut seen = BTreeSet::new();
                let mut captures = Vec::new();
                for var_name in &var_refs {
                    // Skip if already seen
                    if !seen.insert(var_name.as_str()) {
                        continue;
                    }
                    // Skip lambda params
                    if param_names.contains(var_name.as_str()) {
                        continue;
                    }
                    // Skip body-local definitions
                    if body_locals.contains(var_name.as_str()) {
                        continue;
                    }
                    // Skip known functions
                    if self.known_functions.contains(var_name.as_str()) {
                        continue;
                    }
                    // Skip builtins
                    if Self::is_builtin(var_name) {
                        continue;
                    }
                    // This is a captured variable
                    let mode = if self.in_nogc {
                        CaptureMode::Clone
                    } else {
                        CaptureMode::Ref
                    };
                    captures.push(HirCapture {
                        name: var_name.clone(),
                        mode,
                        hir_id: self.fresh_id(),
                    });
                }

                // 5. Lower params and body inside a new scope
                self.push_scope();
                for p in params {
                    self.define_var(&p.name.name, false);
                }
                let hir_params: Vec<HirParam> =
                    params.iter().map(|p| self.lower_param(p)).collect();
                let hir_body = Box::new(self.lower_expr(body));
                self.pop_scope();

                if captures.is_empty() {
                    // No captures → plain Lambda (same as before)
                    HirExprKind::Lambda {
                        params: hir_params,
                        body: hir_body,
                    }
                } else {
                    HirExprKind::Closure {
                        params: hir_params,
                        body: hir_body,
                        captures,
                    }
                }
            }

            cjc_ast::ExprKind::Try(inner) => {
                // Desugar `expr?` to:
                // match expr {
                //     Ok(__try_v) => __try_v,
                //     Err(__try_e) => return Err(__try_e),
                // }
                let hir_inner = self.lower_expr(inner);
                let ok_binding_id = self.fresh_id();
                let ok_body_id = self.fresh_id();
                let err_binding_id = self.fresh_id();
                let err_body_id = self.fresh_id();

                let ok_pattern = HirPattern {
                    kind: HirPatternKind::Variant {
                        enum_name: "Result".into(),
                        variant: "Ok".into(),
                        fields: vec![HirPattern {
                            kind: HirPatternKind::Binding("__try_v".into()),
                            hir_id: ok_binding_id,
                        }],
                    },
                    hir_id: self.fresh_id(),
                };

                let ok_body = HirExpr {
                    kind: HirExprKind::Var("__try_v".into()),
                    hir_id: ok_body_id,
                };

                let err_pattern = HirPattern {
                    kind: HirPatternKind::Variant {
                        enum_name: "Result".into(),
                        variant: "Err".into(),
                        fields: vec![HirPattern {
                            kind: HirPatternKind::Binding("__try_e".into()),
                            hir_id: err_binding_id,
                        }],
                    },
                    hir_id: self.fresh_id(),
                };

                // return Err(__try_e)
                let err_variant = HirExpr {
                    kind: HirExprKind::VariantLit {
                        enum_name: "Result".into(),
                        variant: "Err".into(),
                        fields: vec![HirExpr {
                            kind: HirExprKind::Var("__try_e".into()),
                            hir_id: self.fresh_id(),
                        }],
                    },
                    hir_id: self.fresh_id(),
                };

                // Wrap in a block with a return statement
                let err_body = HirExpr {
                    kind: HirExprKind::Block(HirBlock {
                        stmts: vec![HirStmt {
                            kind: HirStmtKind::Return(Some(err_variant)),
                            hir_id: self.fresh_id(),
                        }],
                        expr: None,
                        hir_id: self.fresh_id(),
                    }),
                    hir_id: err_body_id,
                };

                HirExprKind::Match {
                    scrutinee: Box::new(hir_inner),
                    arms: vec![
                        HirMatchArm {
                            pattern: ok_pattern,
                            body: ok_body,
                            hir_id: self.fresh_id(),
                        },
                        HirMatchArm {
                            pattern: err_pattern,
                            body: err_body,
                            hir_id: self.fresh_id(),
                        },
                    ],
                }
            }

            cjc_ast::ExprKind::VariantLit { enum_name, variant, fields } => {
                let resolved_enum = if let Some(en) = enum_name {
                    en.name.clone()
                } else {
                    self.variant_names
                        .get(&variant.name)
                        .cloned()
                        .unwrap_or_default()
                };
                let hir_fields = fields.iter().map(|f| self.lower_expr(f)).collect();
                HirExprKind::VariantLit {
                    enum_name: resolved_enum,
                    variant: variant.name.clone(),
                    fields: hir_fields,
                }
            }

            // CompoundAssign: `target op= value` desugars to `target = target op value`
            cjc_ast::ExprKind::CompoundAssign { op, target, value } => {
                let hir_target = self.lower_expr(target);
                let hir_value = self.lower_expr(value);
                // Build: target = target op value
                HirExprKind::Assign {
                    target: Box::new(hir_target.clone()),
                    value: Box::new(HirExpr {
                        hir_id: self.fresh_id(),
                        kind: HirExprKind::Binary {
                            op: *op,
                            left: Box::new(hir_target),
                            right: Box::new(hir_value),
                        },
                    }),
                }
            }

            // Cast: `expr as f64` desugars to builtin call
            cjc_ast::ExprKind::Cast { expr, target_type } => {
                let hir_inner = self.lower_expr(expr);
                let builtin = match target_type.name.as_str() {
                    "f64" | "float" | "Float" => "float",
                    "i64" | "int" | "Int" => "int",
                    "String" | "string" => "to_string",
                    "bool" | "Bool" => "float", // bool cast: nonzero → true (desugar to float then compare)
                    other => {
                        // Unknown cast target — emit as a call to the target name
                        // (will produce a runtime error if invalid)
                        other
                    }
                };
                HirExprKind::Call {
                    callee: Box::new(HirExpr {
                        kind: HirExprKind::Var(builtin.to_string()),
                        hir_id: self.fresh_id(),
                    }),
                    args: vec![hir_inner],
                }
            }

            // IfExpr: `if cond { a } else { b }` used as a value expression.
            // Lowered directly to HirExprKind::If — a first-class expression
            // that produces the value of the taken branch.
            cjc_ast::ExprKind::IfExpr { condition, then_block, else_branch } => {
                let cond = Box::new(self.lower_expr(condition));
                let hir_then = self.lower_block(then_block);
                let hir_else = else_branch.as_ref().map(|eb| match eb {
                    cjc_ast::ElseBranch::ElseIf(elif) => {
                        let nested_if = self.lower_if(elif);
                        HirElseBranch::ElseIf(Box::new(nested_if))
                    }
                    cjc_ast::ElseBranch::Else(block) => {
                        HirElseBranch::Else(self.lower_block(block))
                    }
                });
                HirExprKind::If {
                    cond,
                    then_block: hir_then,
                    else_branch: hir_else,
                }
            }
        };
        HirExpr { kind, hir_id }
    }

    /// Recursively collect all `Var` references in an AST expression.
    fn collect_var_refs<'a>(expr: &'a cjc_ast::Expr, out: &mut Vec<String>) {
        match &expr.kind {
            cjc_ast::ExprKind::Ident(id) => out.push(id.name.clone()),
            cjc_ast::ExprKind::Binary { left, right, .. } => {
                Self::collect_var_refs(left, out);
                Self::collect_var_refs(right, out);
            }
            cjc_ast::ExprKind::Unary { operand, .. } => {
                Self::collect_var_refs(operand, out);
            }
            cjc_ast::ExprKind::Call { callee, args } => {
                Self::collect_var_refs(callee, out);
                for a in args {
                    Self::collect_var_refs(&a.value, out);
                }
            }
            cjc_ast::ExprKind::Field { object, .. } => {
                Self::collect_var_refs(object, out);
            }
            cjc_ast::ExprKind::Index { object, index } => {
                Self::collect_var_refs(object, out);
                Self::collect_var_refs(index, out);
            }
            cjc_ast::ExprKind::MultiIndex { object, indices } => {
                Self::collect_var_refs(object, out);
                for i in indices {
                    Self::collect_var_refs(i, out);
                }
            }
            cjc_ast::ExprKind::Assign { target, value } => {
                Self::collect_var_refs(target, out);
                Self::collect_var_refs(value, out);
            }
            cjc_ast::ExprKind::Pipe { left, right } => {
                Self::collect_var_refs(left, out);
                Self::collect_var_refs(right, out);
            }
            cjc_ast::ExprKind::Block(block) => {
                for stmt in &block.stmts {
                    Self::collect_var_refs_stmt(stmt, out);
                }
                if let Some(e) = &block.expr {
                    Self::collect_var_refs(e, out);
                }
            }
            cjc_ast::ExprKind::StructLit { fields, .. } => {
                for f in fields {
                    Self::collect_var_refs(&f.value, out);
                }
            }
            cjc_ast::ExprKind::ArrayLit(elems) => {
                for e in elems {
                    Self::collect_var_refs(e, out);
                }
            }
            cjc_ast::ExprKind::Lambda { body, .. } => {
                // Do NOT descend into nested lambdas for capture analysis
                // at this level — nested lambdas will do their own analysis
                // when lowered. But we do need to find refs to outer vars.
                Self::collect_var_refs(body, out);
            }
            cjc_ast::ExprKind::Match { scrutinee, arms } => {
                Self::collect_var_refs(scrutinee, out);
                for arm in arms {
                    Self::collect_var_refs(&arm.body, out);
                }
            }
            cjc_ast::ExprKind::TupleLit(elems) => {
                for e in elems {
                    Self::collect_var_refs(e, out);
                }
            }
            cjc_ast::ExprKind::FStringLit(segments) => {
                for (_lit, interp) in segments {
                    if let Some(e) = interp {
                        Self::collect_var_refs(e, out);
                    }
                }
            }
            cjc_ast::ExprKind::IntLit(_)
            | cjc_ast::ExprKind::FloatLit(_)
            | cjc_ast::ExprKind::StringLit(_)
            | cjc_ast::ExprKind::ByteStringLit(_)
            | cjc_ast::ExprKind::ByteCharLit(_)
            | cjc_ast::ExprKind::RawStringLit(_)
            | cjc_ast::ExprKind::RawByteStringLit(_)
            | cjc_ast::ExprKind::RegexLit { .. }
            | cjc_ast::ExprKind::BoolLit(_)
            | cjc_ast::ExprKind::NaLit
            | cjc_ast::ExprKind::Col(_) => {}
            cjc_ast::ExprKind::TensorLit { rows } => {
                for row in rows {
                    for expr in row {
                        Self::collect_var_refs(expr, out);
                    }
                }
            }
            cjc_ast::ExprKind::Try(inner) => {
                Self::collect_var_refs(inner, out);
            }
            cjc_ast::ExprKind::VariantLit { fields, .. } => {
                for f in fields {
                    Self::collect_var_refs(f, out);
                }
            }
            cjc_ast::ExprKind::CompoundAssign { target, value, .. } => {
                Self::collect_var_refs(target, out);
                Self::collect_var_refs(value, out);
            }
            cjc_ast::ExprKind::Cast { expr, .. } => {
                Self::collect_var_refs(expr, out);
            }
            cjc_ast::ExprKind::IfExpr { condition, then_block, else_branch } => {
                Self::collect_var_refs(condition, out);
                for stmt in &then_block.stmts {
                    Self::collect_var_refs_stmt(stmt, out);
                }
                if let Some(e) = &then_block.expr {
                    Self::collect_var_refs(e, out);
                }
                if let Some(eb) = else_branch {
                    match eb {
                        cjc_ast::ElseBranch::ElseIf(elif) => {
                            Self::collect_var_refs(&cjc_ast::Expr {
                                kind: cjc_ast::ExprKind::IfExpr {
                                    condition: Box::new(elif.condition.clone()),
                                    then_block: elif.then_block.clone(),
                                    else_branch: elif.else_branch.clone(),
                                },
                                span: elif.condition.span.clone(),
                            }, out);
                        }
                        cjc_ast::ElseBranch::Else(block) => {
                            for stmt in &block.stmts {
                                Self::collect_var_refs_stmt(stmt, out);
                            }
                            if let Some(e) = &block.expr {
                                Self::collect_var_refs(e, out);
                            }
                        }
                    }
                }
            }
        }
    }

    fn collect_var_refs_stmt(stmt: &cjc_ast::Stmt, out: &mut Vec<String>) {
        match &stmt.kind {
            cjc_ast::StmtKind::Let(l) => {
                Self::collect_var_refs(&l.init, out);
            }
            cjc_ast::StmtKind::Expr(e) => {
                Self::collect_var_refs(e, out);
            }
            cjc_ast::StmtKind::Return(e) => {
                if let Some(e) = e {
                    Self::collect_var_refs(e, out);
                }
            }
            cjc_ast::StmtKind::If(if_stmt) => {
                Self::collect_var_refs(&if_stmt.condition, out);
                Self::collect_var_refs_block(&if_stmt.then_block, out);
                if let Some(ref else_branch) = if_stmt.else_branch {
                    match else_branch {
                        cjc_ast::ElseBranch::ElseIf(elif) => {
                            Self::collect_var_refs(&elif.condition, out);
                            Self::collect_var_refs_block(&elif.then_block, out);
                        }
                        cjc_ast::ElseBranch::Else(block) => {
                            Self::collect_var_refs_block(block, out);
                        }
                    }
                }
            }
            cjc_ast::StmtKind::While(w) => {
                Self::collect_var_refs(&w.condition, out);
                Self::collect_var_refs_block(&w.body, out);
            }
            cjc_ast::StmtKind::For(f) => {
                match &f.iter {
                    cjc_ast::ForIter::Range { start, end } => {
                        Self::collect_var_refs(start, out);
                        Self::collect_var_refs(end, out);
                    }
                    cjc_ast::ForIter::Expr(expr) => {
                        Self::collect_var_refs(expr, out);
                    }
                }
                Self::collect_var_refs_block(&f.body, out);
            }
            cjc_ast::StmtKind::NoGcBlock(block) => {
                Self::collect_var_refs_block(block, out);
            }
            // Break/Continue reference no variables.
            cjc_ast::StmtKind::Break | cjc_ast::StmtKind::Continue => {}
        }
    }

    fn collect_var_refs_block(block: &cjc_ast::Block, out: &mut Vec<String>) {
        for stmt in &block.stmts {
            Self::collect_var_refs_stmt(stmt, out);
        }
        if let Some(e) = &block.expr {
            Self::collect_var_refs(e, out);
        }
    }

    /// Collect all variable names defined (via `let`) inside the body expression.
    fn collect_body_locals<'a>(expr: &'a cjc_ast::Expr, out: &mut BTreeSet<&'a str>) {
        match &expr.kind {
            cjc_ast::ExprKind::Block(block) => {
                for stmt in &block.stmts {
                    if let cjc_ast::StmtKind::Let(l) = &stmt.kind {
                        out.insert(&l.name.name);
                    }
                }
            }
            _ => {}
        }
    }

    // -- Pattern lowering --

    fn lower_pattern(&mut self, pat: &cjc_ast::Pattern) -> HirPattern {
        let hir_id = self.fresh_id();
        let kind = match &pat.kind {
            cjc_ast::PatternKind::Wildcard => HirPatternKind::Wildcard,
            cjc_ast::PatternKind::Binding(id) => {
                // Check if this is actually a unit variant name (like `None`)
                if let Some(enum_name) = self.variant_names.get(&id.name).cloned() {
                    HirPatternKind::Variant {
                        enum_name,
                        variant: id.name.clone(),
                        fields: vec![],
                    }
                } else {
                    HirPatternKind::Binding(id.name.clone())
                }
            }
            cjc_ast::PatternKind::LitInt(v) => HirPatternKind::LitInt(*v),
            cjc_ast::PatternKind::LitFloat(v) => HirPatternKind::LitFloat(*v),
            cjc_ast::PatternKind::LitBool(b) => HirPatternKind::LitBool(*b),
            cjc_ast::PatternKind::LitString(s) => HirPatternKind::LitString(s.clone()),
            cjc_ast::PatternKind::Tuple(pats) => {
                HirPatternKind::Tuple(pats.iter().map(|p| self.lower_pattern(p)).collect())
            }
            cjc_ast::PatternKind::Struct { name, fields } => {
                let hir_fields = fields
                    .iter()
                    .map(|f| {
                        let fid = self.fresh_id();
                        let pattern = if let Some(ref p) = f.pattern {
                            self.lower_pattern(p)
                        } else {
                            // Shorthand: `Point { x }` means `Point { x: x }`
                            HirPattern {
                                kind: HirPatternKind::Binding(f.name.name.clone()),
                                hir_id: self.fresh_id(),
                            }
                        };
                        HirPatternField {
                            name: f.name.name.clone(),
                            pattern,
                            hir_id: fid,
                        }
                    })
                    .collect();
                HirPatternKind::Struct {
                    name: name.name.clone(),
                    fields: hir_fields,
                }
            }
            cjc_ast::PatternKind::Variant { enum_name, variant, fields } => {
                let resolved_enum = if let Some(en) = enum_name {
                    en.name.clone()
                } else {
                    self.variant_names
                        .get(&variant.name)
                        .cloned()
                        .unwrap_or_default()
                };
                let hir_fields = fields.iter().map(|f| self.lower_pattern(f)).collect();
                HirPatternKind::Variant {
                    enum_name: resolved_enum,
                    variant: variant.name.clone(),
                    fields: hir_fields,
                }
            }
        };
        HirPattern { kind, hir_id }
    }

    /// Introduce pattern bindings into the current scope so the arm body
    /// can reference them.
    fn define_pattern_bindings(&mut self, pat: &HirPattern) {
        match &pat.kind {
            HirPatternKind::Binding(name) => self.define_var(name, false),
            HirPatternKind::Tuple(pats) => {
                for p in pats {
                    self.define_pattern_bindings(p);
                }
            }
            HirPatternKind::Struct { fields, .. } => {
                for f in fields {
                    self.define_pattern_bindings(&f.pattern);
                }
            }
            HirPatternKind::Variant { fields, .. } => {
                for f in fields {
                    self.define_pattern_bindings(f);
                }
            }
            HirPatternKind::Wildcard
            | HirPatternKind::LitInt(_)
            | HirPatternKind::LitFloat(_)
            | HirPatternKind::LitBool(_)
            | HirPatternKind::LitString(_) => {}
        }
    }

    // -- Helpers --

    fn type_expr_to_string(&self, ty: &cjc_ast::TypeExpr) -> String {
        match &ty.kind {
            cjc_ast::TypeExprKind::Named { name, args } => {
                if args.is_empty() {
                    name.name.clone()
                } else {
                    format!("{}<{}>", name.name, args.len())
                }
            }
            cjc_ast::TypeExprKind::Array { .. } => "Array".to_string(),
            cjc_ast::TypeExprKind::Tuple(elems) => {
                format!("({})", elems.len())
            }
            cjc_ast::TypeExprKind::Fn { params, .. } => {
                format!("fn({})", params.len())
            }
            cjc_ast::TypeExprKind::ShapeLit(_) => "Shape".to_string(),
        }
    }
}

impl Default for AstLowering {
    /// Creates a default [`AstLowering`] instance, equivalent to [`AstLowering::new()`].
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_ast::*;

    fn span() -> Span {
        Span::dummy()
    }

    fn ident(name: &str) -> Ident {
        Ident::dummy(name)
    }

    fn int_expr(v: i64) -> Expr {
        Expr {
            kind: ExprKind::IntLit(v),
            span: span(),
        }
    }

    fn ident_expr(name: &str) -> Expr {
        Expr {
            kind: ExprKind::Ident(ident(name)),
            span: span(),
        }
    }

    fn type_expr(name: &str) -> TypeExpr {
        TypeExpr {
            kind: TypeExprKind::Named {
                name: ident(name),
                args: vec![],
            },
            span: span(),
        }
    }

    #[test]
    fn test_lower_int_literal() {
        let mut lowering = AstLowering::new();
        let ast_expr = int_expr(42);
        let hir = lowering.lower_expr(&ast_expr);
        match hir.kind {
            HirExprKind::IntLit(v) => assert_eq!(v, 42),
            _ => panic!("expected IntLit"),
        }
    }

    #[test]
    fn test_lower_variable() {
        let mut lowering = AstLowering::new();
        let ast_expr = ident_expr("x");
        let hir = lowering.lower_expr(&ast_expr);
        match &hir.kind {
            HirExprKind::Var(name) => assert_eq!(name, "x"),
            _ => panic!("expected Var"),
        }
    }

    #[test]
    fn test_lower_binary_op() {
        let mut lowering = AstLowering::new();
        let ast_expr = Expr {
            kind: ExprKind::Binary {
                op: BinOp::Add,
                left: Box::new(int_expr(1)),
                right: Box::new(int_expr(2)),
            },
            span: span(),
        };
        let hir = lowering.lower_expr(&ast_expr);
        match &hir.kind {
            HirExprKind::Binary { op, left, right } => {
                assert_eq!(*op, BinOp::Add);
                assert!(matches!(left.kind, HirExprKind::IntLit(1)));
                assert!(matches!(right.kind, HirExprKind::IntLit(2)));
            }
            _ => panic!("expected Binary"),
        }
    }

    #[test]
    fn test_lower_pipe_desugaring() {
        let mut lowering = AstLowering::new();
        // `x |> f(y)` should become `f(x, y)`
        let ast_expr = Expr {
            kind: ExprKind::Pipe {
                left: Box::new(ident_expr("x")),
                right: Box::new(Expr {
                    kind: ExprKind::Call {
                        callee: Box::new(ident_expr("f")),
                        args: vec![CallArg {
                            name: None,
                            value: ident_expr("y"),
                            span: span(),
                        }],
                    },
                    span: span(),
                }),
            },
            span: span(),
        };
        let hir = lowering.lower_expr(&ast_expr);
        match &hir.kind {
            HirExprKind::Call { callee, args } => {
                assert!(matches!(callee.kind, HirExprKind::Var(ref n) if n == "f"));
                assert_eq!(args.len(), 2);
                assert!(matches!(args[0].kind, HirExprKind::Var(ref n) if n == "x"));
                assert!(matches!(args[1].kind, HirExprKind::Var(ref n) if n == "y"));
            }
            _ => panic!("expected Call from pipe desugaring"),
        }
    }

    #[test]
    fn test_lower_pipe_simple_ident() {
        let mut lowering = AstLowering::new();
        // `x |> f` should become `f(x)`
        let ast_expr = Expr {
            kind: ExprKind::Pipe {
                left: Box::new(ident_expr("x")),
                right: Box::new(ident_expr("f")),
            },
            span: span(),
        };
        let hir = lowering.lower_expr(&ast_expr);
        match &hir.kind {
            HirExprKind::Call { callee, args } => {
                assert!(matches!(callee.kind, HirExprKind::Var(ref n) if n == "f"));
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0].kind, HirExprKind::Var(ref n) if n == "x"));
            }
            _ => panic!("expected Call from pipe desugaring"),
        }
    }

    #[test]
    fn test_lower_fn_decl() {
        let mut lowering = AstLowering::new();
        let fn_decl = FnDecl {
            name: ident("add"),
            type_params: vec![],
            params: vec![
                Param {
                    name: ident("a"),
                    ty: type_expr("i64"),
                    default: None,
                    is_variadic: false,
                    span: span(),
                },
                Param {
                    name: ident("b"),
                    ty: type_expr("i64"),
                    default: None,
                    is_variadic: false,
                    span: span(),
                },
            ],
            return_type: Some(type_expr("i64")),
            body: Block {
                stmts: vec![],
                expr: Some(Box::new(Expr {
                    kind: ExprKind::Binary {
                        op: BinOp::Add,
                        left: Box::new(ident_expr("a")),
                        right: Box::new(ident_expr("b")),
                    },
                    span: span(),
                })),
                span: span(),
            },
            is_nogc: false,
            effect_annotation: None,
            decorators: vec![],
            vis: Visibility::Private,
        };
        let hir_fn = lowering.lower_fn_decl(&fn_decl);
        assert_eq!(hir_fn.name, "add");
        assert_eq!(hir_fn.params.len(), 2);
        assert_eq!(hir_fn.params[0].name, "a");
        assert_eq!(hir_fn.params[1].name, "b");
        assert_eq!(hir_fn.return_type, Some("i64".to_string()));
        assert!(!hir_fn.is_nogc);
        assert!(hir_fn.body.expr.is_some());
    }

    #[test]
    fn test_lower_struct_literal() {
        let mut lowering = AstLowering::new();
        let ast_expr = Expr {
            kind: ExprKind::StructLit {
                name: ident("Point"),
                fields: vec![
                    FieldInit {
                        name: ident("x"),
                        value: int_expr(1),
                        span: span(),
                    },
                    FieldInit {
                        name: ident("y"),
                        value: int_expr(2),
                        span: span(),
                    },
                ],
            },
            span: span(),
        };
        let hir = lowering.lower_expr(&ast_expr);
        match &hir.kind {
            HirExprKind::StructLit { name, fields } => {
                assert_eq!(name, "Point");
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].0, "x");
                assert_eq!(fields[1].0, "y");
            }
            _ => panic!("expected StructLit"),
        }
    }

    #[test]
    fn test_lower_if_else() {
        let mut lowering = AstLowering::new();
        let if_stmt = IfStmt {
            condition: Expr {
                kind: ExprKind::BoolLit(true),
                span: span(),
            },
            then_block: Block {
                stmts: vec![],
                expr: Some(Box::new(int_expr(1))),
                span: span(),
            },
            else_branch: Some(ElseBranch::Else(Block {
                stmts: vec![],
                expr: Some(Box::new(int_expr(2))),
                span: span(),
            })),
        };
        let hir_if = lowering.lower_if(&if_stmt);
        assert!(matches!(hir_if.cond.kind, HirExprKind::BoolLit(true)));
        assert!(hir_if.else_branch.is_some());
    }

    #[test]
    fn test_lower_while() {
        let mut lowering = AstLowering::new();
        let ast_stmt = Stmt {
            kind: StmtKind::While(WhileStmt {
                condition: Expr {
                    kind: ExprKind::BoolLit(true),
                    span: span(),
                },
                body: Block {
                    stmts: vec![],
                    expr: None,
                    span: span(),
                },
            }),
            span: span(),
        };
        let hir_stmt = lowering.lower_stmt(&ast_stmt);
        match &hir_stmt.kind {
            HirStmtKind::While { cond, body } => {
                assert!(matches!(cond.kind, HirExprKind::BoolLit(true)));
                assert!(body.stmts.is_empty());
            }
            _ => panic!("expected While"),
        }
    }

    #[test]
    fn test_lower_array_literal() {
        let mut lowering = AstLowering::new();
        let ast_expr = Expr {
            kind: ExprKind::ArrayLit(vec![int_expr(1), int_expr(2), int_expr(3)]),
            span: span(),
        };
        let hir = lowering.lower_expr(&ast_expr);
        match &hir.kind {
            HirExprKind::ArrayLit(elems) => {
                assert_eq!(elems.len(), 3);
            }
            _ => panic!("expected ArrayLit"),
        }
    }

    #[test]
    fn test_lower_full_program() {
        let mut lowering = AstLowering::new();
        let program = Program {
            declarations: vec![
                Decl {
                    kind: DeclKind::Let(LetStmt {
                        name: ident("x"),
                        mutable: false,
                        ty: None,
                        init: Box::new(int_expr(42)),
                    }),
                    span: span(),
                },
                Decl {
                    kind: DeclKind::Fn(FnDecl {
                        name: ident("main"),
                        type_params: vec![],
                        params: vec![],
                        return_type: None,
                        body: Block {
                            stmts: vec![],
                            expr: Some(Box::new(ident_expr("x"))),
                            span: span(),
                        },
                        is_nogc: false,
                        effect_annotation: None,
                        decorators: vec![],
                        vis: Visibility::Private,
                    }),
                    span: span(),
                },
            ],
        };
        let hir = lowering.lower_program(&program);
        assert_eq!(hir.items.len(), 2);
        assert!(matches!(hir.items[0], HirItem::Let(_)));
        assert!(matches!(hir.items[1], HirItem::Fn(_)));
    }

    #[test]
    fn test_hir_ids_are_unique() {
        let mut lowering = AstLowering::new();
        let e1 = lowering.lower_expr(&int_expr(1));
        let e2 = lowering.lower_expr(&int_expr(2));
        let e3 = lowering.lower_expr(&int_expr(3));
        assert_ne!(e1.hir_id, e2.hir_id);
        assert_ne!(e2.hir_id, e3.hir_id);
        assert_ne!(e1.hir_id, e3.hir_id);
    }

    #[test]
    fn test_lower_unary_op() {
        let mut lowering = AstLowering::new();
        let ast_expr = Expr {
            kind: ExprKind::Unary {
                op: UnaryOp::Neg,
                operand: Box::new(int_expr(5)),
            },
            span: span(),
        };
        let hir = lowering.lower_expr(&ast_expr);
        match &hir.kind {
            HirExprKind::Unary { op, operand } => {
                assert_eq!(*op, UnaryOp::Neg);
                assert!(matches!(operand.kind, HirExprKind::IntLit(5)));
            }
            _ => panic!("expected Unary"),
        }
    }

    #[test]
    fn test_lower_field_access() {
        let mut lowering = AstLowering::new();
        let ast_expr = Expr {
            kind: ExprKind::Field {
                object: Box::new(ident_expr("point")),
                name: ident("x"),
            },
            span: span(),
        };
        let hir = lowering.lower_expr(&ast_expr);
        match &hir.kind {
            HirExprKind::Field { object, name } => {
                assert!(matches!(object.kind, HirExprKind::Var(ref n) if n == "point"));
                assert_eq!(name, "x");
            }
            _ => panic!("expected Field"),
        }
    }

    #[test]
    fn test_lower_assignment() {
        let mut lowering = AstLowering::new();
        let ast_expr = Expr {
            kind: ExprKind::Assign {
                target: Box::new(ident_expr("x")),
                value: Box::new(int_expr(10)),
            },
            span: span(),
        };
        let hir = lowering.lower_expr(&ast_expr);
        match &hir.kind {
            HirExprKind::Assign { target, value } => {
                assert!(matches!(target.kind, HirExprKind::Var(ref n) if n == "x"));
                assert!(matches!(value.kind, HirExprKind::IntLit(10)));
            }
            _ => panic!("expected Assign"),
        }
    }

    #[test]
    fn test_lower_lambda() {
        let mut lowering = AstLowering::new();
        let ast_expr = Expr {
            kind: ExprKind::Lambda {
                params: vec![Param {
                    name: ident("x"),
                    ty: type_expr("f64"),
                    default: None,
                    is_variadic: false,
                    span: span(),
                }],
                body: Box::new(ident_expr("x")),
            },
            span: span(),
        };
        let hir = lowering.lower_expr(&ast_expr);
        match &hir.kind {
            HirExprKind::Lambda { params, body } => {
                assert_eq!(params.len(), 1);
                assert_eq!(params[0].name, "x");
                assert_eq!(params[0].ty_name, "f64");
                assert!(matches!(body.kind, HirExprKind::Var(ref n) if n == "x"));
            }
            _ => panic!("expected Lambda"),
        }
    }

    #[test]
    fn test_lower_return_stmt() {
        let mut lowering = AstLowering::new();
        let ast_stmt = Stmt {
            kind: StmtKind::Return(Some(int_expr(42))),
            span: span(),
        };
        let hir_stmt = lowering.lower_stmt(&ast_stmt);
        match &hir_stmt.kind {
            HirStmtKind::Return(Some(expr)) => {
                assert!(matches!(expr.kind, HirExprKind::IntLit(42)));
            }
            _ => panic!("expected Return"),
        }
    }

    #[test]
    fn test_lower_call_with_args() {
        let mut lowering = AstLowering::new();
        let ast_expr = Expr {
            kind: ExprKind::Call {
                callee: Box::new(ident_expr("add")),
                args: vec![
                    CallArg {
                        name: None,
                        value: int_expr(1),
                        span: span(),
                    },
                    CallArg {
                        name: None,
                        value: int_expr(2),
                        span: span(),
                    },
                ],
            },
            span: span(),
        };
        let hir = lowering.lower_expr(&ast_expr);
        match &hir.kind {
            HirExprKind::Call { callee, args } => {
                assert!(matches!(callee.kind, HirExprKind::Var(ref n) if n == "add"));
                assert_eq!(args.len(), 2);
            }
            _ => panic!("expected Call"),
        }
    }
}
