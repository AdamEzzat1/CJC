//! Abstract Syntax Tree definitions for the CJC compiler.
//!
//! This crate defines the complete AST node taxonomy produced by [`cjc_parser`]:
//! [`Program`], [`Stmt`]/[`StmtKind`], [`Expr`]/[`ExprKind`], [`Decl`]/[`DeclKind`],
//! type expressions ([`TypeExpr`]/[`TypeExprKind`]), patterns ([`Pattern`]/[`PatternKind`]),
//! and all supporting structures (identifiers, spans, operators, etc.).
//!
//! This is a **leaf crate** with zero internal dependencies, ensuring it can be
//! consumed by every downstream stage of the compiler pipeline without cycles.
//!
//! # Submodules
//!
//! - [`visit`] — Read-only visitor trait and walk functions for AST traversal
//! - [`metrics`] — Structural statistics (node counts, depths, feature flags)
//! - [`validate`] — Lightweight structural validation before HIR lowering
//! - [`inspect`] — Deterministic text dumps for debugging and testing
//! - [`node_utils`] — Pure query methods on [`Expr`], [`Block`], and [`Program`]

pub mod inspect;
pub mod metrics;
pub mod node_utils;
pub mod validate;
pub mod visit;

use std::fmt;

/// Source span: byte offset range in source code.
/// Duplicated from cjc-diag to avoid circular dependency.
/// AST is a leaf crate with no dependencies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    /// Create a new span from a start and end byte offset.
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// Merge two spans into the smallest span that covers both.
    ///
    /// Takes the minimum start and maximum end of the two spans.
    pub fn merge(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }

    /// Create a dummy span at position `0..0` for use in tests and synthetic nodes.
    pub fn dummy() -> Self {
        Self { start: 0, end: 0 }
    }
}

// ── Visibility ──────────────────────────────────────────────────

/// Visibility qualifier for declarations and fields.
/// `pub` makes an item publicly visible; the default is private.
/// NOTE: Enforcement is deferred — single-file programs treat all items as
/// public regardless of annotation. This enum is stored but not checked yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    /// Item is visible outside its defining scope (`pub` keyword).
    Public,
    /// Item is visible only within its defining scope (default).
    Private,
}

impl Default for Visibility {
    fn default() -> Self {
        Visibility::Private
    }
}

// ── AST Node Types ──────────────────────────────────────────────

/// Top-level program: a sequence of declarations.
#[derive(Debug, Clone)]
pub struct Program {
    pub declarations: Vec<Decl>,
}

/// A declaration (top-level item).
#[derive(Debug, Clone)]
pub struct Decl {
    pub kind: DeclKind,
    pub span: Span,
}

/// The kind of a top-level declaration.
///
/// Each variant wraps a dedicated declaration struct that carries the
/// variant-specific fields.
#[derive(Debug, Clone)]
pub enum DeclKind {
    /// Struct declaration: `struct Foo { ... }`
    Struct(StructDecl),
    /// Class declaration: `class Foo { ... }` (mutable fields, heap-allocated)
    Class(ClassDecl),
    /// Record declaration: `record Foo { ... }` (immutable value type)
    Record(RecordDecl),
    /// Function declaration: `fn foo(...) { ... }`
    Fn(FnDecl),
    /// Trait declaration: `trait Foo { ... }`
    Trait(TraitDecl),
    /// Impl block: `impl Foo { ... }` or `impl Trait for Foo { ... }`
    Impl(ImplDecl),
    /// Enum declaration: `enum Foo { A, B(i64) }`
    Enum(EnumDecl),
    /// Top-level let binding: `let x = expr;`
    Let(LetStmt),
    /// Import declaration: `import path.to.module`
    Import(ImportDecl),
    /// Compile-time constant: `const NAME: Type = expr;`
    Const(ConstDecl),
    /// Top-level statement (expression statement, if, while, etc.)
    Stmt(Stmt),
}

/// A compile-time constant declaration: `const PI: f64 = 3.14159;`
///
/// Constants must have an explicit type annotation and a value expression
/// that can be evaluated at compile time.
#[derive(Debug, Clone)]
pub struct ConstDecl {
    /// Constant name.
    pub name: Ident,
    /// Type annotation (required).
    pub ty: TypeExpr,
    /// Value expression (must be compile-time evaluable).
    pub value: Box<Expr>,
    /// Source span of the full declaration.
    pub span: Span,
}

// ── Struct & Class ──────────────────────────────────────────────

/// Struct declaration: `struct Point { x: f64, y: f64 }`
///
/// Structs are product types with named fields. By default, fields are
/// mutable unless the binding is immutable.
#[derive(Debug, Clone)]
pub struct StructDecl {
    /// Name of the struct.
    pub name: Ident,
    /// Generic type parameters (e.g., `<T>`).
    pub type_params: Vec<TypeParam>,
    /// Named fields with types and optional defaults.
    pub fields: Vec<FieldDecl>,
    /// Visibility qualifier (`pub` or private).
    pub vis: Visibility,
}

/// Class declaration: `class Widget { ... }`
///
/// Classes are heap-allocated types with mutable fields, similar to structs
/// but semantically intended for reference-identity objects.
#[derive(Debug, Clone)]
pub struct ClassDecl {
    /// Name of the class.
    pub name: Ident,
    /// Generic type parameters.
    pub type_params: Vec<TypeParam>,
    /// Named fields with types and optional defaults.
    pub fields: Vec<FieldDecl>,
    /// Visibility qualifier.
    pub vis: Visibility,
}

/// Record declaration: immutable value type.
/// `record Point { x: f64, y: f64 }`
/// Records are like structs but always immutable — field reassignment is a type error.
#[derive(Debug, Clone)]
pub struct RecordDecl {
    pub name: Ident,
    pub type_params: Vec<TypeParam>,
    pub fields: Vec<FieldDecl>,
    pub vis: Visibility,
}

// ── Enums ───────────────────────────────────────────────────

/// Enum declaration: `enum Color { Red, Green, Blue(i64) }`
///
/// Enums are sum types with named variants. Each variant may carry
/// zero or more positional payload types.
#[derive(Debug, Clone)]
pub struct EnumDecl {
    /// Name of the enum.
    pub name: Ident,
    /// Generic type parameters.
    pub type_params: Vec<TypeParam>,
    /// Ordered list of variants.
    pub variants: Vec<VariantDecl>,
}

/// A single variant within an [`EnumDecl`].
///
/// Unit variants have an empty `fields` vec; tuple-like variants list
/// their payload types positionally.
#[derive(Debug, Clone)]
pub struct VariantDecl {
    /// Name of the variant.
    pub name: Ident,
    /// Payload types for tuple-like variants. Empty for unit variants.
    pub fields: Vec<TypeExpr>,
    /// Source span of the variant declaration.
    pub span: Span,
}

/// A named field within a struct, class, or record declaration.
///
/// Fields carry a type annotation and an optional default-value expression.
#[derive(Debug, Clone)]
pub struct FieldDecl {
    /// Field name.
    pub name: Ident,
    /// Type annotation for the field.
    pub ty: TypeExpr,
    /// Optional default value expression.
    pub default: Option<Expr>,
    /// Visibility qualifier.
    pub vis: Visibility,
    /// Source span of the field declaration.
    pub span: Span,
}

// ── Functions ───────────────────────────────────────────────────

/// A decorator applied to a function declaration.
///
/// Syntax: `@decorator_name` or `@decorator_name(args...)`
#[derive(Debug, Clone)]
pub struct Decorator {
    pub name: Ident,
    pub args: Vec<Expr>,
    pub span: Span,
}

/// Function declaration: `fn solve(x: f64, tol: f64 = 1e-6) -> f64 { ... }`
///
/// Represents a complete function definition including its signature, body,
/// and metadata (NoGC flag, decorators, visibility).
///
/// Every new builtin must be registered in [`cjc_runtime::builtins`],
/// [`cjc_eval`], and [`cjc_mir_exec`] (the "wiring pattern").
#[derive(Debug, Clone)]
pub struct FnDecl {
    /// Function name.
    pub name: Ident,
    /// Generic type parameters.
    pub type_params: Vec<TypeParam>,
    /// Positional parameters with types, optional defaults, and variadic flag.
    pub params: Vec<Param>,
    /// Optional return-type annotation. `None` means the return type is inferred.
    pub return_type: Option<TypeExpr>,
    /// Function body block.
    pub body: Block,
    /// Whether this function is marked `@nogc` (no GC allocations allowed).
    pub is_nogc: bool,
    /// Effect annotation: `fn foo() -> i64 / pure { ... }`
    /// `None` means "any effect" (backward compatible).
    pub effect_annotation: Option<Vec<String>>,
    /// Decorators applied to this function (e.g., `@log`, `@timed`).
    pub decorators: Vec<Decorator>,
    /// Visibility qualifier.
    pub vis: Visibility,
}

/// Function signature without a body, used in [`TraitDecl`] method declarations.
///
/// Contains everything [`FnDecl`] has except the body block and metadata.
#[derive(Debug, Clone)]
pub struct FnSig {
    /// Function name.
    pub name: Ident,
    /// Generic type parameters.
    pub type_params: Vec<TypeParam>,
    /// Positional parameters.
    pub params: Vec<Param>,
    /// Optional return-type annotation.
    pub return_type: Option<TypeExpr>,
    /// Source span of the signature.
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: Ident,
    pub ty: TypeExpr,
    /// Optional default value expression (e.g., `fn f(x: f64 = 1.0)`).
    pub default: Option<Expr>,
    /// Variadic parameter: `fn f(...args: f64)` collects remaining args into an array.
    /// Only the last parameter may be variadic.
    pub is_variadic: bool,
    pub span: Span,
}

// ── Traits & Impls ──────────────────────────────────────────────

/// Trait declaration: `trait Numeric { fn zero() -> Self; }`
///
/// Traits define abstract interfaces via method signatures.
/// Implementors provide concrete method bodies via [`ImplDecl`].
#[derive(Debug, Clone)]
pub struct TraitDecl {
    /// Trait name.
    pub name: Ident,
    /// Generic type parameters.
    pub type_params: Vec<TypeParam>,
    /// Super-trait bounds this trait extends.
    pub super_traits: Vec<TypeExpr>,
    /// Method signatures (no bodies).
    pub methods: Vec<FnSig>,
}

/// Impl block: `impl Foo { ... }` or `impl Trait for Foo { ... }`
///
/// Associates method implementations with a target type, optionally
/// satisfying a trait contract.
#[derive(Debug, Clone)]
pub struct ImplDecl {
    /// Generic type parameters on the impl.
    pub type_params: Vec<TypeParam>,
    /// The type being implemented (e.g., `Foo<T>`).
    pub target: TypeExpr,
    /// Optional trait being implemented for the target type.
    pub trait_ref: Option<TypeExpr>,
    /// Method implementations.
    pub methods: Vec<FnDecl>,
    /// Source span of the impl block.
    pub span: Span,
}

// ── Import ──────────────────────────────────────────────────────

/// Import declaration: `import math.linalg` or `import stats as s`.
///
/// Brings names from other modules into scope. The `path` field contains
/// the dot-separated segments and `alias` is an optional rename.
#[derive(Debug, Clone)]
pub struct ImportDecl {
    /// Module path segments (e.g., `["math", "linalg"]`).
    pub path: Vec<Ident>,
    /// Optional alias (`as name`).
    pub alias: Option<Ident>,
}

// ── Type Expressions ────────────────────────────────────────────

/// A type expression in source code.
///
/// Wraps a [`TypeExprKind`] discriminant with a source [`Span`].
/// Used in parameter types, return types, field annotations, and
/// generic arguments.
#[derive(Debug, Clone)]
pub struct TypeExpr {
    /// The kind of type expression.
    pub kind: TypeExprKind,
    /// Source span.
    pub span: Span,
}

/// The kind of a type expression.
///
/// Covers named types, arrays, tuples, function types, and shape literals
/// used in CJC's tensor type system.
#[derive(Debug, Clone)]
pub enum TypeExprKind {
    /// Named type with optional generic arguments: `f64`, `Tensor<f32>`, `Vec<T>`.
    Named {
        /// The type name identifier.
        name: Ident,
        /// Generic type arguments (empty for non-generic types).
        args: Vec<TypeArg>,
    },
    /// Fixed-size array type: `[T; N]`.
    Array {
        /// Element type.
        elem: Box<TypeExpr>,
        /// Array size expression (must be a compile-time constant).
        size: Box<Expr>,
    },
    /// Tuple type: `(T, U, V)`.
    Tuple(Vec<TypeExpr>),
    /// Function type: `fn(T, U) -> V`.
    Fn {
        /// Parameter types.
        params: Vec<TypeExpr>,
        /// Return type.
        ret: Box<TypeExpr>,
    },
    /// Shape literal in type position: `[M, N]` for tensor dimensions.
    ShapeLit(Vec<ShapeDim>),
}

/// A type argument in a generic instantiation.
///
/// Generic parameters can accept types, expressions (for const generics),
/// or shape dimensions (for tensor shapes).
#[derive(Debug, Clone)]
pub enum TypeArg {
    /// A type argument: `Tensor<f64>`.
    Type(TypeExpr),
    /// An expression argument (const generic): `Matrix<3>`.
    Expr(Expr),
    /// A shape argument: `Tensor<[M, N]>`.
    Shape(Vec<ShapeDim>),
}

/// A single dimension in a tensor shape specification.
///
/// Dimensions are either symbolic names resolved at compile time or
/// literal integer constants.
#[derive(Debug, Clone)]
pub enum ShapeDim {
    /// Symbolic dimension name (e.g., `M`, `batch`).
    Name(Ident),
    /// Literal integer dimension (e.g., `3`, `128`).
    Lit(i64),
}

/// A generic type parameter declaration: `<T: Numeric>`.
///
/// Appears in function, struct, enum, trait, and impl declarations.
#[derive(Debug, Clone)]
pub struct TypeParam {
    /// Parameter name (e.g., `T`).
    pub name: Ident,
    /// Trait bounds on the parameter.
    pub bounds: Vec<TypeExpr>,
    /// Source span.
    pub span: Span,
}

// ── Statements ──────────────────────────────────────────────────

/// A block of statements with an optional trailing expression.
///
/// Blocks are the body of functions, loops, and if-branches. The optional
/// trailing `expr` is the block's value when used as an expression
/// (e.g., `{ let x = 1; x + 1 }` evaluates to the trailing expression).
#[derive(Debug, Clone)]
pub struct Block {
    /// Ordered list of statements in the block.
    pub stmts: Vec<Stmt>,
    /// Optional trailing expression (the block's value).
    pub expr: Option<Box<Expr>>,
    /// Source span of the entire block including braces.
    pub span: Span,
}

/// A statement node in the AST.
///
/// Wraps a [`StmtKind`] discriminant with a source [`Span`].
#[derive(Debug, Clone)]
pub struct Stmt {
    /// The kind of statement.
    pub kind: StmtKind,
    /// Source span.
    pub span: Span,
}

/// The kind of a statement.
#[derive(Debug, Clone)]
pub enum StmtKind {
    /// Variable binding: `let x = expr;` or `let mut x: T = expr;`
    Let(LetStmt),
    /// Expression statement: `foo();` (result discarded).
    Expr(Expr),
    /// Return statement: `return expr;` or bare `return;`.
    Return(Option<Expr>),
    /// Break out of the innermost loop.
    Break,
    /// Continue to the next iteration of the innermost loop.
    Continue,
    /// If statement (not to be confused with [`ExprKind::IfExpr`]).
    If(IfStmt),
    /// While loop: `while cond { ... }`
    While(WhileStmt),
    /// For loop: `for i in 0..n { ... }` or `for x in arr { ... }`
    For(ForStmt),
    /// NoGC block: `nogc { ... }` (disallows GC allocations inside).
    NoGcBlock(Block),
}

/// Let binding statement: `let x = 1;` or `let mut y: f64 = 3.14;`
#[derive(Debug, Clone)]
pub struct LetStmt {
    /// Binding name.
    pub name: Ident,
    /// Whether the binding is mutable (`let mut`).
    pub mutable: bool,
    /// Optional explicit type annotation.
    pub ty: Option<TypeExpr>,
    /// Initializer expression (required in CJC).
    pub init: Box<Expr>,
}

/// If statement: `if cond { ... } else { ... }`
///
/// See also [`ExprKind::IfExpr`] for the expression form.
#[derive(Debug, Clone)]
pub struct IfStmt {
    /// Condition expression (must evaluate to bool).
    pub condition: Expr,
    /// Then-branch block.
    pub then_block: Block,
    /// Optional else or else-if branch.
    pub else_branch: Option<ElseBranch>,
}

/// The else clause of an if statement or expression.
#[derive(Debug, Clone)]
pub enum ElseBranch {
    /// Chained else-if: `else if cond { ... }`
    ElseIf(Box<IfStmt>),
    /// Terminal else: `else { ... }`
    Else(Block),
}

/// While loop statement: `while cond { ... }`
#[derive(Debug, Clone)]
pub struct WhileStmt {
    /// Loop condition (re-evaluated each iteration).
    pub condition: Expr,
    /// Loop body.
    pub body: Block,
}

/// For loop statement: `for i in 0..n { ... }` or `for x in arr { ... }`
///
/// Desugared to a while loop during HIR lowering.
#[derive(Debug, Clone)]
pub struct ForStmt {
    /// Loop variable name.
    pub ident: Ident,
    /// Iteration source (range or expression).
    pub iter: ForIter,
    /// Loop body.
    pub body: Block,
}

/// The iteration source for a `for` loop.
#[derive(Debug, Clone)]
pub enum ForIter {
    /// Range form: `start..end` (exclusive end)
    Range { start: Box<Expr>, end: Box<Expr> },
    /// Expression form: `for x in expr` (array/tensor iteration)
    Expr(Box<Expr>),
}

// ── Expressions ─────────────────────────────────────────────────

/// An expression node in the AST.
///
/// Wraps an [`ExprKind`] discriminant with a source [`Span`].
/// Expressions produce values and can appear in statement position,
/// as function arguments, in let initializers, etc.
#[derive(Debug, Clone)]
pub struct Expr {
    /// The kind of expression.
    pub kind: ExprKind,
    /// Source span.
    pub span: Span,
}

/// The kind of an expression.
///
/// CJC supports 35+ expression variants covering literals, operators,
/// control flow, data constructors, and pattern matching.
#[derive(Debug, Clone)]
pub enum ExprKind {
    /// Integer literal: `42`
    IntLit(i64),
    /// Float literal: `3.14`
    FloatLit(f64),
    /// String literal: `"hello"`
    StringLit(String),
    /// Byte string literal: `b"hello"` — produces ByteSlice
    ByteStringLit(Vec<u8>),
    /// Byte char literal: `b'A'` — produces u8
    ByteCharLit(u8),
    /// Raw string literal: `r"hello\n"` — produces String (no escape processing)
    RawStringLit(String),
    /// Raw byte string literal: `br"hello\n"` — produces ByteSlice (no escape processing)
    RawByteStringLit(Vec<u8>),
    /// Format string literal: `f"hello {name}!"` — desugared into concat at parse time.
    /// Stored as alternating literal segments and expression strings to format.
    /// Each segment: `(literal_text, Option<expr>)` — literal then optional interp.
    FStringLit(Vec<(String, Option<Box<Expr>>)>),
    /// Regex literal: `/pattern/flags` — produces Regex
    RegexLit { pattern: String, flags: String },
    /// Tensor literal: `[| 1.0, 2.0; 3.0, 4.0 |]` — produces Tensor
    /// `rows` contains each row as a Vec of expressions.
    /// A 1-D tensor has a single row. `;` separates rows for 2-D.
    TensorLit { rows: Vec<Vec<Expr>> },
    /// Bool literal: `true`, `false`
    BoolLit(bool),
    /// Missing value literal: `NA`
    NaLit,
    /// Identifier: `x`, `foo`
    Ident(Ident),
    /// Binary operation: `a + b`
    Binary {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    /// Unary operation: `-x`, `!b`
    Unary {
        op: UnaryOp,
        operand: Box<Expr>,
    },
    /// Function/method call: `foo(a, b)`
    Call {
        callee: Box<Expr>,
        args: Vec<CallArg>,
    },
    /// Field access / method access: `x.foo`
    Field {
        object: Box<Expr>,
        name: Ident,
    },
    /// Index: `a[i]`
    Index {
        object: Box<Expr>,
        index: Box<Expr>,
    },
    /// Multi-index: `a[i, j]`
    MultiIndex {
        object: Box<Expr>,
        indices: Vec<Expr>,
    },
    /// Assignment: `x = expr`
    Assign {
        target: Box<Expr>,
        value: Box<Expr>,
    },
    /// Compound assignment: `x += e`, `x -= e`, etc.
    /// Desugars to `x = x op e` during evaluation.
    CompoundAssign {
        op: BinOp,
        target: Box<Expr>,
        value: Box<Expr>,
    },
    /// If expression: `if cond { a } else { b }` used as a value
    IfExpr {
        condition: Box<Expr>,
        then_block: Block,
        else_branch: Option<ElseBranch>,
    },
    /// Pipe: `a |> f(b)`
    Pipe {
        left: Box<Expr>,
        right: Box<Expr>,
    },
    /// Block expression: `{ stmts; expr }`
    Block(Block),
    /// Struct literal: `Foo { x: 1, y: 2 }`
    StructLit {
        name: Ident,
        fields: Vec<FieldInit>,
    },
    /// Array literal: `[1, 2, 3]`
    ArrayLit(Vec<Expr>),
    /// Column reference in data DSL: `col("name")`
    Col(String),
    /// Lambda expression: `|x, y| x + y`
    Lambda {
        params: Vec<Param>,
        body: Box<Expr>,
    },
    /// Match expression: `match expr { pat => expr, ... }`
    Match {
        scrutinee: Box<Expr>,
        arms: Vec<MatchArm>,
    },
    /// Tuple literal: `(a, b, c)`
    TupleLit(Vec<Expr>),
    /// Type cast expression: `x as f64`, `y as i64`
    Cast {
        expr: Box<Expr>,
        target_type: Ident,
    },
    /// Try operator: `expr?` — desugars to match on Result
    Try(Box<Expr>),
    /// Enum variant constructor: `Some(42)` or `None`
    VariantLit {
        enum_name: Option<Ident>,
        variant: Ident,
        fields: Vec<Expr>,
    },
}

// ── Match Arms & Patterns ───────────────────────────────────────

/// A single arm of a match expression: `pattern => body`
#[derive(Debug, Clone)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub body: Expr,
    pub span: Span,
}

/// A pattern for use in match arms.
#[derive(Debug, Clone)]
pub struct Pattern {
    pub kind: PatternKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum PatternKind {
    /// Wildcard pattern: `_`
    Wildcard,
    /// Binding pattern: `x` (binds the matched value to `x`)
    Binding(Ident),
    /// Literal pattern: `42`, `3.14`, `true`, `"hello"`
    LitInt(i64),
    LitFloat(f64),
    LitBool(bool),
    LitString(String),
    /// Tuple destructuring: `(a, b, c)`
    Tuple(Vec<Pattern>),
    /// Struct destructuring: `Point { x, y }` or `Point { x: px, y: py }`
    Struct {
        name: Ident,
        fields: Vec<PatternField>,
    },
    /// Enum variant pattern: `Some(x)`, `None`, `Ok(v)`, `Err(e)`
    Variant {
        /// Optional enum name qualification (not needed for prelude enums)
        enum_name: Option<Ident>,
        variant: Ident,
        fields: Vec<Pattern>,
    },
}

/// A field pattern inside a struct pattern: `x` or `x: pat`
#[derive(Debug, Clone)]
pub struct PatternField {
    pub name: Ident,
    pub pattern: Option<Pattern>,
    pub span: Span,
}

/// A function call argument, optionally named.
///
/// Positional: `f(42)`. Named: `f(x: 42)`.
#[derive(Debug, Clone)]
pub struct CallArg {
    /// Optional argument name for named/keyword arguments.
    pub name: Option<Ident>,
    /// The argument value expression.
    pub value: Expr,
    /// Source span.
    pub span: Span,
}

/// A field initializer in a struct literal: `x: expr`.
#[derive(Debug, Clone)]
pub struct FieldInit {
    /// Field name.
    pub name: Ident,
    /// Field value expression.
    pub value: Expr,
    /// Source span.
    pub span: Span,
}

// ── Operators ───────────────────────────────────────────────────

/// Binary operator.
///
/// Covers arithmetic, comparison, logical, bitwise, and regex-match operators.
/// Displayed via [`fmt::Display`] as the source-level symbol (e.g., `+`, `==`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    /// Addition: `+`
    Add,
    /// Subtraction: `-`
    Sub,
    /// Multiplication: `*`
    Mul,
    /// Division: `/`
    Div,
    /// Modulo: `%`
    Mod,
    /// Exponentiation: `**`
    Pow,
    /// Equality: `==`
    Eq,
    /// Inequality: `!=`
    Ne,
    /// Less than: `<`
    Lt,
    /// Greater than: `>`
    Gt,
    /// Less than or equal: `<=`
    Le,
    /// Greater than or equal: `>=`
    Ge,
    /// Logical and: `&&`
    And,
    /// Logical or: `||`
    Or,
    /// Regex match: `~=`
    Match,
    /// Negative regex match: `!~`
    NotMatch,
    /// Bitwise and: `&`
    BitAnd,
    /// Bitwise or: `|`
    BitOr,
    /// Bitwise xor: `^`
    BitXor,
    /// Left shift: `<<`
    Shl,
    /// Right shift: `>>`
    Shr,
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinOp::Add => write!(f, "+"),
            BinOp::Sub => write!(f, "-"),
            BinOp::Mul => write!(f, "*"),
            BinOp::Div => write!(f, "/"),
            BinOp::Mod => write!(f, "%"),
            BinOp::Pow => write!(f, "**"),
            BinOp::Eq => write!(f, "=="),
            BinOp::Ne => write!(f, "!="),
            BinOp::Lt => write!(f, "<"),
            BinOp::Gt => write!(f, ">"),
            BinOp::Le => write!(f, "<="),
            BinOp::Ge => write!(f, ">="),
            BinOp::And => write!(f, "&&"),
            BinOp::Or => write!(f, "||"),
            BinOp::Match => write!(f, "~="),
            BinOp::NotMatch => write!(f, "!~"),
            BinOp::BitAnd => write!(f, "&"),
            BinOp::BitOr => write!(f, "|"),
            BinOp::BitXor => write!(f, "^"),
            BinOp::Shl => write!(f, "<<"),
            BinOp::Shr => write!(f, ">>"),
        }
    }
}

/// Unary operator.
///
/// Displayed via [`fmt::Display`] as the source-level symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// Arithmetic negation: `-x`
    Neg,
    /// Logical not: `!b`
    Not,
    /// Bitwise not: `~x`
    BitNot,
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnaryOp::Neg => write!(f, "-"),
            UnaryOp::Not => write!(f, "!"),
            UnaryOp::BitNot => write!(f, "~"),
        }
    }
}

// ── Identifier ──────────────────────────────────────────────────

/// An identifier with its source span.
///
/// Used for variable names, function names, type names, field names, etc.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Ident {
    /// The identifier text.
    pub name: String,
    /// Source span where this identifier appears.
    pub span: Span,
}

impl Ident {
    /// Create a new identifier with an explicit source span.
    pub fn new(name: impl Into<String>, span: Span) -> Self {
        Self {
            name: name.into(),
            span,
        }
    }

    /// Create a dummy identifier with a zero span, for use in tests and synthetic AST nodes.
    pub fn dummy(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            span: Span::dummy(),
        }
    }
}

impl fmt::Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

// ── Pretty Printer ──────────────────────────────────────────────

/// Pretty-printer that converts an AST back into human-readable CJC source.
///
/// Produces deterministic, indented output suitable for debugging and
/// round-trip verification.
///
/// # Examples
///
/// ```rust,ignore
/// use cjc_ast::PrettyPrinter;
///
/// let source = PrettyPrinter::new().print_program(&program);
/// println!("{}", source);
/// ```
pub struct PrettyPrinter {
    indent: usize,
    output: String,
}

impl PrettyPrinter {
    /// Create a new pretty-printer with zero indentation.
    pub fn new() -> Self {
        Self {
            indent: 0,
            output: String::new(),
        }
    }

    /// Consume the printer and return the pretty-printed source for an entire program.
    ///
    /// # Arguments
    ///
    /// * `program` - The AST program to pretty-print.
    ///
    /// # Returns
    ///
    /// A `String` containing the formatted CJC source code.
    pub fn print_program(mut self, program: &Program) -> String {
        for decl in &program.declarations {
            self.print_decl(decl);
            self.output.push('\n');
        }
        self.output
    }

    fn indent_str(&self) -> String {
        "    ".repeat(self.indent)
    }

    fn print_decl(&mut self, decl: &Decl) {
        match &decl.kind {
            DeclKind::Struct(s) => self.print_struct(s),
            DeclKind::Class(c) => self.print_class(c),
            DeclKind::Record(r) => self.print_record(r),
            DeclKind::Fn(f) => self.print_fn(f),
            DeclKind::Trait(t) => self.print_trait(t),
            DeclKind::Impl(i) => self.print_impl(i),
            DeclKind::Enum(e) => self.print_enum(e),
            DeclKind::Let(l) => {
                self.output.push_str(&self.indent_str());
                self.print_let(l);
            }
            DeclKind::Import(i) => self.print_import(i),
            DeclKind::Const(c) => {
                self.output.push_str(&self.indent_str());
                self.output.push_str("const ");
                self.output.push_str(&c.name.name);
                self.output.push_str(": ");
                self.print_type_expr(&c.ty);
                self.output.push_str(" = ");
                self.print_expr(&c.value);
                self.output.push(';');
            }
            DeclKind::Stmt(s) => {
                self.output.push_str(&self.indent_str());
                self.print_stmt(s);
            }
        }
    }

    fn print_struct(&mut self, s: &StructDecl) {
        self.output.push_str(&self.indent_str());
        self.output.push_str(&format!("struct {}", s.name));
        self.print_type_params(&s.type_params);
        self.output.push_str(" {\n");
        self.indent += 1;
        for field in &s.fields {
            self.output
                .push_str(&format!("{}{}: ", self.indent_str(), field.name));
            self.print_type_expr(&field.ty);
            self.output.push('\n');
        }
        self.indent -= 1;
        self.output.push_str(&format!("{}}}", self.indent_str()));
    }

    fn print_class(&mut self, c: &ClassDecl) {
        self.output.push_str(&self.indent_str());
        self.output.push_str(&format!("class {}", c.name));
        self.print_type_params(&c.type_params);
        self.output.push_str(" {\n");
        self.indent += 1;
        for field in &c.fields {
            self.output
                .push_str(&format!("{}{}: ", self.indent_str(), field.name));
            self.print_type_expr(&field.ty);
            self.output.push('\n');
        }
        self.indent -= 1;
        self.output.push_str(&format!("{}}}", self.indent_str()));
    }

    fn print_record(&mut self, r: &RecordDecl) {
        self.output.push_str(&self.indent_str());
        self.output.push_str(&format!("record {}", r.name));
        self.print_type_params(&r.type_params);
        self.output.push_str(" {\n");
        self.indent += 1;
        for field in &r.fields {
            self.output
                .push_str(&format!("{}{}: ", self.indent_str(), field.name));
            self.print_type_expr(&field.ty);
            self.output.push('\n');
        }
        self.indent -= 1;
        self.output.push_str(&format!("{}}}", self.indent_str()));
    }

    fn print_fn(&mut self, f: &FnDecl) {
        self.output.push_str(&self.indent_str());
        if f.is_nogc {
            self.output.push_str("nogc ");
        }
        self.output.push_str(&format!("fn {}", f.name));
        self.print_type_params(&f.type_params);
        self.output.push('(');
        for (i, param) in f.params.iter().enumerate() {
            if i > 0 {
                self.output.push_str(", ");
            }
            if param.is_variadic {
                self.output.push_str("...");
            }
            self.output.push_str(&format!("{}: ", param.name));
            self.print_type_expr(&param.ty);
        }
        self.output.push(')');
        if let Some(ref ret) = f.return_type {
            self.output.push_str(" -> ");
            self.print_type_expr(ret);
        }
        self.output.push(' ');
        self.print_block(&f.body);
    }

    fn print_trait(&mut self, t: &TraitDecl) {
        self.output.push_str(&self.indent_str());
        self.output.push_str(&format!("trait {}", t.name));
        self.print_type_params(&t.type_params);
        if !t.super_traits.is_empty() {
            self.output.push_str(": ");
            for (i, st) in t.super_traits.iter().enumerate() {
                if i > 0 {
                    self.output.push_str(" + ");
                }
                self.print_type_expr(st);
            }
        }
        self.output.push_str(" {\n");
        self.indent += 1;
        for method in &t.methods {
            self.output.push_str(&self.indent_str());
            self.output.push_str(&format!("fn {}", method.name));
            self.print_type_params(&method.type_params);
            self.output.push('(');
            for (i, p) in method.params.iter().enumerate() {
                if i > 0 {
                    self.output.push_str(", ");
                }
                self.output.push_str(&format!("{}: ", p.name));
                self.print_type_expr(&p.ty);
            }
            self.output.push(')');
            if let Some(ref ret) = method.return_type {
                self.output.push_str(" -> ");
                self.print_type_expr(ret);
            }
            self.output.push_str(";\n");
        }
        self.indent -= 1;
        self.output.push_str(&format!("{}}}", self.indent_str()));
    }

    fn print_impl(&mut self, i: &ImplDecl) {
        self.output.push_str(&self.indent_str());
        self.output.push_str("impl");
        self.print_type_params(&i.type_params);
        self.output.push(' ');
        self.print_type_expr(&i.target);
        if let Some(ref tr) = i.trait_ref {
            self.output.push_str(" : ");
            self.print_type_expr(tr);
        }
        self.output.push_str(" {\n");
        self.indent += 1;
        for method in &i.methods {
            self.print_fn(method);
            self.output.push('\n');
        }
        self.indent -= 1;
        self.output.push_str(&format!("{}}}", self.indent_str()));
    }

    fn print_import(&mut self, i: &ImportDecl) {
        self.output.push_str(&self.indent_str());
        self.output.push_str("import ");
        let path: Vec<&str> = i.path.iter().map(|id| id.name.as_str()).collect();
        self.output.push_str(&path.join("."));
        if let Some(ref alias) = i.alias {
            self.output.push_str(&format!(" as {}", alias));
        }
    }

    fn print_enum(&mut self, e: &EnumDecl) {
        self.output.push_str(&self.indent_str());
        self.output.push_str(&format!("enum {}", e.name));
        self.print_type_params(&e.type_params);
        self.output.push_str(" {\n");
        self.indent += 1;
        for variant in &e.variants {
            self.output.push_str(&self.indent_str());
            self.output.push_str(&variant.name.name);
            if !variant.fields.is_empty() {
                self.output.push('(');
                for (i, f) in variant.fields.iter().enumerate() {
                    if i > 0 {
                        self.output.push_str(", ");
                    }
                    self.print_type_expr(f);
                }
                self.output.push(')');
            }
            self.output.push_str(",\n");
        }
        self.indent -= 1;
        self.output.push_str(&format!("{}}}", self.indent_str()));
    }

    fn print_type_params(&mut self, params: &[TypeParam]) {
        if params.is_empty() {
            return;
        }
        self.output.push('<');
        for (i, param) in params.iter().enumerate() {
            if i > 0 {
                self.output.push_str(", ");
            }
            self.output.push_str(&param.name.name);
            if !param.bounds.is_empty() {
                self.output.push_str(": ");
                for (j, bound) in param.bounds.iter().enumerate() {
                    if j > 0 {
                        self.output.push_str(" + ");
                    }
                    self.print_type_expr(bound);
                }
            }
        }
        self.output.push('>');
    }

    fn print_type_expr(&mut self, ty: &TypeExpr) {
        match &ty.kind {
            TypeExprKind::Named { name, args } => {
                self.output.push_str(&name.name);
                if !args.is_empty() {
                    self.output.push('<');
                    for (i, arg) in args.iter().enumerate() {
                        if i > 0 {
                            self.output.push_str(", ");
                        }
                        match arg {
                            TypeArg::Type(t) => self.print_type_expr(t),
                            TypeArg::Expr(e) => self.print_expr(e),
                            TypeArg::Shape(dims) => self.print_shape(dims),
                        }
                    }
                    self.output.push('>');
                }
            }
            TypeExprKind::Array { elem, size } => {
                self.output.push('[');
                self.print_type_expr(elem);
                self.output.push_str("; ");
                self.print_expr(size);
                self.output.push(']');
            }
            TypeExprKind::Tuple(elems) => {
                self.output.push('(');
                for (i, elem) in elems.iter().enumerate() {
                    if i > 0 {
                        self.output.push_str(", ");
                    }
                    self.print_type_expr(elem);
                }
                self.output.push(')');
            }
            TypeExprKind::Fn { params, ret } => {
                self.output.push_str("fn(");
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        self.output.push_str(", ");
                    }
                    self.print_type_expr(p);
                }
                self.output.push_str(") -> ");
                self.print_type_expr(ret);
            }
            TypeExprKind::ShapeLit(dims) => {
                self.print_shape(dims);
            }
        }
    }

    fn print_shape(&mut self, dims: &[ShapeDim]) {
        self.output.push('[');
        for (i, dim) in dims.iter().enumerate() {
            if i > 0 {
                self.output.push_str(", ");
            }
            match dim {
                ShapeDim::Name(n) => self.output.push_str(&n.name),
                ShapeDim::Lit(v) => self.output.push_str(&v.to_string()),
            }
        }
        self.output.push(']');
    }

    fn print_block(&mut self, block: &Block) {
        self.output.push_str("{\n");
        self.indent += 1;
        for stmt in &block.stmts {
            self.print_stmt(stmt);
        }
        if let Some(ref expr) = block.expr {
            self.output.push_str(&self.indent_str());
            self.print_expr(expr);
            self.output.push('\n');
        }
        self.indent -= 1;
        self.output.push_str(&format!("{}}}", self.indent_str()));
    }

    fn print_stmt(&mut self, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::Let(l) => {
                self.output.push_str(&self.indent_str());
                self.print_let(l);
            }
            StmtKind::Expr(e) => {
                self.output.push_str(&self.indent_str());
                self.print_expr(e);
                self.output.push_str(";\n");
            }
            StmtKind::Return(e) => {
                self.output.push_str(&self.indent_str());
                self.output.push_str("return");
                if let Some(expr) = e {
                    self.output.push(' ');
                    self.print_expr(expr);
                }
                self.output.push_str(";\n");
            }
            StmtKind::Break => {
                self.output.push_str(&self.indent_str());
                self.output.push_str("break;\n");
            }
            StmtKind::Continue => {
                self.output.push_str(&self.indent_str());
                self.output.push_str("continue;\n");
            }
            StmtKind::If(if_stmt) => {
                self.output.push_str(&self.indent_str());
                self.print_if(if_stmt);
                self.output.push('\n');
            }
            StmtKind::While(w) => {
                self.output.push_str(&self.indent_str());
                self.output.push_str("while ");
                self.print_expr(&w.condition);
                self.output.push(' ');
                self.print_block(&w.body);
                self.output.push('\n');
            }
            StmtKind::For(f) => {
                self.output.push_str(&self.indent_str());
                self.output.push_str("for ");
                self.output.push_str(&f.ident.name);
                self.output.push_str(" in ");
                match &f.iter {
                    ForIter::Range { start, end } => {
                        self.print_expr(start);
                        self.output.push_str("..");
                        self.print_expr(end);
                    }
                    ForIter::Expr(expr) => {
                        self.print_expr(expr);
                    }
                }
                self.output.push(' ');
                self.print_block(&f.body);
                self.output.push('\n');
            }
            StmtKind::NoGcBlock(block) => {
                self.output.push_str(&self.indent_str());
                self.output.push_str("nogc ");
                self.print_block(block);
                self.output.push('\n');
            }
        }
    }

    fn print_let(&mut self, l: &LetStmt) {
        self.output.push_str("let ");
        if l.mutable {
            self.output.push_str("mut ");
        }
        self.output.push_str(&l.name.name);
        if let Some(ref ty) = l.ty {
            self.output.push_str(": ");
            self.print_type_expr(ty);
        }
        self.output.push_str(" = ");
        self.print_expr(&l.init);
        self.output.push_str(";\n");
    }

    fn print_if(&mut self, if_stmt: &IfStmt) {
        self.output.push_str("if ");
        self.print_expr(&if_stmt.condition);
        self.output.push(' ');
        self.print_block(&if_stmt.then_block);
        if let Some(ref else_branch) = if_stmt.else_branch {
            self.output.push_str(" else ");
            match else_branch {
                ElseBranch::ElseIf(elif) => self.print_if(elif),
                ElseBranch::Else(block) => self.print_block(block),
            }
        }
    }

    fn print_expr(&mut self, expr: &Expr) {
        match &expr.kind {
            ExprKind::IntLit(v) => self.output.push_str(&v.to_string()),
            ExprKind::FloatLit(v) => {
                let s = v.to_string();
                self.output.push_str(&s);
                if !s.contains('.') {
                    self.output.push_str(".0");
                }
            }
            ExprKind::StringLit(s) => {
                self.output.push('"');
                self.output.push_str(s);
                self.output.push('"');
            }
            ExprKind::ByteStringLit(bytes) => {
                self.output.push_str("b\"");
                for &b in bytes {
                    if b.is_ascii_graphic() || b == b' ' {
                        self.output.push(b as char);
                    } else {
                        self.output.push_str(&format!("\\x{:02x}", b));
                    }
                }
                self.output.push('"');
            }
            ExprKind::ByteCharLit(b) => {
                self.output.push_str(&format!("b'\\x{:02x}'", b));
            }
            ExprKind::RawStringLit(s) => {
                self.output.push_str("r\"");
                self.output.push_str(s);
                self.output.push('"');
            }
            ExprKind::RawByteStringLit(bytes) => {
                self.output.push_str("br\"");
                for &b in bytes {
                    self.output.push(b as char);
                }
                self.output.push('"');
            }
            ExprKind::FStringLit(segments) => {
                // Pretty-print as f"..."
                self.output.push_str("f\"");
                for (lit, interp) in segments {
                    self.output.push_str(&lit.replace('"', "\\\""));
                    if let Some(expr) = interp {
                        self.output.push('{');
                        self.print_expr(expr);
                        self.output.push('}');
                    }
                }
                self.output.push('"');
            }
            ExprKind::RegexLit { pattern, flags } => {
                self.output.push('/');
                self.output.push_str(pattern);
                self.output.push('/');
                self.output.push_str(flags);
            }
            ExprKind::TensorLit { rows } => {
                self.output.push_str("[| ");
                for (ri, row) in rows.iter().enumerate() {
                    if ri > 0 {
                        self.output.push_str("; ");
                    }
                    for (ci, expr) in row.iter().enumerate() {
                        if ci > 0 {
                            self.output.push_str(", ");
                        }
                        self.print_expr(expr);
                    }
                }
                self.output.push_str(" |]");
            }
            ExprKind::BoolLit(b) => self.output.push_str(if *b { "true" } else { "false" }),
            ExprKind::NaLit => self.output.push_str("NA"),
            ExprKind::Ident(id) => self.output.push_str(&id.name),
            ExprKind::Binary { op, left, right } => {
                self.output.push('(');
                self.print_expr(left);
                self.output.push_str(&format!(" {} ", op));
                self.print_expr(right);
                self.output.push(')');
            }
            ExprKind::Unary { op, operand } => {
                self.output.push_str(&format!("{}", op));
                self.print_expr(operand);
            }
            ExprKind::Call { callee, args } => {
                self.print_expr(callee);
                self.output.push('(');
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        self.output.push_str(", ");
                    }
                    if let Some(ref name) = arg.name {
                        self.output.push_str(&format!("{}: ", name));
                    }
                    self.print_expr(&arg.value);
                }
                self.output.push(')');
            }
            ExprKind::Field { object, name } => {
                self.print_expr(object);
                self.output.push('.');
                self.output.push_str(&name.name);
            }
            ExprKind::Index { object, index } => {
                self.print_expr(object);
                self.output.push('[');
                self.print_expr(index);
                self.output.push(']');
            }
            ExprKind::MultiIndex { object, indices } => {
                self.print_expr(object);
                self.output.push('[');
                for (i, idx) in indices.iter().enumerate() {
                    if i > 0 {
                        self.output.push_str(", ");
                    }
                    self.print_expr(idx);
                }
                self.output.push(']');
            }
            ExprKind::Assign { target, value } => {
                self.print_expr(target);
                self.output.push_str(" = ");
                self.print_expr(value);
            }
            ExprKind::CompoundAssign { op, target, value } => {
                self.print_expr(target);
                self.output.push_str(&format!(" {}= ", op));
                self.print_expr(value);
            }
            ExprKind::IfExpr { condition, then_block, else_branch } => {
                self.output.push_str("if ");
                self.print_expr(condition);
                self.output.push_str(" ");
                self.print_block(then_block);
                if let Some(eb) = else_branch {
                    self.output.push_str(" else ");
                    match eb {
                        ElseBranch::ElseIf(elif) => {
                            // Print as if statement
                            self.output.push_str("if ...");
                        }
                        ElseBranch::Else(block) => {
                            self.print_block(block);
                        }
                    }
                }
            }
            ExprKind::Pipe { left, right } => {
                self.print_expr(left);
                self.output.push_str(" |> ");
                self.print_expr(right);
            }
            ExprKind::Block(block) => {
                self.print_block(block);
            }
            ExprKind::StructLit { name, fields } => {
                self.output.push_str(&format!("{} {{ ", name));
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        self.output.push_str(", ");
                    }
                    self.output.push_str(&format!("{}: ", field.name));
                    self.print_expr(&field.value);
                }
                self.output.push_str(" }");
            }
            ExprKind::ArrayLit(elems) => {
                self.output.push('[');
                for (i, elem) in elems.iter().enumerate() {
                    if i > 0 {
                        self.output.push_str(", ");
                    }
                    self.print_expr(elem);
                }
                self.output.push(']');
            }
            ExprKind::Col(name) => {
                self.output.push_str(&format!("col(\"{}\")", name));
            }
            ExprKind::Lambda { params, body } => {
                self.output.push('|');
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        self.output.push_str(", ");
                    }
                    self.output.push_str(&format!("{}: ", p.name));
                    self.print_type_expr(&p.ty);
                }
                self.output.push_str("| ");
                self.print_expr(body);
            }
            ExprKind::Match { scrutinee, arms } => {
                self.output.push_str("match ");
                self.print_expr(scrutinee);
                self.output.push_str(" {\n");
                self.indent += 1;
                for arm in arms {
                    self.output.push_str(&self.indent_str());
                    self.print_pattern(&arm.pattern);
                    self.output.push_str(" => ");
                    self.print_expr(&arm.body);
                    self.output.push_str(",\n");
                }
                self.indent -= 1;
                self.output.push_str(&format!("{}}}", self.indent_str()));
            }
            ExprKind::TupleLit(elems) => {
                self.output.push('(');
                for (i, elem) in elems.iter().enumerate() {
                    if i > 0 {
                        self.output.push_str(", ");
                    }
                    self.print_expr(elem);
                }
                self.output.push(')');
            }
            ExprKind::Try(inner) => {
                self.print_expr(inner);
                self.output.push('?');
            }
            ExprKind::VariantLit {
                enum_name,
                variant,
                fields,
            } => {
                if let Some(ref en) = enum_name {
                    self.output.push_str(&en.name);
                    self.output.push_str("::");
                }
                self.output.push_str(&variant.name);
                if !fields.is_empty() {
                    self.output.push('(');
                    for (i, f) in fields.iter().enumerate() {
                        if i > 0 {
                            self.output.push_str(", ");
                        }
                        self.print_expr(f);
                    }
                    self.output.push(')');
                }
            }
            ExprKind::Cast { expr, target_type } => {
                self.print_expr(expr);
                self.output.push_str(" as ");
                self.output.push_str(&target_type.name);
            }
        }
    }

    fn print_pattern(&mut self, pattern: &Pattern) {
        match &pattern.kind {
            PatternKind::Wildcard => self.output.push('_'),
            PatternKind::Binding(id) => self.output.push_str(&id.name),
            PatternKind::LitInt(v) => self.output.push_str(&v.to_string()),
            PatternKind::LitFloat(v) => self.output.push_str(&v.to_string()),
            PatternKind::LitBool(b) => {
                self.output.push_str(if *b { "true" } else { "false" })
            }
            PatternKind::LitString(s) => {
                self.output.push('"');
                self.output.push_str(s);
                self.output.push('"');
            }
            PatternKind::Tuple(pats) => {
                self.output.push('(');
                for (i, p) in pats.iter().enumerate() {
                    if i > 0 {
                        self.output.push_str(", ");
                    }
                    self.print_pattern(p);
                }
                self.output.push(')');
            }
            PatternKind::Struct { name, fields } => {
                self.output.push_str(&name.name);
                self.output.push_str(" { ");
                for (i, f) in fields.iter().enumerate() {
                    if i > 0 {
                        self.output.push_str(", ");
                    }
                    self.output.push_str(&f.name.name);
                    if let Some(ref pat) = f.pattern {
                        self.output.push_str(": ");
                        self.print_pattern(pat);
                    }
                }
                self.output.push_str(" }");
            }
            PatternKind::Variant {
                enum_name,
                variant,
                fields,
            } => {
                if let Some(ref en) = enum_name {
                    self.output.push_str(&en.name);
                    self.output.push_str("::");
                }
                self.output.push_str(&variant.name);
                if !fields.is_empty() {
                    self.output.push('(');
                    for (i, p) in fields.iter().enumerate() {
                        if i > 0 {
                            self.output.push_str(", ");
                        }
                        self.print_pattern(p);
                    }
                    self.output.push(')');
                }
            }
        }
    }
}

impl Default for PrettyPrinter {
    fn default() -> Self {
        Self::new()
    }
}
