//! CJC MIR (Mid-level Intermediate Representation)
//!
//! MIR is a control-flow graph (CFG) of basic blocks. Every value is an
//! explicit temporary. This is the level where:
//! - Pattern matching is compiled to decision trees (Stage 2.2)
//! - Closures are lambda-lifted (Stage 2.1)
//! - `nogc` verification runs (Stage 2.4)
//! - Optimization passes operate (Stage 2.4)
//!
//! For Milestone 2.0, MIR is a simplified representation that mirrors HIR
//! closely — we lower HIR items into MIR functions with basic blocks for
//! straight-line code, if/else, while, and function calls.

pub mod cfg;
pub mod dominators;
pub mod escape;
pub mod inspect;
pub mod loop_analysis;
pub mod monomorph;
pub mod nogc_verify;
pub mod optimize;
pub mod reduction;
pub mod ssa;
pub mod ssa_loop_overlay;
pub mod ssa_optimize;
pub mod verify;

use cjc_ast::{BinOp, UnaryOp, Visibility};
pub use escape::AllocHint;

// ---------------------------------------------------------------------------
// IDs
// ---------------------------------------------------------------------------

/// Unique identifier for a MIR function within a [`MirProgram`].
///
/// Assigned sequentially during HIR-to-MIR lowering. The synthetic `__main`
/// entry function and lambda-lifted closures each receive their own ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MirFnId(pub u32);

/// Unique identifier for a basic block within a [`cfg::MirCfg`].
///
/// Block IDs are dense indices into `MirCfg::basic_blocks`. `BlockId(0)` is
/// always the entry block. IDs are assigned deterministically in creation order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u32);

/// Unique identifier for a temporary value in the MIR.
///
/// Reserved for future use when MIR transitions to explicit temporaries
/// instead of named variables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TempId(pub u32);

// ---------------------------------------------------------------------------
// Program
// ---------------------------------------------------------------------------

/// A MIR program is a collection of functions + struct defs + an entry point.
#[derive(Debug, Clone)]
pub struct MirProgram {
    pub functions: Vec<MirFunction>,
    pub struct_defs: Vec<MirStructDef>,
    pub enum_defs: Vec<MirEnumDef>,
    /// Top-level statements (let bindings, expr stmts) are collected into
    /// a synthetic `__main` function.
    pub entry: MirFnId,
}

/// A struct (or record/class) type definition at the MIR level.
///
/// Struct definitions carry through from HIR without modification.
/// The [`is_record`](MirStructDef::is_record) flag distinguishes immutable
/// value-type records from mutable class-style structs.
#[derive(Debug, Clone)]
pub struct MirStructDef {
    /// Name of the struct type.
    pub name: String,
    /// Fields as `(field_name, type_name)` pairs, in declaration order.
    pub fields: Vec<(String, String)>,
    /// True if this is a record (immutable value type).
    pub is_record: bool,
    /// Visibility of this struct definition.
    pub vis: Visibility,
}

/// An enum type definition at the MIR level.
///
/// Contains the enum name and its ordered list of variant definitions.
#[derive(Debug, Clone)]
pub struct MirEnumDef {
    /// Name of the enum type.
    pub name: String,
    /// Variant definitions in declaration order.
    pub variants: Vec<MirVariantDef>,
}

/// A single variant of a [`MirEnumDef`].
///
/// Each variant can carry zero or more positional fields identified by
/// their type names.
#[derive(Debug, Clone)]
pub struct MirVariantDef {
    /// Name of this variant.
    pub name: String,
    /// Positional field type names.
    pub fields: Vec<String>,
}

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------

/// A MIR function definition.
///
/// Contains both the tree-form [`MirBody`] and an optional CFG representation.
/// The tree-form body is canonical after lowering; the CFG is built on demand
/// via [`build_cfg`](MirFunction::build_cfg) for analyses that require
/// explicit control-flow edges (SSA, dominators, loop analysis).
///
/// Lambda-lifted closures and the synthetic `__main` entry function are
/// represented as regular `MirFunction` instances.
#[derive(Debug, Clone)]
pub struct MirFunction {
    /// Unique function ID within the program.
    pub id: MirFnId,
    /// Function name. Lambda-lifted closures use `__closure_N` names.
    /// Impl methods use `Target.method` qualified names.
    pub name: String,
    /// Generic type parameters as `(param_name, trait_bounds)` pairs.
    pub type_params: Vec<(String, Vec<String>)>,
    /// Function parameters in declaration order.
    pub params: Vec<MirParam>,
    /// Return type name, if explicitly annotated.
    pub return_type: Option<String>,
    /// Tree-form function body (statements + optional tail expression).
    pub body: MirBody,
    /// Whether this function is annotated with `@nogc`.
    /// When true, the [`nogc_verify`] module rejects any GC-triggering operations.
    pub is_nogc: bool,
    /// CFG representation of this function's body.
    /// Built lazily from tree-form `body` via `build_cfg()`.
    /// When present, this is the canonical representation for the CFG executor.
    pub cfg_body: Option<cfg::MirCfg>,
    /// Decorator names applied to this function (e.g., `@memoize`, `@trace`).
    pub decorators: Vec<String>,
    /// Visibility of this function definition.
    pub vis: Visibility,
}

/// A function parameter at the MIR level.
///
/// Parameters carry their type annotation name and optional default value.
/// For lambda-lifted closures, capture parameters appear first with type
/// `"any"` (type-erased at MIR level).
#[derive(Debug, Clone)]
pub struct MirParam {
    /// Parameter name.
    pub name: String,
    /// Type annotation name (e.g., `"i64"`, `"f64"`, `"any"`).
    pub ty_name: String,
    /// Optional default value expression for this parameter.
    pub default: Option<MirExpr>,
    /// Variadic parameter: collects remaining args into an array.
    pub is_variadic: bool,
}

impl MirFunction {
    /// Build the CFG representation from the tree-form body.
    /// Stores the result in `cfg_body`.
    pub fn build_cfg(&mut self) {
        let cfg = cfg::CfgBuilder::build(&self.body);
        self.cfg_body = Some(cfg);
    }

    /// Return a reference to the CFG body, building it on demand if needed.
    ///
    /// Subsequent calls reuse the cached CFG. Prefer [`build_cfg`](Self::build_cfg)
    /// if you need to force a rebuild.
    pub fn cfg(&mut self) -> &cfg::MirCfg {
        if self.cfg_body.is_none() {
            self.build_cfg();
        }
        self.cfg_body.as_ref().unwrap()
    }
}

impl MirProgram {
    /// Build CFG for all functions in this program.
    pub fn build_all_cfgs(&mut self) {
        for func in &mut self.functions {
            func.build_cfg();
        }
    }
}

/// The body of a MIR function — a list of MIR statements.
/// In Milestone 2.0 we use a simplified tree-form (not full CFG with basic
/// blocks). This is extended to a proper CFG in Milestone 2.2+ for pattern
/// matching compilation.
#[derive(Debug, Clone)]
pub struct MirBody {
    pub stmts: Vec<MirStmt>,
    pub result: Option<Box<MirExpr>>,
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

/// A MIR statement.
///
/// Statements represent side-effecting or control-flow operations in the
/// tree-form MIR body. In the CFG representation, control-flow statements
/// (`If`, `While`, `Break`, `Continue`) are compiled into basic block
/// terminators and edges.
#[derive(Debug, Clone)]
pub enum MirStmt {
    /// Variable binding: `let [mut] name = init;`
    ///
    /// The [`alloc_hint`](AllocHint) is populated by escape analysis after
    /// lowering to guide allocation strategy.
    Let {
        /// Binding name.
        name: String,
        /// Whether the binding is mutable.
        mutable: bool,
        /// Initializer expression.
        init: MirExpr,
        /// Escape analysis annotation. `None` before analysis runs.
        alloc_hint: Option<AllocHint>,
    },
    /// A standalone expression statement (e.g., function call, assignment).
    Expr(MirExpr),
    /// Conditional statement: `if cond { then } [else { else_ }]`.
    If {
        /// Condition expression (must evaluate to a boolean).
        cond: MirExpr,
        /// Body executed when the condition is true.
        then_body: MirBody,
        /// Optional body executed when the condition is false.
        else_body: Option<MirBody>,
    },
    /// While loop: `while cond { body }`.
    While {
        /// Loop condition expression.
        cond: MirExpr,
        /// Loop body.
        body: MirBody,
    },
    /// Return from the current function with an optional value.
    Return(Option<MirExpr>),
    /// Break out of the innermost enclosing loop.
    Break,
    /// Continue to the next iteration of the innermost enclosing loop.
    Continue,
    /// A `nogc { ... }` block where GC-triggering operations are forbidden.
    ///
    /// Verified by [`nogc_verify::verify_nogc`].
    NoGcBlock(MirBody),
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

/// A MIR expression node.
///
/// Wraps a [`MirExprKind`] discriminant. All MIR expressions are trees
/// (no sharing / DAG structure).
#[derive(Debug, Clone)]
pub struct MirExpr {
    /// The kind of this expression.
    pub kind: MirExprKind,
}

/// The discriminant for a MIR expression.
///
/// Covers literals, variables, operators, control flow, pattern matching,
/// closures, linalg opcodes, and container constructors. Each variant
/// corresponds to a distinct runtime operation in both the tree-walk
/// interpreter (`cjc-eval`) and the MIR executor (`cjc-mir-exec`).
#[derive(Debug, Clone)]
pub enum MirExprKind {
    /// 64-bit signed integer literal.
    IntLit(i64),
    /// 64-bit IEEE 754 floating-point literal.
    FloatLit(f64),
    /// Boolean literal (`true` or `false`).
    BoolLit(bool),
    /// UTF-8 string literal.
    StringLit(String),
    /// Byte string literal (`b"..."`).
    ByteStringLit(Vec<u8>),
    /// Single byte character literal (`b'x'`).
    ByteCharLit(u8),
    /// Raw string literal (`r"..."`).
    RawStringLit(String),
    /// Raw byte string literal (`rb"..."`).
    RawByteStringLit(Vec<u8>),
    /// Regex literal with pattern and flags.
    RegexLit {
        /// The regex pattern string.
        pattern: String,
        /// Regex flags (e.g., `"gi"`).
        flags: String,
    },
    /// Tensor literal: a 2D grid of expressions (rows x columns).
    TensorLit {
        /// Each inner `Vec` is one row of the tensor.
        rows: Vec<Vec<MirExpr>>,
    },
    /// NA (missing value) literal.
    NaLit,
    /// Variable reference by name.
    Var(String),
    /// Binary operation.
    /// Binary operation.
    Binary {
        /// The binary operator.
        op: BinOp,
        /// Left-hand operand.
        left: Box<MirExpr>,
        /// Right-hand operand.
        right: Box<MirExpr>,
    },
    /// Unary operation (negation, logical not, bitwise not).
    Unary {
        /// The unary operator.
        op: UnaryOp,
        /// The operand.
        operand: Box<MirExpr>,
    },
    /// Function or closure call.
    Call {
        /// The callee expression (usually a [`Var`](MirExprKind::Var) or
        /// [`Field`](MirExprKind::Field) for method calls).
        callee: Box<MirExpr>,
        /// Positional arguments.
        args: Vec<MirExpr>,
    },
    /// Field access: `object.name`.
    Field {
        /// The object being accessed.
        object: Box<MirExpr>,
        /// Field name.
        name: String,
    },
    /// Single-index access: `object[index]`.
    Index {
        /// The collection being indexed.
        object: Box<MirExpr>,
        /// The index expression.
        index: Box<MirExpr>,
    },
    /// Multi-dimensional index access: `object[i, j, ...]`.
    MultiIndex {
        /// The collection being indexed.
        object: Box<MirExpr>,
        /// Index expressions for each dimension.
        indices: Vec<MirExpr>,
    },
    /// Assignment: `target = value`.
    Assign {
        /// Assignment target (variable, field, or index expression).
        target: Box<MirExpr>,
        /// Value being assigned.
        value: Box<MirExpr>,
    },
    /// Block expression: evaluates a [`MirBody`] and returns its result.
    Block(MirBody),
    /// Struct literal: `Name { field1: expr1, field2: expr2, ... }`.
    StructLit {
        /// Struct type name.
        name: String,
        /// Field initializers as `(name, value)` pairs.
        fields: Vec<(String, MirExpr)>,
    },
    /// Array literal: `[expr1, expr2, ...]`.
    ArrayLit(Vec<MirExpr>),
    /// Column reference in a data DSL context (e.g., `col("name")`).
    Col(String),
    /// Lambda expression (non-capturing).
    Lambda {
        /// Lambda parameters.
        params: Vec<MirParam>,
        /// Lambda body expression.
        body: Box<MirExpr>,
    },
    /// Create a closure: captures + a reference to the lifted function.
    /// At runtime, evaluates each capture expression and bundles them with
    /// the function name into a Closure value.
    MakeClosure {
        /// Name of the lambda-lifted top-level function.
        fn_name: String,
        /// Expressions that produce the captured values (evaluated at closure
        /// creation time). Order matches the extra leading params of the
        /// lifted function.
        captures: Vec<MirExpr>,
    },
    /// If expression: `if cond { then } [else { else_ }]`.
    ///
    /// Used as both a statement and an expression (the branch bodies can
    /// produce values).
    If {
        /// Condition expression.
        cond: Box<MirExpr>,
        /// Body evaluated when the condition is true.
        then_body: MirBody,
        /// Optional body evaluated when the condition is false.
        else_body: Option<MirBody>,
    },
    /// Match expression compiled as a decision tree.
    /// Each arm is tried in order; first matching arm's body is evaluated.
    Match {
        /// The value being matched against.
        scrutinee: Box<MirExpr>,
        /// Match arms in order of priority.
        arms: Vec<MirMatchArm>,
    },
    /// Enum variant literal constructor: `EnumName::Variant(fields...)`.
    VariantLit {
        /// Enum type name.
        enum_name: String,
        /// Variant name.
        variant: String,
        /// Positional field values.
        fields: Vec<MirExpr>,
    },
    /// Tuple literal: `(expr1, expr2, ...)`.
    TupleLit(Vec<MirExpr>),
    /// LU decomposition opcode.
    LinalgLU {
        /// Matrix operand.
        operand: Box<MirExpr>,
    },
    /// QR decomposition opcode.
    LinalgQR {
        /// Matrix operand.
        operand: Box<MirExpr>,
    },
    /// Cholesky decomposition opcode.
    LinalgCholesky {
        /// Matrix operand (must be symmetric positive-definite).
        operand: Box<MirExpr>,
    },
    /// Matrix inverse opcode.
    LinalgInv {
        /// Matrix operand.
        operand: Box<MirExpr>,
    },
    /// Broadcast a tensor to a target shape (zero-copy view with stride=0).
    Broadcast {
        /// Tensor operand to broadcast.
        operand: Box<MirExpr>,
        /// Target shape dimensions.
        target_shape: Vec<MirExpr>,
    },
    /// Unit/void value (no meaningful result).
    Void,
}

// ---------------------------------------------------------------------------
// Match / Pattern types (MIR level)
// ---------------------------------------------------------------------------

/// A match arm at MIR level: pattern + body.
#[derive(Debug, Clone)]
pub struct MirMatchArm {
    pub pattern: MirPattern,
    pub body: MirBody,
}

/// A pattern at MIR level.
#[derive(Debug, Clone)]
pub enum MirPattern {
    /// Wildcard: matches anything, binds nothing.
    Wildcard,
    /// Binding: matches anything, binds the value to a name.
    Binding(String),
    /// Literal patterns
    LitInt(i64),
    LitFloat(f64),
    LitBool(bool),
    LitString(String),
    /// Tuple destructuring
    Tuple(Vec<MirPattern>),
    /// Struct destructuring
    Struct {
        name: String,
        fields: Vec<(String, MirPattern)>,
    },
    /// Enum variant pattern
    Variant {
        enum_name: String,
        variant: String,
        fields: Vec<MirPattern>,
    },
}

// ===========================================================================
// HIR -> MIR Lowering
// ===========================================================================

use cjc_hir::*;

/// Lowers HIR into MIR.
///
/// Performs a single-pass traversal of the [`HirProgram`], converting each
/// HIR item into its MIR equivalent. During lowering:
///
/// - Top-level statements are collected into a synthetic `__main` function.
/// - Closures are lambda-lifted into top-level functions with extra leading
///   parameters for captured values, and replaced with [`MirExprKind::MakeClosure`].
/// - Impl methods are flattened to qualified `Target.method` names.
/// - Traits produce no MIR output (metadata only).
///
/// # Usage
///
/// ```rust,ignore
/// let mut lowering = HirToMir::new();
/// let mir_program = lowering.lower_program(&hir_program);
/// ```
pub struct HirToMir {
    next_fn_id: u32,
    next_lambda_id: u32,
    /// Lambda-lifted functions accumulated during lowering.
    /// These are appended to the MirProgram's function list.
    lifted_functions: Vec<MirFunction>,
}

impl HirToMir {
    /// Create a new HIR-to-MIR lowering pass with fresh ID counters.
    pub fn new() -> Self {
        Self {
            next_fn_id: 0,
            next_lambda_id: 0,
            lifted_functions: Vec::new(),
        }
    }

    fn fresh_fn_id(&mut self) -> MirFnId {
        let id = MirFnId(self.next_fn_id);
        self.next_fn_id += 1;
        id
    }

    fn fresh_lambda_name(&mut self) -> String {
        let name = format!("__closure_{}", self.next_lambda_id);
        self.next_lambda_id += 1;
        name
    }

    /// Lower a HIR program to MIR.
    pub fn lower_program(&mut self, hir: &HirProgram) -> MirProgram {
        let mut functions = Vec::new();
        let mut struct_defs = Vec::new();
        let mut enum_defs = Vec::new();
        let mut main_stmts: Vec<MirStmt> = Vec::new();

        for item in &hir.items {
            match item {
                HirItem::Fn(f) => {
                    functions.push(self.lower_fn(f));
                }
                HirItem::Struct(s) => {
                    struct_defs.push(MirStructDef {
                        name: s.name.clone(),
                        fields: s.fields.clone(),
                        is_record: false,
                        vis: s.vis,
                    });
                }
                HirItem::Class(c) => {
                    struct_defs.push(MirStructDef {
                        name: c.name.clone(),
                        fields: c.fields.clone(),
                        is_record: false,
                        vis: c.vis,
                    });
                }
                HirItem::Record(r) => {
                    struct_defs.push(MirStructDef {
                        name: r.name.clone(),
                        fields: r.fields.clone(),
                        is_record: true,
                        vis: r.vis,
                    });
                }
                HirItem::Enum(e) => {
                    enum_defs.push(MirEnumDef {
                        name: e.name.clone(),
                        variants: e
                            .variants
                            .iter()
                            .map(|v| MirVariantDef {
                                name: v.name.clone(),
                                fields: v.fields.clone(),
                            })
                            .collect(),
                    });
                }
                HirItem::Let(l) => {
                    main_stmts.push(MirStmt::Let {
                        name: l.name.clone(),
                        mutable: l.mutable,
                        init: self.lower_expr(&l.init),
                        alloc_hint: None,
                    });
                }
                HirItem::Stmt(s) => {
                    main_stmts.push(self.lower_stmt(s));
                }
                HirItem::Impl(i) => {
                    for method in &i.methods {
                        // Register as qualified name: Target.method
                        let mut mir_fn = self.lower_fn(method);
                        mir_fn.name = format!("{}.{}", i.target, method.name);
                        functions.push(mir_fn);
                    }
                }
                HirItem::Trait(_) => {
                    // Traits are metadata only; no MIR output
                }
            }
        }

        // Create __main entry function from top-level statements
        let main_id = self.fresh_fn_id();
        functions.push(MirFunction {
            id: main_id,
            name: "__main".to_string(),
            type_params: vec![],
            params: vec![],
            return_type: None,
            body: MirBody {
                stmts: main_stmts,
                result: None,
            },
            is_nogc: false,
            cfg_body: None,
            decorators: vec![],
            vis: Visibility::Private,
        });

        // Append all lambda-lifted functions
        functions.append(&mut self.lifted_functions);

        MirProgram {
            functions,
            struct_defs,
            enum_defs,
            entry: main_id,
        }
    }

    /// Lower a single HIR function definition to a [`MirFunction`].
    ///
    /// Assigns a fresh [`MirFnId`] and recursively lowers parameters, body
    /// statements, and the tail expression. Closures encountered within the
    /// body are lambda-lifted and accumulated in `self.lifted_functions`.
    pub fn lower_fn(&mut self, f: &HirFn) -> MirFunction {
        let id = self.fresh_fn_id();
        let params = f
            .params
            .iter()
            .map(|p| MirParam {
                name: p.name.clone(),
                ty_name: p.ty_name.clone(),
                default: p.default.as_ref().map(|d| self.lower_expr(d)),
                is_variadic: p.is_variadic,
            })
            .collect();
        let body = self.lower_block(&f.body);
        MirFunction {
            id,
            name: f.name.clone(),
            type_params: f.type_params.clone(),
            params,
            return_type: f.return_type.clone(),
            body,
            is_nogc: f.is_nogc,
            cfg_body: None,
            decorators: f.decorators.clone(),
            vis: f.vis,
        }
    }

    fn lower_block(&mut self, block: &HirBlock) -> MirBody {
        let stmts = block.stmts.iter().map(|s| self.lower_stmt(s)).collect();
        let result = block.expr.as_ref().map(|e| Box::new(self.lower_expr(e)));
        MirBody { stmts, result }
    }

    fn lower_stmt(&mut self, stmt: &HirStmt) -> MirStmt {
        match &stmt.kind {
            HirStmtKind::Let {
                name,
                mutable,
                init,
                ..
            } => MirStmt::Let {
                name: name.clone(),
                mutable: *mutable,
                init: self.lower_expr(init),
                alloc_hint: None,
            },
            HirStmtKind::Expr(e) => MirStmt::Expr(self.lower_expr(e)),
            HirStmtKind::If(if_expr) => self.lower_if_stmt(if_expr),
            HirStmtKind::While { cond, body } => MirStmt::While {
                cond: self.lower_expr(cond),
                body: self.lower_block(body),
            },
            HirStmtKind::Return(e) => {
                MirStmt::Return(e.as_ref().map(|ex| self.lower_expr(ex)))
            }
            HirStmtKind::Break => MirStmt::Break,
            HirStmtKind::Continue => MirStmt::Continue,
            HirStmtKind::NoGcBlock(block) => MirStmt::NoGcBlock(self.lower_block(block)),
        }
    }

    /// Lower a HIR `if` expression to a [`MirStmt::If`].
    ///
    /// Nested `else if` chains are recursively lowered into nested
    /// [`MirStmt::If`] nodes wrapped in a [`MirBody`].
    pub fn lower_if_stmt(&mut self, if_expr: &HirIfExpr) -> MirStmt {
        let cond = self.lower_expr(&if_expr.cond);
        let then_body = self.lower_block(&if_expr.then_block);
        let else_body = if_expr.else_branch.as_ref().map(|eb| match eb {
            HirElseBranch::ElseIf(elif) => {
                // Nested if-else becomes a block containing the if stmt
                let nested = self.lower_if_stmt(elif);
                MirBody {
                    stmts: vec![nested],
                    result: None,
                }
            }
            HirElseBranch::Else(block) => self.lower_block(block),
        });
        MirStmt::If {
            cond,
            then_body,
            else_body,
        }
    }

    /// Lower a HIR expression to a [`MirExpr`].
    ///
    /// Handles all HIR expression kinds including closures (lambda-lifted),
    /// match expressions (compiled to [`MirExprKind::Match`] decision trees),
    /// and if-expressions.
    pub fn lower_expr(&mut self, expr: &HirExpr) -> MirExpr {
        let kind = match &expr.kind {
            HirExprKind::IntLit(v) => MirExprKind::IntLit(*v),
            HirExprKind::FloatLit(v) => MirExprKind::FloatLit(*v),
            HirExprKind::BoolLit(b) => MirExprKind::BoolLit(*b),
            HirExprKind::NaLit => MirExprKind::NaLit,
            HirExprKind::StringLit(s) => MirExprKind::StringLit(s.clone()),
            HirExprKind::ByteStringLit(bytes) => MirExprKind::ByteStringLit(bytes.clone()),
            HirExprKind::ByteCharLit(b) => MirExprKind::ByteCharLit(*b),
            HirExprKind::RawStringLit(s) => MirExprKind::RawStringLit(s.clone()),
            HirExprKind::RawByteStringLit(bytes) => MirExprKind::RawByteStringLit(bytes.clone()),
            HirExprKind::RegexLit { pattern, flags } => MirExprKind::RegexLit { pattern: pattern.clone(), flags: flags.clone() },
            HirExprKind::TensorLit { rows } => {
                let mir_rows = rows.iter().map(|row| {
                    row.iter().map(|e| self.lower_expr(e)).collect()
                }).collect();
                MirExprKind::TensorLit { rows: mir_rows }
            }
            HirExprKind::Var(name) => MirExprKind::Var(name.clone()),
            HirExprKind::Binary { op, left, right } => MirExprKind::Binary {
                op: *op,
                left: Box::new(self.lower_expr(left)),
                right: Box::new(self.lower_expr(right)),
            },
            HirExprKind::Unary { op, operand } => MirExprKind::Unary {
                op: *op,
                operand: Box::new(self.lower_expr(operand)),
            },
            HirExprKind::Call { callee, args } => MirExprKind::Call {
                callee: Box::new(self.lower_expr(callee)),
                args: args.iter().map(|a| self.lower_expr(a)).collect(),
            },
            HirExprKind::Field { object, name } => MirExprKind::Field {
                object: Box::new(self.lower_expr(object)),
                name: name.clone(),
            },
            HirExprKind::Index { object, index } => MirExprKind::Index {
                object: Box::new(self.lower_expr(object)),
                index: Box::new(self.lower_expr(index)),
            },
            HirExprKind::MultiIndex { object, indices } => MirExprKind::MultiIndex {
                object: Box::new(self.lower_expr(object)),
                indices: indices.iter().map(|i| self.lower_expr(i)).collect(),
            },
            HirExprKind::Assign { target, value } => MirExprKind::Assign {
                target: Box::new(self.lower_expr(target)),
                value: Box::new(self.lower_expr(value)),
            },
            HirExprKind::Block(block) => MirExprKind::Block(self.lower_block(block)),
            HirExprKind::StructLit { name, fields } => MirExprKind::StructLit {
                name: name.clone(),
                fields: fields
                    .iter()
                    .map(|(n, e)| (n.clone(), self.lower_expr(e)))
                    .collect(),
            },
            HirExprKind::ArrayLit(elems) => {
                MirExprKind::ArrayLit(elems.iter().map(|e| self.lower_expr(e)).collect())
            }
            HirExprKind::Col(name) => MirExprKind::Col(name.clone()),
            HirExprKind::Lambda { params, body } => MirExprKind::Lambda {
                params: params
                    .iter()
                    .map(|p| MirParam {
                        name: p.name.clone(),
                        ty_name: p.ty_name.clone(),
                        default: p.default.as_ref().map(|d| self.lower_expr(d)),
                        is_variadic: p.is_variadic,
                    })
                    .collect(),
                body: Box::new(self.lower_expr(body)),
            },
            HirExprKind::Closure {
                params,
                body,
                captures,
            } => {
                // Lambda-lift: create a top-level function with extra
                // leading parameters for the captured values.
                let lifted_name = self.fresh_lambda_name();
                let lifted_id = self.fresh_fn_id();

                // Build params: captures first, then the original params
                let mut lifted_params: Vec<MirParam> = captures
                    .iter()
                    .map(|c| MirParam {
                        name: c.name.clone(),
                        ty_name: "any".to_string(), // Type erasure at MIR level
                        default: None,
                        is_variadic: false,
                    })
                    .collect();
                for p in params {
                    lifted_params.push(MirParam {
                        name: p.name.clone(),
                        ty_name: p.ty_name.clone(),
                        default: p.default.as_ref().map(|d| self.lower_expr(d)),
                        is_variadic: p.is_variadic,
                    });
                }

                let lifted_body = MirBody {
                    stmts: vec![],
                    result: Some(Box::new(self.lower_expr(body))),
                };

                self.lifted_functions.push(MirFunction {
                    id: lifted_id,
                    name: lifted_name.clone(),
                    type_params: vec![],
                    params: lifted_params,
                    return_type: None,
                    body: lifted_body,
                    is_nogc: false,
                    cfg_body: None,
                    decorators: vec![],
                    vis: Visibility::Private,
                });

                // At the call site, emit MakeClosure with the capture
                // variable references as capture expressions
                let capture_exprs: Vec<MirExpr> = captures
                    .iter()
                    .map(|c| MirExpr {
                        kind: MirExprKind::Var(c.name.clone()),
                    })
                    .collect();

                MirExprKind::MakeClosure {
                    fn_name: lifted_name,
                    captures: capture_exprs,
                }
            }
            HirExprKind::Match { scrutinee, arms } => {
                let mir_scrutinee = Box::new(self.lower_expr(scrutinee));
                let mir_arms = arms
                    .iter()
                    .map(|arm| {
                        let pattern = self.lower_pattern(&arm.pattern);
                        let body = MirBody {
                            stmts: vec![],
                            result: Some(Box::new(self.lower_expr(&arm.body))),
                        };
                        MirMatchArm { pattern, body }
                    })
                    .collect();
                MirExprKind::Match {
                    scrutinee: mir_scrutinee,
                    arms: mir_arms,
                }
            }
            HirExprKind::TupleLit(elems) => {
                MirExprKind::TupleLit(elems.iter().map(|e| self.lower_expr(e)).collect())
            }
            HirExprKind::VariantLit {
                enum_name,
                variant,
                fields,
            } => MirExprKind::VariantLit {
                enum_name: enum_name.clone(),
                variant: variant.clone(),
                fields: fields.iter().map(|f| self.lower_expr(f)).collect(),
            },
            HirExprKind::If { cond, then_block, else_branch } => {
                let mir_cond = Box::new(self.lower_expr(cond));
                let mir_then = self.lower_block(then_block);
                let mir_else = else_branch.as_ref().map(|eb| match eb {
                    HirElseBranch::ElseIf(elif) => {
                        // Nested else-if: lower as MirStmt::If inside a MirBody
                        let nested = self.lower_if_stmt(elif);
                        MirBody {
                            stmts: vec![nested],
                            result: None,
                        }
                    }
                    HirElseBranch::Else(block) => self.lower_block(block),
                });
                MirExprKind::If {
                    cond: mir_cond,
                    then_body: mir_then,
                    else_body: mir_else,
                }
            }
            HirExprKind::Void => MirExprKind::Void,
        };
        MirExpr { kind }
    }

    fn lower_pattern(&self, pat: &HirPattern) -> MirPattern {
        match &pat.kind {
            HirPatternKind::Wildcard => MirPattern::Wildcard,
            HirPatternKind::Binding(name) => MirPattern::Binding(name.clone()),
            HirPatternKind::LitInt(v) => MirPattern::LitInt(*v),
            HirPatternKind::LitFloat(v) => MirPattern::LitFloat(*v),
            HirPatternKind::LitBool(b) => MirPattern::LitBool(*b),
            HirPatternKind::LitString(s) => MirPattern::LitString(s.clone()),
            HirPatternKind::Tuple(pats) => {
                MirPattern::Tuple(pats.iter().map(|p| self.lower_pattern(p)).collect())
            }
            HirPatternKind::Struct { name, fields } => MirPattern::Struct {
                name: name.clone(),
                fields: fields
                    .iter()
                    .map(|f| (f.name.clone(), self.lower_pattern(&f.pattern)))
                    .collect(),
            },
            HirPatternKind::Variant {
                enum_name,
                variant,
                fields,
            } => MirPattern::Variant {
                enum_name: enum_name.clone(),
                variant: variant.clone(),
                fields: fields.iter().map(|f| self.lower_pattern(f)).collect(),
            },
        }
    }
}

impl Default for HirToMir {
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
    use cjc_hir::*;

    fn hir_id(n: u32) -> HirId {
        HirId(n)
    }

    fn hir_int(v: i64) -> HirExpr {
        HirExpr {
            kind: HirExprKind::IntLit(v),
            hir_id: hir_id(0),
        }
    }

    fn hir_var(name: &str) -> HirExpr {
        HirExpr {
            kind: HirExprKind::Var(name.to_string()),
            hir_id: hir_id(0),
        }
    }

    #[test]
    fn test_lower_hir_literal() {
        let mut lowering = HirToMir::new();
        let hir = hir_int(42);
        let mir = lowering.lower_expr(&hir);
        assert!(matches!(mir.kind, MirExprKind::IntLit(42)));
    }

    #[test]
    fn test_lower_hir_binary() {
        let mut lowering = HirToMir::new();
        let hir = HirExpr {
            kind: HirExprKind::Binary {
                op: BinOp::Add,
                left: Box::new(hir_int(1)),
                right: Box::new(hir_int(2)),
            },
            hir_id: hir_id(0),
        };
        let mir = lowering.lower_expr(&hir);
        match &mir.kind {
            MirExprKind::Binary { op, .. } => assert_eq!(*op, BinOp::Add),
            _ => panic!("expected Binary"),
        }
    }

    #[test]
    fn test_lower_hir_fn() {
        let mut lowering = HirToMir::new();
        let hir_fn = HirFn {
            name: "add".to_string(),
            type_params: vec![],
            params: vec![
                HirParam {
                    name: "a".to_string(),
                    ty_name: "i64".to_string(),
                    default: None,
                    is_variadic: false,
                    hir_id: hir_id(1),
                },
                HirParam {
                    name: "b".to_string(),
                    ty_name: "i64".to_string(),
                    default: None,
                    is_variadic: false,
                    hir_id: hir_id(2),
                },
            ],
            return_type: Some("i64".to_string()),
            body: HirBlock {
                stmts: vec![],
                expr: Some(Box::new(HirExpr {
                    kind: HirExprKind::Binary {
                        op: BinOp::Add,
                        left: Box::new(hir_var("a")),
                        right: Box::new(hir_var("b")),
                    },
                    hir_id: hir_id(3),
                })),
                hir_id: hir_id(4),
            },
            is_nogc: false,
            hir_id: hir_id(5),
            decorators: vec![],
            vis: cjc_ast::Visibility::Private,
        };
        let mir_fn = lowering.lower_fn(&hir_fn);
        assert_eq!(mir_fn.name, "add");
        assert_eq!(mir_fn.params.len(), 2);
        assert!(mir_fn.body.result.is_some());
    }

    #[test]
    fn test_lower_hir_program_entry() {
        let mut lowering = HirToMir::new();
        let hir = HirProgram {
            items: vec![
                HirItem::Let(HirLetDecl {
                    name: "x".to_string(),
                    mutable: false,
                    ty_name: None,
                    init: hir_int(42),
                    hir_id: hir_id(0),
                }),
                HirItem::Fn(HirFn {
                    name: "f".to_string(),
                    type_params: vec![],
                    params: vec![],
                    return_type: None,
                    body: HirBlock {
                        stmts: vec![],
                        expr: Some(Box::new(hir_var("x"))),
                        hir_id: hir_id(1),
                    },
                    is_nogc: false,
                    hir_id: hir_id(2),
                    decorators: vec![],
                    vis: cjc_ast::Visibility::Private,
                }),
            ],
        };
        let mir = lowering.lower_program(&hir);
        // Should have: function 'f' + synthetic __main
        assert_eq!(mir.functions.len(), 2);
        let main = mir.functions.iter().find(|f| f.name == "__main").unwrap();
        assert_eq!(main.body.stmts.len(), 1); // the let x = 42
        assert_eq!(mir.entry, main.id);
    }

    #[test]
    fn test_lower_hir_if_stmt() {
        let mut lowering = HirToMir::new();
        let hir_if = HirIfExpr {
            cond: Box::new(HirExpr {
                kind: HirExprKind::BoolLit(true),
                hir_id: hir_id(0),
            }),
            then_block: HirBlock {
                stmts: vec![],
                expr: Some(Box::new(hir_int(1))),
                hir_id: hir_id(1),
            },
            else_branch: Some(HirElseBranch::Else(HirBlock {
                stmts: vec![],
                expr: Some(Box::new(hir_int(2))),
                hir_id: hir_id(2),
            })),
            hir_id: hir_id(3),
        };
        let mir_stmt = lowering.lower_if_stmt(&hir_if);
        match &mir_stmt {
            MirStmt::If {
                then_body,
                else_body,
                ..
            } => {
                assert!(then_body.result.is_some());
                assert!(else_body.is_some());
            }
            _ => panic!("expected If"),
        }
    }

    #[test]
    fn test_lower_struct_def() {
        let mut lowering = HirToMir::new();
        let hir = HirProgram {
            items: vec![HirItem::Struct(HirStructDef {
                name: "Point".to_string(),
                fields: vec![
                    ("x".to_string(), "f64".to_string()),
                    ("y".to_string(), "f64".to_string()),
                ],
                hir_id: hir_id(0),
                vis: cjc_ast::Visibility::Private,
            })],
        };
        let mir = lowering.lower_program(&hir);
        assert_eq!(mir.struct_defs.len(), 1);
        assert_eq!(mir.struct_defs[0].name, "Point");
        assert_eq!(mir.struct_defs[0].fields.len(), 2);
    }
}
