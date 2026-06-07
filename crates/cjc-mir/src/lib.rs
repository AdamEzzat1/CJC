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
pub mod fusion_rewrite;
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
use std::collections::BTreeMap;
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
    /// Tier-0 perf: total number of local slots used by this function
    /// (parameters + `let` bindings, including those in nested blocks).
    ///
    /// Populated by the slot-resolution pass in `HirToMir`. Used by the
    /// executor to size the call frame in one allocation rather than
    /// growing it incrementally. `0` means "no slot resolution was
    /// performed; fall back to name-based scope lookup."
    pub local_count: u32,
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
        /// Tier-0 perf (T0-b Stage 3): statically resolved frame slot for
        /// this binding, populated by the slot-resolution pass in
        /// `HirToMir`. `Some(slot)` when slot resolution was active for
        /// the enclosing function (the executor writes
        /// `frame[base + slot] = init_value`); `None` otherwise (the
        /// executor falls back to `self.define(name, val)` for `__main`,
        /// lambda-lifted closure bodies, and match arm bodies).
        slot: Option<u32>,
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
    /// Variable reference by name (unresolved fallback path).
    ///
    /// Used for closures, captured variables, top-level/global references,
    /// and any case where the slot-resolution pass in `HirToMir` couldn't
    /// statically determine a slot. The executor falls back to walking
    /// the scope chain by name for these.
    Var(String),
    /// Tier-0 fast-path: variable reference resolved to a flat slot
    /// index into the current call frame.
    ///
    /// Emitted by `HirToMir` when the variable refers to a function-local
    /// binding (parameter or `let`) and the slot was resolvable
    /// statically. The executor reads directly from `frame[slot]` without
    /// touching the scope chain. The `name` field is retained for
    /// debugging and for the `MirExpr` printer; runtime dispatch uses
    /// `slot` only.
    ///
    /// See ADR-... (Tier-0 perf work) for the design rationale.
    VarLocal {
        /// Original binding name (debugging / printer use only).
        name: String,
        /// 0-indexed slot in the current call frame.
        /// `slot < MirFunction::local_count` is an invariant maintained
        /// by the lowering pass.
        slot: u32,
    },
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
    ///
    /// `slot` is populated by the slot-resolution pass in `HirToMir`
    /// (T0-b Stage 4). `Some(slot)` when the binding is inside a
    /// function whose body is slot-resolved (regular fn bodies, closure
    /// bodies). The executor writes both `frame[base + slot] = value`
    /// AND `self.define(name, value)` when binding. `None` for patterns
    /// outside any slot-resolved function (currently never -- match is
    /// only valid inside a function -- but kept for symmetry with the
    /// `MirStmt::Let.slot` rule).
    Binding { name: String, slot: Option<u32> },
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

    // ---------- Tier-0 perf (T0-b Stage 2) slot-resolution state ----------
    //
    // These fields are active only inside `lower_fn`. The synthetic `__main`
    // and lambda-lifted closures are NOT slot-resolved in Stage 2 (they keep
    // `local_count = 0` and emit `MirExprKind::Var` for every reference, which
    // makes the executor fall back to name-based scope lookup).
    /// Stack of lexical scopes mapping `name -> slot` for the current
    /// function being lowered. `BTreeMap` (not `HashMap`) preserves
    /// deterministic iteration if it ever leaks into output.
    scope_stack: Vec<BTreeMap<String, u32>>,
    /// Monotonic per-function slot counter. Never decrements when scopes
    /// pop, so a function with `if { let x } else { let y }` consumes two
    /// frame slots (one each). This trades a small space cost for a much
    /// simpler implementation.
    slot_counter: u32,
    /// When `false`, `Var` references stay as `Var(name)` instead of
    /// becoming `VarLocal { name, slot }`. Used to skip:
    /// - the `__main` synthetic function body
    /// - lambda-lifted closure bodies (deferred to Stage 4)
    /// - match arm bodies (pattern-bound names are not tracked here;
    ///   deferred to Stage 4. Emitting VarLocal with the outer slot would
    ///   work in Stage 2 (executor still does name lookup) but would bake
    ///   in a latent bug for Stage 3's frame-based reads.)
    slot_resolution_active: bool,
}

/// Snapshot of the slot tracker that can be saved / restored across a
/// nested lowering (closure body, match arm body, etc.).
#[derive(Debug)]
struct TrackerState {
    scope_stack: Vec<BTreeMap<String, u32>>,
    slot_counter: u32,
    slot_resolution_active: bool,
}

impl HirToMir {
    /// Create a new HIR-to-MIR lowering pass with fresh ID counters.
    pub fn new() -> Self {
        Self {
            next_fn_id: 0,
            next_lambda_id: 0,
            lifted_functions: Vec::new(),
            scope_stack: Vec::new(),
            slot_counter: 0,
            slot_resolution_active: false,
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

    // -- Slot tracker helpers (Tier-0 perf, T0-b Stage 2) ------------------

    /// Begin tracking slots for a new function. Clears any previous state,
    /// pushes the initial scope, and assigns each parameter a sequential
    /// slot starting at 0.
    fn enter_function(&mut self, params: &[MirParam]) {
        self.scope_stack.clear();
        self.scope_stack.push(BTreeMap::new());
        self.slot_counter = 0;
        self.slot_resolution_active = true;
        for p in params {
            self.define_local(&p.name);
        }
    }

    /// Finish tracking slots for the current function and return the total
    /// slot count (= `MirFunction.local_count`). Resets the tracker.
    fn exit_function(&mut self) -> u32 {
        let count = self.slot_counter;
        self.scope_stack.clear();
        self.slot_counter = 0;
        self.slot_resolution_active = false;
        count
    }

    fn push_scope(&mut self) {
        if self.slot_resolution_active {
            self.scope_stack.push(BTreeMap::new());
        }
    }

    fn pop_scope(&mut self) {
        if self.slot_resolution_active {
            self.scope_stack.pop();
        }
    }

    /// Register `name` as a local in the top scope, assigning it the next
    /// available slot. Returns the assigned slot.
    fn define_local(&mut self, name: &str) -> u32 {
        let slot = self.slot_counter;
        if let Some(top) = self.scope_stack.last_mut() {
            top.insert(name.to_string(), slot);
        }
        self.slot_counter += 1;
        slot
    }

    /// Walk the scope stack from innermost outward, returning the slot for
    /// `name` if it is bound as a local. Returns `None` for top-level
    /// names, captured variables, function names, etc. — these emit
    /// `MirExprKind::Var(name)` and rely on the executor's scope-chain
    /// fallback.
    fn resolve_local(&self, name: &str) -> Option<u32> {
        if !self.slot_resolution_active {
            return None;
        }
        for scope in self.scope_stack.iter().rev() {
            if let Some(&slot) = scope.get(name) {
                return Some(slot);
            }
        }
        None
    }

    /// Snapshot the tracker so a nested lowering can run with its own
    /// (or disabled) state.
    fn save_tracker(&mut self) -> TrackerState {
        TrackerState {
            scope_stack: std::mem::take(&mut self.scope_stack),
            slot_counter: self.slot_counter,
            slot_resolution_active: self.slot_resolution_active,
        }
    }

    /// Restore a previously saved tracker state.
    fn restore_tracker(&mut self, saved: TrackerState) {
        self.scope_stack = saved.scope_stack;
        self.slot_counter = saved.slot_counter;
        self.slot_resolution_active = saved.slot_resolution_active;
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
                    // __main has no slot tracker (Stage 2 left __main on
                    // name fallback). slot stays None; the executor will
                    // route this through self.define(name, val).
                    main_stmts.push(MirStmt::Let {
                        name: l.name.clone(),
                        mutable: l.mutable,
                        init: self.lower_expr(&l.init),
                        alloc_hint: None,
                        slot: None,
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
            local_count: 0,
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

        // Param defaults are evaluated AT CALL SITES in the caller's scope,
        // not inside the callee's frame. Lower them with the outer tracker
        // state (or whatever state is active when `lower_fn` is called).
        // We snapshot the tracker so `enter_function` below can claim a
        // fresh slot space for the callee body.
        let saved = self.save_tracker();

        let params: Vec<MirParam> = f
            .params
            .iter()
            .map(|p| MirParam {
                name: p.name.clone(),
                ty_name: p.ty_name.clone(),
                default: p.default.as_ref().map(|d| self.lower_expr(d)),
                is_variadic: p.is_variadic,
            })
            .collect();

        // Tier-0 perf (Stage 2): begin slot tracking for this function body.
        // Parameters are assigned slots 0..N in declaration order.
        self.enter_function(&params);
        let body = self.lower_block(&f.body);
        let local_count = self.exit_function();

        // Restore the outer tracker state (typically inactive for top-level
        // fns; may be active when `lower_fn` is invoked from an Impl method
        // iteration after a closure restore -- belt and suspenders).
        self.restore_tracker(saved);

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
            local_count,
        }
    }

    fn lower_block(&mut self, block: &HirBlock) -> MirBody {
        // Tier-0 perf: each block boundary opens a new lexical scope so that
        // shadowing `let` bindings consume distinct slots. The slot counter
        // does NOT reset on pop -- a function with `if { let x } else { let y }`
        // consumes two slots, not one. See `define_local` doc.
        self.push_scope();
        let stmts = block.stmts.iter().map(|s| self.lower_stmt(s)).collect();
        let result = block.expr.as_ref().map(|e| Box::new(self.lower_expr(e)));
        self.pop_scope();
        MirBody { stmts, result }
    }

    fn lower_stmt(&mut self, stmt: &HirStmt) -> MirStmt {
        match &stmt.kind {
            HirStmtKind::Let {
                name,
                mutable,
                init,
                ..
            } => {
                // Lower the initializer BEFORE binding `name` so the RHS
                // resolves to the outer (not the new) binding for shadowing
                // cases like `let x = x + 1`.
                let init = self.lower_expr(init);
                // Tier-0 perf (Stage 3): record the slot the executor will
                // write into. `Some` when slot resolution is active for
                // this function; `None` otherwise (e.g. inside match arm
                // bodies or closure bodies).
                let slot = if self.slot_resolution_active {
                    Some(self.define_local(name))
                } else {
                    None
                };
                MirStmt::Let {
                    name: name.clone(),
                    mutable: *mutable,
                    init,
                    alloc_hint: None,
                    slot,
                }
            }
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
            HirExprKind::Var(name) => {
                // Tier-0 perf (Stage 2): if `name` is bound as a local
                // (parameter or `let`) in the current function, emit the
                // slot-resolved `VarLocal` variant. Otherwise fall back to
                // `Var(name)` (top-level function, captured variable,
                // pattern binding, or any reference outside an active
                // tracker).
                match self.resolve_local(name) {
                    Some(slot) => MirExprKind::VarLocal {
                        name: name.clone(),
                        slot,
                    },
                    None => MirExprKind::Var(name.clone()),
                }
            }
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

                // Build params: captures first, then the original params.
                // Default expressions on params are evaluated in the OUTER
                // (call-site) scope, so lower them BEFORE saving the
                // tracker for the closure body.
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

                // The capture expressions are simple Var references in the
                // OUTER scope. Slot-resolve them while the outer tracker
                // is still active -- `MakeClosure` evaluates these every
                // time it runs to bundle them with the lifted function name.
                let capture_exprs: Vec<MirExpr> = captures
                    .iter()
                    .map(|c| {
                        let kind = match self.resolve_local(&c.name) {
                            Some(slot) => MirExprKind::VarLocal {
                                name: c.name.clone(),
                                slot,
                            },
                            None => MirExprKind::Var(c.name.clone()),
                        };
                        MirExpr { kind }
                    })
                    .collect();

                // Tier-0 perf (Stage 4): slot-resolve the lifted body just
                // like a regular function. The lifted params (captures
                // first, then original) get slots 0..N in declaration
                // order; lets inside the body get slots after. The
                // executor's `call_function` path pushes a frame on entry
                // and binds args (including captures, which the
                // `MakeClosure` mechanism prepends) into the frame slots.
                //
                // Snapshot the outer tracker so the closure body's slot
                // space is fresh -- nested closures and the outer fn
                // each get their own monotonic counter.
                let saved = self.save_tracker();
                self.enter_function(&lifted_params);

                let lifted_body = MirBody {
                    stmts: vec![],
                    result: Some(Box::new(self.lower_expr(body))),
                };

                let local_count = self.exit_function();
                self.restore_tracker(saved);

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
                    local_count,
                });

                MirExprKind::MakeClosure {
                    fn_name: lifted_name,
                    captures: capture_exprs,
                }
            }
            HirExprKind::Match { scrutinee, arms } => {
                // Scrutinee is evaluated in the OUTER scope -- lower with
                // tracker still active so it sees outer locals.
                let mir_scrutinee = Box::new(self.lower_expr(scrutinee));

                // Tier-0 perf (Stage 4): each arm opens its own lexical
                // scope. `lower_pattern` walks the pattern and assigns
                // a slot to every `Binding` via `define_local`, recording
                // the name -> slot mapping in the tracker so the arm
                // body's references resolve to those slots. Outer-scope
                // locals still resolve to outer slots. After the body is
                // lowered, the scope pops so the next arm starts fresh
                // (the slot counter is monotonic, so sibling arms get
                // distinct slot ranges -- same trade-off as sibling
                // `if`/`else` branches).
                let mir_arms = arms
                    .iter()
                    .map(|arm| {
                        self.push_scope();
                        let pattern = self.lower_pattern(&arm.pattern);
                        let body = MirBody {
                            stmts: vec![],
                            result: Some(Box::new(self.lower_expr(&arm.body))),
                        };
                        self.pop_scope();
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

    /// Lower an HIR pattern to MIR, assigning slot indices to every
    /// `Binding` pattern found anywhere in the tree (recurses through
    /// `Tuple`, `Struct`, `Variant` patterns).
    ///
    /// Tier-0 perf (Stage 4): when slot resolution is active for the
    /// enclosing function, every `Binding` slot is `Some(slot)` and
    /// the slot tracker's scope_stack records the name -> slot mapping
    /// so subsequent variable references in the arm body resolve
    /// correctly. When slot resolution is inactive (only happens if a
    /// match expression is somehow encountered outside a fn body),
    /// bindings get `slot: None` and the executor falls back to
    /// name-only definition.
    ///
    /// IMPORTANT: this must be called AFTER `push_scope` and BEFORE
    /// the arm body is lowered, so the bindings are visible to the
    /// arm body's references via the scope_stack lookup.
    fn lower_pattern(&mut self, pat: &HirPattern) -> MirPattern {
        match &pat.kind {
            HirPatternKind::Wildcard => MirPattern::Wildcard,
            HirPatternKind::Binding(name) => {
                let slot = if self.slot_resolution_active {
                    Some(self.define_local(name))
                } else {
                    None
                };
                MirPattern::Binding {
                    name: name.clone(),
                    slot,
                }
            }
            HirPatternKind::LitInt(v) => MirPattern::LitInt(*v),
            HirPatternKind::LitFloat(v) => MirPattern::LitFloat(*v),
            HirPatternKind::LitBool(b) => MirPattern::LitBool(*b),
            HirPatternKind::LitString(s) => MirPattern::LitString(s.clone()),
            HirPatternKind::Tuple(pats) => MirPattern::Tuple(
                pats.iter().map(|p| self.lower_pattern(p)).collect(),
            ),
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

    // -----------------------------------------------------------------
    // Tier-0 perf (T0-b Stage 2) slot-resolution tests
    // -----------------------------------------------------------------
    //
    // These tests pin down the lowering's slot assignment rules so future
    // refactors can't silently break them. The runtime behavior is covered
    // by the existing parity + workspace tests; this module only checks
    // the structure of the lowered MIR.

    /// Build a minimal `HirFn` from a body expression plus param list.
    fn mk_fn(name: &str, params: Vec<(&str, &str)>, body_expr: HirExpr) -> HirFn {
        HirFn {
            name: name.to_string(),
            type_params: vec![],
            params: params
                .into_iter()
                .map(|(n, t)| HirParam {
                    name: n.to_string(),
                    ty_name: t.to_string(),
                    default: None,
                    is_variadic: false,
                    hir_id: hir_id(0),
                })
                .collect(),
            return_type: None,
            body: HirBlock {
                stmts: vec![],
                expr: Some(Box::new(body_expr)),
                hir_id: hir_id(0),
            },
            is_nogc: false,
            hir_id: hir_id(0),
            decorators: vec![],
            vis: cjc_ast::Visibility::Private,
        }
    }

    #[test]
    fn t0b_stage2_params_get_sequential_slots() {
        // `fn f(a, b, c) { a + b + c }` -- a/b/c get slots 0/1/2;
        // local_count = 3; body emits VarLocal for every reference.
        let mut lowering = HirToMir::new();
        let body = HirExpr {
            kind: HirExprKind::Binary {
                op: BinOp::Add,
                left: Box::new(HirExpr {
                    kind: HirExprKind::Binary {
                        op: BinOp::Add,
                        left: Box::new(hir_var("a")),
                        right: Box::new(hir_var("b")),
                    },
                    hir_id: hir_id(0),
                }),
                right: Box::new(hir_var("c")),
            },
            hir_id: hir_id(0),
        };
        let hir_fn = mk_fn("f", vec![("a", "i64"), ("b", "i64"), ("c", "i64")], body);
        let mir_fn = lowering.lower_fn(&hir_fn);

        assert_eq!(mir_fn.local_count, 3, "three params -> three slots");

        // Walk the body looking for VarLocal references.
        fn collect_var_locals(expr: &MirExpr, out: &mut Vec<(String, u32)>) {
            match &expr.kind {
                MirExprKind::VarLocal { name, slot } => out.push((name.clone(), *slot)),
                MirExprKind::Binary { left, right, .. } => {
                    collect_var_locals(left, out);
                    collect_var_locals(right, out);
                }
                _ => {}
            }
        }
        let mut found = Vec::new();
        collect_var_locals(mir_fn.body.result.as_ref().unwrap(), &mut found);
        assert_eq!(
            found,
            vec![
                ("a".to_string(), 0),
                ("b".to_string(), 1),
                ("c".to_string(), 2),
            ],
            "params should slot-resolve to their declaration order"
        );
    }

    #[test]
    fn t0b_stage2_let_binding_gets_next_slot_after_params() {
        // `fn f(a) { let b = a; b }` -- a slot 0, b slot 1, local_count 2.
        let mut lowering = HirToMir::new();
        let let_stmt = HirStmt {
            kind: HirStmtKind::Let {
                name: "b".to_string(),
                mutable: false,
                ty_name: None,
                init: hir_var("a"),
            },
            hir_id: hir_id(0),
        };
        let hir_fn = HirFn {
            name: "f".to_string(),
            type_params: vec![],
            params: vec![HirParam {
                name: "a".to_string(),
                ty_name: "i64".to_string(),
                default: None,
                is_variadic: false,
                hir_id: hir_id(0),
            }],
            return_type: None,
            body: HirBlock {
                stmts: vec![let_stmt],
                expr: Some(Box::new(hir_var("b"))),
                hir_id: hir_id(0),
            },
            is_nogc: false,
            hir_id: hir_id(0),
            decorators: vec![],
            vis: cjc_ast::Visibility::Private,
        };
        let mir_fn = lowering.lower_fn(&hir_fn);
        assert_eq!(mir_fn.local_count, 2);

        // The Let's init expression should resolve `a` to slot 0.
        match &mir_fn.body.stmts[0] {
            MirStmt::Let { init, .. } => match &init.kind {
                MirExprKind::VarLocal { name, slot } => {
                    assert_eq!(name, "a");
                    assert_eq!(*slot, 0);
                }
                other => panic!("expected VarLocal for `a`, got {other:?}"),
            },
            other => panic!("expected Let stmt, got {other:?}"),
        }
        // The body result should resolve `b` to slot 1.
        match &mir_fn.body.result.as_ref().unwrap().kind {
            MirExprKind::VarLocal { name, slot } => {
                assert_eq!(name, "b");
                assert_eq!(*slot, 1);
            }
            other => panic!("expected VarLocal for `b`, got {other:?}"),
        }
    }

    #[test]
    fn t0b_stage2_let_rhs_resolves_to_outer_for_shadowing() {
        // `fn f(x) { let x = x + 1; x }` -- the init's `x` must resolve
        // to the PARAMETER's slot (0), not the new let's slot.
        let mut lowering = HirToMir::new();
        let let_stmt = HirStmt {
            kind: HirStmtKind::Let {
                name: "x".to_string(),
                mutable: false,
                ty_name: None,
                init: HirExpr {
                    kind: HirExprKind::Binary {
                        op: BinOp::Add,
                        left: Box::new(hir_var("x")),
                        right: Box::new(hir_int(1)),
                    },
                    hir_id: hir_id(0),
                },
            },
            hir_id: hir_id(0),
        };
        let hir_fn = HirFn {
            name: "f".to_string(),
            type_params: vec![],
            params: vec![HirParam {
                name: "x".to_string(),
                ty_name: "i64".to_string(),
                default: None,
                is_variadic: false,
                hir_id: hir_id(0),
            }],
            return_type: None,
            body: HirBlock {
                stmts: vec![let_stmt],
                expr: Some(Box::new(hir_var("x"))),
                hir_id: hir_id(0),
            },
            is_nogc: false,
            hir_id: hir_id(0),
            decorators: vec![],
            vis: cjc_ast::Visibility::Private,
        };
        let mir_fn = lowering.lower_fn(&hir_fn);
        assert_eq!(mir_fn.local_count, 2, "param x (slot 0) + let x (slot 1)");

        // Init RHS: x + 1 -- `x` must be slot 0 (the param), not slot 1.
        match &mir_fn.body.stmts[0] {
            MirStmt::Let { init, .. } => match &init.kind {
                MirExprKind::Binary { left, .. } => match &left.kind {
                    MirExprKind::VarLocal { slot, .. } => assert_eq!(*slot, 0),
                    other => panic!("expected VarLocal in RHS, got {other:?}"),
                },
                other => panic!("expected Binary init, got {other:?}"),
            },
            other => panic!("expected Let stmt, got {other:?}"),
        }
        // Body result: `x` must be slot 1 (the new binding shadows the param).
        match &mir_fn.body.result.as_ref().unwrap().kind {
            MirExprKind::VarLocal { slot, .. } => assert_eq!(*slot, 1),
            other => panic!("expected VarLocal in body, got {other:?}"),
        }
    }

    #[test]
    fn t0b_stage2_main_function_not_slot_resolved() {
        // Top-level lets feed __main, which is NOT lowered through
        // `lower_fn`. local_count stays at 0 and references stay as Var.
        let mut lowering = HirToMir::new();
        let hir = HirProgram {
            items: vec![HirItem::Let(HirLetDecl {
                name: "x".to_string(),
                mutable: false,
                ty_name: None,
                init: hir_int(42),
                hir_id: hir_id(0),
            })],
        };
        let mir = lowering.lower_program(&hir);
        let main = mir.functions.iter().find(|f| f.name == "__main").unwrap();
        assert_eq!(main.local_count, 0, "__main left on name fallback in Stage 2");
    }

    #[test]
    fn t0b_stage2_unresolved_name_stays_as_var() {
        // `fn f() { undefined_global }` -- no local of that name; emit Var.
        let mut lowering = HirToMir::new();
        let hir_fn = mk_fn("f", vec![], hir_var("undefined_global"));
        let mir_fn = lowering.lower_fn(&hir_fn);
        assert_eq!(mir_fn.local_count, 0, "no params + no lets -> zero slots");

        match &mir_fn.body.result.as_ref().unwrap().kind {
            MirExprKind::Var(name) => assert_eq!(name, "undefined_global"),
            other => panic!(
                "expected Var(name) for unresolved reference, got {other:?}"
            ),
        }
    }

    #[test]
    fn t0b_stage4_closure_body_is_slot_resolved() {
        // `fn outer() { let x = 1; (|y| x + y) }` -- Stage 4 slot-resolves
        // the lifted closure body just like a regular function. The
        // lifted fn has params [capture x, param y] => local_count >= 2;
        // body references emit VarLocal for both `x` (capture, slot 0)
        // and `y` (param, slot 1).
        let mut lowering = HirToMir::new();
        let lambda_body = HirExpr {
            kind: HirExprKind::Binary {
                op: BinOp::Add,
                left: Box::new(hir_var("x")),
                right: Box::new(hir_var("y")),
            },
            hir_id: hir_id(0),
        };
        let closure = HirExpr {
            kind: HirExprKind::Closure {
                params: vec![HirParam {
                    name: "y".to_string(),
                    ty_name: "i64".to_string(),
                    default: None,
                    is_variadic: false,
                    hir_id: hir_id(0),
                }],
                body: Box::new(lambda_body),
                captures: vec![HirCapture {
                    name: "x".to_string(),
                    mode: CaptureMode::Ref,
                    hir_id: hir_id(0),
                }],
            },
            hir_id: hir_id(0),
        };
        let let_x = HirStmt {
            kind: HirStmtKind::Let {
                name: "x".to_string(),
                mutable: false,
                ty_name: None,
                init: hir_int(1),
            },
            hir_id: hir_id(0),
        };
        let outer = HirFn {
            name: "outer".to_string(),
            type_params: vec![],
            params: vec![],
            return_type: None,
            body: HirBlock {
                stmts: vec![let_x],
                expr: Some(Box::new(closure)),
                hir_id: hir_id(0),
            },
            is_nogc: false,
            hir_id: hir_id(0),
            decorators: vec![],
            vis: cjc_ast::Visibility::Private,
        };

        let _outer_mir = lowering.lower_fn(&outer);
        assert_eq!(lowering.lifted_functions.len(), 1);
        let lifted = &lowering.lifted_functions[0];
        // Stage 4: closures get a real local_count. The lifted params
        // are [x (capture), y (original)] -> 2 slots.
        assert_eq!(
            lifted.local_count, 2,
            "Stage 4 slot-resolves closure bodies; lifted params -> slots 0..N"
        );

        // Body of the lifted function: x + y. Both should be VarLocal
        // with `x` at slot 0 (first capture-param) and `y` at slot 1.
        match &lifted.body.result.as_ref().unwrap().kind {
            MirExprKind::Binary { left, right, .. } => {
                match &left.kind {
                    MirExprKind::VarLocal { name, slot } => {
                        assert_eq!(name, "x");
                        assert_eq!(*slot, 0, "capture-param `x` -> slot 0");
                    }
                    other => panic!(
                        "expected VarLocal for capture `x`, got {other:?}"
                    ),
                }
                match &right.kind {
                    MirExprKind::VarLocal { name, slot } => {
                        assert_eq!(name, "y");
                        assert_eq!(*slot, 1, "original param `y` -> slot 1");
                    }
                    other => {
                        panic!("expected VarLocal for param `y`, got {other:?}")
                    }
                }
            }
            other => panic!("expected Binary in closure body, got {other:?}"),
        }
    }

    #[test]
    fn t0b_stage2_capture_expr_in_outer_is_slot_resolved() {
        // The MakeClosure node's capture *expressions* run in the OUTER
        // scope -- they should slot-resolve. This is the closure-creation
        // hot path.
        //
        // `fn outer() { let x = 1; (|y| x + y) }` -- the MakeClosure's
        // captures vec contains a slot-resolved reference to `x`.
        let mut lowering = HirToMir::new();
        let lambda_body = HirExpr {
            kind: HirExprKind::Binary {
                op: BinOp::Add,
                left: Box::new(hir_var("x")),
                right: Box::new(hir_var("y")),
            },
            hir_id: hir_id(0),
        };
        let closure = HirExpr {
            kind: HirExprKind::Closure {
                params: vec![HirParam {
                    name: "y".to_string(),
                    ty_name: "i64".to_string(),
                    default: None,
                    is_variadic: false,
                    hir_id: hir_id(0),
                }],
                body: Box::new(lambda_body),
                captures: vec![HirCapture {
                    name: "x".to_string(),
                    mode: CaptureMode::Ref,
                    hir_id: hir_id(0),
                }],
            },
            hir_id: hir_id(0),
        };
        let let_x = HirStmt {
            kind: HirStmtKind::Let {
                name: "x".to_string(),
                mutable: false,
                ty_name: None,
                init: hir_int(1),
            },
            hir_id: hir_id(0),
        };
        let outer = HirFn {
            name: "outer".to_string(),
            type_params: vec![],
            params: vec![],
            return_type: None,
            body: HirBlock {
                stmts: vec![let_x],
                expr: Some(Box::new(closure)),
                hir_id: hir_id(0),
            },
            is_nogc: false,
            hir_id: hir_id(0),
            decorators: vec![],
            vis: cjc_ast::Visibility::Private,
        };

        let outer_mir = lowering.lower_fn(&outer);
        assert_eq!(outer_mir.local_count, 1, "outer fn has one let (x)");

        match &outer_mir.body.result.as_ref().unwrap().kind {
            MirExprKind::MakeClosure { captures, .. } => {
                assert_eq!(captures.len(), 1);
                match &captures[0].kind {
                    MirExprKind::VarLocal { name, slot } => {
                        assert_eq!(name, "x");
                        assert_eq!(*slot, 0);
                    }
                    other => panic!(
                        "expected VarLocal in MakeClosure capture, got {other:?}"
                    ),
                }
            }
            other => panic!("expected MakeClosure, got {other:?}"),
        }
    }

    #[test]
    fn t0b_stage2_nested_blocks_use_distinct_slots() {
        // `fn f() { if cond { let x = 1; x } else { let y = 2; y } }`
        // -- x and y must occupy different slots (counter is monotonic).
        let mut lowering = HirToMir::new();
        let then_block = HirBlock {
            stmts: vec![HirStmt {
                kind: HirStmtKind::Let {
                    name: "x".to_string(),
                    mutable: false,
                    ty_name: None,
                    init: hir_int(1),
                },
                hir_id: hir_id(0),
            }],
            expr: Some(Box::new(hir_var("x"))),
            hir_id: hir_id(0),
        };
        let else_block = HirBlock {
            stmts: vec![HirStmt {
                kind: HirStmtKind::Let {
                    name: "y".to_string(),
                    mutable: false,
                    ty_name: None,
                    init: hir_int(2),
                },
                hir_id: hir_id(0),
            }],
            expr: Some(Box::new(hir_var("y"))),
            hir_id: hir_id(0),
        };
        let if_expr = HirExpr {
            kind: HirExprKind::If {
                cond: Box::new(HirExpr {
                    kind: HirExprKind::BoolLit(true),
                    hir_id: hir_id(0),
                }),
                then_block,
                else_branch: Some(HirElseBranch::Else(else_block)),
            },
            hir_id: hir_id(0),
        };
        let hir_fn = mk_fn("f", vec![], if_expr);
        let mir_fn = lowering.lower_fn(&hir_fn);

        // Two distinct slots even though only one branch executes at runtime
        // -- slot counter never decrements.
        assert_eq!(
            mir_fn.local_count, 2,
            "shadowing across siblings consumes two slots"
        );
    }

    #[test]
    fn t0b_stage4_match_arm_pattern_bindings_get_slots() {
        // `fn f(outer) { match outer { inner => outer + inner } }`
        //
        // Stage 4: the pattern binding `inner` gets a slot (1, after
        // param `outer` at slot 0); references to `outer` and `inner`
        // in the arm body both slot-resolve. The pattern itself carries
        // the assigned slot so the executor knows where to write the
        // binding value.
        let mut lowering = HirToMir::new();
        let arm_body = HirExpr {
            kind: HirExprKind::Binary {
                op: BinOp::Add,
                left: Box::new(hir_var("outer")),
                right: Box::new(hir_var("inner")),
            },
            hir_id: hir_id(0),
        };
        let match_expr = HirExpr {
            kind: HirExprKind::Match {
                scrutinee: Box::new(hir_var("outer")),
                arms: vec![HirMatchArm {
                    pattern: HirPattern {
                        kind: HirPatternKind::Binding("inner".to_string()),
                        hir_id: hir_id(0),
                    },
                    body: arm_body,
                    hir_id: hir_id(0),
                }],
            },
            hir_id: hir_id(0),
        };
        let hir_fn = mk_fn("f", vec![("outer", "i64")], match_expr);
        let mir_fn = lowering.lower_fn(&hir_fn);

        // Param outer is slot 0; pattern binding inner is slot 1.
        // local_count is 2.
        assert_eq!(
            mir_fn.local_count, 2,
            "param outer (slot 0) + pattern binding inner (slot 1)"
        );

        match &mir_fn.body.result.as_ref().unwrap().kind {
            MirExprKind::Match { scrutinee, arms } => {
                // Scrutinee in outer scope -> VarLocal slot 0.
                match &scrutinee.kind {
                    MirExprKind::VarLocal { name, slot } => {
                        assert_eq!(name, "outer");
                        assert_eq!(*slot, 0);
                    }
                    other => {
                        panic!("scrutinee should slot-resolve, got {other:?}")
                    }
                }
                // Pattern itself carries slot.
                match &arms[0].pattern {
                    MirPattern::Binding { name, slot } => {
                        assert_eq!(name, "inner");
                        assert_eq!(
                            *slot,
                            Some(1),
                            "pattern binding -> slot 1"
                        );
                    }
                    other => panic!(
                        "expected Binding pattern, got {other:?}"
                    ),
                }
                // Arm body: both `outer` and `inner` are VarLocal.
                match &arms[0].body.result.as_ref().unwrap().kind {
                    MirExprKind::Binary { left, right, .. } => {
                        match &left.kind {
                            MirExprKind::VarLocal { name, slot } => {
                                assert_eq!(name, "outer");
                                assert_eq!(*slot, 0);
                            }
                            other => panic!(
                                "expected VarLocal for `outer`, got {other:?}"
                            ),
                        }
                        match &right.kind {
                            MirExprKind::VarLocal { name, slot } => {
                                assert_eq!(name, "inner");
                                assert_eq!(*slot, 1);
                            }
                            other => panic!(
                                "expected VarLocal for `inner`, got {other:?}"
                            ),
                        }
                    }
                    other => {
                        panic!("expected Binary in arm body, got {other:?}")
                    }
                }
            }
            other => panic!("expected Match, got {other:?}"),
        }
    }

    #[test]
    fn t0b_stage2_function_calls_dont_disturb_outer_slots() {
        // Lowering an inner `lower_fn` from inside an outer function (via
        // an Impl method, say) must save/restore the outer tracker.
        //
        // Here we manually trigger this by lowering two functions in
        // sequence; the second must have local_count = its own params,
        // not the first's.
        let mut lowering = HirToMir::new();
        let f1 = mk_fn("f1", vec![("a", "i64"), ("b", "i64")], hir_var("a"));
        let f2 = mk_fn("f2", vec![("x", "i64")], hir_var("x"));

        let m1 = lowering.lower_fn(&f1);
        let m2 = lowering.lower_fn(&f2);
        assert_eq!(m1.local_count, 2);
        assert_eq!(m2.local_count, 1, "f2 should not inherit f1's slot count");
    }
}
