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
pub mod escape;
pub mod monomorph;
pub mod nogc_verify;
pub mod optimize;

use cjc_ast::{BinOp, UnaryOp};
pub use escape::AllocHint;

// ---------------------------------------------------------------------------
// IDs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MirFnId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

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

#[derive(Debug, Clone)]
pub struct MirStructDef {
    pub name: String,
    pub fields: Vec<(String, String)>, // (name, type_name)
}

#[derive(Debug, Clone)]
pub struct MirEnumDef {
    pub name: String,
    pub variants: Vec<MirVariantDef>,
}

#[derive(Debug, Clone)]
pub struct MirVariantDef {
    pub name: String,
    pub fields: Vec<String>, // type names
}

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MirFunction {
    pub id: MirFnId,
    pub name: String,
    pub type_params: Vec<(String, Vec<String>)>, // (param_name, bounds)
    pub params: Vec<MirParam>,
    pub return_type: Option<String>,
    pub body: MirBody,
    pub is_nogc: bool,
}

#[derive(Debug, Clone)]
pub struct MirParam {
    pub name: String,
    pub ty_name: String,
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

#[derive(Debug, Clone)]
pub enum MirStmt {
    Let {
        name: String,
        mutable: bool,
        init: MirExpr,
        /// Escape analysis annotation. `None` before analysis runs.
        alloc_hint: Option<AllocHint>,
    },
    Expr(MirExpr),
    If {
        cond: MirExpr,
        then_body: MirBody,
        else_body: Option<MirBody>,
    },
    While {
        cond: MirExpr,
        body: MirBody,
    },
    Return(Option<MirExpr>),
    Break,
    Continue,
    NoGcBlock(MirBody),
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MirExpr {
    pub kind: MirExprKind,
}

#[derive(Debug, Clone)]
pub enum MirExprKind {
    IntLit(i64),
    FloatLit(f64),
    BoolLit(bool),
    StringLit(String),
    ByteStringLit(Vec<u8>),
    ByteCharLit(u8),
    RawStringLit(String),
    RawByteStringLit(Vec<u8>),
    RegexLit { pattern: String, flags: String },
    TensorLit { rows: Vec<Vec<MirExpr>> },
    Var(String),
    Binary {
        op: BinOp,
        left: Box<MirExpr>,
        right: Box<MirExpr>,
    },
    Unary {
        op: UnaryOp,
        operand: Box<MirExpr>,
    },
    Call {
        callee: Box<MirExpr>,
        args: Vec<MirExpr>,
    },
    Field {
        object: Box<MirExpr>,
        name: String,
    },
    Index {
        object: Box<MirExpr>,
        index: Box<MirExpr>,
    },
    MultiIndex {
        object: Box<MirExpr>,
        indices: Vec<MirExpr>,
    },
    Assign {
        target: Box<MirExpr>,
        value: Box<MirExpr>,
    },
    Block(MirBody),
    StructLit {
        name: String,
        fields: Vec<(String, MirExpr)>,
    },
    ArrayLit(Vec<MirExpr>),
    Col(String),
    Lambda {
        params: Vec<MirParam>,
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
    If {
        cond: Box<MirExpr>,
        then_body: MirBody,
        else_body: Option<MirBody>,
    },
    /// Match expression compiled as a decision tree.
    /// Each arm is tried in order; first matching arm's body is evaluated.
    Match {
        scrutinee: Box<MirExpr>,
        arms: Vec<MirMatchArm>,
    },
    /// Enum variant literal
    VariantLit {
        enum_name: String,
        variant: String,
        fields: Vec<MirExpr>,
    },
    /// Tuple literal
    TupleLit(Vec<MirExpr>),
    /// Linalg opcodes — dedicated MIR nodes for matrix decompositions.
    LinalgLU { operand: Box<MirExpr> },
    LinalgQR { operand: Box<MirExpr> },
    LinalgCholesky { operand: Box<MirExpr> },
    LinalgInv { operand: Box<MirExpr> },
    /// Broadcast a tensor to a target shape (zero-copy view with stride=0).
    Broadcast {
        operand: Box<MirExpr>,
        target_shape: Vec<MirExpr>,
    },
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
pub struct HirToMir {
    next_fn_id: u32,
    next_lambda_id: u32,
    /// Lambda-lifted functions accumulated during lowering.
    /// These are appended to the MirProgram's function list.
    lifted_functions: Vec<MirFunction>,
}

impl HirToMir {
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
                    });
                }
                HirItem::Class(c) => {
                    struct_defs.push(MirStructDef {
                        name: c.name.clone(),
                        fields: c.fields.clone(),
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

    pub fn lower_fn(&mut self, f: &HirFn) -> MirFunction {
        let id = self.fresh_fn_id();
        let params = f
            .params
            .iter()
            .map(|p| MirParam {
                name: p.name.clone(),
                ty_name: p.ty_name.clone(),
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

    pub fn lower_expr(&mut self, expr: &HirExpr) -> MirExpr {
        let kind = match &expr.kind {
            HirExprKind::IntLit(v) => MirExprKind::IntLit(*v),
            HirExprKind::FloatLit(v) => MirExprKind::FloatLit(*v),
            HirExprKind::BoolLit(b) => MirExprKind::BoolLit(*b),
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
                    })
                    .collect();
                for p in params {
                    lifted_params.push(MirParam {
                        name: p.name.clone(),
                        ty_name: p.ty_name.clone(),
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
                    hir_id: hir_id(1),
                },
                HirParam {
                    name: "b".to_string(),
                    ty_name: "i64".to_string(),
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
            })],
        };
        let mir = lowering.lower_program(&hir);
        assert_eq!(mir.struct_defs.len(), 1);
        assert_eq!(mir.struct_defs[0].name, "Point");
        assert_eq!(mir.struct_defs[0].fields.len(), 2);
    }
}
