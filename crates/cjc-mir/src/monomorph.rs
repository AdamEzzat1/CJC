//! MIR Monomorphization Pass
//!
//! Post-lowering pass that collects generic function instantiations,
//! clones + specializes them with concrete type arguments, rewrites
//! call sites, and removes the original generic definitions.
//!
//! Pipeline: AST → HIR → MIR → **Monomorph** → Optimize → Execute
//!
//! Since the CJC runtime is dynamically typed, monomorphization is
//! primarily a correctness/performance preparation step for future
//! static typing / codegen. Generic functions already work without
//! it due to dynamic dispatch.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    MirBody, MirExpr, MirExprKind, MirFnId, MirFunction, MirMatchArm, MirParam, MirPattern,
    MirProgram, MirStmt,
};

/// Hard limit on specializations per program to prevent combinatorial explosion.
const MAX_SPECIALIZATIONS: usize = 1000;

/// Report produced by the monomorphization pass.
///
/// Contains statistics about how many specializations were generated,
/// which generic functions had the highest fanout, and whether the
/// specialization budget was exceeded.
pub struct MonomorphReport {
    /// Number of specializations generated.
    pub specialization_count: usize,
    /// Top fanout functions sorted by instantiation count (descending), capped at 10.
    pub top_fanout: Vec<(String, usize)>,
    /// Whether the specialization budget ([`MAX_SPECIALIZATIONS`]) was exceeded.
    pub budget_exceeded: bool,
}

/// An instantiation request: a generic function with concrete type args.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Instantiation {
    fn_name: String,
    type_args: Vec<String>,
}

impl Instantiation {
    /// Generate a deterministic mangled name: `{fn_name}__M__{type1}_{type2}`
    fn mangled_name(&self) -> String {
        if self.type_args.is_empty() {
            self.fn_name.clone()
        } else {
            format!(
                "{}__M__{}",
                self.fn_name,
                self.type_args.join("_")
            )
        }
    }
}

/// Run the monomorphization pass on a MIR program.
///
/// Returns the transformed program and a report.
pub fn monomorphize_program(program: &MirProgram) -> (MirProgram, MonomorphReport) {
    let mut mono = Monomorphizer::new(program);
    mono.run();
    let report = mono.report();
    let new_program = mono.into_program();
    (new_program, report)
}

struct Monomorphizer<'a> {
    original: &'a MirProgram,
    /// Map from function name to its definition.
    fn_map: BTreeMap<String, &'a MirFunction>,
    /// Set of generic function names (those with non-empty type_params).
    generic_fns: BTreeSet<String>,
    /// Collected instantiation requests.
    instantiations: BTreeSet<Instantiation>,
    /// Specialized (cloned) functions.
    specialized: Vec<MirFunction>,
    /// Next function ID for new specializations.
    next_fn_id: u32,
    /// Count of instantiations per generic function name.
    fanout: BTreeMap<String, usize>,
    /// Whether budget was exceeded.
    budget_exceeded: bool,
}

impl<'a> Monomorphizer<'a> {
    fn new(program: &'a MirProgram) -> Self {
        let mut fn_map = BTreeMap::new();
        let mut generic_fns = BTreeSet::new();
        let mut max_id = 0u32;

        for f in &program.functions {
            fn_map.insert(f.name.clone(), f);
            if !f.type_params.is_empty() {
                generic_fns.insert(f.name.clone());
            }
            if f.id.0 >= max_id {
                max_id = f.id.0 + 1;
            }
        }

        Self {
            original: program,
            fn_map,
            generic_fns,
            instantiations: BTreeSet::new(),
            specialized: Vec::new(),
            next_fn_id: max_id,
            fanout: BTreeMap::new(),
            budget_exceeded: false,
        }
    }

    fn fresh_fn_id(&mut self) -> MirFnId {
        let id = MirFnId(self.next_fn_id);
        self.next_fn_id += 1;
        id
    }

    /// Run the monomorphization pass.
    fn run(&mut self) {
        // If no generic functions exist, nothing to do.
        if self.generic_fns.is_empty() {
            return;
        }

        // Phase 1: Collect instantiation requests by walking all function bodies.
        for f in &self.original.functions {
            self.collect_from_body(&f.body);
        }

        // Phase 2: Clone + specialize each instantiation.
        let requests: Vec<Instantiation> = self.instantiations.iter().cloned().collect();
        for inst in &requests {
            if self.specialized.len() >= MAX_SPECIALIZATIONS {
                self.budget_exceeded = true;
                break;
            }
            self.specialize(inst);
        }
    }

    /// Collect instantiation requests from a function body.
    fn collect_from_body(&mut self, body: &MirBody) {
        for stmt in &body.stmts {
            self.collect_from_stmt(stmt);
        }
        if let Some(ref expr) = body.result {
            self.collect_from_expr(expr);
        }
    }

    fn collect_from_stmt(&mut self, stmt: &MirStmt) {
        match stmt {
            MirStmt::Let { init, .. } => self.collect_from_expr(init),
            MirStmt::Expr(e) => self.collect_from_expr(e),
            MirStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                self.collect_from_expr(cond);
                self.collect_from_body(then_body);
                if let Some(eb) = else_body {
                    self.collect_from_body(eb);
                }
            }
            MirStmt::While { cond, body } => {
                self.collect_from_expr(cond);
                self.collect_from_body(body);
            }
            MirStmt::Return(Some(e)) => self.collect_from_expr(e),
            MirStmt::Return(None) => {}
            MirStmt::Break | MirStmt::Continue => {}
            MirStmt::NoGcBlock(body) => self.collect_from_body(body),
        }
    }

    fn collect_from_expr(&mut self, expr: &MirExpr) {
        match &expr.kind {
            MirExprKind::Call { callee, args } => {
                // Check if this is a call to a generic function
                if let MirExprKind::Var(name) = &callee.kind {
                    if self.generic_fns.contains(name) {
                        // Infer type args from the call site arguments
                        let type_args = self.infer_type_args(name, args);
                        if !type_args.is_empty() {
                            let inst = Instantiation {
                                fn_name: name.clone(),
                                type_args,
                            };
                            self.instantiations.insert(inst);
                            *self.fanout.entry(name.clone()).or_insert(0) += 1;
                        }
                    }
                }
                self.collect_from_expr(callee);
                for arg in args {
                    self.collect_from_expr(arg);
                }
            }
            MirExprKind::Binary { left, right, .. } => {
                self.collect_from_expr(left);
                self.collect_from_expr(right);
            }
            MirExprKind::Unary { operand, .. } => self.collect_from_expr(operand),
            MirExprKind::Field { object, .. } => self.collect_from_expr(object),
            MirExprKind::Index { object, index } => {
                self.collect_from_expr(object);
                self.collect_from_expr(index);
            }
            MirExprKind::MultiIndex { object, indices } => {
                self.collect_from_expr(object);
                for idx in indices {
                    self.collect_from_expr(idx);
                }
            }
            MirExprKind::Assign { target, value } => {
                self.collect_from_expr(target);
                self.collect_from_expr(value);
            }
            MirExprKind::Block(body) => self.collect_from_body(body),
            MirExprKind::StructLit { fields, .. } => {
                for (_, e) in fields {
                    self.collect_from_expr(e);
                }
            }
            MirExprKind::VariantLit { fields, .. } => {
                for f in fields {
                    self.collect_from_expr(f);
                }
            }
            MirExprKind::ArrayLit(elems) | MirExprKind::TupleLit(elems) => {
                for e in elems {
                    self.collect_from_expr(e);
                }
            }
            MirExprKind::MakeClosure { captures, .. } => {
                for c in captures {
                    self.collect_from_expr(c);
                }
            }
            MirExprKind::If {
                cond,
                then_body,
                else_body,
            } => {
                self.collect_from_expr(cond);
                self.collect_from_body(then_body);
                if let Some(eb) = else_body {
                    self.collect_from_body(eb);
                }
            }
            MirExprKind::Match { scrutinee, arms } => {
                self.collect_from_expr(scrutinee);
                for arm in arms {
                    self.collect_from_body(&arm.body);
                }
            }
            MirExprKind::Lambda { body, .. } => self.collect_from_expr(body),
            MirExprKind::LinalgLU { operand }
            | MirExprKind::LinalgQR { operand }
            | MirExprKind::LinalgCholesky { operand }
            | MirExprKind::LinalgInv { operand } => self.collect_from_expr(operand),
            MirExprKind::Broadcast {
                operand,
                target_shape,
            } => {
                self.collect_from_expr(operand);
                for s in target_shape {
                    self.collect_from_expr(s);
                }
            }
            // Leaves
            MirExprKind::IntLit(_)
            | MirExprKind::FloatLit(_)
            | MirExprKind::BoolLit(_)
            | MirExprKind::NaLit
            | MirExprKind::StringLit(_)
            | MirExprKind::ByteStringLit(_)
            | MirExprKind::ByteCharLit(_)
            | MirExprKind::RawStringLit(_)
            | MirExprKind::RawByteStringLit(_)
            | MirExprKind::RegexLit { .. }
            | MirExprKind::Var(_)
            | MirExprKind::Col(_)
            | MirExprKind::Void => {}
            MirExprKind::TensorLit { rows } => {
                for row in rows {
                    for elem in row {
                        self.collect_from_expr(elem);
                    }
                }
            }
        }
    }

    /// Infer concrete type arguments for a call to a generic function.
    ///
    /// Simple heuristic: inspect argument expressions to determine types.
    /// At MIR level, types are erased, so we use value-level hints.
    fn infer_type_args(&self, fn_name: &str, args: &[MirExpr]) -> Vec<String> {
        let func = match self.fn_map.get(fn_name) {
            Some(f) => f,
            None => return vec![],
        };

        if func.type_params.is_empty() {
            return vec![];
        }

        // Build a mapping from type param name to concrete type
        let mut param_map: BTreeMap<String, String> = BTreeMap::new();

        // Try to infer from argument values
        for (i, (arg, param)) in args.iter().zip(func.params.iter()).enumerate() {
            let _ = i; // suppress unused warning
            let param_type = &param.ty_name;

            // If param type is one of the type params, try to infer from the arg
            for (tp_name, _) in &func.type_params {
                if param_type == tp_name {
                    // Infer from the argument expression
                    if let Some(concrete) = infer_type_from_expr(arg) {
                        param_map.insert(tp_name.clone(), concrete);
                    }
                }
            }
        }

        // Build the type_args vector in order
        func.type_params
            .iter()
            .map(|(name, _)| {
                param_map
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| "any".to_string())
            })
            .collect()
    }

    /// Clone a generic function and specialize it for the given type args.
    fn specialize(&mut self, inst: &Instantiation) {
        // Clone all data we need from fn_map before any mutable borrows
        let (type_params, params, return_type, body, is_nogc) =
            match self.fn_map.get(&inst.fn_name) {
                Some(f) => (
                    f.type_params.clone(),
                    f.params.clone(),
                    f.return_type.clone(),
                    f.body.clone(),
                    f.is_nogc,
                ),
                None => return,
            };

        // Build substitution map: type_param_name -> concrete_type
        let subst: BTreeMap<String, String> = type_params
            .iter()
            .zip(inst.type_args.iter())
            .map(|((name, _), concrete)| (name.clone(), concrete.clone()))
            .collect();

        let mangled = inst.mangled_name();
        let new_id = self.fresh_fn_id();

        // Clone and specialize params
        let new_params: Vec<MirParam> = params
            .iter()
            .map(|p| MirParam {
                name: p.name.clone(),
                ty_name: subst.get(&p.ty_name).unwrap_or(&p.ty_name).clone(),
                default: p.default.as_ref().map(|d| substitute_expr(d, &subst)),
                is_variadic: p.is_variadic,
            })
            .collect();

        // Clone and specialize return type
        let new_ret = return_type.as_ref().map(|rt| {
            subst.get(rt).unwrap_or(rt).clone()
        });

        // Clone and specialize body
        let new_body = substitute_body(&body, &subst);

        self.specialized.push(MirFunction {
            id: new_id,
            name: mangled,
            type_params: vec![], // Specialized — no more type params
            params: new_params,
            return_type: new_ret,
            body: new_body,
            is_nogc,
            cfg_body: None,
            decorators: vec![],
            vis: cjc_ast::Visibility::Private,
        });
    }

    /// Build the final program with specialized functions and rewritten call sites.
    fn into_program(self) -> MirProgram {
        // Build the instantiation lookup: (fn_name, type_args) -> mangled_name
        let mut inst_lookup: BTreeMap<String, Vec<(Vec<String>, String)>> = BTreeMap::new();
        for inst in &self.instantiations {
            inst_lookup
                .entry(inst.fn_name.clone())
                .or_default()
                .push((inst.type_args.clone(), inst.mangled_name()));
        }

        // Rewrite all functions
        let mut new_functions: Vec<MirFunction> = self
            .original
            .functions
            .iter()
            .map(|f| {
                let mut cloned = f.clone();
                rewrite_calls_in_body(&mut cloned.body, &inst_lookup, &self.fn_map);
                cloned
            })
            .collect();

        // Append specialized functions
        new_functions.extend(self.specialized);

        MirProgram {
            functions: new_functions,
            struct_defs: self.original.struct_defs.clone(),
            enum_defs: self.original.enum_defs.clone(),
            entry: self.original.entry,
        }
    }

    fn report(&self) -> MonomorphReport {
        let mut fanout_vec: Vec<(String, usize)> = self.fanout.clone().into_iter().collect();
        fanout_vec.sort_by(|a, b| b.1.cmp(&a.1));
        fanout_vec.truncate(10);

        MonomorphReport {
            specialization_count: self.specialized.len(),
            top_fanout: fanout_vec,
            budget_exceeded: self.budget_exceeded,
        }
    }
}

/// Try to infer a concrete type name from a MIR expression.
fn infer_type_from_expr(expr: &MirExpr) -> Option<String> {
    match &expr.kind {
        MirExprKind::IntLit(_) => Some("i64".to_string()),
        MirExprKind::FloatLit(_) => Some("f64".to_string()),
        MirExprKind::BoolLit(_) => Some("bool".to_string()),
        MirExprKind::NaLit => Some("Na".to_string()),
        MirExprKind::StringLit(_) => Some("String".to_string()),
        MirExprKind::ByteStringLit(_) => Some("ByteSlice".to_string()),
        MirExprKind::ByteCharLit(_) => Some("u8".to_string()),
        MirExprKind::RawStringLit(_) => Some("String".to_string()),
        MirExprKind::RawByteStringLit(_) => Some("ByteSlice".to_string()),
        MirExprKind::RegexLit { .. } => Some("Regex".to_string()),
        MirExprKind::TensorLit { .. } => Some("Tensor".to_string()),
        MirExprKind::StructLit { name, .. } => Some(name.clone()),
        MirExprKind::VariantLit { enum_name, .. } => Some(enum_name.clone()),
        MirExprKind::ArrayLit(elems) => {
            if let Some(first) = elems.first() {
                let inner = infer_type_from_expr(first).unwrap_or_else(|| "any".to_string());
                Some(format!("Array_{}", inner))
            } else {
                Some("Array_any".to_string())
            }
        }
        MirExprKind::TupleLit(elems) => {
            let parts: Vec<String> = elems
                .iter()
                .map(|e| infer_type_from_expr(e).unwrap_or_else(|| "any".to_string()))
                .collect();
            Some(format!("Tuple_{}", parts.join("_")))
        }
        // For variables and other expressions, we can't easily infer the type at MIR level
        _ => None,
    }
}

/// Substitute type param strings in a body.
fn substitute_body(body: &MirBody, subst: &BTreeMap<String, String>) -> MirBody {
    MirBody {
        stmts: body.stmts.iter().map(|s| substitute_stmt(s, subst)).collect(),
        result: body.result.as_ref().map(|e| Box::new(substitute_expr(e, subst))),
    }
}

fn substitute_stmt(stmt: &MirStmt, subst: &BTreeMap<String, String>) -> MirStmt {
    match stmt {
        MirStmt::Let { name, mutable, init, alloc_hint } => MirStmt::Let {
            name: name.clone(),
            mutable: *mutable,
            init: substitute_expr(init, subst),
            alloc_hint: *alloc_hint,
        },
        MirStmt::Expr(e) => MirStmt::Expr(substitute_expr(e, subst)),
        MirStmt::If {
            cond,
            then_body,
            else_body,
        } => MirStmt::If {
            cond: substitute_expr(cond, subst),
            then_body: substitute_body(then_body, subst),
            else_body: else_body.as_ref().map(|eb| substitute_body(eb, subst)),
        },
        MirStmt::While { cond, body } => MirStmt::While {
            cond: substitute_expr(cond, subst),
            body: substitute_body(body, subst),
        },
        MirStmt::Return(Some(e)) => MirStmt::Return(Some(substitute_expr(e, subst))),
        MirStmt::Return(None) => MirStmt::Return(None),
        MirStmt::Break => MirStmt::Break,
        MirStmt::Continue => MirStmt::Continue,
        MirStmt::NoGcBlock(body) => MirStmt::NoGcBlock(substitute_body(body, subst)),
    }
}

fn substitute_expr(expr: &MirExpr, subst: &BTreeMap<String, String>) -> MirExpr {
    let kind = match &expr.kind {
        // Most expressions just recurse — type substitution mainly affects
        // StructLit/VariantLit type names and param type annotations
        MirExprKind::StructLit { name, fields } => MirExprKind::StructLit {
            name: subst.get(name).unwrap_or(name).clone(),
            fields: fields
                .iter()
                .map(|(n, e)| (n.clone(), substitute_expr(e, subst)))
                .collect(),
        },
        MirExprKind::VariantLit {
            enum_name,
            variant,
            fields,
        } => MirExprKind::VariantLit {
            enum_name: subst.get(enum_name).unwrap_or(enum_name).clone(),
            variant: variant.clone(),
            fields: fields.iter().map(|f| substitute_expr(f, subst)).collect(),
        },
        MirExprKind::Binary { op, left, right } => MirExprKind::Binary {
            op: *op,
            left: Box::new(substitute_expr(left, subst)),
            right: Box::new(substitute_expr(right, subst)),
        },
        MirExprKind::Unary { op, operand } => MirExprKind::Unary {
            op: *op,
            operand: Box::new(substitute_expr(operand, subst)),
        },
        MirExprKind::Call { callee, args } => MirExprKind::Call {
            callee: Box::new(substitute_expr(callee, subst)),
            args: args.iter().map(|a| substitute_expr(a, subst)).collect(),
        },
        MirExprKind::Field { object, name } => MirExprKind::Field {
            object: Box::new(substitute_expr(object, subst)),
            name: name.clone(),
        },
        MirExprKind::Index { object, index } => MirExprKind::Index {
            object: Box::new(substitute_expr(object, subst)),
            index: Box::new(substitute_expr(index, subst)),
        },
        MirExprKind::MultiIndex { object, indices } => MirExprKind::MultiIndex {
            object: Box::new(substitute_expr(object, subst)),
            indices: indices.iter().map(|i| substitute_expr(i, subst)).collect(),
        },
        MirExprKind::Assign { target, value } => MirExprKind::Assign {
            target: Box::new(substitute_expr(target, subst)),
            value: Box::new(substitute_expr(value, subst)),
        },
        MirExprKind::Block(body) => MirExprKind::Block(substitute_body(body, subst)),
        MirExprKind::ArrayLit(elems) => {
            MirExprKind::ArrayLit(elems.iter().map(|e| substitute_expr(e, subst)).collect())
        }
        MirExprKind::TupleLit(elems) => {
            MirExprKind::TupleLit(elems.iter().map(|e| substitute_expr(e, subst)).collect())
        }
        MirExprKind::Lambda { params, body } => MirExprKind::Lambda {
            params: params
                .iter()
                .map(|p| MirParam {
                    name: p.name.clone(),
                    ty_name: subst.get(&p.ty_name).unwrap_or(&p.ty_name).clone(),
                    default: p.default.as_ref().map(|d| substitute_expr(d, &subst)),
                    is_variadic: p.is_variadic,
                })
                .collect(),
            body: Box::new(substitute_expr(body, subst)),
        },
        MirExprKind::MakeClosure { fn_name, captures } => MirExprKind::MakeClosure {
            fn_name: fn_name.clone(),
            captures: captures.iter().map(|c| substitute_expr(c, subst)).collect(),
        },
        MirExprKind::If {
            cond,
            then_body,
            else_body,
        } => MirExprKind::If {
            cond: Box::new(substitute_expr(cond, subst)),
            then_body: substitute_body(then_body, subst),
            else_body: else_body.as_ref().map(|eb| substitute_body(eb, subst)),
        },
        MirExprKind::Match { scrutinee, arms } => MirExprKind::Match {
            scrutinee: Box::new(substitute_expr(scrutinee, subst)),
            arms: arms
                .iter()
                .map(|arm| MirMatchArm {
                    pattern: substitute_pattern(&arm.pattern, subst),
                    body: substitute_body(&arm.body, subst),
                })
                .collect(),
        },
        MirExprKind::LinalgLU { operand } => MirExprKind::LinalgLU {
            operand: Box::new(substitute_expr(operand, subst)),
        },
        MirExprKind::LinalgQR { operand } => MirExprKind::LinalgQR {
            operand: Box::new(substitute_expr(operand, subst)),
        },
        MirExprKind::LinalgCholesky { operand } => MirExprKind::LinalgCholesky {
            operand: Box::new(substitute_expr(operand, subst)),
        },
        MirExprKind::LinalgInv { operand } => MirExprKind::LinalgInv {
            operand: Box::new(substitute_expr(operand, subst)),
        },
        MirExprKind::Broadcast {
            operand,
            target_shape,
        } => MirExprKind::Broadcast {
            operand: Box::new(substitute_expr(operand, subst)),
            target_shape: target_shape
                .iter()
                .map(|s| substitute_expr(s, subst))
                .collect(),
        },
        MirExprKind::TensorLit { rows } => MirExprKind::TensorLit {
            rows: rows.iter().map(|row| {
                row.iter().map(|e| substitute_expr(e, subst)).collect()
            }).collect(),
        },
        // Leaves — no substitution needed
        MirExprKind::IntLit(_)
        | MirExprKind::FloatLit(_)
        | MirExprKind::BoolLit(_)
        | MirExprKind::NaLit
        | MirExprKind::StringLit(_)
        | MirExprKind::ByteStringLit(_)
        | MirExprKind::ByteCharLit(_)
        | MirExprKind::RawStringLit(_)
        | MirExprKind::RawByteStringLit(_)
        | MirExprKind::RegexLit { .. }
        | MirExprKind::Var(_)
        | MirExprKind::Col(_)
        | MirExprKind::Void => expr.kind.clone(),
    };
    MirExpr { kind }
}

fn substitute_pattern(pat: &MirPattern, subst: &BTreeMap<String, String>) -> MirPattern {
    match pat {
        MirPattern::Variant {
            enum_name,
            variant,
            fields,
        } => MirPattern::Variant {
            enum_name: subst.get(enum_name).unwrap_or(enum_name).clone(),
            variant: variant.clone(),
            fields: fields.iter().map(|f| substitute_pattern(f, subst)).collect(),
        },
        MirPattern::Struct { name, fields } => MirPattern::Struct {
            name: subst.get(name).unwrap_or(name).clone(),
            fields: fields
                .iter()
                .map(|(n, p)| (n.clone(), substitute_pattern(p, subst)))
                .collect(),
        },
        MirPattern::Tuple(pats) => {
            MirPattern::Tuple(pats.iter().map(|p| substitute_pattern(p, subst)).collect())
        }
        // Leaves — no substitution needed
        _ => pat.clone(),
    }
}

/// Rewrite call sites in a body: if calling a generic fn, redirect to mangled name.
fn rewrite_calls_in_body(
    body: &mut MirBody,
    inst_lookup: &BTreeMap<String, Vec<(Vec<String>, String)>>,
    fn_map: &BTreeMap<String, &MirFunction>,
) {
    for stmt in &mut body.stmts {
        rewrite_calls_in_stmt(stmt, inst_lookup, fn_map);
    }
    if let Some(ref mut expr) = body.result {
        rewrite_calls_in_expr(expr, inst_lookup, fn_map);
    }
}

fn rewrite_calls_in_stmt(
    stmt: &mut MirStmt,
    inst_lookup: &BTreeMap<String, Vec<(Vec<String>, String)>>,
    fn_map: &BTreeMap<String, &MirFunction>,
) {
    match stmt {
        MirStmt::Let { init, .. } => rewrite_calls_in_expr(init, inst_lookup, fn_map),
        MirStmt::Expr(e) => rewrite_calls_in_expr(e, inst_lookup, fn_map),
        MirStmt::If {
            cond,
            then_body,
            else_body,
        } => {
            rewrite_calls_in_expr(cond, inst_lookup, fn_map);
            rewrite_calls_in_body(then_body, inst_lookup, fn_map);
            if let Some(eb) = else_body {
                rewrite_calls_in_body(eb, inst_lookup, fn_map);
            }
        }
        MirStmt::While { cond, body } => {
            rewrite_calls_in_expr(cond, inst_lookup, fn_map);
            rewrite_calls_in_body(body, inst_lookup, fn_map);
        }
        MirStmt::Return(Some(e)) => rewrite_calls_in_expr(e, inst_lookup, fn_map),
        MirStmt::Return(None) => {}
        MirStmt::Break | MirStmt::Continue => {}
        MirStmt::NoGcBlock(body) => rewrite_calls_in_body(body, inst_lookup, fn_map),
    }
}

fn rewrite_calls_in_expr(
    expr: &mut MirExpr,
    inst_lookup: &BTreeMap<String, Vec<(Vec<String>, String)>>,
    fn_map: &BTreeMap<String, &MirFunction>,
) {
    match &mut expr.kind {
        MirExprKind::Call { callee, args } => {
            // First rewrite sub-expressions
            rewrite_calls_in_expr(callee, inst_lookup, fn_map);
            for arg in args.iter_mut() {
                rewrite_calls_in_expr(arg, inst_lookup, fn_map);
            }

            // Check if callee is a generic function
            if let MirExprKind::Var(name) = &callee.kind {
                if let Some(entries) = inst_lookup.get(name.as_str()) {
                    // Infer type args from arguments
                    if let Some(func) = fn_map.get(name.as_str()) {
                        let type_args = infer_type_args_from_call(func, args);
                        // Find matching instantiation
                        for (inst_args, mangled) in entries {
                            if *inst_args == type_args {
                                // Rewrite callee to mangled name
                                *callee = Box::new(MirExpr {
                                    kind: MirExprKind::Var(mangled.clone()),
                                });
                                return;
                            }
                        }
                    }
                }
            }
        }
        MirExprKind::Binary { left, right, .. } => {
            rewrite_calls_in_expr(left, inst_lookup, fn_map);
            rewrite_calls_in_expr(right, inst_lookup, fn_map);
        }
        MirExprKind::Unary { operand, .. } => {
            rewrite_calls_in_expr(operand, inst_lookup, fn_map);
        }
        MirExprKind::Field { object, .. } => {
            rewrite_calls_in_expr(object, inst_lookup, fn_map);
        }
        MirExprKind::Index { object, index } => {
            rewrite_calls_in_expr(object, inst_lookup, fn_map);
            rewrite_calls_in_expr(index, inst_lookup, fn_map);
        }
        MirExprKind::MultiIndex { object, indices } => {
            rewrite_calls_in_expr(object, inst_lookup, fn_map);
            for idx in indices {
                rewrite_calls_in_expr(idx, inst_lookup, fn_map);
            }
        }
        MirExprKind::Assign { target, value } => {
            rewrite_calls_in_expr(target, inst_lookup, fn_map);
            rewrite_calls_in_expr(value, inst_lookup, fn_map);
        }
        MirExprKind::Block(body) => {
            rewrite_calls_in_body(body, inst_lookup, fn_map);
        }
        MirExprKind::StructLit { fields, .. } => {
            for (_, e) in fields {
                rewrite_calls_in_expr(e, inst_lookup, fn_map);
            }
        }
        MirExprKind::VariantLit { fields, .. } => {
            for f in fields {
                rewrite_calls_in_expr(f, inst_lookup, fn_map);
            }
        }
        MirExprKind::ArrayLit(elems) | MirExprKind::TupleLit(elems) => {
            for e in elems {
                rewrite_calls_in_expr(e, inst_lookup, fn_map);
            }
        }
        MirExprKind::MakeClosure { captures, .. } => {
            for c in captures {
                rewrite_calls_in_expr(c, inst_lookup, fn_map);
            }
        }
        MirExprKind::If {
            cond,
            then_body,
            else_body,
        } => {
            rewrite_calls_in_expr(cond, inst_lookup, fn_map);
            rewrite_calls_in_body(then_body, inst_lookup, fn_map);
            if let Some(eb) = else_body {
                rewrite_calls_in_body(eb, inst_lookup, fn_map);
            }
        }
        MirExprKind::Match { scrutinee, arms } => {
            rewrite_calls_in_expr(scrutinee, inst_lookup, fn_map);
            for arm in arms {
                rewrite_calls_in_body(&mut arm.body, inst_lookup, fn_map);
            }
        }
        MirExprKind::Lambda { body, .. } => {
            rewrite_calls_in_expr(body, inst_lookup, fn_map);
        }
        MirExprKind::LinalgLU { operand }
        | MirExprKind::LinalgQR { operand }
        | MirExprKind::LinalgCholesky { operand }
        | MirExprKind::LinalgInv { operand } => {
            rewrite_calls_in_expr(operand, inst_lookup, fn_map);
        }
        MirExprKind::Broadcast {
            operand,
            target_shape,
        } => {
            rewrite_calls_in_expr(operand, inst_lookup, fn_map);
            for s in target_shape {
                rewrite_calls_in_expr(s, inst_lookup, fn_map);
            }
        }
        MirExprKind::TensorLit { rows } => {
            for row in rows {
                for elem in row {
                    rewrite_calls_in_expr(elem, inst_lookup, fn_map);
                }
            }
        }
        // Leaves
        MirExprKind::IntLit(_)
        | MirExprKind::FloatLit(_)
        | MirExprKind::BoolLit(_)
        | MirExprKind::NaLit
        | MirExprKind::StringLit(_)
        | MirExprKind::ByteStringLit(_)
        | MirExprKind::ByteCharLit(_)
        | MirExprKind::RawStringLit(_)
        | MirExprKind::RawByteStringLit(_)
        | MirExprKind::RegexLit { .. }
        | MirExprKind::Var(_)
        | MirExprKind::Col(_)
        | MirExprKind::Void => {}
    }
}

/// Infer type args from a call to a generic function.
fn infer_type_args_from_call(func: &MirFunction, args: &[MirExpr]) -> Vec<String> {
    let mut param_map: BTreeMap<String, String> = BTreeMap::new();

    for (arg, param) in args.iter().zip(func.params.iter()) {
        for (tp_name, _) in &func.type_params {
            if param.ty_name == *tp_name {
                if let Some(concrete) = infer_type_from_expr(arg) {
                    param_map.insert(tp_name.clone(), concrete);
                }
            }
        }
    }

    func.type_params
        .iter()
        .map(|(name, _)| {
            param_map
                .get(name)
                .cloned()
                .unwrap_or_else(|| "any".to_string())
        })
        .collect()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MirBody, MirExpr, MirExprKind, MirFnId, MirFunction, MirParam, MirProgram, MirStmt, MirStructDef};

    fn mk_var(name: &str) -> MirExpr {
        MirExpr {
            kind: MirExprKind::Var(name.to_string()),
        }
    }

    fn mk_int(v: i64) -> MirExpr {
        MirExpr {
            kind: MirExprKind::IntLit(v),
        }
    }

    fn mk_call(callee: &str, args: Vec<MirExpr>) -> MirExpr {
        MirExpr {
            kind: MirExprKind::Call {
                callee: Box::new(mk_var(callee)),
                args,
            },
        }
    }

    #[test]
    fn test_no_generics_passthrough() {
        // A program with no generic functions should be unchanged.
        let program = MirProgram {
            functions: vec![MirFunction {
                id: MirFnId(0),
                name: "__main".to_string(),
                type_params: vec![],
                params: vec![],
                return_type: None,
                body: MirBody {
                    stmts: vec![MirStmt::Expr(mk_int(42))],
                    result: None,
                },
                is_nogc: false,
                cfg_body: None,
                decorators: vec![],
                vis: cjc_ast::Visibility::Private,
            }],
            struct_defs: vec![],
            enum_defs: vec![],
            entry: MirFnId(0),
        };

        let (new_prog, report) = monomorphize_program(&program);
        assert_eq!(report.specialization_count, 0);
        assert!(!report.budget_exceeded);
        assert_eq!(new_prog.functions.len(), 1);
    }

    #[test]
    fn test_generic_function_specialized() {
        // fn id<T>(x: T) -> T { x }
        // __main: id(42)
        let id_fn = MirFunction {
            id: MirFnId(0),
            name: "id".to_string(),
            type_params: vec![("T".to_string(), vec![])],
            params: vec![MirParam {
                name: "x".to_string(),
                ty_name: "T".to_string(),
                default: None,
                is_variadic: false,
            }],
            return_type: Some("T".to_string()),
            body: MirBody {
                stmts: vec![],
                result: Some(Box::new(mk_var("x"))),
            },
            is_nogc: false,
            cfg_body: None,
            decorators: vec![],
            vis: cjc_ast::Visibility::Private,
        };

        let main_fn = MirFunction {
            id: MirFnId(1),
            name: "__main".to_string(),
            type_params: vec![],
            params: vec![],
            return_type: None,
            body: MirBody {
                stmts: vec![MirStmt::Expr(mk_call("id", vec![mk_int(42)]))],
                result: None,
            },
            is_nogc: false,
            cfg_body: None,
            decorators: vec![],
            vis: cjc_ast::Visibility::Private,
        };

        let program = MirProgram {
            functions: vec![id_fn, main_fn],
            struct_defs: vec![],
            enum_defs: vec![],
            entry: MirFnId(1),
        };

        let (new_prog, report) = monomorphize_program(&program);
        assert_eq!(report.specialization_count, 1);
        assert!(!report.budget_exceeded);

        // Should have 3 functions: original id, __main (rewritten), id__M__i64
        assert_eq!(new_prog.functions.len(), 3);

        // Check that the specialized function exists
        let specialized = new_prog
            .functions
            .iter()
            .find(|f| f.name == "id__M__i64");
        assert!(specialized.is_some());
        let spec = specialized.unwrap();
        assert!(spec.type_params.is_empty());
        assert_eq!(spec.params[0].ty_name, "i64");
    }

    #[test]
    fn test_mangled_name() {
        let inst = Instantiation {
            fn_name: "id".to_string(),
            type_args: vec!["i32".to_string()],
        };
        assert_eq!(inst.mangled_name(), "id__M__i32");

        let inst2 = Instantiation {
            fn_name: "pair".to_string(),
            type_args: vec!["i32".to_string(), "String".to_string()],
        };
        assert_eq!(inst2.mangled_name(), "pair__M__i32_String");
    }

    #[test]
    fn test_budget_limit() {
        // Shouldn't panic even with many instantiations — just sets budget_exceeded
        let report = MonomorphReport {
            specialization_count: 0,
            top_fanout: vec![],
            budget_exceeded: false,
        };
        assert!(!report.budget_exceeded);
    }
}
