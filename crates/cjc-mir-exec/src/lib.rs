//! CJC MIR Executor (Reference Interpreter)
//!
//! Executes a MIR program by interpreting it directly. This replaces the
//! tree-walk AST interpreter (`cjc-eval`) with one that operates on the
//! lowered MIR representation.
//!
//! The behavior must be **identical** to `cjc-eval` for all existing programs
//! (Parity Gate G-1 / G-2).

use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;
use std::time::Instant;

use cjc_ast::{BinOp, UnaryOp};
use cjc_data::{Column, CsvConfig, CsvReader, DataFrame, StreamingCsvProcessor};
use cjc_mir::*;
use cjc_repro::Rng;
use cjc_runtime::{GcHeap, Tensor, Value};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum MirExecError {
    Return(Value),
    /// A `break` statement — caught by the innermost loop.
    Break,
    /// A `continue` statement — caught by the innermost loop.
    Continue,
    Runtime(String),
    RuntimeError(cjc_runtime::RuntimeError),
    /// Compile-time type errors — collected by the type checker before execution.
    /// Each entry is a rendered diagnostic message.
    TypeErrors(Vec<String>),
    /// P1-2: Tail-call trampoline signal.
    /// Used internally by call_function — never propagates to the user.
    TailCall { name: String, args: Vec<Value> },
}

impl fmt::Display for MirExecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MirExecError::Return(_) => write!(f, "uncaught return"),
            MirExecError::Break => write!(f, "break outside of loop"),
            MirExecError::Continue => write!(f, "continue outside of loop"),
            MirExecError::Runtime(msg) => write!(f, "runtime error: {msg}"),
            MirExecError::RuntimeError(e) => write!(f, "runtime error: {e}"),
            MirExecError::TypeErrors(errs) => {
                write!(f, "{} compile error(s):\n{}", errs.len(), errs.join("\n"))
            }
            MirExecError::TailCall { name, .. } => {
                write!(f, "internal: uncaught tail call to `{name}`")
            }
        }
    }
}

impl std::error::Error for MirExecError {}

impl From<cjc_runtime::RuntimeError> for MirExecError {
    fn from(e: cjc_runtime::RuntimeError) -> Self {
        MirExecError::RuntimeError(e)
    }
}

pub type MirExecResult = Result<Value, MirExecError>;

// ---------------------------------------------------------------------------
// MIR Executor
// ---------------------------------------------------------------------------

pub struct MirExecutor {
    /// P1-5: Function bodies stored as Rc to avoid cloning on every call.
    functions: HashMap<String, Rc<MirFunction>>,
    struct_defs: HashMap<String, MirStructDef>,
    scopes: Vec<HashMap<String, Value>>,
    pub gc_heap: GcHeap,
    pub rng: Rng,
    pub output: Vec<String>,
    start_time: Instant,
    pub gc_collections: u64,
    /// Name of the currently-executing function (for TCO detection).
    current_fn: Option<String>,
    /// Per-call-frame arena stack. Provides bulk-free discipline:
    /// push on function entry, reset on tail-call, pop on return/error.
    arena_stack: Vec<cjc_runtime::ArenaStore>,
    /// Running count of arena allocations (for diagnostics/testing).
    pub arena_alloc_count: u64,
}

impl MirExecutor {
    pub fn new(seed: u64) -> Self {
        Self {
            functions: HashMap::new(),
            struct_defs: HashMap::new(),
            scopes: vec![HashMap::new()],
            gc_heap: GcHeap::new(1024),
            rng: Rng::seeded(seed),
            output: Vec::new(),
            start_time: Instant::now(),
            gc_collections: 0,
            current_fn: None,
            arena_stack: Vec::new(),
            arena_alloc_count: 0,
        }
    }

    // -- Scope management ---------------------------------------------------

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    pub fn define(&mut self, name: &str, val: Value) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), val);
        }
    }

    fn lookup(&self, name: &str) -> Option<&Value> {
        for scope in self.scopes.iter().rev() {
            if let Some(v) = scope.get(name) {
                return Some(v);
            }
        }
        None
    }

    fn assign(&mut self, name: &str, val: Value) -> Result<(), MirExecError> {
        for scope in self.scopes.iter_mut().rev() {
            if scope.contains_key(name) {
                scope.insert(name.to_string(), val);
                return Ok(());
            }
        }
        Err(MirExecError::Runtime(format!(
            "assignment to undefined variable `{name}`"
        )))
    }

    // -- Program execution --------------------------------------------------

    pub fn exec(&mut self, program: &MirProgram) -> MirExecResult {
        // Register all functions and struct defs
        // P1-5: Store as Rc to avoid cloning bodies on every call.
        for func in &program.functions {
            self.functions.insert(func.name.clone(), Rc::new(func.clone()));
        }
        for sd in &program.struct_defs {
            self.struct_defs.insert(sd.name.clone(), sd.clone());
        }

        // Execute __main
        let main_fn = self.functions.get("__main").cloned();
        if let Some(main_fn) = main_fn {
            let result = self.exec_body(&main_fn.body)?;

            // If there is a user-defined `main` function, call it
            if self.functions.contains_key("main") {
                return self.call_function("main", &[]);
            }

            return Ok(result);
        }

        Ok(Value::Void)
    }

    fn exec_body(&mut self, body: &MirBody) -> MirExecResult {
        let mut last = Value::Void;
        for stmt in &body.stmts {
            last = self.exec_stmt(stmt)?;
        }
        if let Some(ref expr) = body.result {
            // P1-2: Detect tail call in the result-expression position.
            // A body like `fn foo(n) { foo(n-1) }` lowers to a body with no
            // stmts and a result expr that is a direct call.  Emit TailCall so
            // the trampoline in call_function can loop instead of recursing.
            if let MirExprKind::Call { callee, args } = &expr.kind {
                if let MirExprKind::Var(callee_name) = &callee.kind {
                    if self.functions.contains_key(callee_name.as_str()) {
                        let evaluated_args: Result<Vec<Value>, _> =
                            args.iter().map(|a| self.eval_expr(a)).collect();
                        let evaluated_args = evaluated_args?;
                        return Err(MirExecError::TailCall {
                            name: callee_name.clone(),
                            args: evaluated_args,
                        });
                    }
                }
            }
            self.eval_expr(expr)
        } else {
            Ok(last)
        }
    }

    fn exec_body_scoped(&mut self, body: &MirBody) -> MirExecResult {
        self.push_scope();
        let result = self.exec_body(body);
        self.pop_scope();
        result
    }

    fn exec_stmt(&mut self, stmt: &MirStmt) -> MirExecResult {
        match stmt {
            MirStmt::Let { name, init, alloc_hint, .. } => {
                let val = self.eval_expr(init)?;
                // Track arena-classified allocations for diagnostics.
                if let Some(cjc_mir::AllocHint::Arena) = alloc_hint {
                    self.arena_alloc_count += 1;
                }
                self.define(name, val);
                Ok(Value::Void)
            }
            MirStmt::Expr(expr) => {
                self.eval_expr(expr)?;
                Ok(Value::Void)
            }
            MirStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                let cond_val = self.eval_expr(cond)?;
                let cond_bool = match cond_val {
                    Value::Bool(b) => b,
                    other => {
                        return Err(MirExecError::Runtime(format!(
                            "if condition must be Bool, got {}",
                            other.type_name()
                        )));
                    }
                };
                if cond_bool {
                    self.exec_body_scoped(then_body)
                } else if let Some(else_b) = else_body {
                    self.exec_body_scoped(else_b)
                } else {
                    Ok(Value::Void)
                }
            }
            MirStmt::While { cond, body } => {
                loop {
                    let cond_val = self.eval_expr(cond)?;
                    let cond_bool = match cond_val {
                        Value::Bool(b) => b,
                        other => {
                            return Err(MirExecError::Runtime(format!(
                                "while condition must be Bool, got {}",
                                other.type_name()
                            )));
                        }
                    };
                    if !cond_bool {
                        break;
                    }
                    match self.exec_body_scoped(body) {
                        Ok(_) => {}
                        Err(MirExecError::Break) => break,
                        Err(MirExecError::Continue) => continue,
                        Err(e) => return Err(e),
                    }
                }
                Ok(Value::Void)
            }
            MirStmt::Return(opt_expr) => {
                // P1-2: Tail-call optimization.
                // If the return value is a direct call to a user-defined function,
                // emit TailCall instead of recursing. The trampoline in
                // call_function will restart the loop without growing the stack.
                if let Some(expr) = opt_expr {
                    if let MirExprKind::Call { callee, args } = &expr.kind {
                        if let MirExprKind::Var(callee_name) = &callee.kind {
                            if self.functions.contains_key(callee_name.as_str()) {
                                // Evaluate all arguments before trampolining.
                                let evaluated_args: Result<Vec<Value>, _> =
                                    args.iter().map(|a| self.eval_expr(a)).collect();
                                let evaluated_args = evaluated_args?;
                                return Err(MirExecError::TailCall {
                                    name: callee_name.clone(),
                                    args: evaluated_args,
                                });
                            }
                        }
                    }
                    // Not a tail call — evaluate normally.
                    let val = self.eval_expr(expr)?;
                    Err(MirExecError::Return(val))
                } else {
                    Err(MirExecError::Return(Value::Void))
                }
            }
            MirStmt::Break => Err(MirExecError::Break),
            MirStmt::Continue => Err(MirExecError::Continue),
            MirStmt::NoGcBlock(body) => self.exec_body_scoped(body),
        }
    }

    // -- Expression evaluation ----------------------------------------------

    pub fn eval_expr(&mut self, expr: &MirExpr) -> MirExecResult {
        match &expr.kind {
            MirExprKind::IntLit(v) => Ok(Value::Int(*v)),
            MirExprKind::FloatLit(v) => Ok(Value::Float(*v)),
            MirExprKind::BoolLit(b) => Ok(Value::Bool(*b)),
            MirExprKind::StringLit(s) => Ok(Value::String(Rc::new(s.clone()))),
            MirExprKind::ByteStringLit(bytes) => Ok(Value::ByteSlice(Rc::new(bytes.clone()))),
            MirExprKind::ByteCharLit(b) => Ok(Value::U8(*b)),
            MirExprKind::RawStringLit(s) => Ok(Value::String(Rc::new(s.clone()))),
            MirExprKind::RawByteStringLit(bytes) => Ok(Value::ByteSlice(Rc::new(bytes.clone()))),
            MirExprKind::RegexLit { pattern, flags } => Ok(Value::Regex { pattern: pattern.clone(), flags: flags.clone() }),
            MirExprKind::TensorLit { rows } => {
                let n_rows = rows.len();
                if n_rows == 0 {
                    return Ok(Value::Tensor(Tensor::from_vec(vec![], &[0]).map_err(|e| MirExecError::Runtime(format!("{e}")))?));
                }
                let n_cols = rows[0].len();
                let mut data = Vec::with_capacity(n_rows * n_cols);
                for row in rows {
                    if n_rows > 1 && row.len() != n_cols {
                        return Err(MirExecError::Runtime(format!(
                            "tensor literal: row length mismatch, expected {} but got {}",
                            n_cols, row.len()
                        )));
                    }
                    for expr in row {
                        let val = self.eval_expr(expr)?;
                        match val {
                            Value::Float(f) => data.push(f),
                            Value::Int(i) => data.push(i as f64),
                            _ => return Err(MirExecError::Runtime(
                                "tensor literal elements must be numbers".to_string(),
                            )),
                        }
                    }
                }
                let shape = if n_rows == 1 {
                    vec![n_cols]
                } else {
                    vec![n_rows, n_cols]
                };
                Ok(Value::Tensor(Tensor::from_vec(data, &shape).map_err(|e| MirExecError::Runtime(format!("{e}")))?))
            }
            MirExprKind::Var(name) => {
                if let Some(v) = self.lookup(name).cloned() {
                    return Ok(v);
                }
                // Fall back: if name is a registered function, produce a Fn value.
                // This allows passing functions as first-class values (e.g. opt.map(double)).
                if let Some(func) = self.functions.get(name.as_str()) {
                    let arity = func.params.len();
                    return Ok(Value::Fn(cjc_runtime::FnValue {
                        name: name.clone(),
                        arity,
                        body_id: 0,
                    }));
                }
                Err(MirExecError::Runtime(format!("undefined variable `{name}`")))
            }
            MirExprKind::Binary { op, left, right } => self.eval_binary(*op, left, right),
            MirExprKind::Unary { op, operand } => self.eval_unary(*op, operand),
            MirExprKind::Call { callee, args } => self.eval_call(callee, args),
            MirExprKind::Field { object, name } => self.eval_field(object, name),
            MirExprKind::Index { object, index } => self.eval_index(object, index),
            MirExprKind::MultiIndex { object, indices } => {
                self.eval_multi_index(object, indices)
            }
            MirExprKind::Assign { target, value } => {
                let val = self.eval_expr(value)?;
                self.exec_assign(target, val)?;
                Ok(Value::Void)
            }
            MirExprKind::Block(body) => self.exec_body_scoped(body),
            MirExprKind::StructLit { name, fields } => {
                let mut field_map = HashMap::new();
                for (fname, fexpr) in fields {
                    let val = self.eval_expr(fexpr)?;
                    field_map.insert(fname.clone(), val);
                }
                Ok(Value::Struct {
                    name: name.clone(),
                    fields: field_map,
                })
            }
            MirExprKind::ArrayLit(elems) => {
                let mut vals = Vec::with_capacity(elems.len());
                for e in elems {
                    vals.push(self.eval_expr(e)?);
                }
                Ok(Value::Array(Rc::new(vals)))
            }
            MirExprKind::Col(name) => Ok(Value::String(Rc::new(format!("col:{name}")))),
            MirExprKind::Lambda { params, body } => {
                let lambda_name = format!("<mir_lambda@{}>", params.len());
                // Create a MIR function for the lambda
                let mir_fn = MirFunction {
                    id: MirFnId(u32::MAX),
                    name: lambda_name.clone(),
                    type_params: vec![],
                    params: params.clone(),
                    return_type: None,
                    body: MirBody {
                        stmts: vec![],
                        result: Some(body.clone()),
                    },
                    is_nogc: false,
                };
                self.functions.insert(lambda_name.clone(), Rc::new(mir_fn));
                Ok(Value::Fn(cjc_runtime::FnValue {
                    name: lambda_name,
                    arity: params.len(),
                    body_id: 0,
                }))
            }
            MirExprKind::MakeClosure {
                fn_name,
                captures,
            } => {
                // Evaluate each capture expression to build the env
                let mut env = Vec::with_capacity(captures.len());
                for cap in captures {
                    env.push(self.eval_expr(cap)?);
                }
                // Look up the lifted function to determine the original
                // arity (total params minus captures)
                let total_params = self
                    .functions
                    .get(fn_name)
                    .map(|f| f.params.len())
                    .unwrap_or(0);
                let arity = total_params - env.len();
                Ok(Value::Closure {
                    fn_name: fn_name.clone(),
                    env,
                    arity,
                })
            }
            MirExprKind::If {
                cond,
                then_body,
                else_body,
            } => {
                let cond_val = self.eval_expr(cond)?;
                let cond_bool = match cond_val {
                    Value::Bool(b) => b,
                    other => {
                        return Err(MirExecError::Runtime(format!(
                            "if condition must be Bool, got {}",
                            other.type_name()
                        )));
                    }
                };
                if cond_bool {
                    self.exec_body_scoped(then_body)
                } else if let Some(else_b) = else_body {
                    self.exec_body_scoped(else_b)
                } else {
                    Ok(Value::Void)
                }
            }
            MirExprKind::Match { scrutinee, arms } => {
                let scrut_val = self.eval_expr(scrutinee)?;
                for arm in arms {
                    if let Some(bindings) = self.match_pattern(&arm.pattern, &scrut_val) {
                        self.push_scope();
                        for (name, val) in bindings {
                            self.define(&name, val);
                        }
                        let result = self.exec_body(&arm.body);
                        self.pop_scope();
                        return result;
                    }
                }
                Err(MirExecError::Runtime(
                    "non-exhaustive match: no arm matched".to_string(),
                ))
            }
            MirExprKind::TupleLit(elems) => {
                let mut vals = Vec::with_capacity(elems.len());
                for e in elems {
                    vals.push(self.eval_expr(e)?);
                }
                Ok(Value::Tuple(Rc::new(vals)))
            }
            // Linalg opcodes
            MirExprKind::LinalgLU { operand } => {
                let val = self.eval_expr(operand)?;
                match val {
                    Value::Tensor(t) => {
                        let (l, u, _pivots) = t.lu_decompose()?;
                        Ok(Value::Tuple(Rc::new(vec![Value::Tensor(l), Value::Tensor(u)])))
                    }
                    _ => Err(MirExecError::Runtime("LU: expected Tensor".into())),
                }
            }
            MirExprKind::LinalgQR { operand } => {
                let val = self.eval_expr(operand)?;
                match val {
                    Value::Tensor(t) => {
                        let (q, r) = t.qr_decompose()?;
                        Ok(Value::Tuple(Rc::new(vec![Value::Tensor(q), Value::Tensor(r)])))
                    }
                    _ => Err(MirExecError::Runtime("QR: expected Tensor".into())),
                }
            }
            MirExprKind::LinalgCholesky { operand } => {
                let val = self.eval_expr(operand)?;
                match val {
                    Value::Tensor(t) => {
                        let l = t.cholesky()?;
                        Ok(Value::Tensor(l))
                    }
                    _ => Err(MirExecError::Runtime("Cholesky: expected Tensor".into())),
                }
            }
            MirExprKind::LinalgInv { operand } => {
                let val = self.eval_expr(operand)?;
                match val {
                    Value::Tensor(t) => {
                        let inv = t.inverse()?;
                        Ok(Value::Tensor(inv))
                    }
                    _ => Err(MirExecError::Runtime("Inv: expected Tensor".into())),
                }
            }
            MirExprKind::Broadcast { operand, target_shape } => {
                let val = self.eval_expr(operand)?;
                let shape: Result<Vec<usize>, _> = target_shape
                    .iter()
                    .map(|e| {
                        self.eval_expr(e).and_then(|v| match v {
                            Value::Int(n) => Ok(n as usize),
                            _ => Err(MirExecError::Runtime("Broadcast: shape must be integers".into())),
                        })
                    })
                    .collect();
                match val {
                    Value::Tensor(t) => {
                        let broadcasted = t.broadcast_to(&shape?)?;
                        Ok(Value::Tensor(broadcasted))
                    }
                    _ => Err(MirExecError::Runtime("Broadcast: expected Tensor".into())),
                }
            }
            MirExprKind::VariantLit {
                enum_name,
                variant,
                fields,
            } => {
                let mut field_vals = Vec::with_capacity(fields.len());
                for f in fields {
                    field_vals.push(self.eval_expr(f)?);
                }
                Ok(Value::Enum {
                    enum_name: enum_name.clone(),
                    variant: variant.clone(),
                    fields: field_vals,
                })
            }
            MirExprKind::Void => Ok(Value::Void),
        }
    }

    /// Try to match a value against a pattern. Returns Some(bindings) if it
    /// matches, None otherwise. Bindings are (name, value) pairs.
    fn match_pattern(
        &self,
        pattern: &MirPattern,
        value: &Value,
    ) -> Option<Vec<(String, Value)>> {
        match pattern {
            MirPattern::Wildcard => Some(vec![]),
            MirPattern::Binding(name) => Some(vec![(name.clone(), value.clone())]),
            MirPattern::LitInt(v) => match value {
                Value::Int(i) => {
                    if i == v {
                        Some(vec![])
                    } else {
                        None
                    }
                }
                _ => None,
            },
            MirPattern::LitFloat(v) => match value {
                Value::Float(f) => {
                    if f == v {
                        Some(vec![])
                    } else {
                        None
                    }
                }
                _ => None,
            },
            MirPattern::LitBool(v) => match value {
                Value::Bool(b) => {
                    if b == v {
                        Some(vec![])
                    } else {
                        None
                    }
                }
                _ => None,
            },
            MirPattern::LitString(v) => match value {
                Value::String(s) => {
                    if s.as_str() == v {
                        Some(vec![])
                    } else {
                        None
                    }
                }
                _ => None,
            },
            MirPattern::Tuple(pats) => match value {
                Value::Tuple(vals) => {
                    if pats.len() != vals.len() {
                        return None;
                    }
                    let mut all_bindings = Vec::new();
                    for (pat, val) in pats.iter().zip(vals.iter()) {
                        match self.match_pattern(pat, val) {
                            Some(bindings) => all_bindings.extend(bindings),
                            None => return None,
                        }
                    }
                    Some(all_bindings)
                }
                _ => None,
            },
            MirPattern::Struct { name, fields } => match value {
                Value::Struct {
                    name: val_name,
                    fields: val_fields,
                } => {
                    if name != val_name {
                        return None;
                    }
                    let mut all_bindings = Vec::new();
                    for (field_name, field_pat) in fields {
                        match val_fields.get(field_name) {
                            Some(field_val) => {
                                match self.match_pattern(field_pat, field_val) {
                                    Some(bindings) => all_bindings.extend(bindings),
                                    None => return None,
                                }
                            }
                            None => return None,
                        }
                    }
                    Some(all_bindings)
                }
                _ => None,
            },
            MirPattern::Variant {
                enum_name: _,
                variant,
                fields,
            } => match value {
                Value::Enum {
                    enum_name: _,
                    variant: val_variant,
                    fields: val_fields,
                } => {
                    if variant != val_variant {
                        return None;
                    }
                    if fields.len() != val_fields.len() {
                        return None;
                    }
                    let mut all_bindings = Vec::new();
                    for (pat, val) in fields.iter().zip(val_fields.iter()) {
                        match self.match_pattern(pat, val) {
                            Some(bindings) => all_bindings.extend(bindings),
                            None => return None,
                        }
                    }
                    Some(all_bindings)
                }
                _ => None,
            },
        }
    }

    // -- Binary operations --------------------------------------------------

    fn eval_binary(
        &mut self,
        op: BinOp,
        left: &MirExpr,
        right: &MirExpr,
    ) -> MirExecResult {
        // Short-circuit for logical operators
        if op == BinOp::And {
            let lv = self.eval_expr(left)?;
            return match lv {
                Value::Bool(false) => Ok(Value::Bool(false)),
                Value::Bool(true) => {
                    let rv = self.eval_expr(right)?;
                    match rv {
                        Value::Bool(b) => Ok(Value::Bool(b)),
                        _ => Err(MirExecError::Runtime(
                            "`&&` requires Bool operands".to_string(),
                        )),
                    }
                }
                _ => Err(MirExecError::Runtime(
                    "`&&` requires Bool operands".to_string(),
                )),
            };
        }
        if op == BinOp::Or {
            let lv = self.eval_expr(left)?;
            return match lv {
                Value::Bool(true) => Ok(Value::Bool(true)),
                Value::Bool(false) => {
                    let rv = self.eval_expr(right)?;
                    match rv {
                        Value::Bool(b) => Ok(Value::Bool(b)),
                        _ => Err(MirExecError::Runtime(
                            "`||` requires Bool operands".to_string(),
                        )),
                    }
                }
                _ => Err(MirExecError::Runtime(
                    "`||` requires Bool operands".to_string(),
                )),
            };
        }

        // Regex match/not-match operators
        if op == BinOp::Match || op == BinOp::NotMatch {
            let lv = self.eval_expr(left)?;
            let rv = self.eval_expr(right)?;
            let (hay, pat, flags) = match (&lv, &rv) {
                (Value::ByteSlice(b), Value::Regex { pattern, flags }) => {
                    (b.as_slice(), pattern.as_str(), flags.as_str())
                }
                (Value::String(s), Value::Regex { pattern, flags }) => {
                    (s.as_bytes(), pattern.as_str(), flags.as_str())
                }
                (Value::StrView(b), Value::Regex { pattern, flags }) => {
                    (b.as_slice(), pattern.as_str(), flags.as_str())
                }
                _ => {
                    return Err(MirExecError::Runtime(format!(
                        "`{}` requires (ByteSlice|String|StrView) ~= Regex, got {} and {}",
                        op, lv.type_name(), rv.type_name()
                    )));
                }
            };
            let matched = cjc_regex::is_match(pat, flags, hay);
            return if op == BinOp::Match {
                Ok(Value::Bool(matched))
            } else {
                Ok(Value::Bool(!matched))
            };
        }

        let lv = self.eval_expr(left)?;
        let rv = self.eval_expr(right)?;

        match (&lv, &rv) {
            (Value::Int(a), Value::Int(b)) => self.binop_int(op, *a, *b),
            (Value::Float(a), Value::Float(b)) => self.binop_float(op, *a, *b),
            (Value::Int(a), Value::Float(b)) => self.binop_float(op, *a as f64, *b),
            (Value::Float(a), Value::Int(b)) => self.binop_float(op, *a, *b as f64),
            (Value::Bool(a), Value::Bool(b)) => match op {
                BinOp::Eq => Ok(Value::Bool(a == b)),
                BinOp::Ne => Ok(Value::Bool(a != b)),
                _ => Err(MirExecError::Runtime(format!(
                    "cannot apply `{op}` to Bool values"
                ))),
            },
            (Value::String(a), Value::String(b)) => match op {
                BinOp::Add => Ok(Value::String(Rc::new(format!("{a}{b}")))),
                BinOp::Eq => Ok(Value::Bool(a == b)),
                BinOp::Ne => Ok(Value::Bool(a != b)),
                _ => Err(MirExecError::Runtime(format!(
                    "cannot apply `{op}` to String values"
                ))),
            },
            // F16 x F16 arithmetic
            (Value::F16(a), Value::F16(b)) => match op {
                BinOp::Add => Ok(Value::F16(a.add(*b))),
                BinOp::Sub => Ok(Value::F16(a.sub(*b))),
                BinOp::Mul => Ok(Value::F16(a.mul(*b))),
                BinOp::Div => Ok(Value::F16(a.div(*b))),
                BinOp::Eq => Ok(Value::Bool(a.to_f64() == b.to_f64())),
                BinOp::Ne => Ok(Value::Bool(a.to_f64() != b.to_f64())),
                BinOp::Lt => Ok(Value::Bool(a.to_f64() < b.to_f64())),
                BinOp::Le => Ok(Value::Bool(a.to_f64() <= b.to_f64())),
                BinOp::Gt => Ok(Value::Bool(a.to_f64() > b.to_f64())),
                BinOp::Ge => Ok(Value::Bool(a.to_f64() >= b.to_f64())),
                _ => Err(MirExecError::Runtime(format!(
                    "cannot apply `{op}` to f16 values"
                ))),
            },
            // Complex x Complex arithmetic
            (Value::Complex(a), Value::Complex(b)) => match op {
                BinOp::Add => Ok(Value::Complex(a.add(*b))),
                BinOp::Sub => Ok(Value::Complex(a.sub(*b))),
                BinOp::Mul => Ok(Value::Complex(a.mul_fixed(*b))),
                BinOp::Div => Ok(Value::Complex(a.div_fixed(*b))),
                BinOp::Eq => Ok(Value::Bool(a.re == b.re && a.im == b.im)),
                BinOp::Ne => Ok(Value::Bool(a.re != b.re || a.im != b.im)),
                _ => Err(MirExecError::Runtime(format!(
                    "cannot apply `{op}` to Complex values"
                ))),
            },
            (Value::Tensor(a), Value::Tensor(b)) => match op {
                BinOp::Add => Ok(Value::Tensor(a.add(b)?)),
                BinOp::Sub => Ok(Value::Tensor(a.sub(b)?)),
                BinOp::Mul => Ok(Value::Tensor(a.mul_elem(b)?)),
                BinOp::Div => Ok(Value::Tensor(a.div_elem(b)?)),
                _ => Err(MirExecError::Runtime(format!(
                    "cannot apply `{op}` to Tensor values"
                ))),
            },
            _ => Err(MirExecError::Runtime(format!(
                "cannot apply `{op}` to {} and {}",
                lv.type_name(),
                rv.type_name()
            ))),
        }
    }

    fn binop_int(&self, op: BinOp, a: i64, b: i64) -> MirExecResult {
        match op {
            BinOp::Add => Ok(Value::Int(a.wrapping_add(b))),
            BinOp::Sub => Ok(Value::Int(a.wrapping_sub(b))),
            BinOp::Mul => Ok(Value::Int(a.wrapping_mul(b))),
            BinOp::Div => {
                if b == 0 {
                    Err(MirExecError::Runtime("division by zero".to_string()))
                } else {
                    Ok(Value::Int(a / b))
                }
            }
            BinOp::Mod => {
                if b == 0 {
                    Err(MirExecError::Runtime("modulo by zero".to_string()))
                } else {
                    Ok(Value::Int(a % b))
                }
            }
            BinOp::Eq => Ok(Value::Bool(a == b)),
            BinOp::Ne => Ok(Value::Bool(a != b)),
            BinOp::Lt => Ok(Value::Bool(a < b)),
            BinOp::Gt => Ok(Value::Bool(a > b)),
            BinOp::Le => Ok(Value::Bool(a <= b)),
            BinOp::Ge => Ok(Value::Bool(a >= b)),
            BinOp::And | BinOp::Or | BinOp::Match | BinOp::NotMatch => Err(MirExecError::Runtime(format!(
                "cannot apply `{op}` to Int values"
            ))),
        }
    }

    fn binop_float(&self, op: BinOp, a: f64, b: f64) -> MirExecResult {
        match op {
            BinOp::Add => Ok(Value::Float(a + b)),
            BinOp::Sub => Ok(Value::Float(a - b)),
            BinOp::Mul => Ok(Value::Float(a * b)),
            BinOp::Div => Ok(Value::Float(a / b)),
            BinOp::Mod => Ok(Value::Float(a % b)),
            BinOp::Eq => Ok(Value::Bool(a == b)),
            BinOp::Ne => Ok(Value::Bool(a != b)),
            BinOp::Lt => Ok(Value::Bool(a < b)),
            BinOp::Gt => Ok(Value::Bool(a > b)),
            BinOp::Le => Ok(Value::Bool(a <= b)),
            BinOp::Ge => Ok(Value::Bool(a >= b)),
            BinOp::And | BinOp::Or | BinOp::Match | BinOp::NotMatch => Err(MirExecError::Runtime(format!(
                "cannot apply `{op}` to Float values"
            ))),
        }
    }

    // -- Unary operations ---------------------------------------------------

    fn eval_unary(&mut self, op: UnaryOp, operand: &MirExpr) -> MirExecResult {
        let val = self.eval_expr(operand)?;
        match (op, &val) {
            (UnaryOp::Neg, Value::Int(v)) => Ok(Value::Int(-v)),
            (UnaryOp::Neg, Value::Float(v)) => Ok(Value::Float(-v)),
            (UnaryOp::Neg, Value::Tensor(t)) => Ok(Value::Tensor(t.map(|x| -x))),
            (UnaryOp::Neg, Value::F16(v)) => Ok(Value::F16(v.neg())),
            (UnaryOp::Neg, Value::Complex(z)) => Ok(Value::Complex(z.neg())),
            (UnaryOp::Not, Value::Bool(b)) => Ok(Value::Bool(!b)),
            _ => Err(MirExecError::Runtime(format!(
                "cannot apply `{op}` to {}",
                val.type_name()
            ))),
        }
    }

    // -- Call dispatch -------------------------------------------------------

    fn eval_call(&mut self, callee: &MirExpr, args: &[MirExpr]) -> MirExecResult {
        let mut arg_vals: Vec<Value> = Vec::with_capacity(args.len());
        for arg in args {
            arg_vals.push(self.eval_expr(arg)?);
        }

        match &callee.kind {
            MirExprKind::Var(name) => self.dispatch_call(name, arg_vals),
            MirExprKind::Field { object, name } => {
                // Static method: Tensor.zeros(...) etc.
                if let MirExprKind::Var(obj_name) = &object.kind {
                    let qualified = format!("{obj_name}.{name}");
                    if self.is_known_builtin(&qualified)
                        || self.functions.contains_key(&qualified)
                    {
                        return self.dispatch_call(&qualified, arg_vals);
                    }
                }
                // Instance method
                let obj_val = self.eval_expr(object)?;
                self.dispatch_method(obj_val, name, arg_vals)
            }
            _ => {
                let callee_val = self.eval_expr(callee)?;
                match callee_val {
                    Value::Fn(fv) => self.call_function(&fv.name, &arg_vals),
                    Value::Closure {
                        fn_name,
                        env,
                        ..
                    } => {
                        // Prepend captured env values to the argument list
                        let mut full_args = env;
                        full_args.extend(arg_vals);
                        self.call_function(&fn_name, &full_args)
                    }
                    _ => Err(MirExecError::Runtime(format!(
                        "cannot call value of type {}",
                        callee_val.type_name()
                    ))),
                }
            }
        }
    }

    fn is_known_builtin(&self, name: &str) -> bool {
        matches!(
            name,
            "print"
                | "to_string"
                | "Tensor.zeros"
                | "Tensor.ones"
                | "Tensor.randn"
                | "Tensor.from_vec"
                | "matmul"
                | "Buffer.alloc"
                | "len"
                | "push"
                | "assert"
                | "assert_eq"
                | "clock"
                | "gc_alloc"
                | "gc_collect"
                | "gc_live_count"
                | "linalg.lu"
                | "linalg.qr"
                | "linalg.cholesky"
                | "linalg.inv"
                | "Map.new"
                | "Map.insert"
                | "Map.get"
                | "Map.remove"
                | "Map.len"
                | "Map.contains_key"
                | "SparseCsr.matvec"
                | "SparseCsr.to_dense"
                | "SparseCoo.to_csr"
                | "Tensor.from_bytes"
                | "Scratchpad.new"
                | "PagedKvCache.new"
                | "AlignedByteSlice.from_bytes"
                | "attention"
                // Phase 8: CSV / Data Logistics
                | "Csv.parse"
                | "Csv.parse_tsv"
                | "Csv.stream_sum"
                | "Csv.stream_minmax"
                | "f16_to_f64"
                | "f64_to_f16"
                | "f16_to_f32"
                | "f32_to_f16"
                | "bf16_to_f32"
                | "f32_to_bf16"
                | "Complex"
                // JSON builtins
                | "json_parse"
                | "json_stringify"
                // DateTime builtins
                | "datetime_now"
                | "datetime_from_epoch"
                | "datetime_from_parts"
                | "datetime_year"
                | "datetime_month"
                | "datetime_day"
                | "datetime_hour"
                | "datetime_minute"
                | "datetime_second"
                | "datetime_diff"
                | "datetime_add_millis"
                | "datetime_format"
                // File I/O builtins
                | "file_read"
                | "file_write"
                | "file_exists"
                | "file_lines"
                // Window functions
                | "window_sum"
                | "window_mean"
                | "window_min"
                | "window_max"
        )
    }

    fn dispatch_call(&mut self, name: &str, args: Vec<Value>) -> MirExecResult {
        // If the name refers to a variable holding a Closure or Fn value,
        // dispatch through it rather than looking for a named function.
        if !self.is_known_builtin(name) && !self.functions.contains_key(name) {
            if let Some(val) = self.lookup(name).cloned() {
                match val {
                    Value::Closure {
                        fn_name,
                        env,
                        ..
                    } => {
                        let mut full_args = env;
                        full_args.extend(args);
                        return self.call_function(&fn_name, &full_args);
                    }
                    Value::Fn(fv) => {
                        return self.call_function(&fv.name, &args);
                    }
                    _ => {}
                }
            }
        }
        // Stateful builtins that need interpreter state
        match name {
            "print" => {
                let parts: Vec<String> = args.iter().map(|v| format!("{v}")).collect();
                let line = parts.join(" ");
                println!("{line}");
                self.output.push(line);
                return Ok(Value::Void);
            }
            "Tensor.randn" => {
                let shape = cjc_runtime::builtins::value_to_shape(&args[0])
                    .map_err(MirExecError::Runtime)?;
                let t = Tensor::randn(&shape, &mut self.rng);
                return Ok(Value::Tensor(t));
            }
            "clock" => {
                let elapsed = self.start_time.elapsed().as_secs_f64();
                return Ok(Value::Float(elapsed));
            }
            "datetime_now" => {
                return Ok(Value::Int(cjc_runtime::datetime::datetime_now()));
            }
            "gc_alloc" => {
                let tag = if args.is_empty() {
                    "anon".to_string()
                } else {
                    format!("{}", args[0])
                };
                let _ref = self.gc_heap.alloc(tag);
                return Ok(Value::Void);
            }
            "gc_collect" => {
                self.gc_heap.collect(&[]);
                self.gc_collections += 1;
                return Ok(Value::Void);
            }
            "gc_live_count" => {
                return Ok(Value::Int(self.gc_heap.live_count() as i64));
            }
            _ => {}
        }

        // Try shared (stateless) builtins
        match cjc_runtime::builtins::dispatch_builtin(name, &args) {
            Ok(Some(value)) => return Ok(value),
            Err(msg) => return Err(MirExecError::Runtime(msg)),
            Ok(None) => {} // not a shared builtin, fall through
        }

        // CSV / Data Logistics builtins (depend on cjc-data, not in shared module)
        match name {
            "Csv.parse" | "Csv.parse_tsv" => {
                if args.is_empty() || args.len() > 2 {
                    return Err(MirExecError::Runtime(
                        "Csv.parse requires 1 argument: bytes (+ optional max_rows)".to_string(),
                    ));
                }
                let bytes = cjc_runtime::builtins::value_to_bytes(&args[0])
                    .map_err(MirExecError::Runtime)?;
                let max_rows = if args.len() == 2 {
                    Some(cjc_runtime::builtins::value_to_usize(&args[1])
                        .map_err(MirExecError::Runtime)?)
                } else {
                    None
                };
                let delim = if name == "Csv.parse_tsv" { b'\t' } else { b',' };
                let config = CsvConfig { delimiter: delim, max_rows, ..CsvConfig::default() };
                let df = CsvReader::new(config)
                    .parse(&bytes)
                    .map_err(|e| MirExecError::Runtime(format!("Csv.parse error: {}", e)))?;
                return Ok(dataframe_to_value(df));
            }
            "Csv.stream_sum" => {
                if args.is_empty() {
                    return Err(MirExecError::Runtime(
                        "Csv.stream_sum requires 1 argument: bytes".to_string(),
                    ));
                }
                let bytes = cjc_runtime::builtins::value_to_bytes(&args[0])
                    .map_err(MirExecError::Runtime)?;
                let config = CsvConfig::default();
                let (names, sums, count) = StreamingCsvProcessor::new(config)
                    .sum_columns(&bytes)
                    .map_err(|e| MirExecError::Runtime(format!("Csv.stream_sum error: {}", e)))?;
                let mut fields = std::collections::HashMap::new();
                for (name, sum) in names.iter().zip(sums.iter()) {
                    fields.insert(name.clone(), Value::Float(*sum));
                }
                fields.insert("__row_count".to_string(), Value::Int(count as i64));
                return Ok(Value::Struct { name: "CsvStats".to_string(), fields });
            }
            "Csv.stream_minmax" => {
                if args.is_empty() {
                    return Err(MirExecError::Runtime(
                        "Csv.stream_minmax requires 1 argument: bytes".to_string(),
                    ));
                }
                let bytes = cjc_runtime::builtins::value_to_bytes(&args[0])
                    .map_err(MirExecError::Runtime)?;
                let config = CsvConfig::default();
                let (names, mins, maxs, count) = StreamingCsvProcessor::new(config)
                    .minmax_columns(&bytes)
                    .map_err(|e| MirExecError::Runtime(format!("Csv.stream_minmax error: {}", e)))?;
                let mut fields = std::collections::HashMap::new();
                for i in 0..names.len() {
                    fields.insert(format!("{}_min", names[i]), Value::Float(mins[i]));
                    fields.insert(format!("{}_max", names[i]), Value::Float(maxs[i]));
                }
                fields.insert("__row_count".to_string(), Value::Int(count as i64));
                return Ok(Value::Struct { name: "CsvMinMax".to_string(), fields });
            }
            _ => {}
        }

        // Try user-defined function.
        if self.functions.contains_key(name) {
            self.call_function(name, &args)
        } else {
            Err(MirExecError::Runtime(format!(
                "undefined function `{name}`"
            )))
        }
    }

    fn dispatch_method(
        &mut self,
        receiver: Value,
        method: &str,
        args: Vec<Value>,
    ) -> MirExecResult {
        match (&receiver, method) {
            (Value::Tensor(t), "sum") => Ok(Value::Float(t.sum())),
            (Value::Tensor(t), "mean") => Ok(Value::Float(t.mean())),
            (Value::Tensor(t), "binned_sum") => Ok(Value::Float(t.binned_sum())),

            // Complex methods
            (Value::Complex(z), "re") => Ok(Value::Float(z.re)),
            (Value::Complex(z), "im") => Ok(Value::Float(z.im)),
            (Value::Complex(z), "abs") => Ok(Value::Float(z.abs())),
            (Value::Complex(z), "conj") => Ok(Value::Complex(z.conj())),
            (Value::Complex(z), "norm_sq") => Ok(Value::Float(z.norm_sq())),
            (Value::Complex(z), "add") => {
                if let Some(Value::Complex(w)) = args.first() {
                    Ok(Value::Complex(z.add(*w)))
                } else {
                    Err(MirExecError::Runtime("Complex.add requires a Complex argument".to_string()))
                }
            }
            (Value::Complex(z), "mul") => {
                if let Some(Value::Complex(w)) = args.first() {
                    Ok(Value::Complex(z.mul_fixed(*w)))
                } else {
                    Err(MirExecError::Runtime("Complex.mul requires a Complex argument".to_string()))
                }
            }
            (Value::Complex(z), "sub") => {
                if let Some(Value::Complex(w)) = args.first() {
                    Ok(Value::Complex(z.sub(*w)))
                } else {
                    Err(MirExecError::Runtime("Complex.sub requires a Complex argument".to_string()))
                }
            }
            (Value::Complex(z), "div") => {
                if let Some(Value::Complex(w)) = args.first() {
                    Ok(Value::Complex(z.div_fixed(*w)))
                } else {
                    Err(MirExecError::Runtime("Complex.div requires a Complex argument".to_string()))
                }
            }
            (Value::Complex(z), "neg") => Ok(Value::Complex(z.neg())),
            (Value::Complex(z), "scale") => {
                if let Some(Value::Float(s)) = args.first() {
                    Ok(Value::Complex(z.scale(*s)))
                } else if let Some(Value::Int(s)) = args.first() {
                    Ok(Value::Complex(z.scale(*s as f64)))
                } else {
                    Err(MirExecError::Runtime("Complex.scale requires a numeric argument".to_string()))
                }
            }
            (Value::Complex(z), "is_nan") => Ok(Value::Bool(z.is_nan())),
            (Value::Complex(z), "is_finite") => Ok(Value::Bool(z.is_finite())),

            // F16 methods
            (Value::F16(v), "to_f64") => Ok(Value::Float(v.to_f64())),
            (Value::F16(v), "to_f32") => Ok(Value::Float(v.to_f32() as f64)),

            (Value::Tensor(t), "shape") => {
                let shape_vals: Vec<Value> =
                    t.shape().iter().map(|&d| Value::Int(d as i64)).collect();
                Ok(Value::Array(Rc::new(shape_vals)))
            }
            (Value::Tensor(t), "len") => Ok(Value::Int(t.len() as i64)),
            (Value::Tensor(t), "to_vec") => {
                let data: Vec<Value> = t.to_vec().iter().map(|&v| Value::Float(v)).collect();
                Ok(Value::Array(Rc::new(data)))
            }
            (Value::Tensor(t), "matmul") => {
                if args.len() != 1 {
                    return Err(MirExecError::Runtime(
                        "matmul requires 1 Tensor argument".to_string(),
                    ));
                }
                let other = self.value_to_tensor(&args[0])?;
                Ok(Value::Tensor(t.matmul(&other)?))
            }
            (Value::Tensor(t), "add") => {
                if args.len() != 1 {
                    return Err(MirExecError::Runtime(
                        "add requires 1 Tensor argument".to_string(),
                    ));
                }
                let other = self.value_to_tensor(&args[0])?;
                Ok(Value::Tensor(t.add(&other)?))
            }
            (Value::Tensor(t), "sub") => {
                if args.len() != 1 {
                    return Err(MirExecError::Runtime(
                        "sub requires 1 Tensor argument".to_string(),
                    ));
                }
                let other = self.value_to_tensor(&args[0])?;
                Ok(Value::Tensor(t.sub(&other)?))
            }
            (Value::Tensor(t), "reshape") => {
                if args.len() != 1 {
                    return Err(MirExecError::Runtime(
                        "reshape requires 1 shape argument".to_string(),
                    ));
                }
                let shape = self.value_to_shape(&args[0])?;
                Ok(Value::Tensor(t.reshape(&shape)?))
            }
            (Value::Tensor(t), "get") => {
                if args.len() != 1 {
                    return Err(MirExecError::Runtime(
                        "get requires 1 index argument".to_string(),
                    ));
                }
                let indices = self.value_to_usize_vec(&args[0])?;
                Ok(Value::Float(t.get(&indices)?))
            }

            // -- Transformer kernel methods --
            (Value::Tensor(t), "softmax") => {
                Ok(Value::Tensor(t.softmax()?))
            }
            (Value::Tensor(t), "layer_norm") => {
                if args.len() < 2 || args.len() > 3 {
                    return Err(MirExecError::Runtime(
                        "layer_norm requires 2-3 arguments: gamma, beta, [eps]".to_string(),
                    ));
                }
                let gamma = self.value_to_tensor(&args[0])?;
                let beta = self.value_to_tensor(&args[1])?;
                let eps = if args.len() == 3 {
                    match &args[2] {
                        Value::Float(f) => *f,
                        Value::Int(i) => *i as f64,
                        _ => return Err(MirExecError::Runtime(
                            "layer_norm eps must be a number".to_string(),
                        )),
                    }
                } else {
                    1e-5
                };
                Ok(Value::Tensor(t.layer_norm(&gamma, &beta, eps)?))
            }
            (Value::Tensor(t), "relu") => {
                Ok(Value::Tensor(t.relu()))
            }
            (Value::Tensor(t), "gelu") => {
                Ok(Value::Tensor(t.gelu()))
            }
            (Value::Tensor(t), "conv1d") => {
                if args.len() < 2 || args.len() > 2 {
                    return Err(MirExecError::Runtime(
                        "conv1d requires 2 arguments: filters, bias".to_string(),
                    ));
                }
                let filters = self.value_to_tensor(&args[0])?;
                let bias = self.value_to_tensor(&args[1])?;
                Ok(Value::Tensor(t.conv1d(&filters, &bias)?))
            }
            // Phase 7: 2D Spatial Vision
            (Value::Tensor(t), "conv2d") => {
                if args.len() < 2 || args.len() > 3 {
                    return Err(MirExecError::Runtime(
                        "conv2d requires 2-3 arguments: filters, bias[, stride]".to_string(),
                    ));
                }
                let filters = self.value_to_tensor(&args[0])?;
                let bias    = self.value_to_tensor(&args[1])?;
                let stride  = if args.len() == 3 {
                    self.value_to_usize(&args[2])?
                } else {
                    1
                };
                Ok(Value::Tensor(t.conv2d(&filters, &bias, stride)?))
            }
            (Value::Tensor(t), "maxpool2d") => {
                if args.len() != 2 {
                    return Err(MirExecError::Runtime(
                        "maxpool2d requires 2 arguments: pool_h, pool_w".to_string(),
                    ));
                }
                let ph = self.value_to_usize(&args[0])?;
                let pw = self.value_to_usize(&args[1])?;
                Ok(Value::Tensor(t.maxpool2d(ph, pw)?))
            }
            (Value::Tensor(t), "bmm") => {
                if args.len() != 1 {
                    return Err(MirExecError::Runtime(
                        "bmm requires 1 Tensor argument".to_string(),
                    ));
                }
                let other = self.value_to_tensor(&args[0])?;
                Ok(Value::Tensor(t.bmm(&other)?))
            }
            (Value::Tensor(t), "linear") => {
                if args.len() != 2 {
                    return Err(MirExecError::Runtime(
                        "linear requires 2 arguments: weight, bias".to_string(),
                    ));
                }
                let weight = self.value_to_tensor(&args[0])?;
                let bias = self.value_to_tensor(&args[1])?;
                Ok(Value::Tensor(t.linear(&weight, &bias)?))
            }
            (Value::Tensor(t), "transpose_last_two") => {
                Ok(Value::Tensor(t.transpose_last_two()?))
            }

            // -- Phase 3: Zero-Copy / Multi-Head methods --
            (Value::Tensor(t), "split_heads") => {
                if args.len() != 1 {
                    return Err(MirExecError::Runtime(
                        "split_heads requires 1 argument: num_heads".to_string(),
                    ));
                }
                let num_heads = self.value_to_usize(&args[0])?;
                Ok(Value::Tensor(t.split_heads(num_heads)?))
            }
            (Value::Tensor(t), "merge_heads") => {
                Ok(Value::Tensor(t.merge_heads()?))
            }
            (Value::Tensor(t), "view_reshape") => {
                if args.len() != 1 {
                    return Err(MirExecError::Runtime(
                        "view_reshape requires 1 argument: new_shape".to_string(),
                    ));
                }
                let new_shape = self.value_to_shape(&args[0])?;
                Ok(Value::Tensor(t.view_reshape(&new_shape)?))
            }
            (Value::Tensor(_t), "from_bytes") => {
                return Err(MirExecError::Runtime(
                    "use Tensor.from_bytes(bytes, shape, [dtype]) as a constructor".to_string(),
                ));
            }

            // -- KV-Cache Scratchpad methods --
            (Value::Scratchpad(s), "len") => {
                Ok(Value::Int(s.borrow().len() as i64))
            }
            (Value::Scratchpad(s), "capacity") => {
                Ok(Value::Int(s.borrow().capacity() as i64))
            }
            (Value::Scratchpad(s), "dim") => {
                Ok(Value::Int(s.borrow().dim() as i64))
            }
            (Value::Scratchpad(s), "is_empty") => {
                Ok(Value::Bool(s.borrow().is_empty()))
            }
            (Value::Scratchpad(s), "append") => {
                if args.len() != 1 {
                    return Err(MirExecError::Runtime(
                        "Scratchpad.append requires 1 argument: a Tensor or Array".to_string(),
                    ));
                }
                match &args[0] {
                    Value::Tensor(t) => {
                        if t.ndim() == 1 {
                            s.borrow_mut().append(&t.to_vec())
                                .map_err(|e| MirExecError::Runtime(format!("{e}")))?;
                        } else {
                            s.borrow_mut().append_tensor(t)
                                .map_err(|e| MirExecError::Runtime(format!("{e}")))?;
                        }
                    }
                    Value::Array(arr) => {
                        let data: Vec<f64> = arr.iter().map(|v| match v {
                            Value::Float(f) => Ok(*f),
                            Value::Int(i) => Ok(*i as f64),
                            _ => Err(MirExecError::Runtime("append: array must contain numbers".into())),
                        }).collect::<Result<Vec<_>, _>>()?;
                        s.borrow_mut().append(&data)
                            .map_err(|e| MirExecError::Runtime(format!("{e}")))?;
                    }
                    _ => return Err(MirExecError::Runtime(
                        "Scratchpad.append requires a Tensor or Array".to_string(),
                    )),
                }
                Ok(Value::Void)
            }
            (Value::Scratchpad(s), "as_tensor") => {
                Ok(Value::Tensor(s.borrow().as_tensor()))
            }
            (Value::Scratchpad(s), "clear") => {
                s.borrow_mut().clear();
                Ok(Value::Void)
            }

            // PagedKvCache methods (Phase 4: block-paged KV-cache, zero-alloc)
            (Value::PagedKvCache(c), "len") => {
                Ok(Value::Int(c.borrow().len() as i64))
            }
            (Value::PagedKvCache(c), "is_empty") => {
                Ok(Value::Bool(c.borrow().is_empty()))
            }
            (Value::PagedKvCache(c), "max_tokens") => {
                Ok(Value::Int(c.borrow().max_tokens() as i64))
            }
            (Value::PagedKvCache(c), "dim") => {
                Ok(Value::Int(c.borrow().dim() as i64))
            }
            (Value::PagedKvCache(c), "num_blocks") => {
                Ok(Value::Int(c.borrow().num_blocks() as i64))
            }
            (Value::PagedKvCache(c), "blocks_in_use") => {
                Ok(Value::Int(c.borrow().blocks_in_use() as i64))
            }
            (Value::PagedKvCache(c), "append") => {
                if args.len() != 1 {
                    return Err(MirExecError::Runtime(
                        "PagedKvCache.append requires 1 argument".to_string(),
                    ));
                }
                match &args[0] {
                    Value::Tensor(t) => {
                        c.borrow_mut().append_tensor(t)
                            .map_err(|e| MirExecError::Runtime(format!("{e}")))?;
                    }
                    Value::Array(arr) => {
                        let data: Vec<f64> = arr.iter().map(|v| match v {
                            Value::Float(f) => Ok(*f),
                            Value::Int(i) => Ok(*i as f64),
                            _ => Err(MirExecError::Runtime("append: array must contain numbers".into())),
                        }).collect::<Result<Vec<_>, _>>()?;
                        c.borrow_mut().append(&data)
                            .map_err(|e| MirExecError::Runtime(format!("{e}")))?;
                    }
                    _ => return Err(MirExecError::Runtime(
                        "PagedKvCache.append requires a Tensor or Array".to_string(),
                    )),
                }
                Ok(Value::Void)
            }
            (Value::PagedKvCache(c), "as_tensor") => {
                Ok(Value::Tensor(c.borrow().as_tensor()))
            }
            (Value::PagedKvCache(c), "clear") => {
                c.borrow_mut().clear();
                Ok(Value::Void)
            }
            (Value::PagedKvCache(c), "get_token") => {
                if args.len() != 1 {
                    return Err(MirExecError::Runtime(
                        "PagedKvCache.get_token requires 1 argument: index".to_string(),
                    ));
                }
                let idx = self.value_to_usize(&args[0])?;
                let token = c.borrow().get_token(idx)
                    .map_err(|e| MirExecError::Runtime(format!("{e}")))?;
                let dim = c.borrow().dim();
                Ok(Value::Tensor(Tensor::from_vec(token, &[dim])
                    .map_err(|e| MirExecError::Runtime(format!("{e}")))?))
            }

            // AlignedByteSlice methods (Phase 4: alignment-aware bytes)
            (Value::AlignedBytes(a), "as_tensor") => {
                if args.is_empty() || args.len() > 2 {
                    return Err(MirExecError::Runtime(
                        "AlignedByteSlice.as_tensor requires 1-2 arguments: shape, [dtype]".to_string(),
                    ));
                }
                let shape = self.value_to_shape(&args[0])?;
                let dtype_str = if args.len() == 2 {
                    match &args[1] {
                        Value::String(s) => (**s).clone(),
                        _ => return Err(MirExecError::Runtime("dtype must be a string".into())),
                    }
                } else {
                    "f64".to_string()
                };
                Ok(Value::Tensor(a.as_tensor(&shape, &dtype_str)
                    .map_err(|e| MirExecError::Runtime(format!("{e}")))?))
            }
            (Value::AlignedBytes(a), "was_realigned") => {
                Ok(Value::Bool(a.was_realigned()))
            }
            (Value::AlignedBytes(a), "len") => {
                Ok(Value::Int(a.len() as i64))
            }
            (Value::AlignedBytes(a), "is_empty") => {
                Ok(Value::Bool(a.is_empty()))
            }

            (Value::Array(arr), "len") => Ok(Value::Int(arr.len() as i64)),
            (Value::String(s), "len") => Ok(Value::Int(s.len() as i64)),
            (Value::String(s), "as_bytes") => {
                Ok(Value::ByteSlice(Rc::new(s.as_bytes().to_vec())))
            }

            // ByteSlice methods (NoGC-safe)
            (Value::ByteSlice(b), "len") => Ok(Value::Int(b.len() as i64)),
            (Value::ByteSlice(b), "is_empty") => Ok(Value::Bool(b.is_empty())),
            (Value::ByteSlice(b), "get") => {
                if args.len() != 1 {
                    return Err(MirExecError::Runtime("get requires 1 index argument".into()));
                }
                let idx = self.value_to_usize(&args[0])?;
                if idx >= b.len() {
                    return Err(MirExecError::Runtime(format!(
                        "index {} out of bounds for ByteSlice of length {}", idx, b.len()
                    )));
                }
                Ok(Value::U8(b[idx]))
            }
            (Value::ByteSlice(b), "slice") => {
                if args.len() != 2 {
                    return Err(MirExecError::Runtime("slice requires 2 arguments: start, end".into()));
                }
                let start = self.value_to_usize(&args[0])?;
                let end = self.value_to_usize(&args[1])?;
                if start > end || end > b.len() {
                    return Err(MirExecError::Runtime(format!(
                        "slice bounds [{}, {}) out of range for ByteSlice of length {}", start, end, b.len()
                    )));
                }
                Ok(Value::ByteSlice(Rc::new(b[start..end].to_vec())))
            }
            (Value::ByteSlice(b), "find_byte") => {
                if args.len() != 1 {
                    return Err(MirExecError::Runtime("find_byte requires 1 u8 argument".into()));
                }
                let needle = match &args[0] {
                    Value::U8(v) => *v,
                    Value::Int(v) => *v as u8,
                    _ => return Err(MirExecError::Runtime("find_byte requires a byte argument".into())),
                };
                match b.iter().position(|&x| x == needle) {
                    Some(pos) => Ok(Value::Int(pos as i64)),
                    None => Ok(Value::Int(-1)),
                }
            }
            (Value::ByteSlice(b), "split_byte") => {
                if args.len() != 1 {
                    return Err(MirExecError::Runtime("split_byte requires 1 u8 argument".into()));
                }
                let delim = match &args[0] {
                    Value::U8(v) => *v,
                    Value::Int(v) => *v as u8,
                    _ => return Err(MirExecError::Runtime("split_byte requires a byte argument".into())),
                };
                let parts: Vec<Value> = b.split(|&x| x == delim)
                    .map(|part| Value::ByteSlice(Rc::new(part.to_vec())))
                    .collect();
                Ok(Value::Array(Rc::new(parts)))
            }
            (Value::ByteSlice(b), "trim_ascii") => {
                let trimmed: &[u8] = b.as_slice();
                let start = trimmed.iter().position(|c| !c.is_ascii_whitespace()).unwrap_or(trimmed.len());
                let end = trimmed.iter().rposition(|c| !c.is_ascii_whitespace()).map(|p| p + 1).unwrap_or(start);
                Ok(Value::ByteSlice(Rc::new(trimmed[start..end].to_vec())))
            }
            (Value::ByteSlice(b), "strip_prefix") => {
                if args.len() != 1 {
                    return Err(MirExecError::Runtime("strip_prefix requires 1 ByteSlice argument".into()));
                }
                let prefix = match &args[0] {
                    Value::ByteSlice(p) => p.clone(),
                    _ => return Err(MirExecError::Runtime("strip_prefix requires a ByteSlice argument".into())),
                };
                if b.starts_with(&prefix) {
                    Ok(Value::Enum {
                        enum_name: "Result".into(),
                        variant: "Ok".into(),
                        fields: vec![Value::ByteSlice(Rc::new(b[prefix.len()..].to_vec()))],
                    })
                } else {
                    Ok(Value::Enum {
                        enum_name: "Result".into(),
                        variant: "Err".into(),
                        fields: vec![Value::String(Rc::new("prefix not found".into()))],
                    })
                }
            }
            (Value::ByteSlice(b), "strip_suffix") => {
                if args.len() != 1 {
                    return Err(MirExecError::Runtime("strip_suffix requires 1 ByteSlice argument".into()));
                }
                let suffix = match &args[0] {
                    Value::ByteSlice(s) => s.clone(),
                    _ => return Err(MirExecError::Runtime("strip_suffix requires a ByteSlice argument".into())),
                };
                if b.ends_with(&suffix) {
                    Ok(Value::Enum {
                        enum_name: "Result".into(),
                        variant: "Ok".into(),
                        fields: vec![Value::ByteSlice(Rc::new(b[..b.len() - suffix.len()].to_vec()))],
                    })
                } else {
                    Ok(Value::Enum {
                        enum_name: "Result".into(),
                        variant: "Err".into(),
                        fields: vec![Value::String(Rc::new("suffix not found".into()))],
                    })
                }
            }
            (Value::ByteSlice(b), "starts_with") => {
                if args.len() != 1 {
                    return Err(MirExecError::Runtime("starts_with requires 1 ByteSlice argument".into()));
                }
                let prefix = match &args[0] {
                    Value::ByteSlice(p) => p.clone(),
                    _ => return Err(MirExecError::Runtime("starts_with requires a ByteSlice argument".into())),
                };
                Ok(Value::Bool(b.starts_with(&prefix)))
            }
            (Value::ByteSlice(b), "ends_with") => {
                if args.len() != 1 {
                    return Err(MirExecError::Runtime("ends_with requires 1 ByteSlice argument".into()));
                }
                let suffix = match &args[0] {
                    Value::ByteSlice(s) => s.clone(),
                    _ => return Err(MirExecError::Runtime("ends_with requires a ByteSlice argument".into())),
                };
                Ok(Value::Bool(b.ends_with(&suffix)))
            }
            (Value::ByteSlice(b), "count_byte") => {
                if args.len() != 1 {
                    return Err(MirExecError::Runtime("count_byte requires 1 u8 argument".into()));
                }
                let needle = match &args[0] {
                    Value::U8(v) => *v,
                    Value::Int(v) => *v as u8,
                    _ => return Err(MirExecError::Runtime("count_byte requires a byte argument".into())),
                };
                let count = b.iter().filter(|&&x| x == needle).count();
                Ok(Value::Int(count as i64))
            }
            (Value::ByteSlice(b), "as_str_utf8") => {
                match std::str::from_utf8(b) {
                    Ok(_) => Ok(Value::Enum {
                        enum_name: "Result".into(),
                        variant: "Ok".into(),
                        fields: vec![Value::StrView(b.clone())],
                    }),
                    Err(e) => Ok(Value::Enum {
                        enum_name: "Result".into(),
                        variant: "Err".into(),
                        fields: vec![Value::Struct {
                            name: "Utf8Error".into(),
                            fields: {
                                let mut m = HashMap::new();
                                m.insert("valid_up_to".into(), Value::Int(e.valid_up_to() as i64));
                                m.insert("error_len".into(), Value::Int(e.error_len().unwrap_or(0) as i64));
                                m
                            },
                        }],
                    }),
                }
            }
            (Value::ByteSlice(b), "eq") => {
                if args.len() != 1 {
                    return Err(MirExecError::Runtime("eq requires 1 ByteSlice argument".into()));
                }
                match &args[0] {
                    Value::ByteSlice(other) => Ok(Value::Bool(**b == **other)),
                    _ => Ok(Value::Bool(false)),
                }
            }
            (Value::ByteSlice(b), "as_tensor") => {
                if args.is_empty() || args.len() > 2 {
                    return Err(MirExecError::Runtime(
                        "as_tensor requires 1-2 arguments: shape, [dtype='f64']".to_string(),
                    ));
                }
                let shape = self.value_to_shape(&args[0])?;
                let dtype = if args.len() == 2 {
                    match &args[1] {
                        Value::String(s) => s.as_str().to_string(),
                        _ => return Err(MirExecError::Runtime(
                            "as_tensor: dtype must be a string".to_string(),
                        )),
                    }
                } else {
                    "f64".to_string()
                };
                Ok(Value::Tensor(Tensor::from_bytes(b, &shape, &dtype)
                    .map_err(|e| MirExecError::Runtime(format!("{e}")))?))
            }

            // StrView methods (NoGC-safe)
            (Value::StrView(b), "len_bytes") => Ok(Value::Int(b.len() as i64)),
            (Value::StrView(b), "as_bytes") => Ok(Value::ByteSlice(b.clone())),
            (Value::StrView(b), "to_string") => {
                let s = std::str::from_utf8(b).unwrap_or("");
                Ok(Value::String(Rc::new(s.to_string())))
            }
            (Value::StrView(b), "eq") => {
                if args.len() != 1 {
                    return Err(MirExecError::Runtime("eq requires 1 StrView argument".into()));
                }
                match &args[0] {
                    Value::StrView(other) => Ok(Value::Bool(**b == **other)),
                    _ => Ok(Value::Bool(false)),
                }
            }

            // Phase 8: DataFrame methods
            (Value::Struct { name: sname, fields }, method_name)
                if sname == "DataFrame" =>
            {
                match method_name {
                    "nrows" => {
                        Ok(fields.get("__nrows").cloned().unwrap_or(Value::Int(0)))
                    }
                    "ncols" => {
                        let n = fields.keys().filter(|k| !k.starts_with("__")).count();
                        Ok(Value::Int(n as i64))
                    }
                    "column_names" => {
                        Ok(fields.get("__columns").cloned().unwrap_or(Value::Array(Rc::new(vec![]))))
                    }
                    "column" => {
                        if args.len() != 1 {
                            return Err(MirExecError::Runtime(
                                "DataFrame.column requires 1 argument: column_name".to_string(),
                            ));
                        }
                        let col_name = match &args[0] {
                            Value::String(s) => s.as_ref().clone(),
                            _ => return Err(MirExecError::Runtime(
                                "DataFrame.column: column name must be a String".to_string(),
                            )),
                        };
                        fields.get(&col_name).cloned().ok_or_else(|| {
                            MirExecError::Runtime(format!("column '{}' not found", col_name))
                        })
                    }
                    "to_tensor" => {
                        // df.to_tensor(["x", "y"]) → Tensor[nrows, ncols]
                        if args.len() != 1 {
                            return Err(MirExecError::Runtime(
                                "DataFrame.to_tensor requires 1 argument: column_names array".to_string(),
                            ));
                        }
                        let col_names: Vec<String> = match &args[0] {
                            Value::Array(arr) => {
                                arr.iter()
                                    .map(|v| match v {
                                        Value::String(s) => Ok(s.as_ref().clone()),
                                        _ => Err(MirExecError::Runtime(
                                            "to_tensor: column names must be strings".to_string(),
                                        )),
                                    })
                                    .collect::<Result<Vec<_>, _>>()?
                            }
                            _ => return Err(MirExecError::Runtime(
                                "to_tensor: argument must be an array of column names".to_string(),
                            )),
                        };

                        // Rebuild a DataFrame from the Struct fields, then use to_tensor.
                        let _nrows = match fields.get("__nrows") {
                            Some(Value::Int(n)) => *n as usize,
                            _ => 0,
                        };
                        let mut df_cols: Vec<(String, Column)> = Vec::new();
                        for col_name in &col_names {
                            let arr_val = fields.get(col_name).ok_or_else(|| {
                                MirExecError::Runtime(format!("column '{}' not found", col_name))
                            })?;
                            let col = match arr_val {
                                Value::Array(arr) => {
                                    let floats: Vec<f64> = arr.iter()
                                        .map(|v| match v {
                                            Value::Float(x) => Ok(*x),
                                            Value::Int(x)   => Ok(*x as f64),
                                            _ => Err(MirExecError::Runtime(format!(
                                                "column '{}' contains non-numeric values", col_name
                                            ))),
                                        })
                                        .collect::<Result<Vec<_>, _>>()?;
                                    Column::Float(floats)
                                }
                                _ => return Err(MirExecError::Runtime(format!(
                                    "column '{}' has unexpected type", col_name
                                ))),
                            };
                            df_cols.push((col_name.clone(), col));
                        }
                        let df = DataFrame::from_columns(df_cols)
                            .map_err(|e| MirExecError::Runtime(format!("DataFrame: {}", e)))?;
                        let col_refs: Vec<&str> = col_names.iter().map(|s| s.as_str()).collect();
                        let t = df.to_tensor(&col_refs)
                            .map_err(|e| MirExecError::Runtime(format!("to_tensor: {}", e)))?;
                        Ok(Value::Tensor(t))
                    }
                    other => {
                        return Err(MirExecError::Runtime(format!(
                            "no method `{}` on DataFrame", other
                        )));
                    }
                }
            }

            // P2-6: Result<T,E> and Option<T> builtin methods.
            (Value::Enum { enum_name, variant, fields }, "unwrap") => {
                match (enum_name.as_str(), variant.as_str()) {
                    ("Option", "Some") | ("Result", "Ok") => {
                        Ok(fields.first().cloned().unwrap_or(Value::Void))
                    }
                    ("Option", "None") => Err(MirExecError::Runtime(
                        "called `unwrap` on a `None` value".to_string(),
                    )),
                    ("Result", "Err") => Err(MirExecError::Runtime(format!(
                        "called `unwrap` on an `Err` value: {}",
                        fields.first().map(|v| format!("{v}")).unwrap_or_default()
                    ))),
                    _ => Err(MirExecError::Runtime(format!(
                        "unwrap not supported on {enum_name}::{variant}"
                    ))),
                }
            }
            (Value::Enum { enum_name, variant, fields }, "unwrap_or") => {
                let default = args.into_iter().next().unwrap_or(Value::Void);
                match (enum_name.as_str(), variant.as_str()) {
                    ("Option", "Some") | ("Result", "Ok") => {
                        Ok(fields.first().cloned().unwrap_or(Value::Void))
                    }
                    ("Option", "None") | ("Result", "Err") => Ok(default),
                    _ => Err(MirExecError::Runtime(format!(
                        "unwrap_or not supported on {enum_name}::{variant}"
                    ))),
                }
            }
            (Value::Enum { enum_name, variant, fields: _ }, "is_some")
                if enum_name == "Option" =>
            {
                Ok(Value::Bool(variant == "Some"))
            }
            (Value::Enum { enum_name, variant, fields: _ }, "is_none")
                if enum_name == "Option" =>
            {
                Ok(Value::Bool(variant == "None"))
            }
            (Value::Enum { enum_name, variant, fields: _ }, "is_ok")
                if enum_name == "Result" =>
            {
                Ok(Value::Bool(variant == "Ok"))
            }
            (Value::Enum { enum_name, variant, fields: _ }, "is_err")
                if enum_name == "Result" =>
            {
                Ok(Value::Bool(variant == "Err"))
            }
            (Value::Enum { enum_name, variant, fields }, "map") => {
                // map(|x| expr) — apply closure to Ok/Some value
                let f = args.into_iter().next().unwrap_or(Value::Void);
                match (enum_name.as_str(), variant.as_str()) {
                    ("Option", "None") => Ok(Value::Enum {
                        enum_name: "Option".into(),
                        variant: "None".into(),
                        fields: vec![],
                    }),
                    ("Result", "Err") => Ok(Value::Enum {
                        enum_name: "Result".into(),
                        variant: "Err".into(),
                        fields: fields.clone(),
                    }),
                    ("Option", "Some") | ("Result", "Ok") => {
                        let inner = fields.first().cloned().unwrap_or(Value::Void);
                        let mapped = match f {
                            Value::Fn(fv) => self.call_function(&fv.name, &[inner])?,
                            Value::Closure { fn_name, env, .. } => {
                                let mut full = env;
                                full.push(inner);
                                self.call_function(&fn_name, &full)?
                            }
                            _ => return Err(MirExecError::Runtime(
                                "map: argument must be a function".to_string()
                            )),
                        };
                        let out_variant = if enum_name == "Option" { "Some" } else { "Ok" };
                        Ok(Value::Enum {
                            enum_name: enum_name.clone(),
                            variant: out_variant.into(),
                            fields: vec![mapped],
                        })
                    }
                    _ => Err(MirExecError::Runtime(format!(
                        "map not supported on {enum_name}::{variant}"
                    ))),
                }
            }
            (Value::Enum { enum_name, variant, fields }, "and_then") => {
                // and_then(|x| expr) — flatMap
                let f = args.into_iter().next().unwrap_or(Value::Void);
                match (enum_name.as_str(), variant.as_str()) {
                    ("Option", "None") => Ok(Value::Enum {
                        enum_name: "Option".into(),
                        variant: "None".into(),
                        fields: vec![],
                    }),
                    ("Result", "Err") => Ok(Value::Enum {
                        enum_name: "Result".into(),
                        variant: "Err".into(),
                        fields: fields.clone(),
                    }),
                    ("Option", "Some") | ("Result", "Ok") => {
                        let inner = fields.first().cloned().unwrap_or(Value::Void);
                        match f {
                            Value::Fn(fv) => self.call_function(&fv.name, &[inner]),
                            Value::Closure { fn_name, env, .. } => {
                                let mut full = env;
                                full.push(inner);
                                self.call_function(&fn_name, &full)
                            }
                            _ => Err(MirExecError::Runtime(
                                "and_then: argument must be a function".to_string()
                            )),
                        }
                    }
                    _ => Err(MirExecError::Runtime(format!(
                        "and_then not supported on {enum_name}::{variant}"
                    ))),
                }
            }

            // String methods (len/is_empty handled above with NoGC methods)
            (Value::String(s), "to_upper") | (Value::String(s), "to_uppercase") => {
                Ok(Value::String(Rc::new(s.to_uppercase())))
            }
            (Value::String(s), "to_lower") | (Value::String(s), "to_lowercase") => {
                Ok(Value::String(Rc::new(s.to_lowercase())))
            }
            (Value::String(s), "trim") => {
                Ok(Value::String(Rc::new(s.trim().to_string())))
            }
            (Value::String(s), "contains") => {
                let needle = match args.first() {
                    Some(Value::String(n)) => n.as_str().to_string(),
                    Some(Value::Int(n)) => ((*n as u8) as char).to_string(),
                    _ => return Err(MirExecError::Runtime("contains requires a string argument".to_string())),
                };
                Ok(Value::Bool(s.contains(needle.as_str())))
            }
            (Value::String(s), "starts_with") => {
                let prefix = match args.first() {
                    Some(Value::String(n)) => n.as_str().to_string(),
                    _ => return Err(MirExecError::Runtime("starts_with requires a string argument".to_string())),
                };
                Ok(Value::Bool(s.starts_with(prefix.as_str())))
            }
            (Value::String(s), "ends_with") => {
                let suffix = match args.first() {
                    Some(Value::String(n)) => n.as_str().to_string(),
                    _ => return Err(MirExecError::Runtime("ends_with requires a string argument".to_string())),
                };
                Ok(Value::Bool(s.ends_with(suffix.as_str())))
            }
            (Value::String(s), "split") => {
                let sep = match args.first() {
                    Some(Value::String(n)) => n.as_str().to_string(),
                    _ => return Err(MirExecError::Runtime("split requires a string argument".to_string())),
                };
                let parts: Vec<Value> = s.split(sep.as_str())
                    .map(|p| Value::String(Rc::new(p.to_string())))
                    .collect();
                Ok(Value::Array(Rc::new(parts)))
            }

            // Array methods (len handled above with NoGC methods)
            (Value::Array(arr), "is_empty") => Ok(Value::Bool(arr.is_empty())),
            (Value::Array(arr), "first") => {
                Ok(arr.first().map(|v| Value::Enum {
                    enum_name: "Option".into(),
                    variant: "Some".into(),
                    fields: vec![v.clone()],
                }).unwrap_or(Value::Enum {
                    enum_name: "Option".into(),
                    variant: "None".into(),
                    fields: vec![],
                }))
            }
            (Value::Array(arr), "last") => {
                Ok(arr.last().map(|v| Value::Enum {
                    enum_name: "Option".into(),
                    variant: "Some".into(),
                    fields: vec![v.clone()],
                }).unwrap_or(Value::Enum {
                    enum_name: "Option".into(),
                    variant: "None".into(),
                    fields: vec![],
                }))
            }

            (Value::Struct { name: sname, .. }, _) => {
                let qualified = format!("{sname}.{method}");
                if self.functions.contains_key(&qualified) {
                    let mut full_args = vec![receiver.clone()];
                    full_args.extend(args);
                    self.call_function(&qualified, &full_args)
                } else {
                    Err(MirExecError::Runtime(format!(
                        "no method `{method}` on struct `{sname}`"
                    )))
                }
            }
            _ => Err(MirExecError::Runtime(format!(
                "no method `{method}` on type {}",
                receiver.type_name()
            ))),
        }
    }

    fn call_function(&mut self, name: &str, args: &[Value]) -> MirExecResult {
        // P1-2: Tail-call trampoline. The first call to call_function sets up
        // a scope and executes the body. If the body ends with a self-tail-call,
        // the current scope is reused (params rebound) and execution restarts
        // instead of growing the Rust call stack.
        let mut current_name = name.to_string();
        let mut current_args: Vec<Value> = args.to_vec();

        loop {
            let func = self.functions.get(&current_name).cloned().ok_or_else(|| {
                MirExecError::Runtime(format!("undefined function `{}`", current_name))
            })?;

            if func.params.len() != current_args.len() {
                return Err(MirExecError::Runtime(format!(
                    "function `{}` expects {} arguments, got {}",
                    current_name,
                    func.params.len(),
                    current_args.len()
                )));
            }

            self.push_scope();
            self.arena_stack.push(cjc_runtime::ArenaStore::new());
            for (param, val) in func.params.iter().zip(current_args.iter()) {
                self.define(&param.name, val.clone());
            }

            // Track which function we're in for TCO detection.
            let prev_fn = self.current_fn.take();
            self.current_fn = Some(current_name.clone());

            let result = match self.exec_body(&func.body) {
                Ok(val) => val,
                Err(MirExecError::Return(val)) => val,
                Err(MirExecError::TailCall { name: tco_name, args: tco_args }) => {
                    // Tail call detected — reset arena for reuse, pop scope, loop.
                    if let Some(arena) = self.arena_stack.last_mut() {
                        arena.reset();
                    }
                    self.pop_scope();
                    self.current_fn = prev_fn;
                    current_name = tco_name;
                    current_args = tco_args;
                    continue;
                }
                Err(e) => {
                    self.arena_stack.pop();
                    self.pop_scope();
                    self.current_fn = prev_fn;
                    return Err(e);
                }
            };

            self.arena_stack.pop();
            self.pop_scope();
            self.current_fn = prev_fn;
            return Ok(result);
        }
    }

    // -- Field access -------------------------------------------------------

    fn eval_field(&mut self, object: &MirExpr, field: &str) -> MirExecResult {
        let obj = self.eval_expr(object)?;
        match &obj {
            Value::Struct { fields, name, .. } => {
                fields.get(field).cloned().ok_or_else(|| {
                    MirExecError::Runtime(format!("no field `{field}` on struct `{name}`"))
                })
            }
            Value::Tensor(t) => match field {
                "shape" => {
                    let shape_vals: Vec<Value> =
                        t.shape().iter().map(|&d| Value::Int(d as i64)).collect();
                    Ok(Value::Array(Rc::new(shape_vals)))
                }
                _ => Err(MirExecError::Runtime(format!(
                    "no field `{field}` on Tensor"
                ))),
            },
            Value::Array(arr) => match field {
                "len" => Ok(Value::Int(arr.len() as i64)),
                _ => Err(MirExecError::Runtime(format!(
                    "no field `{field}` on Array"
                ))),
            },
            _ => Err(MirExecError::Runtime(format!(
                "cannot access field `{field}` on {}",
                obj.type_name()
            ))),
        }
    }

    // -- Indexing ------------------------------------------------------------

    fn eval_index(&mut self, object: &MirExpr, index: &MirExpr) -> MirExecResult {
        let obj = self.eval_expr(object)?;
        let idx = self.eval_expr(index)?;

        match (&obj, &idx) {
            (Value::Array(arr), Value::Int(i)) => {
                let i = *i as usize;
                arr.get(i).cloned().ok_or_else(|| {
                    MirExecError::Runtime(format!(
                        "index {i} out of bounds for array of length {}",
                        arr.len()
                    ))
                })
            }
            (Value::Tensor(t), Value::Int(i)) => {
                let i = *i as usize;
                if t.ndim() == 1 {
                    Ok(Value::Float(t.get(&[i])?))
                } else {
                    Err(MirExecError::Runtime(
                        "use multi-index for tensors with ndim > 1".to_string(),
                    ))
                }
            }
            _ => Err(MirExecError::Runtime(format!(
                "cannot index {} with {}",
                obj.type_name(),
                idx.type_name()
            ))),
        }
    }

    fn eval_multi_index(
        &mut self,
        object: &MirExpr,
        indices: &[MirExpr],
    ) -> MirExecResult {
        let obj = self.eval_expr(object)?;
        let mut idx_vals = Vec::with_capacity(indices.len());
        for idx_expr in indices {
            let v = self.eval_expr(idx_expr)?;
            match v {
                Value::Int(i) => idx_vals.push(i as usize),
                _ => {
                    return Err(MirExecError::Runtime(format!(
                        "index must be Int, got {}",
                        v.type_name()
                    )));
                }
            }
        }
        match &obj {
            Value::Tensor(t) => Ok(Value::Float(t.get(&idx_vals)?)),
            _ => Err(MirExecError::Runtime(format!(
                "multi-index not supported on {}",
                obj.type_name()
            ))),
        }
    }

    // -- Assignment ---------------------------------------------------------

    fn exec_assign(&mut self, target: &MirExpr, val: Value) -> Result<(), MirExecError> {
        match &target.kind {
            MirExprKind::Var(name) => self.assign(name, val),
            MirExprKind::Field { object, name } => {
                if let MirExprKind::Var(obj_name) = &object.kind {
                    let obj_name = obj_name.clone();
                    let field_name = name.clone();
                    let mut obj_val = self
                        .lookup(&obj_name)
                        .cloned()
                        .ok_or_else(|| {
                            MirExecError::Runtime(format!("undefined variable `{obj_name}`"))
                        })?;
                    match &mut obj_val {
                        Value::Struct { fields, .. } => {
                            fields.insert(field_name, val);
                        }
                        _ => {
                            return Err(MirExecError::Runtime(format!(
                                "cannot assign field on {}",
                                obj_val.type_name()
                            )));
                        }
                    }
                    self.assign(&obj_name, obj_val)
                } else {
                    Err(MirExecError::Runtime(
                        "complex field assignment not supported".to_string(),
                    ))
                }
            }
            MirExprKind::Index { object, index } => {
                if let MirExprKind::Var(obj_name) = &object.kind {
                    let obj_name = obj_name.clone();
                    let idx = self.eval_expr(index)?;
                    let idx = match idx {
                        Value::Int(i) => i as usize,
                        _ => {
                            return Err(MirExecError::Runtime(
                                "index must be Int".to_string(),
                            ))
                        }
                    };
                    let mut obj_val = self
                        .lookup(&obj_name)
                        .cloned()
                        .ok_or_else(|| {
                            MirExecError::Runtime(format!("undefined variable `{obj_name}`"))
                        })?;
                    match &mut obj_val {
                        Value::Array(arr) => {
                            if idx >= arr.len() {
                                return Err(MirExecError::Runtime(format!(
                                    "index {idx} out of bounds for array of length {}",
                                    arr.len()
                                )));
                            }
                            Rc::make_mut(arr)[idx] = val;
                        }
                        _ => {
                            return Err(MirExecError::Runtime(format!(
                                "cannot index-assign on {}",
                                obj_val.type_name()
                            )));
                        }
                    }
                    self.assign(&obj_name, obj_val)
                } else {
                    Err(MirExecError::Runtime(
                        "complex index assignment not supported".to_string(),
                    ))
                }
            }
            _ => Err(MirExecError::Runtime(
                "invalid assignment target".to_string(),
            )),
        }
    }

    // -- Value conversion helpers -------------------------------------------

    fn value_to_shape(&self, val: &Value) -> Result<Vec<usize>, MirExecError> {
        match val {
            Value::Array(arr) => {
                let mut shape = Vec::with_capacity(arr.len());
                for v in arr.iter() {
                    shape.push(self.value_to_usize(v)?);
                }
                Ok(shape)
            }
            _ => Err(MirExecError::Runtime(format!(
                "expected Array for shape, got {}",
                val.type_name()
            ))),
        }
    }

    fn value_to_usize(&self, val: &Value) -> Result<usize, MirExecError> {
        match val {
            Value::Int(i) => {
                if *i < 0 {
                    Err(MirExecError::Runtime(format!(
                        "expected non-negative integer, got {i}"
                    )))
                } else {
                    Ok(*i as usize)
                }
            }
            _ => Err(MirExecError::Runtime(format!(
                "expected Int, got {}",
                val.type_name()
            ))),
        }
    }

    fn value_to_usize_vec(&self, val: &Value) -> Result<Vec<usize>, MirExecError> {
        match val {
            Value::Array(arr) => {
                let mut indices = Vec::with_capacity(arr.len());
                for v in arr.iter() {
                    indices.push(self.value_to_usize(v)?);
                }
                Ok(indices)
            }
            _ => Err(MirExecError::Runtime(format!(
                "expected Array for indices, got {}",
                val.type_name()
            ))),
        }
    }

    fn value_to_tensor<'a>(&self, val: &'a Value) -> Result<&'a Tensor, MirExecError> {
        match val {
            Value::Tensor(t) => Ok(t),
            _ => Err(MirExecError::Runtime(format!(
                "expected Tensor, got {}",
                val.type_name()
            ))),
        }
    }

}

impl Default for MirExecutor {
    fn default() -> Self {
        Self::new(0)
    }
}

// ---------------------------------------------------------------------------
// Convenience: full pipeline AST -> HIR -> MIR -> Execute
// ---------------------------------------------------------------------------

/// Run a full AST program through the HIR -> MIR -> MIR-Exec pipeline.
// ---------------------------------------------------------------------------
// Type-checking gate
// ---------------------------------------------------------------------------

/// Run the CJC type checker on `program`.
///
/// Returns `Ok(())` if there are no type errors, or
/// `Err(MirExecError::TypeErrors(...))` with rendered diagnostics on failure.
///
/// This is the **compile-time safety gate** that enforces:
/// - Non-exhaustive match is an error (not a warning).
/// - Undefined variables are an error.
/// - Type mismatches in binary ops / function calls are errors.
///
/// All diagnostics carry `Span` information (file offset, line, column) because
/// the type checker stores them in `DiagnosticBag` with `Diagnostic::error(..., span)`.
pub fn type_check_program(program: &cjc_ast::Program) -> Result<(), MirExecError> {
    let mut checker = cjc_types::TypeChecker::new();
    checker.check_program(program);
    if checker.diagnostics.has_errors() {
        let messages: Vec<String> = checker
            .diagnostics
            .diagnostics
            .iter()
            .filter(|d| d.severity == cjc_diag::Severity::Error)
            .map(|d| {
                // Include span information in the rendered message
                let span = &d.span;
                format!("error[{}] at {}..{}: {}", d.code, span.start, span.end, d.message)
            })
            .collect();
        return Err(MirExecError::TypeErrors(messages));
    }
    Ok(())
}

pub fn run_program(program: &cjc_ast::Program, seed: u64) -> MirExecResult {
    let mut ast_lowering = cjc_hir::AstLowering::new();
    let hir = ast_lowering.lower_program(program);

    let mut hir_to_mir = cjc_mir::HirToMir::new();
    let mut mir = hir_to_mir.lower_program(&hir);
    cjc_mir::escape::annotate_program(&mut mir);

    let mut executor = MirExecutor::new(seed);
    executor.exec(&mir)
}

/// Run a full AST program and return the executor (for inspecting output, etc.)
pub fn run_program_with_executor(
    program: &cjc_ast::Program,
    seed: u64,
) -> Result<(Value, MirExecutor), MirExecError> {
    let mut ast_lowering = cjc_hir::AstLowering::new();
    let hir = ast_lowering.lower_program(program);

    let mut hir_to_mir = cjc_mir::HirToMir::new();
    let mut mir = hir_to_mir.lower_program(&hir);
    cjc_mir::escape::annotate_program(&mut mir);

    let mut executor = MirExecutor::new(seed);
    let result = executor.exec(&mir)?;
    Ok((result, executor))
}

/// Run a full AST program through the optimized MIR pipeline.
pub fn run_program_optimized(program: &cjc_ast::Program, seed: u64) -> MirExecResult {
    let mut ast_lowering = cjc_hir::AstLowering::new();
    let hir = ast_lowering.lower_program(program);

    let mut hir_to_mir = cjc_mir::HirToMir::new();
    let mir = hir_to_mir.lower_program(&hir);

    let mut optimized = cjc_mir::optimize::optimize_program(&mir);
    cjc_mir::escape::annotate_program(&mut optimized);

    let mut executor = MirExecutor::new(seed);
    executor.exec(&optimized)
}

/// Run a full AST program through the optimized MIR pipeline, returning executor.
pub fn run_program_optimized_with_executor(
    program: &cjc_ast::Program,
    seed: u64,
) -> Result<(Value, MirExecutor), MirExecError> {
    let mut ast_lowering = cjc_hir::AstLowering::new();
    let hir = ast_lowering.lower_program(program);

    let mut hir_to_mir = cjc_mir::HirToMir::new();
    let mir = hir_to_mir.lower_program(&hir);

    let mut optimized = cjc_mir::optimize::optimize_program(&mir);
    cjc_mir::escape::annotate_program(&mut optimized);

    let mut executor = MirExecutor::new(seed);
    let result = executor.exec(&optimized)?;
    Ok((result, executor))
}

/// Run a full AST program with the **type-checking compile-time gate** enabled.
///
/// This is the production-safe variant. It runs `type_check_program` first and
/// returns `Err(MirExecError::TypeErrors(...))` if there are any type errors
/// (including non-exhaustive matches, type mismatches, or undefined variables).
///
/// Use this when you want compile-time safety enforcement. The standard
/// `run_program` skips the type checker to maintain backward compatibility while
/// the type checker's builtin coverage is still being expanded.
pub fn run_program_type_checked(program: &cjc_ast::Program, seed: u64) -> MirExecResult {
    // Phase 0: Type-check (compile-time safety gate).
    type_check_program(program)?;

    let mut ast_lowering = cjc_hir::AstLowering::new();
    let hir = ast_lowering.lower_program(program);

    let mut hir_to_mir = cjc_mir::HirToMir::new();
    let mut mir = hir_to_mir.lower_program(&hir);
    cjc_mir::escape::annotate_program(&mut mir);

    let mut executor = MirExecutor::new(seed);
    executor.exec(&mir)
}

/// Run a full AST program through the MIR pipeline with monomorphization + optimization.
pub fn run_program_monomorphized(program: &cjc_ast::Program, seed: u64) -> MirExecResult {
    let mut ast_lowering = cjc_hir::AstLowering::new();
    let hir = ast_lowering.lower_program(program);

    let mut hir_to_mir = cjc_mir::HirToMir::new();
    let mir = hir_to_mir.lower_program(&hir);

    // Monomorphize before optimization
    let (monomorphized, _report) = cjc_mir::monomorph::monomorphize_program(&mir);

    let mut optimized = cjc_mir::optimize::optimize_program(&monomorphized);
    cjc_mir::escape::annotate_program(&mut optimized);

    let mut executor = MirExecutor::new(seed);
    executor.exec(&optimized)
}

/// Run a multi-file CJC program starting from the entry file path.
///
/// This function:
/// 1. Builds the module graph (resolving imports, detecting cycles)
/// 2. Merges all modules into a single MIR program (symbol-prefixed)
/// 3. Runs escape analysis annotation
/// 4. Executes the merged program
///
/// Returns the execution result or an error.
pub fn run_program_with_modules(
    entry_path: &std::path::Path,
    seed: u64,
) -> MirExecResult {
    let graph = cjc_module::build_module_graph(entry_path)
        .map_err(|e| MirExecError::Runtime(format!("module error: {}", e)))?;

    let mut mir = cjc_module::merge_programs(&graph)
        .map_err(|e| MirExecError::Runtime(format!("module merge error: {}", e)))?;

    cjc_mir::escape::annotate_program(&mut mir);

    let mut executor = MirExecutor::new(seed);
    executor.exec(&mir)
}

/// Run a multi-file CJC program and return the executor for inspection.
pub fn run_program_with_modules_executor(
    entry_path: &std::path::Path,
    seed: u64,
) -> Result<(Value, MirExecutor), MirExecError> {
    let graph = cjc_module::build_module_graph(entry_path)
        .map_err(|e| MirExecError::Runtime(format!("module error: {}", e)))?;

    let mut mir = cjc_module::merge_programs(&graph)
        .map_err(|e| MirExecError::Runtime(format!("module merge error: {}", e)))?;

    cjc_mir::escape::annotate_program(&mut mir);

    let mut executor = MirExecutor::new(seed);
    let result = executor.exec(&mir)?;
    Ok((result, executor))
}

/// Run NoGC verification on a program's MIR.
/// Returns Ok(()) if clean, or a descriptive error message.
pub fn verify_nogc(program: &cjc_ast::Program) -> Result<(), String> {
    let mut ast_lowering = cjc_hir::AstLowering::new();
    let hir = ast_lowering.lower_program(program);

    let mut hir_to_mir = cjc_mir::HirToMir::new();
    let mir = hir_to_mir.lower_program(&hir);

    match cjc_mir::nogc_verify::verify_nogc(&mir) {
        Ok(()) => Ok(()),
        Err(errors) => {
            let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
            Err(msgs.join("\n"))
        }
    }
}

/// Lower an AST program to MIR and return it (for inspection).
pub fn lower_to_mir(program: &cjc_ast::Program) -> cjc_mir::MirProgram {
    let mut ast_lowering = cjc_hir::AstLowering::new();
    let hir = ast_lowering.lower_program(program);

    let mut hir_to_mir = cjc_mir::HirToMir::new();
    hir_to_mir.lower_program(&hir)
}

// ---------------------------------------------------------------------------
// Phase 8: DataFrame helper (mirrors cjc-eval's dataframe_to_value)
// ---------------------------------------------------------------------------

/// Convert a `DataFrame` into a `Value::Struct { name: "DataFrame", fields }`.
///
/// Column data is stored as `Value::Array` keyed by column name.
/// A special `"__columns"` field stores ordered column names.
/// A special `"__nrows"` field stores the row count.
fn dataframe_to_value(df: DataFrame) -> Value {
    let col_names: Vec<Value> = df
        .column_names()
        .iter()
        .map(|&n| Value::String(Rc::new(n.to_string())))
        .collect();

    let mut fields = std::collections::HashMap::new();
    fields.insert("__columns".to_string(), Value::Array(Rc::new(col_names)));
    fields.insert("__nrows".to_string(), Value::Int(df.nrows() as i64));

    for (name, col) in &df.columns {
        let arr: Value = match col {
            Column::Float(v) => Value::Array(Rc::new(v.iter().map(|&x| Value::Float(x)).collect())),
            Column::Int(v)   => Value::Array(Rc::new(v.iter().map(|&x| Value::Int(x)).collect())),
            Column::Str(v)   => Value::Array(Rc::new(
                v.iter().map(|s| Value::String(Rc::new(s.clone()))).collect(),
            )),
            Column::Bool(v)  => Value::Array(Rc::new(v.iter().map(|&x| Value::Bool(x)).collect())),
        };
        fields.insert(name.clone(), arr);
    }

    Value::Struct { name: "DataFrame".to_string(), fields }
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
        Expr { kind: ExprKind::IntLit(v), span: span() }
    }

    fn float_expr(v: f64) -> Expr {
        Expr { kind: ExprKind::FloatLit(v), span: span() }
    }

    fn bool_expr(v: bool) -> Expr {
        Expr { kind: ExprKind::BoolLit(v), span: span() }
    }

    fn ident_expr(name: &str) -> Expr {
        Expr { kind: ExprKind::Ident(ident(name)), span: span() }
    }

    fn binary(op: BinOp, left: Expr, right: Expr) -> Expr {
        Expr {
            kind: ExprKind::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            },
            span: span(),
        }
    }

    fn call(callee: Expr, args: Vec<Expr>) -> Expr {
        let call_args: Vec<CallArg> = args
            .into_iter()
            .map(|value| CallArg { name: None, value, span: span() })
            .collect();
        Expr {
            kind: ExprKind::Call {
                callee: Box::new(callee),
                args: call_args,
            },
            span: span(),
        }
    }

    fn field_expr(object: Expr, name: &str) -> Expr {
        Expr {
            kind: ExprKind::Field {
                object: Box::new(object),
                name: ident(name),
            },
            span: span(),
        }
    }

    fn pipe_expr(left: Expr, right: Expr) -> Expr {
        Expr {
            kind: ExprKind::Pipe {
                left: Box::new(left),
                right: Box::new(right),
            },
            span: span(),
        }
    }

    fn array_expr(elems: Vec<Expr>) -> Expr {
        Expr { kind: ExprKind::ArrayLit(elems), span: span() }
    }

    fn assign_expr(target: Expr, value: Expr) -> Expr {
        Expr {
            kind: ExprKind::Assign {
                target: Box::new(target),
                value: Box::new(value),
            },
            span: span(),
        }
    }

    fn let_stmt_ast(name: &str, init: Expr) -> Stmt {
        Stmt {
            kind: StmtKind::Let(LetStmt {
                name: ident(name),
                mutable: false,
                ty: None,
                init: Box::new(init),
            }),
            span: span(),
        }
    }

    fn let_mut_stmt(name: &str, init: Expr) -> Stmt {
        Stmt {
            kind: StmtKind::Let(LetStmt {
                name: ident(name),
                mutable: true,
                ty: None,
                init: Box::new(init),
            }),
            span: span(),
        }
    }

    fn expr_stmt(expr: Expr) -> Stmt {
        Stmt { kind: StmtKind::Expr(expr), span: span() }
    }

    fn return_stmt(expr: Option<Expr>) -> Stmt {
        Stmt { kind: StmtKind::Return(expr), span: span() }
    }

    fn dummy_type_expr() -> TypeExpr {
        TypeExpr {
            kind: TypeExprKind::Named { name: ident("i64"), args: vec![] },
            span: span(),
        }
    }

    fn make_param(name: &str) -> Param {
        Param { name: ident(name), ty: dummy_type_expr(), span: span() }
    }

    fn make_fn_decl(name: &str, params: Vec<&str>, body: Block) -> Decl {
        Decl {
            kind: DeclKind::Fn(FnDecl {
                name: ident(name),
                type_params: vec![],
                params: params.into_iter().map(make_param).collect(),
                return_type: None,
                body,
                is_nogc: false,
            }),
            span: span(),
        }
    }

    fn make_block(stmts: Vec<Stmt>, expr: Option<Expr>) -> Block {
        Block { stmts, expr: expr.map(Box::new), span: span() }
    }

    #[test]
    fn test_mir_pipeline_arithmetic() {
        let program = Program {
            declarations: vec![make_fn_decl(
                "main",
                vec![],
                make_block(
                    vec![],
                    Some(binary(BinOp::Add, int_expr(2), int_expr(3))),
                ),
            )],
        };
        let result = run_program(&program, 0).unwrap();
        assert!(matches!(result, Value::Int(5)));
    }

    #[test]
    fn test_mir_pipeline_function_call() {
        let program = Program {
            declarations: vec![
                make_fn_decl(
                    "add",
                    vec!["a", "b"],
                    make_block(
                        vec![],
                        Some(binary(BinOp::Add, ident_expr("a"), ident_expr("b"))),
                    ),
                ),
                make_fn_decl(
                    "main",
                    vec![],
                    make_block(
                        vec![],
                        Some(call(ident_expr("add"), vec![int_expr(3), int_expr(4)])),
                    ),
                ),
            ],
        };
        let result = run_program(&program, 0).unwrap();
        assert!(matches!(result, Value::Int(7)));
    }

    #[test]
    fn test_mir_pipeline_while_loop() {
        let program = Program {
            declarations: vec![make_fn_decl(
                "main",
                vec![],
                make_block(
                    vec![
                        let_mut_stmt("i", int_expr(0)),
                        let_mut_stmt("sum", int_expr(0)),
                        Stmt {
                            kind: StmtKind::While(WhileStmt {
                                condition: binary(BinOp::Lt, ident_expr("i"), int_expr(5)),
                                body: make_block(
                                    vec![
                                        expr_stmt(assign_expr(
                                            ident_expr("sum"),
                                            binary(BinOp::Add, ident_expr("sum"), ident_expr("i")),
                                        )),
                                        expr_stmt(assign_expr(
                                            ident_expr("i"),
                                            binary(BinOp::Add, ident_expr("i"), int_expr(1)),
                                        )),
                                    ],
                                    None,
                                ),
                            }),
                            span: span(),
                        },
                    ],
                    Some(ident_expr("sum")),
                ),
            )],
        };
        let result = run_program(&program, 0).unwrap();
        assert!(matches!(result, Value::Int(10)));
    }

    #[test]
    fn test_mir_pipeline_pipe_desugaring() {
        let program = Program {
            declarations: vec![
                make_fn_decl(
                    "double",
                    vec!["x"],
                    make_block(
                        vec![],
                        Some(binary(BinOp::Mul, ident_expr("x"), int_expr(2))),
                    ),
                ),
                make_fn_decl(
                    "main",
                    vec![],
                    make_block(
                        vec![],
                        Some(pipe_expr(
                            int_expr(5),
                            call(ident_expr("double"), vec![]),
                        )),
                    ),
                ),
            ],
        };
        let result = run_program(&program, 0).unwrap();
        assert!(matches!(result, Value::Int(10)));
    }

    #[test]
    fn test_mir_pipeline_tensor_operations() {
        let program = Program {
            declarations: vec![make_fn_decl(
                "main",
                vec![],
                make_block(
                    vec![let_stmt_ast(
                        "t",
                        call(
                            field_expr(ident_expr("Tensor"), "zeros"),
                            vec![array_expr(vec![int_expr(2), int_expr(3)])],
                        ),
                    )],
                    Some(call(
                        field_expr(ident_expr("Tensor"), "zeros"),
                        vec![array_expr(vec![int_expr(2), int_expr(3)])],
                    )),
                ),
            )],
        };
        let result = run_program(&program, 0).unwrap();
        match &result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[2, 3]);
                assert_eq!(t.len(), 6);
            }
            _ => panic!("expected Tensor"),
        }
    }

    #[test]
    fn test_mir_pipeline_recursive_factorial() {
        let fact_body = make_block(
            vec![
                Stmt {
                    kind: StmtKind::If(IfStmt {
                        condition: binary(BinOp::Le, ident_expr("n"), int_expr(1)),
                        then_block: make_block(vec![return_stmt(Some(int_expr(1)))], None),
                        else_branch: None,
                    }),
                    span: span(),
                },
                return_stmt(Some(binary(
                    BinOp::Mul,
                    ident_expr("n"),
                    call(
                        ident_expr("factorial"),
                        vec![binary(BinOp::Sub, ident_expr("n"), int_expr(1))],
                    ),
                ))),
            ],
            None,
        );

        let program = Program {
            declarations: vec![
                make_fn_decl("factorial", vec!["n"], fact_body),
                make_fn_decl(
                    "main",
                    vec![],
                    make_block(
                        vec![],
                        Some(call(ident_expr("factorial"), vec![int_expr(5)])),
                    ),
                ),
            ],
        };
        let result = run_program(&program, 0).unwrap();
        assert!(matches!(result, Value::Int(120)));
    }

    #[test]
    fn test_mir_pipeline_if_else() {
        let program = Program {
            declarations: vec![make_fn_decl(
                "main",
                vec![],
                make_block(
                    vec![Stmt {
                        kind: StmtKind::If(IfStmt {
                            condition: bool_expr(false),
                            then_block: make_block(vec![], Some(int_expr(1))),
                            else_branch: Some(ElseBranch::Else(
                                make_block(vec![], Some(int_expr(2))),
                            )),
                        }),
                        span: span(),
                    }],
                    None,
                ),
            )],
        };
        let result = run_program(&program, 0).unwrap();
        assert!(matches!(result, Value::Int(2)));
    }

    #[test]
    fn test_mir_pipeline_print_output() {
        let program = Program {
            declarations: vec![make_fn_decl(
                "main",
                vec![],
                make_block(
                    vec![expr_stmt(call(
                        ident_expr("print"),
                        vec![Expr {
                            kind: ExprKind::StringLit("hello world".to_string()),
                            span: span(),
                        }],
                    ))],
                    None,
                ),
            )],
        };
        let (_, executor) = run_program_with_executor(&program, 0).unwrap();
        assert_eq!(executor.output, vec!["hello world"]);
    }
}
