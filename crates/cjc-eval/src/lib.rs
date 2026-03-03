//! CJC Tree-Walk Interpreter
//!
//! Evaluates a CJC `Program` AST by walking the tree directly. Supports:
//! - Arithmetic on integers, floats, booleans, and tensors
//! - User-defined functions with lexical scoping
//! - Structs (value types) and field access
//! - Built-in functions: print, Tensor constructors, matmul, Buffer.alloc
//! - Pipe operator (`|>`)
//! - If/else, while loops, early return
//! - Reproducible RNG via `cjc_repro::Rng`

use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;
use std::time::Instant;

use cjc_ast::{
    BinOp, Block, CallArg, Decl, DeclKind, ElseBranch, Expr, ExprKind, FnDecl, ForIter, ForStmt,
    IfStmt, LetStmt, Program, Stmt, StmtKind, UnaryOp, WhileStmt,
};
use cjc_data::{Column, CsvConfig, CsvReader, DataFrame, StreamingCsvProcessor, TidyView};
use cjc_data::tidy_dispatch;
use cjc_repro::Rng;
use cjc_runtime::{GcHeap, Tensor, Value};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Evaluation errors, including a `Return` variant used for control flow.
#[derive(Debug)]
pub enum EvalError {
    /// A `return` statement was executed. The interpreter unwinds the call
    /// stack until a function boundary catches this variant.
    Return(Value),

    /// A `break` statement was executed. Caught by the innermost loop.
    Break,

    /// A `continue` statement was executed. Caught by the innermost loop.
    Continue,

    /// A runtime error with a human-readable message.
    Runtime(String),

    /// An error propagated from the runtime crate.
    RuntimeError(cjc_runtime::RuntimeError),
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvalError::Return(_) => write!(f, "uncaught return"),
            EvalError::Break => write!(f, "break outside of loop"),
            EvalError::Continue => write!(f, "continue outside of loop"),
            EvalError::Runtime(msg) => write!(f, "runtime error: {msg}"),
            EvalError::RuntimeError(e) => write!(f, "runtime error: {e}"),
        }
    }
}

impl std::error::Error for EvalError {}

impl From<cjc_runtime::RuntimeError> for EvalError {
    fn from(e: cjc_runtime::RuntimeError) -> Self {
        EvalError::RuntimeError(e)
    }
}

/// Convenience type alias.
pub type EvalResult = Result<Value, EvalError>;

// ---------------------------------------------------------------------------
// Interpreter
// ---------------------------------------------------------------------------

/// Tree-walk interpreter for CJC programs.
pub struct Interpreter {
    /// User-defined functions indexed by name.
    functions: HashMap<String, FnDecl>,

    /// User-defined struct declarations indexed by name.
    struct_defs: HashMap<String, cjc_ast::StructDecl>,

    /// Maps variant name → enum name for variant resolution.
    variant_to_enum: HashMap<String, String>,

    /// Scope stack. The last entry is the innermost scope.
    scopes: Vec<HashMap<String, Value>>,

    /// GC heap for class instances.
    pub gc_heap: GcHeap,

    /// Deterministic RNG for reproducibility.
    pub rng: Rng,

    /// Captured print output (for testing).
    pub output: Vec<String>,

    /// High-resolution clock epoch for `clock()` builtin.
    start_time: Instant,

    /// Number of GC collections triggered so far.
    pub gc_collections: u64,
}

impl Interpreter {
    /// Create a new interpreter with the given RNG seed.
    pub fn new(seed: u64) -> Self {
        let mut variant_to_enum = HashMap::new();
        // Register prelude variant names
        variant_to_enum.insert("Some".into(), "Option".into());
        variant_to_enum.insert("None".into(), "Option".into());
        variant_to_enum.insert("Ok".into(), "Result".into());
        variant_to_enum.insert("Err".into(), "Result".into());

        Self {
            functions: HashMap::new(),
            struct_defs: HashMap::new(),
            variant_to_enum,
            scopes: vec![HashMap::new()],
            gc_heap: GcHeap::new(1024),
            rng: Rng::seeded(seed),
            output: Vec::new(),
            start_time: Instant::now(),
            gc_collections: 0,
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

    fn assign(&mut self, name: &str, val: Value) -> Result<(), EvalError> {
        for scope in self.scopes.iter_mut().rev() {
            if scope.contains_key(name) {
                scope.insert(name.to_string(), val);
                return Ok(());
            }
        }
        Err(EvalError::Runtime(format!(
            "assignment to undefined variable `{name}`"
        )))
    }

    // -- Program execution --------------------------------------------------

    /// Execute a full program.
    pub fn exec(&mut self, program: &Program) -> EvalResult {
        // First pass: register all function and struct declarations.
        for decl in &program.declarations {
            self.register_decl(decl);
        }

        // Second pass: execute top-level let bindings and statements.
        let mut last = Value::Void;
        for decl in &program.declarations {
            match &decl.kind {
                DeclKind::Let(let_stmt) => {
                    self.exec_let(let_stmt)?;
                }
                DeclKind::Stmt(stmt) => {
                    last = self.exec_stmt(stmt)?;
                }
                DeclKind::Const(c) => {
                    // P2-3: Evaluate const initializer and bind as immutable.
                    let val = self.eval_expr(&c.value)?;
                    self.define(&c.name.name, val);
                }
                DeclKind::Fn(_) | DeclKind::Struct(_) | DeclKind::Class(_)
                | DeclKind::Trait(_) | DeclKind::Impl(_) | DeclKind::Import(_)
                | DeclKind::Enum(_) => {
                    // Already registered or not needed at runtime.
                }
            }
        }

        // If there is a `main` function, call it.
        if self.functions.contains_key("main") {
            last = self.call_function("main", &[])?;
        }

        Ok(last)
    }

    fn register_decl(&mut self, decl: &Decl) {
        match &decl.kind {
            DeclKind::Fn(f) => {
                self.functions.insert(f.name.name.clone(), f.clone());
            }
            DeclKind::Struct(s) => {
                self.struct_defs.insert(s.name.name.clone(), s.clone());
            }
            DeclKind::Impl(impl_decl) => {
                // Register methods with qualified names: `TypeName.method_name`
                let type_name = match &impl_decl.target.kind {
                    cjc_ast::TypeExprKind::Named { name, .. } => name.name.clone(),
                    _ => return,
                };
                for method in &impl_decl.methods {
                    let qualified = format!("{}.{}", type_name, method.name.name);
                    self.functions.insert(qualified, method.clone());
                }
            }
            DeclKind::Enum(e) => {
                for v in &e.variants {
                    self.variant_to_enum
                        .insert(v.name.name.clone(), e.name.name.clone());
                }
            }
            _ => {}
        }
    }

    // -- Statement execution ------------------------------------------------

    fn exec_block(&mut self, block: &Block) -> EvalResult {
        self.push_scope();
        let result = self.exec_block_inner(block);
        self.pop_scope();
        result
    }

    fn exec_block_inner(&mut self, block: &Block) -> EvalResult {
        let mut last = Value::Void;
        for stmt in &block.stmts {
            last = self.exec_stmt(stmt)?;
        }
        if let Some(ref expr) = block.expr {
            self.eval_expr(expr)
        } else {
            Ok(last)
        }
    }

    fn exec_stmt(&mut self, stmt: &Stmt) -> EvalResult {
        match &stmt.kind {
            StmtKind::Let(let_stmt) => {
                self.exec_let(let_stmt)?;
                Ok(Value::Void)
            }
            StmtKind::Expr(expr) => {
                self.eval_expr(expr)?;
                Ok(Value::Void)
            }
            StmtKind::Return(opt_expr) => {
                let val = match opt_expr {
                    Some(expr) => self.eval_expr(expr)?,
                    None => Value::Void,
                };
                Err(EvalError::Return(val))
            }
            StmtKind::Break => Err(EvalError::Break),
            StmtKind::Continue => Err(EvalError::Continue),
            StmtKind::If(if_stmt) => self.exec_if(if_stmt),
            StmtKind::While(while_stmt) => self.exec_while(while_stmt),
            StmtKind::For(for_stmt) => self.exec_for(for_stmt),
            StmtKind::NoGcBlock(block) => {
                // NoGc enforcement is done by the type checker; at runtime,
                // just execute the block normally.
                self.exec_block(block)
            }
        }
    }

    fn exec_let(&mut self, let_stmt: &LetStmt) -> Result<(), EvalError> {
        let val = self.eval_expr(&let_stmt.init)?;
        self.define(&let_stmt.name.name, val);
        Ok(())
    }

    pub fn exec_if(&mut self, if_stmt: &IfStmt) -> EvalResult {
        let cond = self.eval_expr(&if_stmt.condition)?;
        let cond_bool = match cond {
            Value::Bool(b) => b,
            other => {
                return Err(EvalError::Runtime(format!(
                    "if condition must be Bool, got {}",
                    other.type_name()
                )));
            }
        };

        if cond_bool {
            self.exec_block(&if_stmt.then_block)
        } else if let Some(ref else_branch) = if_stmt.else_branch {
            match else_branch {
                ElseBranch::ElseIf(elif) => self.exec_if(elif),
                ElseBranch::Else(block) => self.exec_block(block),
            }
        } else {
            Ok(Value::Void)
        }
    }

    fn exec_while(&mut self, while_stmt: &WhileStmt) -> EvalResult {
        loop {
            let cond = self.eval_expr(&while_stmt.condition)?;
            let cond_bool = match cond {
                Value::Bool(b) => b,
                other => {
                    return Err(EvalError::Runtime(format!(
                        "while condition must be Bool, got {}",
                        other.type_name()
                    )));
                }
            };
            if !cond_bool {
                break;
            }
            match self.exec_block(&while_stmt.body) {
                Ok(_) => {}
                Err(EvalError::Break) => break,
                Err(EvalError::Continue) => continue,
                Err(e) => return Err(e),
            }
        }
        Ok(Value::Void)
    }

    fn exec_for(&mut self, for_stmt: &ForStmt) -> EvalResult {
        match &for_stmt.iter {
            ForIter::Range { start, end } => {
                let start_val = self.eval_expr(start)?;
                let end_val = self.eval_expr(end)?;
                let start_int = match start_val {
                    Value::Int(v) => v,
                    other => {
                        return Err(EvalError::Runtime(format!(
                            "for range start must be Int, got {}",
                            other.type_name()
                        )));
                    }
                };
                let end_int = match end_val {
                    Value::Int(v) => v,
                    other => {
                        return Err(EvalError::Runtime(format!(
                            "for range end must be Int, got {}",
                            other.type_name()
                        )));
                    }
                };
                let mut i = start_int;
                while i < end_int {
                    self.push_scope();
                    self.define(&for_stmt.ident.name, Value::Int(i));
                    match self.exec_block(&for_stmt.body) {
                        Ok(_) => {}
                        Err(EvalError::Break) => {
                            self.pop_scope();
                            break;
                        }
                        Err(EvalError::Continue) => {
                            self.pop_scope();
                            i += 1;
                            continue;
                        }
                        Err(e) => {
                            self.pop_scope();
                            return Err(e);
                        }
                    }
                    self.pop_scope();
                    i += 1;
                }
                Ok(Value::Void)
            }
            ForIter::Expr(expr) => {
                let collection = self.eval_expr(expr)?;
                let items: Vec<Value> = match &collection {
                    Value::Array(arr) => (**arr).clone(),
                    other => {
                        return Err(EvalError::Runtime(format!(
                            "for-in expression must be an Array, got {}",
                            other.type_name()
                        )));
                    }
                };
                for item in items {
                    self.push_scope();
                    self.define(&for_stmt.ident.name, item);
                    match self.exec_block(&for_stmt.body) {
                        Ok(_) => {}
                        Err(EvalError::Break) => {
                            self.pop_scope();
                            break;
                        }
                        Err(EvalError::Continue) => {
                            self.pop_scope();
                            continue;
                        }
                        Err(e) => {
                            self.pop_scope();
                            return Err(e);
                        }
                    }
                    self.pop_scope();
                }
                Ok(Value::Void)
            }
        }
    }

    // -- Expression evaluation ----------------------------------------------

    pub fn eval_expr(&mut self, expr: &Expr) -> EvalResult {
        match &expr.kind {
            ExprKind::IntLit(v) => Ok(Value::Int(*v)),
            ExprKind::FloatLit(v) => Ok(Value::Float(*v)),
            ExprKind::StringLit(s) => Ok(Value::String(Rc::new(s.clone()))),
            ExprKind::ByteStringLit(bytes) => Ok(Value::ByteSlice(Rc::new(bytes.clone()))),
            ExprKind::ByteCharLit(b) => Ok(Value::U8(*b)),
            ExprKind::RawStringLit(s) => Ok(Value::String(Rc::new(s.clone()))),
            ExprKind::RawByteStringLit(bytes) => Ok(Value::ByteSlice(Rc::new(bytes.clone()))),
            ExprKind::FStringLit(segments) => {
                // P2-5: Evaluate each interpolated expression and concat into a String.
                let mut result = String::new();
                for (lit, interp) in segments {
                    result.push_str(lit);
                    if let Some(e) = interp {
                        let val = self.eval_expr(e)?;
                        result.push_str(&format!("{val}"));
                    }
                }
                Ok(Value::String(Rc::new(result)))
            }
            ExprKind::RegexLit { pattern, flags } => Ok(Value::Regex { pattern: pattern.clone(), flags: flags.clone() }),
            ExprKind::TensorLit { rows } => {
                // Evaluate all elements, flatten to f64 vec, infer shape
                let n_rows = rows.len();
                if n_rows == 0 {
                    return Ok(Value::Tensor(Tensor::from_vec(vec![], &[0])?));
                }
                let n_cols = rows[0].len();
                let mut data = Vec::with_capacity(n_rows * n_cols);
                for row in rows {
                    if n_rows > 1 && row.len() != n_cols {
                        return Err(EvalError::Runtime(format!(
                            "tensor literal: row length mismatch, expected {} but got {}",
                            n_cols, row.len()
                        )));
                    }
                    for expr in row {
                        let val = self.eval_expr(expr)?;
                        match val {
                            Value::Float(f) => data.push(f),
                            Value::Int(i) => data.push(i as f64),
                            _ => return Err(EvalError::Runtime(
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
                Ok(Value::Tensor(Tensor::from_vec(data, &shape)?))
            }
            ExprKind::BoolLit(b) => Ok(Value::Bool(*b)),

            ExprKind::Ident(id) => {
                // First check if it's a local variable
                if let Some(val) = self.lookup(&id.name) {
                    return Ok(val.clone());
                }
                // Check if it's a unit variant (like None)
                if let Some(enum_name) = self.variant_to_enum.get(&id.name).cloned() {
                    return Ok(Value::Enum {
                        enum_name,
                        variant: id.name.clone(),
                        fields: vec![],
                    });
                }
                Err(EvalError::Runtime(format!(
                    "undefined variable `{}`",
                    id.name
                )))
            }

            ExprKind::Binary { op, left, right } => self.eval_binary(*op, left, right),
            ExprKind::Unary { op, operand } => self.eval_unary(*op, operand),
            ExprKind::Call { callee, args } => self.eval_call(callee, args),
            ExprKind::Field { object, name } => self.eval_field(object, &name.name),
            ExprKind::Index { object, index } => self.eval_index(object, index),
            ExprKind::MultiIndex { object, indices } => self.eval_multi_index(object, indices),

            ExprKind::Assign { target, value } => {
                let val = self.eval_expr(value)?;
                self.exec_assign(target, val)?;
                Ok(Value::Void)
            }

            ExprKind::Pipe { left, right } => self.eval_pipe(left, right),
            ExprKind::Block(block) => self.exec_block(block),

            ExprKind::StructLit { name, fields } => {
                let mut field_map = HashMap::new();
                for fi in fields {
                    let val = self.eval_expr(&fi.value)?;
                    field_map.insert(fi.name.name.clone(), val);
                }
                Ok(Value::Struct {
                    name: name.name.clone(),
                    fields: field_map,
                })
            }

            ExprKind::ArrayLit(elems) => {
                let mut vals = Vec::with_capacity(elems.len());
                for e in elems {
                    vals.push(self.eval_expr(e)?);
                }
                Ok(Value::Array(Rc::new(vals)))
            }

            ExprKind::Col(name) => {
                // Build a proper DExpr struct so that downstream tidy dispatch
                // correctly resolves the column name without any prefix.
                Ok(tidy_dispatch::build_col_expr(name))
            }

            ExprKind::Lambda { params, body } => {
                // For v1, lambdas are not first-class closures. We store them
                // as named functions with a synthetic name and register them.
                let lambda_name = format!("<lambda@{}>", expr.span.start);
                let fn_decl = FnDecl {
                    name: cjc_ast::Ident::dummy(&lambda_name),
                    type_params: vec![],
                    params: params.clone(),
                    return_type: None,
                    body: Block {
                        stmts: vec![],
                        expr: Some(body.clone()),
                        span: body.span,
                    },
                    is_nogc: false,
                };
                self.functions.insert(lambda_name.clone(), fn_decl);
                Ok(Value::Fn(cjc_runtime::FnValue {
                    name: lambda_name,
                    arity: params.len(),
                    body_id: 0,
                }))
            }

            ExprKind::Match { scrutinee, arms } => {
                let scrut_val = self.eval_expr(scrutinee)?;
                for arm in arms {
                    if let Some(bindings) = Self::match_pattern(&arm.pattern, &scrut_val) {
                        self.push_scope();
                        for (name, val) in bindings {
                            self.define(&name, val);
                        }
                        let result = self.eval_expr(&arm.body);
                        self.pop_scope();
                        return result;
                    }
                }
                Err(EvalError::Runtime(
                    "non-exhaustive match: no arm matched".to_string(),
                ))
            }

            ExprKind::TupleLit(elems) => {
                let mut vals = Vec::with_capacity(elems.len());
                for e in elems {
                    vals.push(self.eval_expr(e)?);
                }
                Ok(Value::Tuple(Rc::new(vals)))
            }

            ExprKind::Try(inner) => {
                // `expr?` desugars to: match on Result, Ok(v) => v, Err(e) => return Err(e)
                let inner_val = self.eval_expr(inner)?;
                match &inner_val {
                    Value::Enum {
                        enum_name,
                        variant,
                        fields,
                    } if enum_name == "Result" => {
                        if variant == "Ok" {
                            Ok(fields.first().cloned().unwrap_or(Value::Void))
                        } else {
                            // Err — propagate via early return
                            Err(EvalError::Return(Value::Enum {
                                enum_name: "Result".into(),
                                variant: "Err".into(),
                                fields: fields.clone(),
                            }))
                        }
                    }
                    _ => Err(EvalError::Runtime(format!(
                        "`?` operator requires Result value, got {}",
                        inner_val.type_name()
                    ))),
                }
            }

            ExprKind::VariantLit {
                enum_name,
                variant,
                fields,
            } => {
                let resolved_enum = if let Some(en) = enum_name {
                    en.name.clone()
                } else {
                    self.variant_to_enum
                        .get(&variant.name)
                        .cloned()
                        .unwrap_or_default()
                };
                let mut field_vals = Vec::with_capacity(fields.len());
                for f in fields {
                    field_vals.push(self.eval_expr(f)?);
                }
                Ok(Value::Enum {
                    enum_name: resolved_enum,
                    variant: variant.name.clone(),
                    fields: field_vals,
                })
            }

            ExprKind::CompoundAssign { op, target, value } => {
                let current = self.eval_expr(target)?;
                let rhs = self.eval_expr(value)?;
                let result = self.eval_binary_values(*op, current, rhs)?;
                self.exec_assign(target, result)?;
                Ok(Value::Void)
            }

            ExprKind::IfExpr { condition, then_block, else_branch } => {
                let cond = self.eval_expr(condition)?;
                let cond_bool = match cond {
                    Value::Bool(b) => b,
                    other => {
                        return Err(EvalError::Runtime(format!(
                            "if condition must be Bool, got {}",
                            other.type_name()
                        )));
                    }
                };
                if cond_bool {
                    self.exec_block(then_block)
                } else if let Some(ref else_br) = else_branch {
                    match else_br {
                        ElseBranch::ElseIf(elif) => self.exec_if(elif),
                        ElseBranch::Else(block) => self.exec_block(block),
                    }
                } else {
                    Ok(Value::Void)
                }
            }
        }
    }

    /// Try to match a value against a pattern. Returns Some(bindings) on match.
    fn match_pattern(
        pattern: &cjc_ast::Pattern,
        value: &Value,
    ) -> Option<Vec<(String, Value)>> {
        use cjc_ast::PatternKind;
        match &pattern.kind {
            PatternKind::Wildcard => Some(vec![]),
            PatternKind::Binding(id) => {
                // Check for well-known unit variant names that should match
                // rather than bind (e.g., None, Ok, Err used as unit patterns)
                let known_unit_variants = ["None"];
                if known_unit_variants.contains(&id.name.as_str()) {
                    // Treat as a variant pattern match
                    match value {
                        Value::Enum { variant, fields, .. }
                            if variant == &id.name && fields.is_empty() =>
                        {
                            return Some(vec![]);
                        }
                        Value::Enum { .. } => return None,
                        _ => {} // Fall through to normal binding for non-enum values
                    }
                }
                Some(vec![(id.name.clone(), value.clone())])
            }
            PatternKind::LitInt(v) => match value {
                Value::Int(i) if i == v => Some(vec![]),
                _ => None,
            },
            PatternKind::LitFloat(v) => match value {
                Value::Float(f) if f == v => Some(vec![]),
                _ => None,
            },
            PatternKind::LitBool(v) => match value {
                Value::Bool(b) if b == v => Some(vec![]),
                _ => None,
            },
            PatternKind::LitString(v) => match value {
                Value::String(s) if s.as_str() == v => Some(vec![]),
                _ => None,
            },
            PatternKind::Tuple(pats) => match value {
                Value::Tuple(vals) if pats.len() == vals.len() => {
                    let mut all = Vec::new();
                    for (p, v) in pats.iter().zip(vals.iter()) {
                        match Self::match_pattern(p, v) {
                            Some(bindings) => all.extend(bindings),
                            None => return None,
                        }
                    }
                    Some(all)
                }
                _ => None,
            },
            PatternKind::Struct { name, fields } => match value {
                Value::Struct {
                    name: val_name,
                    fields: val_fields,
                } if &name.name == val_name => {
                    let mut all = Vec::new();
                    for f in fields {
                        let field_val = val_fields.get(&f.name.name)?;
                        if let Some(ref p) = f.pattern {
                            match Self::match_pattern(p, field_val) {
                                Some(bindings) => all.extend(bindings),
                                None => return None,
                            }
                        } else {
                            // Shorthand: `Point { x }` means bind x
                            all.push((f.name.name.clone(), field_val.clone()));
                        }
                    }
                    Some(all)
                }
                _ => None,
            },
            PatternKind::Variant {
                enum_name: _,
                variant,
                fields,
            } => match value {
                Value::Enum {
                    enum_name: _,
                    variant: val_variant,
                    fields: val_fields,
                } => {
                    if variant.name != *val_variant {
                        return None;
                    }
                    if fields.len() != val_fields.len() {
                        return None;
                    }
                    let mut all = Vec::new();
                    for (pat, val) in fields.iter().zip(val_fields.iter()) {
                        match Self::match_pattern(pat, val) {
                            Some(bindings) => all.extend(bindings),
                            None => return None,
                        }
                    }
                    Some(all)
                }
                _ => None,
            },
        }
    }

    // -- Binary operations --------------------------------------------------

    fn eval_binary(&mut self, op: BinOp, left: &Expr, right: &Expr) -> EvalResult {
        // Short-circuit for logical operators.
        if op == BinOp::And {
            let lv = self.eval_expr(left)?;
            return match lv {
                Value::Bool(false) => Ok(Value::Bool(false)),
                Value::Bool(true) => {
                    let rv = self.eval_expr(right)?;
                    match rv {
                        Value::Bool(b) => Ok(Value::Bool(b)),
                        _ => Err(EvalError::Runtime(
                            "`&&` requires Bool operands".to_string(),
                        )),
                    }
                }
                _ => Err(EvalError::Runtime(
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
                        _ => Err(EvalError::Runtime(
                            "`||` requires Bool operands".to_string(),
                        )),
                    }
                }
                _ => Err(EvalError::Runtime(
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
                    return Err(EvalError::Runtime(format!(
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
            // Int x Int
            (Value::Int(a), Value::Int(b)) => self.binop_int(op, *a, *b),
            // Float x Float
            (Value::Float(a), Value::Float(b)) => self.binop_float(op, *a, *b),
            // Int x Float or Float x Int -- promote Int to Float
            (Value::Int(a), Value::Float(b)) => self.binop_float(op, *a as f64, *b),
            (Value::Float(a), Value::Int(b)) => self.binop_float(op, *a, *b as f64),
            // Bool equality
            (Value::Bool(a), Value::Bool(b)) => match op {
                BinOp::Eq => Ok(Value::Bool(a == b)),
                BinOp::Ne => Ok(Value::Bool(a != b)),
                _ => Err(EvalError::Runtime(format!(
                    "cannot apply `{op}` to Bool values"
                ))),
            },
            // String concatenation
            (Value::String(a), Value::String(b)) => match op {
                BinOp::Add => Ok(Value::String(Rc::new(format!("{a}{b}")))),
                BinOp::Eq => Ok(Value::Bool(a == b)),
                BinOp::Ne => Ok(Value::Bool(a != b)),
                _ => Err(EvalError::Runtime(format!(
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
                _ => Err(EvalError::Runtime(format!(
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
                _ => Err(EvalError::Runtime(format!(
                    "cannot apply `{op}` to Complex values"
                ))),
            },
            // Tensor x Tensor
            (Value::Tensor(a), Value::Tensor(b)) => match op {
                BinOp::Add => Ok(Value::Tensor(a.add(b)?)),
                BinOp::Sub => Ok(Value::Tensor(a.sub(b)?)),
                BinOp::Mul => Ok(Value::Tensor(a.mul_elem(b)?)),
                BinOp::Div => Ok(Value::Tensor(a.div_elem(b)?)),
                _ => Err(EvalError::Runtime(format!(
                    "cannot apply `{op}` to Tensor values"
                ))),
            },
            // Tensor x Float (scalar broadcast)
            (Value::Tensor(t), Value::Float(s)) => match op {
                BinOp::Mul => Ok(Value::Tensor(t.scalar_mul(*s))),
                BinOp::Div => Ok(Value::Tensor(t.scalar_mul(1.0 / *s))),
                BinOp::Add => Ok(Value::Tensor(t.map(|x| x + *s))),
                BinOp::Sub => Ok(Value::Tensor(t.map(|x| x - *s))),
                _ => Err(EvalError::Runtime(format!(
                    "cannot apply `{op}` to Tensor and Float"
                ))),
            },
            // Float x Tensor (scalar broadcast)
            (Value::Float(s), Value::Tensor(t)) => match op {
                BinOp::Mul => Ok(Value::Tensor(t.scalar_mul(*s))),
                BinOp::Add => Ok(Value::Tensor(t.map(|x| *s + x))),
                BinOp::Sub => Ok(Value::Tensor(t.map(|x| *s - x))),
                BinOp::Div => Ok(Value::Tensor(t.map(|x| *s / x))),
                _ => Err(EvalError::Runtime(format!(
                    "cannot apply `{op}` to Float and Tensor"
                ))),
            },
            // Tensor x Int (scalar broadcast, promote to float)
            (Value::Tensor(t), Value::Int(s)) => match op {
                BinOp::Mul => Ok(Value::Tensor(t.scalar_mul(*s as f64))),
                BinOp::Div => Ok(Value::Tensor(t.scalar_mul(1.0 / *s as f64))),
                _ => Err(EvalError::Runtime(format!(
                    "cannot apply `{op}` to Tensor and Int"
                ))),
            },
            // Int x Tensor (scalar broadcast, promote to float)
            (Value::Int(s), Value::Tensor(t)) => match op {
                BinOp::Mul => Ok(Value::Tensor(t.scalar_mul(*s as f64))),
                _ => Err(EvalError::Runtime(format!(
                    "cannot apply `{op}` to Int and Tensor"
                ))),
            },
            _ => Err(EvalError::Runtime(format!(
                "cannot apply `{op}` to {} and {}",
                lv.type_name(),
                rv.type_name()
            ))),
        }
    }

    /// Apply a binary operation on two already-evaluated values.
    /// Used by CompoundAssign desugaring.
    fn eval_binary_values(&mut self, op: BinOp, lv: Value, rv: Value) -> EvalResult {
        match (&lv, &rv) {
            (Value::Int(a), Value::Int(b)) => self.binop_int(op, *a, *b),
            (Value::Float(a), Value::Float(b)) => self.binop_float(op, *a, *b),
            (Value::Int(a), Value::Float(b)) => self.binop_float(op, *a as f64, *b),
            (Value::Float(a), Value::Int(b)) => self.binop_float(op, *a, *b as f64),
            (Value::String(a), Value::String(b)) => match op {
                BinOp::Add => Ok(Value::String(Rc::new(format!("{a}{b}")))),
                _ => Err(EvalError::Runtime(format!(
                    "cannot apply `{op}` to String values"
                ))),
            },
            (Value::Tensor(a), Value::Tensor(b)) => match op {
                BinOp::Add => Ok(Value::Tensor(a.add(b)?)),
                BinOp::Sub => Ok(Value::Tensor(a.sub(b)?)),
                BinOp::Mul => Ok(Value::Tensor(a.mul_elem(b)?)),
                BinOp::Div => Ok(Value::Tensor(a.div_elem(b)?)),
                _ => Err(EvalError::Runtime(format!(
                    "cannot apply `{op}` to Tensor values"
                ))),
            },
            (Value::Tensor(t), Value::Float(s)) => match op {
                BinOp::Mul => Ok(Value::Tensor(t.scalar_mul(*s))),
                BinOp::Div => Ok(Value::Tensor(t.scalar_mul(1.0 / *s))),
                BinOp::Add => Ok(Value::Tensor(t.map(|x| x + *s))),
                BinOp::Sub => Ok(Value::Tensor(t.map(|x| x - *s))),
                _ => Err(EvalError::Runtime(format!(
                    "cannot apply `{op}` to Tensor and Float"
                ))),
            },
            (Value::Float(s), Value::Tensor(t)) => match op {
                BinOp::Mul => Ok(Value::Tensor(t.scalar_mul(*s))),
                BinOp::Add => Ok(Value::Tensor(t.map(|x| *s + x))),
                BinOp::Sub => Ok(Value::Tensor(t.map(|x| *s - x))),
                BinOp::Div => Ok(Value::Tensor(t.map(|x| *s / x))),
                _ => Err(EvalError::Runtime(format!(
                    "cannot apply `{op}` to Float and Tensor"
                ))),
            },
            _ => Err(EvalError::Runtime(format!(
                "cannot apply `{op}` to {} and {}",
                lv.type_name(),
                rv.type_name()
            ))),
        }
    }

    fn binop_int(&self, op: BinOp, a: i64, b: i64) -> EvalResult {
        match op {
            BinOp::Add => Ok(Value::Int(a.wrapping_add(b))),
            BinOp::Sub => Ok(Value::Int(a.wrapping_sub(b))),
            BinOp::Mul => Ok(Value::Int(a.wrapping_mul(b))),
            BinOp::Div => {
                if b == 0 {
                    Err(EvalError::Runtime("division by zero".to_string()))
                } else {
                    Ok(Value::Int(a / b))
                }
            }
            BinOp::Mod => {
                if b == 0 {
                    Err(EvalError::Runtime("modulo by zero".to_string()))
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
            BinOp::Pow => Ok(Value::Int((a as f64).powf(b as f64) as i64)),
            BinOp::BitAnd => Ok(Value::Int(a & b)),
            BinOp::BitOr => Ok(Value::Int(a | b)),
            BinOp::BitXor => Ok(Value::Int(a ^ b)),
            BinOp::Shl => Ok(Value::Int(a.wrapping_shl(b as u32))),
            BinOp::Shr => Ok(Value::Int(a.wrapping_shr(b as u32))),
            BinOp::And | BinOp::Or | BinOp::Match | BinOp::NotMatch => Err(EvalError::Runtime(format!(
                "cannot apply `{op}` to Int values"
            ))),
        }
    }

    fn binop_float(&self, op: BinOp, a: f64, b: f64) -> EvalResult {
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
            BinOp::Pow => Ok(Value::Float(a.powf(b))),
            BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => Err(EvalError::Runtime(format!(
                "cannot apply `{op}` to Float values"
            ))),
            BinOp::And | BinOp::Or | BinOp::Match | BinOp::NotMatch => Err(EvalError::Runtime(format!(
                "cannot apply `{op}` to Float values"
            ))),
        }
    }

    // -- Unary operations ---------------------------------------------------

    fn eval_unary(&mut self, op: UnaryOp, operand: &Expr) -> EvalResult {
        let val = self.eval_expr(operand)?;
        match (op, &val) {
            (UnaryOp::Neg, Value::Int(v)) => Ok(Value::Int(-v)),
            (UnaryOp::Neg, Value::Float(v)) => Ok(Value::Float(-v)),
            (UnaryOp::Neg, Value::Tensor(t)) => Ok(Value::Tensor(t.map(|x| -x))),
            (UnaryOp::Neg, Value::F16(v)) => Ok(Value::F16(v.neg())),
            (UnaryOp::Neg, Value::Complex(z)) => Ok(Value::Complex(z.neg())),
            (UnaryOp::Not, Value::Bool(b)) => Ok(Value::Bool(!b)),
            (UnaryOp::BitNot, Value::Int(v)) => Ok(Value::Int(!v)),
            _ => Err(EvalError::Runtime(format!(
                "cannot apply `{op}` to {}",
                val.type_name()
            ))),
        }
    }

    // -- Function / method calls --------------------------------------------

    fn eval_call(&mut self, callee: &Expr, args: &[CallArg]) -> EvalResult {
        // Evaluate arguments eagerly.
        let mut arg_vals: Vec<Value> = Vec::with_capacity(args.len());
        for arg in args {
            arg_vals.push(self.eval_expr(&arg.value)?);
        }

        match &callee.kind {
            ExprKind::Ident(id) => {
                // Check if this is a variant constructor call (e.g., Some(42), Ok(v))
                if let Some(enum_name) = self.variant_to_enum.get(&id.name).cloned() {
                    return Ok(Value::Enum {
                        enum_name,
                        variant: id.name.clone(),
                        fields: arg_vals,
                    });
                }
                self.dispatch_call(&id.name, arg_vals)
            }
            ExprKind::Field { object, name } => {
                // Could be a static method like `Tensor.zeros(...)` or an
                // instance method like `t.sum()`.
                //
                // Try static dispatch first (object is an identifier that
                // names a type).
                if let ExprKind::Ident(obj_id) = &object.kind {
                    let qualified = format!("{}.{}", obj_id.name, name.name);
                    if self.is_known_builtin(&qualified) || self.functions.contains_key(&qualified) {
                        return self.dispatch_call(&qualified, arg_vals);
                    }
                }

                // Instance method: evaluate the object and call the method.
                let obj_val = self.eval_expr(object)?;
                self.dispatch_method(obj_val, &name.name, arg_vals)
            }
            _ => {
                // Callee might be a FnValue.
                let callee_val = self.eval_expr(callee)?;
                match callee_val {
                    Value::Fn(fv) => self.call_function(&fv.name, &arg_vals),
                    _ => Err(EvalError::Runtime(format!(
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
                | "Tensor.zeros"
                | "Tensor.ones"
                | "Tensor.randn"
                | "Tensor.from_vec"
                | "matmul"
                | "Buffer.alloc"
                | "len"
                | "push"
                | "sort"
                | "sqrt"
                | "floor"
                | "abs"
                | "int"
                | "float"
                | "isnan"
                | "isinf"
                | "assert"
                | "assert_eq"
                | "clock"
                | "gc_alloc"
                | "gc_collect"
                | "gc_live_count"
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
                // Tidy builder builtins
                | "col"
                | "desc"
                | "asc"
                | "dexpr_binop"
                | "tidy_count"
                | "tidy_sum"
                | "tidy_mean"
                | "tidy_min"
                | "tidy_max"
                | "tidy_first"
                | "tidy_last"
                // stringr builtins
                | "str_detect"
                | "str_extract"
                | "str_extract_all"
                | "str_replace"
                | "str_replace_all"
                | "str_split"
                | "str_count"
                | "str_trim"
                | "str_to_upper"
                | "str_to_lower"
                | "str_starts"
                | "str_ends"
                | "str_sub"
                | "str_len"
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
                // Sprint 1: Descriptive statistics
                | "variance"
                | "sd"
                | "se"
                | "median"
                | "quantile"
                | "iqr"
                | "skewness"
                | "kurtosis"
                | "z_score"
                | "standardize"
                | "n_distinct"
                // Sprint 2: Correlation + Inference
                | "cor"
                | "cov"
                | "normal_cdf"
                | "normal_pdf"
                | "normal_ppf"
                | "t_cdf"
                | "chi2_cdf"
                | "f_cdf"
                | "t_test"
                | "t_test_two_sample"
                | "chi_squared_test"
                // Sprint 3: Linalg
                | "det"
                | "solve"
                | "lstsq"
                | "trace"
                | "norm_frobenius"
                | "eigh"
                | "matrix_rank"
                | "kron"
                // Sprint 4: ML
                | "mse_loss"
                | "cross_entropy_loss"
                | "huber_loss"
                | "binary_cross_entropy"
                | "hinge_loss"
                | "confusion_matrix"
                | "auc_roc"
                // Sprint 5: Analyst QoL
                | "cumsum"
                | "cumprod"
                | "cummax"
                | "cummin"
                | "lag"
                | "lead"
                | "rank"
                | "dense_rank"
                | "row_number"
                | "histogram"
                | "sample_variance"
                | "sample_sd"
                | "sample_cov"
                // Sprint 6: Advanced
                | "t_ppf"
                | "chi2_ppf"
                | "f_ppf"
                | "binomial_pmf"
                | "binomial_cdf"
                | "poisson_pmf"
                | "poisson_cdf"
                | "t_test_paired"
                | "anova_oneway"
                | "f_test"
                | "lm"
                | "rfft"
                | "psd"
                // Tensor activations & utilities
                | "sigmoid"
                | "tanh_activation"
                | "leaky_relu"
                | "silu"
                | "mish"
                | "argmax"
                | "argmin"
                | "clamp"
                | "one_hot"
                // Phase B1: Weighted & robust statistics
                | "weighted_mean"
                | "weighted_var"
                | "trimmed_mean"
                | "winsorize"
                | "mad"
                | "mode"
                | "percentile_rank"
                // Phase B2: Rank correlations
                | "spearman_cor"
                | "kendall_cor"
                | "partial_cor"
                | "cor_ci"
                // Phase B3: Linear algebra extensions
                | "cond"
                | "norm_1"
                | "norm_inf"
                | "schur"
                | "matrix_exp"
                // Phase B4: ML training extensions
                | "cat"
                | "stack"
                | "topk"
                | "batch_norm"
                | "dropout_mask"
                | "lr_step_decay"
                | "lr_cosine"
                | "lr_linear_warmup"
                | "l1_penalty"
                | "l2_penalty"
                // Phase B5: Analyst QoL extensions
                | "case_when"
                | "ntile"
                | "percent_rank"
                | "cume_dist"
                | "wls"
                // Phase B6: Advanced FFT & Distributions
                | "hann"
                | "hamming"
                | "blackman"
                | "fft_arbitrary"
                | "fft_2d"
                | "ifft_2d"
                | "beta_pdf"
                | "beta_cdf"
                | "gamma_pdf"
                | "gamma_cdf"
                | "exp_pdf"
                | "exp_cdf"
                | "weibull_pdf"
                | "weibull_cdf"
                // Phase B7: Non-parametric tests & multiple comparisons
                | "tukey_hsd"
                | "mann_whitney"
                | "kruskal_wallis"
                | "wilcoxon_signed_rank"
                | "bonferroni"
                | "fdr_bh"
                | "logistic_regression"
                // Phase C1: GradGraph Language API
                | "GradGraph.new"
                // Phase C2: Optimizer constructors
                | "Adam.new"
                | "Sgd.new"
                // Phase C6: I/O & Collection Utilities
                | "read_line"
                | "array_push"
                | "array_pop"
                | "array_contains"
                | "array_reverse"
                | "array_flatten"
                | "array_len"
                | "array_slice"
                // Phase C5: Map & Set constructors
                | "Map.new"
                | "Set.new"
                // Phase C4: Sorting & Tensor Indexing
                | "argsort"
                | "gather"
                | "scatter"
                | "index_select"
                // Phase C3: Bitwise operations
                | "bit_and"
                | "bit_or"
                | "bit_xor"
                | "bit_not"
                | "bit_shl"
                | "bit_shr"
                | "popcount"
                // Phase D: RL primitives
                | "log"
                | "exp"
                | "categorical_sample"
                // Phase E: Mathematics Hardening
                | "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "atan2"
                | "sinh" | "cosh" | "tanh_scalar"
                | "pow" | "log2" | "log10" | "log1p" | "expm1"
                | "ceil" | "round"
                | "min" | "max" | "sign"
                | "hypot"
                | "PI" | "E" | "TAU" | "INF" | "NAN_VAL"
                | "dot" | "outer" | "cross" | "norm"
                | "Tensor.linspace" | "Tensor.arange" | "Tensor.eye"
                | "Tensor.full" | "Tensor.diag" | "Tensor.uniform"
                // ML Autodiff builtins
                | "stop_gradient" | "grad_checkpoint" | "clip_grad" | "grad_scale"
        )
    }

    fn dispatch_call(&mut self, name: &str, args: Vec<Value>) -> EvalResult {
        // User-defined functions always take priority over builtins
        // (allows shadowing builtin names like "outer", "min", etc.)
        if self.functions.contains_key(name) {
            return self.call_function(name, &args);
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
                    .map_err(EvalError::Runtime)?;
                let t = Tensor::randn(&shape, &mut self.rng);
                return Ok(Value::Tensor(t));
            }
            "Tensor.uniform" => {
                let shape = cjc_runtime::builtins::value_to_shape(&args[0])
                    .map_err(EvalError::Runtime)?;
                let total: usize = shape.iter().product();
                let data: Vec<f64> = (0..total).map(|_| {
                    let u = self.rng.next_f64().abs();
                    u - u.floor()
                }).collect();
                return Ok(Value::Tensor(Tensor::from_vec(data, &shape).map_err(|e| EvalError::Runtime(format!("{e}")))?));
            }
            "categorical_sample" => {
                if args.len() != 1 {
                    return Err(EvalError::Runtime("categorical_sample requires 1 argument: probs tensor".into()));
                }
                let probs = match &args[0] {
                    Value::Tensor(t) => t,
                    _ => return Err(EvalError::Runtime(format!(
                        "categorical_sample requires a Tensor, got {}", args[0].type_name()
                    ))),
                };
                let u = self.rng.next_f64().abs();
                let u = u - u.floor(); // ensure [0, 1)
                let idx = cjc_runtime::builtins::categorical_sample_with_u(probs, u)
                    .map_err(EvalError::Runtime)?;
                return Ok(Value::Int(idx));
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
            // Phase C6: read_line (needs IO)
            "read_line" => {
                let mut line = String::new();
                std::io::stdin().read_line(&mut line).map_err(|e| EvalError::Runtime(format!("read_line: {e}")))?;
                // Strip trailing newline
                if line.ends_with('\n') { line.pop(); }
                if line.ends_with('\r') { line.pop(); }
                return Ok(Value::String(Rc::new(line)));
            }
            _ => {}
        }

        // Try shared (stateless) builtins
        match cjc_runtime::builtins::dispatch_builtin(name, &args) {
            Ok(Some(value)) => return Ok(value),
            Err(msg) => return Err(EvalError::Runtime(msg)),
            Ok(None) => {} // not a shared builtin, fall through
        }

        // Phase C1: GradGraph constructor (bypasses builtins.rs — cjc-ad dep)
        if name == "GradGraph.new" {
            use std::any::Any;
            use std::cell::RefCell;
            if !args.is_empty() {
                return Err(EvalError::Runtime("GradGraph.new takes 0 arguments".into()));
            }
            let g = cjc_ad::GradGraph::new();
            let erased: Rc<RefCell<dyn Any>> = Rc::new(RefCell::new(g));
            return Ok(Value::GradGraph(erased));
        }

        // CSV / Data Logistics builtins (depend on cjc-data, not in shared module)
        match name {
            "Csv.parse" | "Csv.parse_tsv" => {
                if args.is_empty() || args.len() > 2 {
                    return Err(EvalError::Runtime(
                        "Csv.parse requires 1 argument: bytes (+ optional max_rows)".to_string(),
                    ));
                }
                let bytes = cjc_runtime::builtins::value_to_bytes(&args[0])
                    .map_err(EvalError::Runtime)?;
                let max_rows = if args.len() == 2 {
                    Some(cjc_runtime::builtins::value_to_usize(&args[1])
                        .map_err(EvalError::Runtime)?)
                } else {
                    None
                };
                let delim = if name == "Csv.parse_tsv" { b'\t' } else { b',' };
                let config = CsvConfig { delimiter: delim, max_rows, ..CsvConfig::default() };
                let df = CsvReader::new(config)
                    .parse(&bytes)
                    .map_err(|e| EvalError::Runtime(format!("Csv.parse error: {}", e)))?;
                return Ok(dataframe_to_value(df));
            }
            "Csv.stream_sum" => {
                if args.is_empty() {
                    return Err(EvalError::Runtime(
                        "Csv.stream_sum requires 1 argument: bytes".to_string(),
                    ));
                }
                let bytes = cjc_runtime::builtins::value_to_bytes(&args[0])
                    .map_err(EvalError::Runtime)?;
                let config = CsvConfig::default();
                let (names, sums, count) = StreamingCsvProcessor::new(config)
                    .sum_columns(&bytes)
                    .map_err(|e| EvalError::Runtime(format!("Csv.stream_sum error: {}", e)))?;
                let mut fields = std::collections::HashMap::new();
                for (name, sum) in names.iter().zip(sums.iter()) {
                    fields.insert(name.clone(), Value::Float(*sum));
                }
                fields.insert("__row_count".to_string(), Value::Int(count as i64));
                return Ok(Value::Struct { name: "CsvStats".to_string(), fields });
            }
            "Csv.stream_minmax" => {
                if args.is_empty() {
                    return Err(EvalError::Runtime(
                        "Csv.stream_minmax requires 1 argument: bytes".to_string(),
                    ));
                }
                let bytes = cjc_runtime::builtins::value_to_bytes(&args[0])
                    .map_err(EvalError::Runtime)?;
                let config = CsvConfig::default();
                let (names, mins, maxs, count) = StreamingCsvProcessor::new(config)
                    .minmax_columns(&bytes)
                    .map_err(|e| EvalError::Runtime(format!("Csv.stream_minmax error: {}", e)))?;
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

        // Tidy builder builtins: col(), desc(), asc(), tidy_sum(), etc.
        match tidy_dispatch::dispatch_tidy_builtin(name, &args) {
            Ok(Some(value)) => return Ok(value),
            Err(msg) => return Err(EvalError::Runtime(msg)),
            Ok(None) => {} // not a tidy builtin, fall through
        }

        // Try user-defined function.
        if self.functions.contains_key(name) {
            self.call_function(name, &args)
        } else {
            Err(EvalError::Runtime(format!(
                "undefined function `{name}`"
            )))
        }
    }

    fn dispatch_method(&mut self, receiver: Value, method: &str, args: Vec<Value>) -> EvalResult {
        match (&receiver, method) {
            // Tensor methods
            (Value::Tensor(t), "sum") => Ok(Value::Float(t.sum())),
            (Value::Tensor(t), "mean") => Ok(Value::Float(t.mean())),
            (Value::Tensor(t), "binned_sum") => Ok(Value::Float(t.binned_sum())),
            // Mathematics Hardening Phase: tensor reductions
            (Value::Tensor(t), "max") => {
                let data = t.to_vec();
                if data.is_empty() { return Err(EvalError::Runtime("max on empty tensor".to_string())); }
                Ok(Value::Float(data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)))
            }
            (Value::Tensor(t), "min") => {
                let data = t.to_vec();
                if data.is_empty() { return Err(EvalError::Runtime("min on empty tensor".to_string())); }
                Ok(Value::Float(data.iter().cloned().fold(f64::INFINITY, f64::min)))
            }
            (Value::Tensor(t), "var") => {
                let data = t.to_vec();
                let n = data.len() as f64;
                if n == 0.0 { return Err(EvalError::Runtime("var on empty tensor".to_string())); }
                let mean = data.iter().sum::<f64>() / n;
                let var = data.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
                Ok(Value::Float(var))
            }
            (Value::Tensor(t), "std") => {
                let data = t.to_vec();
                let n = data.len() as f64;
                if n == 0.0 { return Err(EvalError::Runtime("std on empty tensor".to_string())); }
                let mean = data.iter().sum::<f64>() / n;
                let var = data.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
                Ok(Value::Float(var.sqrt()))
            }
            (Value::Tensor(t), "abs") => {
                Ok(Value::Tensor(t.map(|x| x.abs())))
            }
            (Value::Tensor(t), "mean_axis") => {
                if args.is_empty() { return Err(EvalError::Runtime("mean_axis requires an axis argument".to_string())); }
                let axis = match &args[0] {
                    Value::Int(i) => *i as usize,
                    _ => return Err(EvalError::Runtime("mean_axis: axis must be an integer".to_string())),
                };
                if t.ndim() != 2 { return Err(EvalError::Runtime("mean_axis currently requires a 2D tensor".to_string())); }
                let sum_t = t.sum_axis(axis).map_err(|e| EvalError::Runtime(format!("{e}")))?;
                let divisor = t.shape()[axis] as f64;
                Ok(Value::Tensor(sum_t.scalar_mul(1.0 / divisor)))
            }
            (Value::Tensor(t), "max_axis") => {
                if args.is_empty() { return Err(EvalError::Runtime("max_axis requires an axis argument".to_string())); }
                let axis = match &args[0] {
                    Value::Int(i) => *i as usize,
                    _ => return Err(EvalError::Runtime("max_axis: axis must be an integer".to_string())),
                };
                if t.ndim() != 2 { return Err(EvalError::Runtime("max_axis currently requires a 2D tensor".to_string())); }
                let rows = t.shape()[0];
                let cols = t.shape()[1];
                let data_vec = t.to_vec();
                if axis == 0 {
                    let mut data = vec![f64::NEG_INFINITY; cols];
                    for r in 0..rows { for c in 0..cols { data[c] = data[c].max(data_vec[r * cols + c]); } }
                    Ok(Value::Tensor(Tensor::from_vec(data, &[1, cols]).map_err(|e| EvalError::Runtime(format!("{e}")))?))
                } else {
                    let mut data = vec![f64::NEG_INFINITY; rows];
                    for r in 0..rows { for c in 0..cols { data[r] = data[r].max(data_vec[r * cols + c]); } }
                    Ok(Value::Tensor(Tensor::from_vec(data, &[rows, 1]).map_err(|e| EvalError::Runtime(format!("{e}")))?))
                }
            }
            (Value::Tensor(t), "min_axis") => {
                if args.is_empty() { return Err(EvalError::Runtime("min_axis requires an axis argument".to_string())); }
                let axis = match &args[0] {
                    Value::Int(i) => *i as usize,
                    _ => return Err(EvalError::Runtime("min_axis: axis must be an integer".to_string())),
                };
                if t.ndim() != 2 { return Err(EvalError::Runtime("min_axis currently requires a 2D tensor".to_string())); }
                let rows = t.shape()[0];
                let cols = t.shape()[1];
                let data_vec = t.to_vec();
                if axis == 0 {
                    let mut data = vec![f64::INFINITY; cols];
                    for r in 0..rows { for c in 0..cols { data[c] = data[c].min(data_vec[r * cols + c]); } }
                    Ok(Value::Tensor(Tensor::from_vec(data, &[1, cols]).map_err(|e| EvalError::Runtime(format!("{e}")))?))
                } else {
                    let mut data = vec![f64::INFINITY; rows];
                    for r in 0..rows { for c in 0..cols { data[r] = data[r].min(data_vec[r * cols + c]); } }
                    Ok(Value::Tensor(Tensor::from_vec(data, &[rows, 1]).map_err(|e| EvalError::Runtime(format!("{e}")))?))
                }
            }

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
                    Err(EvalError::Runtime("Complex.add requires a Complex argument".to_string()))
                }
            }
            (Value::Complex(z), "mul") => {
                if let Some(Value::Complex(w)) = args.first() {
                    Ok(Value::Complex(z.mul_fixed(*w)))
                } else {
                    Err(EvalError::Runtime("Complex.mul requires a Complex argument".to_string()))
                }
            }
            (Value::Complex(z), "sub") => {
                if let Some(Value::Complex(w)) = args.first() {
                    Ok(Value::Complex(z.sub(*w)))
                } else {
                    Err(EvalError::Runtime("Complex.sub requires a Complex argument".to_string()))
                }
            }
            (Value::Complex(z), "div") => {
                if let Some(Value::Complex(w)) = args.first() {
                    Ok(Value::Complex(z.div_fixed(*w)))
                } else {
                    Err(EvalError::Runtime("Complex.div requires a Complex argument".to_string()))
                }
            }
            (Value::Complex(z), "neg") => Ok(Value::Complex(z.neg())),
            (Value::Complex(z), "scale") => {
                if let Some(Value::Float(s)) = args.first() {
                    Ok(Value::Complex(z.scale(*s)))
                } else if let Some(Value::Int(s)) = args.first() {
                    Ok(Value::Complex(z.scale(*s as f64)))
                } else {
                    Err(EvalError::Runtime("Complex.scale requires a numeric argument".to_string()))
                }
            }
            (Value::Complex(z), "is_nan") => Ok(Value::Bool(z.is_nan())),
            (Value::Complex(z), "is_finite") => Ok(Value::Bool(z.is_finite())),

            // F16 methods
            (Value::F16(v), "to_f64") => Ok(Value::Float(v.to_f64())),
            (Value::F16(v), "to_f32") => Ok(Value::Float(v.to_f32() as f64)),
            (Value::Tensor(t), "shape") => {
                let shape_vals: Vec<Value> = t.shape().iter().map(|&d| Value::Int(d as i64)).collect();
                Ok(Value::Array(Rc::new(shape_vals)))
            }
            (Value::Tensor(t), "len") => Ok(Value::Int(t.len() as i64)),
            (Value::Tensor(t), "to_vec") => {
                let data: Vec<Value> = t.to_vec().iter().map(|&v| Value::Float(v)).collect();
                Ok(Value::Array(Rc::new(data)))
            }
            (Value::Tensor(t), "matmul") => {
                if args.len() != 1 {
                    return Err(EvalError::Runtime(
                        "matmul requires 1 Tensor argument".to_string(),
                    ));
                }
                let other = self.value_to_tensor(&args[0])?;
                Ok(Value::Tensor(t.matmul(&other)?))
            }
            (Value::Tensor(t), "add") => {
                if args.len() != 1 {
                    return Err(EvalError::Runtime(
                        "add requires 1 Tensor argument".to_string(),
                    ));
                }
                let other = self.value_to_tensor(&args[0])?;
                Ok(Value::Tensor(t.add(&other)?))
            }
            (Value::Tensor(t), "sub") => {
                if args.len() != 1 {
                    return Err(EvalError::Runtime(
                        "sub requires 1 Tensor argument".to_string(),
                    ));
                }
                let other = self.value_to_tensor(&args[0])?;
                Ok(Value::Tensor(t.sub(&other)?))
            }
            (Value::Tensor(t), "reshape") => {
                if args.len() != 1 {
                    return Err(EvalError::Runtime(
                        "reshape requires 1 shape argument".to_string(),
                    ));
                }
                let shape = self.value_to_shape(&args[0])?;
                Ok(Value::Tensor(t.reshape(&shape)?))
            }
            (Value::Tensor(t), "get") => {
                if args.len() != 1 {
                    return Err(EvalError::Runtime(
                        "get requires 1 index argument".to_string(),
                    ));
                }
                let indices = self.value_to_usize_vec(&args[0])?;
                Ok(Value::Float(t.get(&indices)?))
            }
            (Value::Tensor(t), "transpose") => {
                if t.ndim() != 2 {
                    return Err(EvalError::Runtime(
                        "transpose requires a 2-D tensor".to_string(),
                    ));
                }
                Ok(Value::Tensor(t.transpose()))
            }
            (Value::Tensor(t), "neg") => {
                Ok(Value::Tensor(t.neg()))
            }
            (Value::Tensor(t), "scalar_mul") => {
                if args.len() != 1 {
                    return Err(EvalError::Runtime(
                        "scalar_mul requires 1 Float argument".to_string(),
                    ));
                }
                match &args[0] {
                    Value::Float(s) => Ok(Value::Tensor(t.scalar_mul(*s))),
                    Value::Int(s) => Ok(Value::Tensor(t.scalar_mul(*s as f64))),
                    _ => Err(EvalError::Runtime(format!(
                        "scalar_mul requires a number, got {}", args[0].type_name()
                    ))),
                }
            }
            (Value::Tensor(t), "mul") => {
                if args.len() != 1 {
                    return Err(EvalError::Runtime(
                        "mul requires 1 Tensor argument".to_string(),
                    ));
                }
                let other = self.value_to_tensor(&args[0])?;
                Ok(Value::Tensor(t.mul_elem(&other)?))
            }
            (Value::Tensor(t), "sum_axis") => {
                if args.len() != 1 {
                    return Err(EvalError::Runtime(
                        "sum_axis requires 1 Int argument (axis)".to_string(),
                    ));
                }
                let axis = self.value_to_usize(&args[0])?;
                Ok(Value::Tensor(t.sum_axis(axis)?))
            }
            (Value::Tensor(t), "set") => {
                if args.len() != 2 {
                    return Err(EvalError::Runtime(
                        "set requires 2 arguments: indices and value".to_string(),
                    ));
                }
                let indices = self.value_to_usize_vec(&args[0])?;
                let val = match &args[1] {
                    Value::Float(f) => *f,
                    Value::Int(i) => *i as f64,
                    _ => return Err(EvalError::Runtime(
                        "set value must be a number".to_string(),
                    )),
                };
                let mut t_mut = t.clone();
                t_mut.set(&indices, val)?;
                Ok(Value::Tensor(t_mut))
            }

            // -- Transformer kernel methods --
            (Value::Tensor(t), "softmax") => {
                Ok(Value::Tensor(t.softmax()?))
            }
            (Value::Tensor(t), "layer_norm") => {
                if args.len() < 2 || args.len() > 3 {
                    return Err(EvalError::Runtime(
                        "layer_norm requires 2-3 arguments: gamma, beta, [eps]".to_string(),
                    ));
                }
                let gamma = self.value_to_tensor(&args[0])?;
                let beta = self.value_to_tensor(&args[1])?;
                let eps = if args.len() == 3 {
                    match &args[2] {
                        Value::Float(f) => *f,
                        Value::Int(i) => *i as f64,
                        _ => return Err(EvalError::Runtime(
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
                    return Err(EvalError::Runtime(
                        "conv1d requires 2 arguments: filters, bias".to_string(),
                    ));
                }
                let filters = self.value_to_tensor(&args[0])?;
                let bias = self.value_to_tensor(&args[1])?;
                Ok(Value::Tensor(t.conv1d(&filters, &bias)?))
            }
            // Phase 7: 2D Spatial Vision
            (Value::Tensor(t), "conv2d") => {
                // conv2d(filters, bias)  — stride defaults to 1
                // conv2d(filters, bias, stride)
                if args.len() < 2 || args.len() > 3 {
                    return Err(EvalError::Runtime(
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
                // maxpool2d(ph, pw)
                if args.len() != 2 {
                    return Err(EvalError::Runtime(
                        "maxpool2d requires 2 arguments: pool_h, pool_w".to_string(),
                    ));
                }
                let ph = self.value_to_usize(&args[0])?;
                let pw = self.value_to_usize(&args[1])?;
                Ok(Value::Tensor(t.maxpool2d(ph, pw)?))
            }
            (Value::Tensor(t), "bmm") => {
                if args.len() != 1 {
                    return Err(EvalError::Runtime(
                        "bmm requires 1 Tensor argument".to_string(),
                    ));
                }
                let other = self.value_to_tensor(&args[0])?;
                Ok(Value::Tensor(t.bmm(&other)?))
            }
            (Value::Tensor(t), "linear") => {
                if args.len() != 2 {
                    return Err(EvalError::Runtime(
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
                    return Err(EvalError::Runtime(
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
                    return Err(EvalError::Runtime(
                        "view_reshape requires 1 argument: new_shape".to_string(),
                    ));
                }
                let new_shape = self.value_to_shape(&args[0])?;
                Ok(Value::Tensor(t.view_reshape(&new_shape)?))
            }
            (Value::Tensor(_t), "from_bytes") => {
                // instance method variant: tensor_bytes.from_bytes(shape, [dtype])
                // Not typical — prefer Tensor.from_bytes(). But wiring for completeness.
                return Err(EvalError::Runtime(
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
                    return Err(EvalError::Runtime(
                        "Scratchpad.append requires 1 argument: a Tensor or Array".to_string(),
                    ));
                }
                match &args[0] {
                    Value::Tensor(t) => {
                        if t.ndim() == 1 {
                            s.borrow_mut().append(&t.to_vec())?;
                        } else {
                            s.borrow_mut().append_tensor(t)?;
                        }
                    }
                    Value::Array(arr) => {
                        let data: Vec<f64> = arr.iter().map(|v| match v {
                            Value::Float(f) => Ok(*f),
                            Value::Int(i) => Ok(*i as f64),
                            _ => Err(EvalError::Runtime("append: array must contain numbers".into())),
                        }).collect::<Result<Vec<_>, _>>()?;
                        s.borrow_mut().append(&data)?;
                    }
                    _ => return Err(EvalError::Runtime(
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
                    return Err(EvalError::Runtime(
                        "PagedKvCache.append requires 1 argument".to_string(),
                    ));
                }
                match &args[0] {
                    Value::Tensor(t) => {
                        c.borrow_mut().append_tensor(t)?;
                    }
                    Value::Array(arr) => {
                        let data: Vec<f64> = arr.iter().map(|v| match v {
                            Value::Float(f) => Ok(*f),
                            Value::Int(i) => Ok(*i as f64),
                            _ => Err(EvalError::Runtime("append: array must contain numbers".into())),
                        }).collect::<Result<Vec<_>, _>>()?;
                        c.borrow_mut().append(&data)?;
                    }
                    _ => return Err(EvalError::Runtime(
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
                    return Err(EvalError::Runtime(
                        "PagedKvCache.get_token requires 1 argument: index".to_string(),
                    ));
                }
                let idx = self.value_to_usize(&args[0])?;
                let token = c.borrow().get_token(idx)?;
                Ok(Value::Tensor(Tensor::from_vec(token, &[c.borrow().dim()])?))
            }

            // AlignedByteSlice methods (Phase 4: alignment-aware bytes)
            (Value::AlignedBytes(a), "as_tensor") => {
                if args.is_empty() || args.len() > 2 {
                    return Err(EvalError::Runtime(
                        "AlignedByteSlice.as_tensor requires 1-2 arguments: shape, [dtype]".to_string(),
                    ));
                }
                let shape = self.value_to_shape(&args[0])?;
                let dtype_str = if args.len() == 2 {
                    match &args[1] {
                        Value::String(s) => (**s).clone(),
                        _ => return Err(EvalError::Runtime("dtype must be a string".into())),
                    }
                } else {
                    "f64".to_string()
                };
                Ok(Value::Tensor(a.as_tensor(&shape, &dtype_str)?))
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

            // Array methods
            (Value::Array(arr), "len") => Ok(Value::Int(arr.len() as i64)),

            // String methods
            (Value::String(s), "len") => Ok(Value::Int(s.len() as i64)),
            (Value::String(s), "as_bytes") => {
                Ok(Value::ByteSlice(Rc::new(s.as_bytes().to_vec())))
            }

            // ByteSlice methods (NoGC-safe)
            (Value::ByteSlice(b), "len") => Ok(Value::Int(b.len() as i64)),
            (Value::ByteSlice(b), "is_empty") => Ok(Value::Bool(b.is_empty())),
            (Value::ByteSlice(b), "get") => {
                if args.len() != 1 {
                    return Err(EvalError::Runtime("get requires 1 index argument".into()));
                }
                let idx = self.value_to_usize(&args[0])?;
                if idx >= b.len() {
                    return Err(EvalError::Runtime(format!(
                        "index {} out of bounds for ByteSlice of length {}", idx, b.len()
                    )));
                }
                Ok(Value::U8(b[idx]))
            }
            (Value::ByteSlice(b), "slice") => {
                if args.len() != 2 {
                    return Err(EvalError::Runtime("slice requires 2 arguments: start, end".into()));
                }
                let start = self.value_to_usize(&args[0])?;
                let end = self.value_to_usize(&args[1])?;
                if start > end || end > b.len() {
                    return Err(EvalError::Runtime(format!(
                        "slice bounds [{}, {}) out of range for ByteSlice of length {}", start, end, b.len()
                    )));
                }
                Ok(Value::ByteSlice(Rc::new(b[start..end].to_vec())))
            }
            (Value::ByteSlice(b), "find_byte") => {
                if args.len() != 1 {
                    return Err(EvalError::Runtime("find_byte requires 1 u8 argument".into()));
                }
                let needle = match &args[0] {
                    Value::U8(v) => *v,
                    Value::Int(v) => *v as u8,
                    _ => return Err(EvalError::Runtime("find_byte requires a byte argument".into())),
                };
                match b.iter().position(|&x| x == needle) {
                    Some(pos) => Ok(Value::Int(pos as i64)),
                    None => Ok(Value::Int(-1)),
                }
            }
            (Value::ByteSlice(b), "split_byte") => {
                if args.len() != 1 {
                    return Err(EvalError::Runtime("split_byte requires 1 u8 argument".into()));
                }
                let delim = match &args[0] {
                    Value::U8(v) => *v,
                    Value::Int(v) => *v as u8,
                    _ => return Err(EvalError::Runtime("split_byte requires a byte argument".into())),
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
                    return Err(EvalError::Runtime("strip_prefix requires 1 ByteSlice argument".into()));
                }
                let prefix = match &args[0] {
                    Value::ByteSlice(p) => p.clone(),
                    _ => return Err(EvalError::Runtime("strip_prefix requires a ByteSlice argument".into())),
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
                    return Err(EvalError::Runtime("strip_suffix requires 1 ByteSlice argument".into()));
                }
                let suffix = match &args[0] {
                    Value::ByteSlice(s) => s.clone(),
                    _ => return Err(EvalError::Runtime("strip_suffix requires a ByteSlice argument".into())),
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
                    return Err(EvalError::Runtime("starts_with requires 1 ByteSlice argument".into()));
                }
                let prefix = match &args[0] {
                    Value::ByteSlice(p) => p.clone(),
                    _ => return Err(EvalError::Runtime("starts_with requires a ByteSlice argument".into())),
                };
                Ok(Value::Bool(b.starts_with(&prefix)))
            }
            (Value::ByteSlice(b), "ends_with") => {
                if args.len() != 1 {
                    return Err(EvalError::Runtime("ends_with requires 1 ByteSlice argument".into()));
                }
                let suffix = match &args[0] {
                    Value::ByteSlice(s) => s.clone(),
                    _ => return Err(EvalError::Runtime("ends_with requires a ByteSlice argument".into())),
                };
                Ok(Value::Bool(b.ends_with(&suffix)))
            }
            (Value::ByteSlice(b), "count_byte") => {
                if args.len() != 1 {
                    return Err(EvalError::Runtime("count_byte requires 1 u8 argument".into()));
                }
                let needle = match &args[0] {
                    Value::U8(v) => *v,
                    Value::Int(v) => *v as u8,
                    _ => return Err(EvalError::Runtime("count_byte requires a byte argument".into())),
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
                    return Err(EvalError::Runtime("eq requires 1 ByteSlice argument".into()));
                }
                match &args[0] {
                    Value::ByteSlice(other) => Ok(Value::Bool(**b == **other)),
                    _ => Ok(Value::Bool(false)),
                }
            }
            (Value::ByteSlice(b), "as_tensor") => {
                // as_tensor(shape, [dtype])
                if args.is_empty() || args.len() > 2 {
                    return Err(EvalError::Runtime(
                        "as_tensor requires 1-2 arguments: shape, [dtype='f64']".to_string(),
                    ));
                }
                let shape = self.value_to_shape(&args[0])?;
                let dtype = if args.len() == 2 {
                    match &args[1] {
                        Value::String(s) => s.as_str().to_string(),
                        _ => return Err(EvalError::Runtime(
                            "as_tensor: dtype must be a string".to_string(),
                        )),
                    }
                } else {
                    "f64".to_string()
                };
                Ok(Value::Tensor(Tensor::from_bytes(b, &shape, &dtype)?))
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
                    return Err(EvalError::Runtime("eq requires 1 StrView argument".into()));
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
                        // Count fields that are not meta fields (__columns, __nrows).
                        let n = fields.keys().filter(|k| !k.starts_with("__")).count();
                        Ok(Value::Int(n as i64))
                    }
                    "column_names" => {
                        Ok(fields.get("__columns").cloned().unwrap_or(Value::Array(Rc::new(vec![]))))
                    }
                    "column" => {
                        // df.column("salary") → Array of values
                        if args.len() != 1 {
                            return Err(EvalError::Runtime(
                                "DataFrame.column requires 1 argument: column_name".to_string(),
                            ));
                        }
                        let col_name = match &args[0] {
                            Value::String(s) => s.as_ref().clone(),
                            _ => return Err(EvalError::Runtime(
                                "DataFrame.column: column name must be a String".to_string(),
                            )),
                        };
                        fields.get(&col_name).cloned().ok_or_else(|| {
                            EvalError::Runtime(format!("column '{}' not found", col_name))
                        })
                    }
                    "view" => {
                        // df.view() → TidyView
                        let df = rebuild_dataframe_from_struct(fields)?;
                        Ok(tidy_dispatch::wrap_view(TidyView::from_df(df)))
                    }
                    "to_tensor" => {
                        // df.to_tensor(["x", "y"]) → Tensor[nrows, ncols]
                        if args.len() != 1 {
                            return Err(EvalError::Runtime(
                                "DataFrame.to_tensor requires 1 argument: column_names array".to_string(),
                            ));
                        }
                        let col_names: Vec<String> = match &args[0] {
                            Value::Array(arr) => {
                                arr.iter()
                                    .map(|v| match v {
                                        Value::String(s) => Ok(s.as_ref().clone()),
                                        _ => Err(EvalError::Runtime(
                                            "to_tensor: column names must be strings".to_string(),
                                        )),
                                    })
                                    .collect::<Result<Vec<_>, _>>()?
                            }
                            _ => return Err(EvalError::Runtime(
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
                                EvalError::Runtime(format!("column '{}' not found", col_name))
                            })?;
                            let col = match arr_val {
                                Value::Array(arr) => {
                                    let floats: Vec<f64> = arr.iter()
                                        .map(|v| match v {
                                            Value::Float(x) => Ok(*x),
                                            Value::Int(x)   => Ok(*x as f64),
                                            _ => Err(EvalError::Runtime(format!(
                                                "column '{}' contains non-numeric values", col_name
                                            ))),
                                        })
                                        .collect::<Result<Vec<_>, _>>()?;
                                    Column::Float(floats)
                                }
                                _ => return Err(EvalError::Runtime(format!(
                                    "column '{}' has unexpected type", col_name
                                ))),
                            };
                            df_cols.push((col_name.clone(), col));
                        }
                        let df = DataFrame::from_columns(df_cols)
                            .map_err(|e| EvalError::Runtime(format!("DataFrame: {}", e)))?;
                        let col_refs: Vec<&str> = col_names.iter().map(|s| s.as_str()).collect();
                        let t = df.to_tensor(&col_refs)
                            .map_err(|e| EvalError::Runtime(format!("to_tensor: {}", e)))?;
                        Ok(Value::Tensor(t))
                    }
                    other => {
                        return Err(EvalError::Runtime(format!(
                            "no method `{}` on DataFrame", other
                        )));
                    }
                }
            }

            // -- Phase C5: Map method dispatch --
            (Value::Map(m), "insert") => {
                if args.len() != 2 { return Err(EvalError::Runtime("Map.insert requires 2 args: key, value".into())); }
                m.borrow_mut().insert(args[0].clone(), args[1].clone());
                Ok(Value::Void)
            }
            (Value::Map(m), "get") => {
                if args.len() != 1 { return Err(EvalError::Runtime("Map.get requires 1 arg: key".into())); }
                match m.borrow().get(&args[0]) {
                    Some(v) => Ok(v.clone()),
                    None => Ok(Value::Void),
                }
            }
            (Value::Map(m), "remove") => {
                if args.len() != 1 { return Err(EvalError::Runtime("Map.remove requires 1 arg: key".into())); }
                m.borrow_mut().remove(&args[0]);
                Ok(Value::Void)
            }
            (Value::Map(m), "len") => {
                Ok(Value::Int(m.borrow().len() as i64))
            }
            (Value::Map(m), "contains_key") | (Value::Map(m), "contains") => {
                if args.len() != 1 { return Err(EvalError::Runtime("contains_key requires 1 arg: key".into())); }
                Ok(Value::Bool(m.borrow().contains_key(&args[0])))
            }
            (Value::Map(m), "keys") => {
                Ok(Value::Array(Rc::new(m.borrow().keys())))
            }
            (Value::Map(m), "values") => {
                Ok(Value::Array(Rc::new(m.borrow().values_vec())))
            }
            // Set methods (Set is a Map with all values = Void)
            (Value::Map(m), "add") => {
                if args.len() != 1 { return Err(EvalError::Runtime("Set.add requires 1 arg: value".into())); }
                m.borrow_mut().insert(args[0].clone(), Value::Void);
                Ok(Value::Void)
            }
            (Value::Map(m), "to_array") => {
                Ok(Value::Array(Rc::new(m.borrow().keys())))
            }

            // -- Phase C1: GradGraph method dispatch --
            (Value::GradGraph(inner), method) => {
                use std::any::Any;
                match method {
                    "parameter" => {
                        if args.len() != 1 { return Err(EvalError::Runtime("parameter requires 1 arg: Tensor".into())); }
                        let t = match &args[0] { Value::Tensor(t) => t.clone(), _ => return Err(EvalError::Runtime("expected Tensor".into())) };
                        let mut borrow = inner.borrow_mut();
                        let graph = borrow.downcast_mut::<cjc_ad::GradGraph>().unwrap();
                        Ok(Value::Int(graph.parameter(t) as i64))
                    }
                    "input" => {
                        if args.len() != 1 { return Err(EvalError::Runtime("input requires 1 arg: Tensor".into())); }
                        let t = match &args[0] { Value::Tensor(t) => t.clone(), _ => return Err(EvalError::Runtime("expected Tensor".into())) };
                        let mut borrow = inner.borrow_mut();
                        let graph = borrow.downcast_mut::<cjc_ad::GradGraph>().unwrap();
                        Ok(Value::Int(graph.input(t) as i64))
                    }
                    "add" | "sub" | "mul" | "div" | "matmul" => {
                        if args.len() != 2 { return Err(EvalError::Runtime(format!("{method} requires 2 args: node_a, node_b"))); }
                        let a = match &args[0] { Value::Int(i) => *i as usize, _ => return Err(EvalError::Runtime("expected Int (node index)".into())) };
                        let b = match &args[1] { Value::Int(i) => *i as usize, _ => return Err(EvalError::Runtime("expected Int (node index)".into())) };
                        let mut borrow = inner.borrow_mut();
                        let graph = borrow.downcast_mut::<cjc_ad::GradGraph>().unwrap();
                        let idx = match method {
                            "add" => graph.add(a, b),
                            "sub" => graph.sub(a, b),
                            "mul" => graph.mul(a, b),
                            "div" => graph.div(a, b),
                            "matmul" => graph.matmul(a, b),
                            _ => unreachable!(),
                        };
                        Ok(Value::Int(idx as i64))
                    }
                    "neg" | "sum" | "mean" | "sigmoid" | "relu" | "tanh" | "sin" | "cos" | "sqrt" | "exp" | "ln" => {
                        if args.len() != 1 { return Err(EvalError::Runtime(format!("{method} requires 1 arg: node_index"))); }
                        let a = match &args[0] { Value::Int(i) => *i as usize, _ => return Err(EvalError::Runtime("expected Int (node index)".into())) };
                        let mut borrow = inner.borrow_mut();
                        let graph = borrow.downcast_mut::<cjc_ad::GradGraph>().unwrap();
                        let idx = match method {
                            "neg" => graph.neg(a),
                            "sum" => graph.sum(a),
                            "mean" => graph.mean(a),
                            "sigmoid" => graph.sigmoid(a),
                            "relu" => graph.relu(a),
                            "tanh" => graph.tanh_act(a),
                            "sin" => graph.sin(a),
                            "cos" => graph.cos(a),
                            "sqrt" => graph.sqrt(a),
                            "exp" => graph.exp(a),
                            "ln" => graph.ln(a),
                            _ => unreachable!(),
                        };
                        Ok(Value::Int(idx as i64))
                    }
                    "pow" => {
                        if args.len() != 2 { return Err(EvalError::Runtime("pow requires 2 args: node_index, exponent".into())); }
                        let a = match &args[0] { Value::Int(i) => *i as usize, _ => return Err(EvalError::Runtime("expected Int".into())) };
                        let n = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err(EvalError::Runtime("expected number".into())) };
                        let mut borrow = inner.borrow_mut();
                        let graph = borrow.downcast_mut::<cjc_ad::GradGraph>().unwrap();
                        Ok(Value::Int(graph.pow(a, n) as i64))
                    }
                    "scalar_mul" => {
                        if args.len() != 2 { return Err(EvalError::Runtime("scalar_mul requires 2 args: node_index, scalar".into())); }
                        let a = match &args[0] { Value::Int(i) => *i as usize, _ => return Err(EvalError::Runtime("expected Int".into())) };
                        let s = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err(EvalError::Runtime("expected number".into())) };
                        let mut borrow = inner.borrow_mut();
                        let graph = borrow.downcast_mut::<cjc_ad::GradGraph>().unwrap();
                        Ok(Value::Int(graph.scalar_mul(a, s) as i64))
                    }
                    "backward" => {
                        if args.len() != 1 { return Err(EvalError::Runtime("backward requires 1 arg: loss_node_index".into())); }
                        let loss_idx = match &args[0] { Value::Int(i) => *i as usize, _ => return Err(EvalError::Runtime("expected Int".into())) };
                        let borrow = inner.borrow();
                        let graph = borrow.downcast_ref::<cjc_ad::GradGraph>().unwrap();
                        graph.backward(loss_idx);
                        Ok(Value::Void)
                    }
                    "value" => {
                        if args.len() != 1 { return Err(EvalError::Runtime("value requires 1 arg: node_index".into())); }
                        let idx = match &args[0] { Value::Int(i) => *i as usize, _ => return Err(EvalError::Runtime("expected Int".into())) };
                        let borrow = inner.borrow();
                        let graph = borrow.downcast_ref::<cjc_ad::GradGraph>().unwrap();
                        Ok(Value::Float(graph.value(idx)))
                    }
                    "tensor" => {
                        if args.len() != 1 { return Err(EvalError::Runtime("tensor requires 1 arg: node_index".into())); }
                        let idx = match &args[0] { Value::Int(i) => *i as usize, _ => return Err(EvalError::Runtime("expected Int".into())) };
                        let borrow = inner.borrow();
                        let graph = borrow.downcast_ref::<cjc_ad::GradGraph>().unwrap();
                        Ok(Value::Tensor(graph.tensor(idx)))
                    }
                    "grad" => {
                        if args.len() != 1 { return Err(EvalError::Runtime("grad requires 1 arg: node_index".into())); }
                        let idx = match &args[0] { Value::Int(i) => *i as usize, _ => return Err(EvalError::Runtime("expected Int".into())) };
                        let borrow = inner.borrow();
                        let graph = borrow.downcast_ref::<cjc_ad::GradGraph>().unwrap();
                        match graph.grad(idx) {
                            Some(t) => Ok(Value::Tensor(t)),
                            None => Ok(Value::Void),
                        }
                    }
                    "set_tensor" => {
                        if args.len() != 2 { return Err(EvalError::Runtime("set_tensor requires 2 args: node_index, tensor".into())); }
                        let idx = match &args[0] { Value::Int(i) => *i as usize, _ => return Err(EvalError::Runtime("expected Int".into())) };
                        let t = match &args[1] { Value::Tensor(t) => t.clone(), _ => return Err(EvalError::Runtime("expected Tensor".into())) };
                        let borrow = inner.borrow();
                        let graph = borrow.downcast_ref::<cjc_ad::GradGraph>().unwrap();
                        graph.set_tensor(idx, t);
                        Ok(Value::Void)
                    }
                    "zero_grad" => {
                        if !args.is_empty() { return Err(EvalError::Runtime("zero_grad takes 0 arguments".into())); }
                        let borrow = inner.borrow();
                        let graph = borrow.downcast_ref::<cjc_ad::GradGraph>().unwrap();
                        graph.zero_grad();
                        Ok(Value::Void)
                    }
                    _ => Err(EvalError::Runtime(format!("no method `{method}` on GradGraph"))),
                }
            }

            // -- Phase C2: OptimizerState method dispatch --
            (Value::OptimizerState(inner), "step") => {
                use std::any::Any;
                if args.len() != 2 {
                    return Err(EvalError::Runtime("step requires 2 args: params_tensor, grads_tensor".into()));
                }
                let params_t = match &args[0] { Value::Tensor(t) => t.clone(), _ => return Err(EvalError::Runtime("expected Tensor for params".into())) };
                let grads_t = match &args[1] { Value::Tensor(t) => t.clone(), _ => return Err(EvalError::Runtime("expected Tensor for grads".into())) };
                let mut params = params_t.to_vec();
                let grads = grads_t.to_vec();
                if params.len() != grads.len() {
                    return Err(EvalError::Runtime("params and grads must have same length".into()));
                }
                let mut borrow = inner.borrow_mut();
                // Try Adam first, then SGD
                if let Some(adam) = borrow.downcast_mut::<cjc_runtime::ml::AdamState>() {
                    if params.len() != adam.m.len() {
                        return Err(EvalError::Runtime(format!(
                            "param size mismatch: optimizer expects {}, got {}",
                            adam.m.len(), params.len()
                        )));
                    }
                    cjc_runtime::ml::adam_step(&mut params, &grads, adam);
                } else if let Some(sgd) = borrow.downcast_mut::<cjc_runtime::ml::SgdState>() {
                    if params.len() != sgd.velocity.len() {
                        return Err(EvalError::Runtime(format!(
                            "param size mismatch: optimizer expects {}, got {}",
                            sgd.velocity.len(), params.len()
                        )));
                    }
                    cjc_runtime::ml::sgd_step(&mut params, &grads, sgd);
                } else {
                    return Err(EvalError::Runtime("unknown optimizer type".into()));
                }
                Ok(Value::Tensor(Tensor::from_vec(params, params_t.shape())?))
            }
            (Value::OptimizerState(_), method) => {
                Err(EvalError::Runtime(format!("no method `{method}` on OptimizerState")))
            }

            // -- TidyView dispatch (all tidy verbs) --
            (Value::TidyView(inner), _) => {
                match tidy_dispatch::dispatch_tidy_method(inner, method, &args) {
                    Ok(Some(Value::String(s))) if method == "print" => {
                        // print returns a formatted string; output it
                        let line = s.as_ref().clone();
                        println!("{line}");
                        self.output.push(line);
                        Ok(Value::Void)
                    }
                    Ok(Some(val)) => Ok(val),
                    Ok(None) => Err(EvalError::Runtime(format!(
                        "no method `{method}` on TidyView"
                    ))),
                    Err(msg) => Err(EvalError::Runtime(msg)),
                }
            }

            // -- GroupedTidyView dispatch --
            (Value::GroupedTidyView(inner), _) => {
                match tidy_dispatch::dispatch_grouped_method(inner, method, &args) {
                    Ok(Some(val)) => Ok(val),
                    Ok(None) => Err(EvalError::Runtime(format!(
                        "no method `{method}` on GroupedTidyView"
                    ))),
                    Err(msg) => Err(EvalError::Runtime(msg)),
                }
            }

            // Struct: try qualified method lookup
            (Value::Struct { name: sname, .. }, _) => {
                let qualified = format!("{sname}.{method}");
                if self.functions.contains_key(&qualified) {
                    let mut full_args = vec![receiver.clone()];
                    full_args.extend(args);
                    self.call_function(&qualified, &full_args)
                } else {
                    Err(EvalError::Runtime(format!(
                        "no method `{method}` on struct `{sname}`"
                    )))
                }
            }

            _ => Err(EvalError::Runtime(format!(
                "no method `{method}` on type {}",
                receiver.type_name()
            ))),
        }
    }

    fn call_function(&mut self, name: &str, args: &[Value]) -> EvalResult {
        let func = self.functions.get(name).cloned().ok_or_else(|| {
            EvalError::Runtime(format!("undefined function `{name}`"))
        })?;

        if func.params.len() != args.len() {
            return Err(EvalError::Runtime(format!(
                "function `{name}` expects {} arguments, got {}",
                func.params.len(),
                args.len()
            )));
        }

        self.push_scope();

        // Bind parameters.
        for (param, val) in func.params.iter().zip(args.iter()) {
            self.define(&param.name.name, val.clone());
        }

        // Execute body, catching Return errors.
        let result = match self.exec_block_inner(&func.body) {
            Ok(val) => val,
            Err(EvalError::Return(val)) => val,
            Err(e) => {
                self.pop_scope();
                return Err(e);
            }
        };

        self.pop_scope();
        Ok(result)
    }

    // -- Field access -------------------------------------------------------

    fn eval_field(&mut self, object: &Expr, field: &str) -> EvalResult {
        let obj = self.eval_expr(object)?;
        match &obj {
            Value::Struct { fields, name, .. } => {
                fields.get(field).cloned().ok_or_else(|| {
                    EvalError::Runtime(format!("no field `{field}` on struct `{name}`"))
                })
            }
            Value::Tensor(t) => match field {
                "shape" => {
                    let shape_vals: Vec<Value> =
                        t.shape().iter().map(|&d| Value::Int(d as i64)).collect();
                    Ok(Value::Array(Rc::new(shape_vals)))
                }
                _ => Err(EvalError::Runtime(format!(
                    "no field `{field}` on Tensor"
                ))),
            },
            Value::Array(arr) => match field {
                "len" => Ok(Value::Int(arr.len() as i64)),
                _ => Err(EvalError::Runtime(format!(
                    "no field `{field}` on Array"
                ))),
            },
            _ => Err(EvalError::Runtime(format!(
                "cannot access field `{field}` on {}",
                obj.type_name()
            ))),
        }
    }

    // -- Indexing ------------------------------------------------------------

    fn eval_index(&mut self, object: &Expr, index: &Expr) -> EvalResult {
        let obj = self.eval_expr(object)?;
        let idx = self.eval_expr(index)?;

        match (&obj, &idx) {
            (Value::Array(arr), Value::Int(i)) => {
                let i = *i as usize;
                arr.get(i).cloned().ok_or_else(|| {
                    EvalError::Runtime(format!(
                        "index {i} out of bounds for array of length {}",
                        arr.len()
                    ))
                })
            }
            (Value::Tensor(t), Value::Int(i)) => {
                // 1-D tensor indexing: t[i]
                let i = *i as usize;
                if t.ndim() == 1 {
                    Ok(Value::Float(t.get(&[i])?))
                } else {
                    Err(EvalError::Runtime(
                        "use multi-index for tensors with ndim > 1".to_string(),
                    ))
                }
            }
            _ => Err(EvalError::Runtime(format!(
                "cannot index {} with {}",
                obj.type_name(),
                idx.type_name()
            ))),
        }
    }

    fn eval_multi_index(&mut self, object: &Expr, indices: &[Expr]) -> EvalResult {
        let obj = self.eval_expr(object)?;
        let mut idx_vals = Vec::with_capacity(indices.len());
        for idx_expr in indices {
            let v = self.eval_expr(idx_expr)?;
            match v {
                Value::Int(i) => idx_vals.push(i as usize),
                _ => {
                    return Err(EvalError::Runtime(format!(
                        "index must be Int, got {}",
                        v.type_name()
                    )));
                }
            }
        }

        match &obj {
            Value::Tensor(t) => Ok(Value::Float(t.get(&idx_vals)?)),
            _ => Err(EvalError::Runtime(format!(
                "multi-index not supported on {}",
                obj.type_name()
            ))),
        }
    }

    // -- Assignment ---------------------------------------------------------

    fn exec_assign(&mut self, target: &Expr, val: Value) -> Result<(), EvalError> {
        match &target.kind {
            ExprKind::Ident(id) => self.assign(&id.name, val),
            ExprKind::Field { object, name } => {
                // Field assignment on a struct.
                if let ExprKind::Ident(obj_id) = &object.kind {
                    let obj_name = obj_id.name.clone();
                    let field_name = name.name.clone();
                    let mut obj_val = self
                        .lookup(&obj_name)
                        .cloned()
                        .ok_or_else(|| {
                            EvalError::Runtime(format!("undefined variable `{obj_name}`"))
                        })?;
                    match &mut obj_val {
                        Value::Struct { fields, .. } => {
                            fields.insert(field_name, val);
                        }
                        _ => {
                            return Err(EvalError::Runtime(format!(
                                "cannot assign field on {}",
                                obj_val.type_name()
                            )));
                        }
                    }
                    self.assign(&obj_name, obj_val)
                } else {
                    Err(EvalError::Runtime(
                        "complex field assignment not supported".to_string(),
                    ))
                }
            }
            ExprKind::Index { object, index } => {
                if let ExprKind::Ident(obj_id) = &object.kind {
                    let obj_name = obj_id.name.clone();
                    let idx = self.eval_expr(index)?;
                    let idx = match idx {
                        Value::Int(i) => i as usize,
                        _ => return Err(EvalError::Runtime("index must be Int".to_string())),
                    };
                    let mut obj_val = self
                        .lookup(&obj_name)
                        .cloned()
                        .ok_or_else(|| {
                            EvalError::Runtime(format!("undefined variable `{obj_name}`"))
                        })?;
                    match &mut obj_val {
                        Value::Array(arr) => {
                            if idx >= arr.len() {
                                return Err(EvalError::Runtime(format!(
                                    "index {idx} out of bounds for array of length {}",
                                    arr.len()
                                )));
                            }
                            Rc::make_mut(arr)[idx] = val;
                        }
                        _ => {
                            return Err(EvalError::Runtime(format!(
                                "cannot index-assign on {}",
                                obj_val.type_name()
                            )));
                        }
                    }
                    self.assign(&obj_name, obj_val)
                } else {
                    Err(EvalError::Runtime(
                        "complex index assignment not supported".to_string(),
                    ))
                }
            }
            _ => Err(EvalError::Runtime(
                "invalid assignment target".to_string(),
            )),
        }
    }

    // -- Pipe operator ------------------------------------------------------

    fn eval_pipe(&mut self, left: &Expr, right: &Expr) -> EvalResult {
        // `a |> f(b, c)` desugars to `f(a, b, c)`.
        let left_val = self.eval_expr(left)?;

        match &right.kind {
            ExprKind::Call { callee, args } => {
                // Insert `left_val` as the first argument.
                let mut new_args = Vec::with_capacity(args.len() + 1);
                new_args.push(left_val);
                for arg in args {
                    new_args.push(self.eval_expr(&arg.value)?);
                }

                match &callee.kind {
                    ExprKind::Ident(id) => self.dispatch_call(&id.name, new_args),
                    ExprKind::Field { object, name } => {
                        if let ExprKind::Ident(obj_id) = &object.kind {
                            let qualified = format!("{}.{}", obj_id.name, name.name);
                            if self.is_known_builtin(&qualified)
                                || self.functions.contains_key(&qualified)
                            {
                                return self.dispatch_call(&qualified, new_args);
                            }
                        }
                        let obj_val = self.eval_expr(object)?;
                        self.dispatch_method(obj_val, &name.name, new_args)
                    }
                    _ => {
                        let callee_val = self.eval_expr(callee)?;
                        match callee_val {
                            Value::Fn(fv) => self.call_function(&fv.name, &new_args),
                            _ => Err(EvalError::Runtime(
                                "pipe target must be a function call".to_string(),
                            )),
                        }
                    }
                }
            }
            ExprKind::Ident(id) => {
                // `a |> f` is shorthand for `f(a)`.
                self.dispatch_call(&id.name, vec![left_val])
            }
            _ => Err(EvalError::Runtime(
                "right side of pipe must be a function call or identifier".to_string(),
            )),
        }
    }

    // -- Value conversion helpers -------------------------------------------

    fn value_to_shape(&self, val: &Value) -> Result<Vec<usize>, EvalError> {
        match val {
            Value::Array(arr) => {
                let mut shape = Vec::with_capacity(arr.len());
                for v in arr.iter() {
                    shape.push(self.value_to_usize(v)?);
                }
                Ok(shape)
            }
            _ => Err(EvalError::Runtime(format!(
                "expected Array for shape, got {}",
                val.type_name()
            ))),
        }
    }

    fn value_to_usize(&self, val: &Value) -> Result<usize, EvalError> {
        match val {
            Value::Int(i) => {
                if *i < 0 {
                    Err(EvalError::Runtime(format!(
                        "expected non-negative integer, got {i}"
                    )))
                } else {
                    Ok(*i as usize)
                }
            }
            _ => Err(EvalError::Runtime(format!(
                "expected Int, got {}",
                val.type_name()
            ))),
        }
    }

    fn value_to_usize_vec(&self, val: &Value) -> Result<Vec<usize>, EvalError> {
        match val {
            Value::Array(arr) => {
                let mut indices = Vec::with_capacity(arr.len());
                for v in arr.iter() {
                    indices.push(self.value_to_usize(v)?);
                }
                Ok(indices)
            }
            _ => Err(EvalError::Runtime(format!(
                "expected Array for indices, got {}",
                val.type_name()
            ))),
        }
    }

    fn value_to_tensor<'a>(&self, val: &'a Value) -> Result<&'a Tensor, EvalError> {
        match val {
            Value::Tensor(t) => Ok(t),
            _ => Err(EvalError::Runtime(format!(
                "expected Tensor, got {}",
                val.type_name()
            ))),
        }
    }

}

// ---------------------------------------------------------------------------
// Phase 8 helpers
// ---------------------------------------------------------------------------

/// Convert a `cjc_data::DataFrame` into a `Value::Struct` with name
/// `"DataFrame"`. Each column becomes a named field:
/// - `Float` column → `Value::Array` of `Value::Float`
/// - `Int`   column → `Value::Array` of `Value::Int`
/// - `Str`   column → `Value::Array` of `Value::String`
/// - `Bool`  column → `Value::Array` of `Value::Bool`
///
/// A special `"__columns"` field holds a `Value::Array` of
/// `Value::String` with the ordered column names (for to_tensor ordering).
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

/// Rebuild a `cjc_data::DataFrame` from a `Value::Struct { name: "DataFrame", fields }`.
///
/// Reads `__columns` for column ordering, then converts each `Value::Array`
/// back into a `Column`. Used by `DataFrame.view()` to bridge the legacy
/// Struct-encoded representation into the typed TidyView layer.
fn rebuild_dataframe_from_struct(
    fields: &std::collections::HashMap<String, Value>,
) -> Result<DataFrame, EvalError> {
    let col_names: Vec<String> = match fields.get("__columns") {
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| {
                if let Value::String(s) = v { Some(s.as_ref().clone()) } else { None }
            })
            .collect(),
        _ => {
            // fallback: use all non-meta keys
            fields
                .keys()
                .filter(|k| !k.starts_with("__"))
                .cloned()
                .collect()
        }
    };
    let mut df_cols: Vec<(String, Column)> = Vec::new();
    for name in &col_names {
        let arr_val = fields.get(name).ok_or_else(|| {
            EvalError::Runtime(format!("column '{}' not found in DataFrame struct", name))
        })?;
        let col = value_array_to_column(arr_val, name)?;
        df_cols.push((name.clone(), col));
    }
    DataFrame::from_columns(df_cols)
        .map_err(|e| EvalError::Runtime(format!("DataFrame rebuild: {e}")))
}

/// Convert a `Value::Array` to a `Column`, inferring the type from the first element.
fn value_array_to_column(v: &Value, col_name: &str) -> Result<Column, EvalError> {
    match v {
        Value::Array(arr) => {
            if arr.is_empty() {
                return Ok(Column::Float(vec![])); // default to float for empty
            }
            match &arr[0] {
                Value::Float(_) | Value::Int(_) => {
                    // Try float first
                    let mut floats = Vec::with_capacity(arr.len());
                    let mut all_int = true;
                    for v in arr.iter() {
                        match v {
                            Value::Float(f) => { floats.push(*f); all_int = false; }
                            Value::Int(i) => { floats.push(*i as f64); }
                            _ => return Err(EvalError::Runtime(format!(
                                "column '{}' has mixed types", col_name
                            ))),
                        }
                    }
                    if all_int {
                        Ok(Column::Int(arr.iter().map(|v| if let Value::Int(i) = v { *i } else { 0 }).collect()))
                    } else {
                        Ok(Column::Float(floats))
                    }
                }
                Value::String(_) => {
                    let strs: Result<Vec<_>, _> = arr.iter().map(|v| match v {
                        Value::String(s) => Ok(s.as_ref().clone()),
                        _ => Err(EvalError::Runtime(format!("column '{}' has mixed types", col_name))),
                    }).collect();
                    Ok(Column::Str(strs?))
                }
                Value::Bool(_) => {
                    let bools: Result<Vec<_>, _> = arr.iter().map(|v| match v {
                        Value::Bool(b) => Ok(*b),
                        _ => Err(EvalError::Runtime(format!("column '{}' has mixed types", col_name))),
                    }).collect();
                    Ok(Column::Bool(bools?))
                }
                other => Err(EvalError::Runtime(format!(
                    "unsupported column element type: {}", other.type_name()
                ))),
            }
        }
        _ => Err(EvalError::Runtime(format!(
            "expected Array for column '{}', got {}", col_name, v.type_name()
        ))),
    }
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new(0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_ast::*;

    // -- Helpers for building AST nodes -------------------------------------

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

    fn float_expr(v: f64) -> Expr {
        Expr {
            kind: ExprKind::FloatLit(v),
            span: span(),
        }
    }

    fn bool_expr(v: bool) -> Expr {
        Expr {
            kind: ExprKind::BoolLit(v),
            span: span(),
        }
    }

    fn string_expr(s: &str) -> Expr {
        Expr {
            kind: ExprKind::StringLit(s.to_string()),
            span: span(),
        }
    }

    fn ident_expr(name: &str) -> Expr {
        Expr {
            kind: ExprKind::Ident(ident(name)),
            span: span(),
        }
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

    fn unary(op: UnaryOp, operand: Expr) -> Expr {
        Expr {
            kind: ExprKind::Unary {
                op,
                operand: Box::new(operand),
            },
            span: span(),
        }
    }

    fn call(callee: Expr, args: Vec<Expr>) -> Expr {
        let call_args: Vec<CallArg> = args
            .into_iter()
            .map(|value| CallArg {
                name: None,
                value,
                span: span(),
            })
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

    fn assign_expr(target: Expr, value: Expr) -> Expr {
        Expr {
            kind: ExprKind::Assign {
                target: Box::new(target),
                value: Box::new(value),
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
        Expr {
            kind: ExprKind::ArrayLit(elems),
            span: span(),
        }
    }

    fn struct_lit(name: &str, fields: Vec<(&str, Expr)>) -> Expr {
        Expr {
            kind: ExprKind::StructLit {
                name: ident(name),
                fields: fields
                    .into_iter()
                    .map(|(n, v)| FieldInit {
                        name: ident(n),
                        value: v,
                        span: span(),
                    })
                    .collect(),
            },
            span: span(),
        }
    }

    fn index_expr(object: Expr, index: Expr) -> Expr {
        Expr {
            kind: ExprKind::Index {
                object: Box::new(object),
                index: Box::new(index),
            },
            span: span(),
        }
    }

    fn multi_index_expr(object: Expr, indices: Vec<Expr>) -> Expr {
        Expr {
            kind: ExprKind::MultiIndex {
                object: Box::new(object),
                indices,
            },
            span: span(),
        }
    }

    fn let_stmt(name: &str, init: Expr) -> Stmt {
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
        Stmt {
            kind: StmtKind::Expr(expr),
            span: span(),
        }
    }

    fn return_stmt(expr: Option<Expr>) -> Stmt {
        Stmt {
            kind: StmtKind::Return(expr),
            span: span(),
        }
    }

    fn dummy_type_expr() -> TypeExpr {
        TypeExpr {
            kind: TypeExprKind::Named {
                name: ident("i64"),
                args: vec![],
            },
            span: span(),
        }
    }

    fn make_param(name: &str) -> Param {
        Param {
            name: ident(name),
            ty: dummy_type_expr(),
            span: span(),
        }
    }

    fn make_fn_decl(name: &str, params: Vec<&str>, body: Block) -> Decl {
        Decl {
            kind: DeclKind::Fn(FnDecl {
                name: ident(name),
                type_params: vec![],
                params: params.into_iter().map(|n| make_param(n)).collect(),
                return_type: None,
                body,
                is_nogc: false,
            }),
            span: span(),
        }
    }

    fn make_block(stmts: Vec<Stmt>, expr: Option<Expr>) -> Block {
        Block {
            stmts,
            expr: expr.map(Box::new),
            span: span(),
        }
    }

    fn make_let_decl(name: &str, init: Expr) -> Decl {
        Decl {
            kind: DeclKind::Let(LetStmt {
                name: ident(name),
                mutable: false,
                ty: None,
                init: Box::new(init),
            }),
            span: span(),
        }
    }

    fn make_struct_decl(name: &str, fields: Vec<&str>) -> Decl {
        Decl {
            kind: DeclKind::Struct(StructDecl {
                name: ident(name),
                type_params: vec![],
                fields: fields
                    .into_iter()
                    .map(|f| FieldDecl {
                        name: ident(f),
                        ty: dummy_type_expr(),
                        default: None,
                        span: span(),
                    })
                    .collect(),
            }),
            span: span(),
        }
    }

    // -- Tests --------------------------------------------------------------

    #[test]
    fn test_basic_arithmetic_int() {
        let mut interp = Interpreter::new(0);
        // 2 + 3
        let result = interp
            .eval_expr(&binary(BinOp::Add, int_expr(2), int_expr(3)))
            .unwrap();
        assert!(matches!(result, Value::Int(5)));

        // 10 - 4
        let result = interp
            .eval_expr(&binary(BinOp::Sub, int_expr(10), int_expr(4)))
            .unwrap();
        assert!(matches!(result, Value::Int(6)));

        // 3 * 7
        let result = interp
            .eval_expr(&binary(BinOp::Mul, int_expr(3), int_expr(7)))
            .unwrap();
        assert!(matches!(result, Value::Int(21)));

        // 15 / 4
        let result = interp
            .eval_expr(&binary(BinOp::Div, int_expr(15), int_expr(4)))
            .unwrap();
        assert!(matches!(result, Value::Int(3)));

        // 17 % 5
        let result = interp
            .eval_expr(&binary(BinOp::Mod, int_expr(17), int_expr(5)))
            .unwrap();
        assert!(matches!(result, Value::Int(2)));
    }

    #[test]
    fn test_basic_arithmetic_float() {
        let mut interp = Interpreter::new(0);
        let result = interp
            .eval_expr(&binary(BinOp::Add, float_expr(1.5), float_expr(2.5)))
            .unwrap();
        match result {
            Value::Float(v) => assert!((v - 4.0).abs() < 1e-12),
            _ => panic!("expected Float"),
        }

        let result = interp
            .eval_expr(&binary(BinOp::Mul, float_expr(3.0), float_expr(2.0)))
            .unwrap();
        match result {
            Value::Float(v) => assert!((v - 6.0).abs() < 1e-12),
            _ => panic!("expected Float"),
        }
    }

    #[test]
    fn test_comparison_operators() {
        let mut interp = Interpreter::new(0);

        let result = interp
            .eval_expr(&binary(BinOp::Lt, int_expr(3), int_expr(5)))
            .unwrap();
        assert!(matches!(result, Value::Bool(true)));

        let result = interp
            .eval_expr(&binary(BinOp::Ge, int_expr(3), int_expr(5)))
            .unwrap();
        assert!(matches!(result, Value::Bool(false)));

        let result = interp
            .eval_expr(&binary(BinOp::Eq, int_expr(7), int_expr(7)))
            .unwrap();
        assert!(matches!(result, Value::Bool(true)));

        let result = interp
            .eval_expr(&binary(BinOp::Ne, int_expr(7), int_expr(8)))
            .unwrap();
        assert!(matches!(result, Value::Bool(true)));
    }

    #[test]
    fn test_boolean_logic() {
        let mut interp = Interpreter::new(0);

        let result = interp
            .eval_expr(&binary(BinOp::And, bool_expr(true), bool_expr(false)))
            .unwrap();
        assert!(matches!(result, Value::Bool(false)));

        let result = interp
            .eval_expr(&binary(BinOp::Or, bool_expr(false), bool_expr(true)))
            .unwrap();
        assert!(matches!(result, Value::Bool(true)));
    }

    #[test]
    fn test_unary_operators() {
        let mut interp = Interpreter::new(0);

        let result = interp
            .eval_expr(&unary(UnaryOp::Neg, int_expr(42)))
            .unwrap();
        assert!(matches!(result, Value::Int(-42)));

        let result = interp
            .eval_expr(&unary(UnaryOp::Not, bool_expr(true)))
            .unwrap();
        assert!(matches!(result, Value::Bool(false)));
    }

    #[test]
    fn test_function_call() {
        // fn add(a, b) { a + b }
        // add(3, 4)  => 7
        let func_decl = make_fn_decl(
            "add",
            vec!["a", "b"],
            make_block(
                vec![],
                Some(binary(BinOp::Add, ident_expr("a"), ident_expr("b"))),
            ),
        );

        let program = Program {
            declarations: vec![
                func_decl,
                make_fn_decl(
                    "main",
                    vec![],
                    make_block(vec![], Some(call(ident_expr("add"), vec![int_expr(3), int_expr(4)]))),
                ),
            ],
        };

        let mut interp = Interpreter::new(0);
        let result = interp.exec(&program).unwrap();
        assert!(matches!(result, Value::Int(7)));
    }

    #[test]
    fn test_recursive_function() {
        // fn factorial(n) {
        //     if n <= 1 { return 1; }
        //     return n * factorial(n - 1);
        // }
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

        let mut interp = Interpreter::new(0);
        let result = interp.exec(&program).unwrap();
        assert!(matches!(result, Value::Int(120)));
    }

    #[test]
    fn test_if_else() {
        let mut interp = Interpreter::new(0);

        let if_stmt = IfStmt {
            condition: bool_expr(true),
            then_block: make_block(vec![], Some(int_expr(42))),
            else_branch: Some(ElseBranch::Else(make_block(vec![], Some(int_expr(99))))),
        };

        let result = interp.exec_if(&if_stmt).unwrap();
        assert!(matches!(result, Value::Int(42)));

        let if_stmt_false = IfStmt {
            condition: bool_expr(false),
            then_block: make_block(vec![], Some(int_expr(42))),
            else_branch: Some(ElseBranch::Else(make_block(vec![], Some(int_expr(99))))),
        };

        let result = interp.exec_if(&if_stmt_false).unwrap();
        assert!(matches!(result, Value::Int(99)));
    }

    #[test]
    fn test_while_loop() {
        // let mut i = 0;
        // let mut sum = 0;
        // while i < 5 { sum = sum + i; i = i + 1; }
        // sum  => 0+1+2+3+4 = 10
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

        let mut interp = Interpreter::new(0);
        let result = interp.exec(&program).unwrap();
        assert!(matches!(result, Value::Int(10)));
    }

    #[test]
    fn test_struct_creation_and_field_access() {
        // struct Point { x, y }
        // let p = Point { x: 10, y: 20 };
        // p.x + p.y  => 30
        let program = Program {
            declarations: vec![
                make_struct_decl("Point", vec!["x", "y"]),
                make_fn_decl(
                    "main",
                    vec![],
                    make_block(
                        vec![let_stmt(
                            "p",
                            struct_lit("Point", vec![("x", int_expr(10)), ("y", int_expr(20))]),
                        )],
                        Some(binary(
                            BinOp::Add,
                            field_expr(ident_expr("p"), "x"),
                            field_expr(ident_expr("p"), "y"),
                        )),
                    ),
                ),
            ],
        };

        let mut interp = Interpreter::new(0);
        let result = interp.exec(&program).unwrap();
        assert!(matches!(result, Value::Int(30)));
    }

    #[test]
    fn test_tensor_operations() {
        let mut interp = Interpreter::new(42);

        // Tensor.zeros([2, 3])
        let result = interp
            .eval_expr(&call(
                field_expr(ident_expr("Tensor"), "zeros"),
                vec![array_expr(vec![int_expr(2), int_expr(3)])],
            ))
            .unwrap();
        match &result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[2, 3]);
                assert_eq!(t.len(), 6);
                assert!((t.sum() - 0.0).abs() < 1e-12);
            }
            _ => panic!("expected Tensor"),
        }

        // Tensor.ones([3])
        let result = interp
            .eval_expr(&call(
                field_expr(ident_expr("Tensor"), "ones"),
                vec![array_expr(vec![int_expr(3)])],
            ))
            .unwrap();
        match &result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[3]);
                assert!((t.sum() - 3.0).abs() < 1e-12);
            }
            _ => panic!("expected Tensor"),
        }

        // Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2])
        let result = interp
            .eval_expr(&call(
                field_expr(ident_expr("Tensor"), "from_vec"),
                vec![
                    array_expr(vec![
                        float_expr(1.0),
                        float_expr(2.0),
                        float_expr(3.0),
                        float_expr(4.0),
                    ]),
                    array_expr(vec![int_expr(2), int_expr(2)]),
                ],
            ))
            .unwrap();
        match &result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[2, 2]);
                assert!((t.get(&[0, 0]).unwrap() - 1.0).abs() < 1e-12);
                assert!((t.get(&[1, 1]).unwrap() - 4.0).abs() < 1e-12);
            }
            _ => panic!("expected Tensor"),
        }
    }

    #[test]
    fn test_tensor_arithmetic() {
        let mut interp = Interpreter::new(0);

        // Create two tensors and add them.
        interp.define("a", Value::Tensor(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(),
        ));
        interp.define("b", Value::Tensor(
            Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap(),
        ));

        let result = interp
            .eval_expr(&binary(BinOp::Add, ident_expr("a"), ident_expr("b")))
            .unwrap();
        match &result {
            Value::Tensor(t) => {
                assert_eq!(t.to_vec(), vec![5.0, 7.0, 9.0]);
            }
            _ => panic!("expected Tensor"),
        }

        let result = interp
            .eval_expr(&binary(BinOp::Mul, ident_expr("a"), ident_expr("b")))
            .unwrap();
        match &result {
            Value::Tensor(t) => {
                assert_eq!(t.to_vec(), vec![4.0, 10.0, 18.0]);
            }
            _ => panic!("expected Tensor"),
        }
    }

    #[test]
    fn test_matmul() {
        let mut interp = Interpreter::new(0);

        interp.define("a", Value::Tensor(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap(),
        ));
        interp.define("b", Value::Tensor(
            Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap(),
        ));

        let result = interp
            .eval_expr(&call(
                ident_expr("matmul"),
                vec![ident_expr("a"), ident_expr("b")],
            ))
            .unwrap();

        match &result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[2, 2]);
                assert!((t.get(&[0, 0]).unwrap() - 19.0).abs() < 1e-12);
                assert!((t.get(&[0, 1]).unwrap() - 22.0).abs() < 1e-12);
                assert!((t.get(&[1, 0]).unwrap() - 43.0).abs() < 1e-12);
                assert!((t.get(&[1, 1]).unwrap() - 50.0).abs() < 1e-12);
            }
            _ => panic!("expected Tensor"),
        }
    }

    #[test]
    fn test_pipe_operator() {
        // fn double(x) { x * 2 }
        // fn add_one(x) { x + 1 }
        // 5 |> double() |> add_one()  => 11
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
                    "add_one",
                    vec!["x"],
                    make_block(
                        vec![],
                        Some(binary(BinOp::Add, ident_expr("x"), int_expr(1))),
                    ),
                ),
                make_fn_decl(
                    "main",
                    vec![],
                    make_block(
                        vec![],
                        Some(pipe_expr(
                            pipe_expr(
                                int_expr(5),
                                call(ident_expr("double"), vec![]),
                            ),
                            call(ident_expr("add_one"), vec![]),
                        )),
                    ),
                ),
            ],
        };

        let mut interp = Interpreter::new(0);
        let result = interp.exec(&program).unwrap();
        assert!(matches!(result, Value::Int(11)));
    }

    #[test]
    fn test_pipe_with_extra_args() {
        // fn add(a, b) { a + b }
        // 10 |> add(5)  => 15
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
                        Some(pipe_expr(
                            int_expr(10),
                            call(ident_expr("add"), vec![int_expr(5)]),
                        )),
                    ),
                ),
            ],
        };

        let mut interp = Interpreter::new(0);
        let result = interp.exec(&program).unwrap();
        assert!(matches!(result, Value::Int(15)));
    }

    #[test]
    fn test_print_builtin() {
        let mut interp = Interpreter::new(0);

        interp
            .eval_expr(&call(
                ident_expr("print"),
                vec![string_expr("hello"), int_expr(42)],
            ))
            .unwrap();

        assert_eq!(interp.output.len(), 1);
        assert_eq!(interp.output[0], "hello 42");
    }

    #[test]
    fn test_array_literal_and_indexing() {
        let mut interp = Interpreter::new(0);

        interp.define(
            "arr",
            Value::Array(Rc::new(vec![Value::Int(10), Value::Int(20), Value::Int(30)])),
        );

        let result = interp
            .eval_expr(&index_expr(ident_expr("arr"), int_expr(1)))
            .unwrap();
        assert!(matches!(result, Value::Int(20)));
    }

    #[test]
    fn test_tensor_multi_index() {
        let mut interp = Interpreter::new(0);

        interp.define(
            "t",
            Value::Tensor(
                Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap(),
            ),
        );

        // t[1, 2] => 6.0
        let result = interp
            .eval_expr(&multi_index_expr(
                ident_expr("t"),
                vec![int_expr(1), int_expr(2)],
            ))
            .unwrap();
        match result {
            Value::Float(v) => assert!((v - 6.0).abs() < 1e-12),
            _ => panic!("expected Float"),
        }
    }

    #[test]
    fn test_variable_assignment() {
        let mut interp = Interpreter::new(0);

        let program = Program {
            declarations: vec![make_fn_decl(
                "main",
                vec![],
                make_block(
                    vec![
                        let_mut_stmt("x", int_expr(10)),
                        expr_stmt(assign_expr(ident_expr("x"), int_expr(20))),
                    ],
                    Some(ident_expr("x")),
                ),
            )],
        };

        let result = interp.exec(&program).unwrap();
        assert!(matches!(result, Value::Int(20)));
    }

    #[test]
    fn test_nested_scopes() {
        let mut interp = Interpreter::new(0);

        let program = Program {
            declarations: vec![make_fn_decl(
                "main",
                vec![],
                make_block(
                    vec![
                        let_stmt("x", int_expr(1)),
                        let_stmt(
                            "y",
                            Expr {
                                kind: ExprKind::Block(make_block(
                                    vec![let_stmt("x", int_expr(99))],
                                    Some(ident_expr("x")),
                                )),
                                span: span(),
                            },
                        ),
                    ],
                    // x should still be 1 (inner x was in its own scope).
                    Some(binary(BinOp::Add, ident_expr("x"), ident_expr("y"))),
                ),
            )],
        };

        let result = interp.exec(&program).unwrap();
        // x=1 (outer), y=99 (from inner block), 1+99=100
        assert!(matches!(result, Value::Int(100)));
    }

    #[test]
    fn test_early_return() {
        // fn early(x) {
        //     if x > 0 { return x; }
        //     return 0;
        // }
        let func = make_fn_decl(
            "early",
            vec!["x"],
            make_block(
                vec![
                    Stmt {
                        kind: StmtKind::If(IfStmt {
                            condition: binary(BinOp::Gt, ident_expr("x"), int_expr(0)),
                            then_block: make_block(
                                vec![return_stmt(Some(ident_expr("x")))],
                                None,
                            ),
                            else_branch: None,
                        }),
                        span: span(),
                    },
                    return_stmt(Some(int_expr(0))),
                ],
                None,
            ),
        );

        let program = Program {
            declarations: vec![
                func,
                make_fn_decl(
                    "main",
                    vec![],
                    make_block(
                        vec![],
                        Some(binary(
                            BinOp::Add,
                            call(ident_expr("early"), vec![int_expr(5)]),
                            call(ident_expr("early"), vec![int_expr(-3)]),
                        )),
                    ),
                ),
            ],
        };

        let mut interp = Interpreter::new(0);
        let result = interp.exec(&program).unwrap();
        // early(5)=5, early(-3)=0, 5+0=5
        assert!(matches!(result, Value::Int(5)));
    }

    #[test]
    fn test_string_concatenation() {
        let mut interp = Interpreter::new(0);

        let result = interp
            .eval_expr(&binary(
                BinOp::Add,
                string_expr("hello "),
                string_expr("world"),
            ))
            .unwrap();
        match result {
            Value::String(s) => assert_eq!(s.as_str(), "hello world"),
            _ => panic!("expected String"),
        }
    }

    #[test]
    fn test_division_by_zero() {
        let mut interp = Interpreter::new(0);
        let result = interp.eval_expr(&binary(BinOp::Div, int_expr(10), int_expr(0)));
        assert!(result.is_err());
    }

    #[test]
    fn test_undefined_variable() {
        let mut interp = Interpreter::new(0);
        let result = interp.eval_expr(&ident_expr("nonexistent"));
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_randn_deterministic() {
        // Two interpreters with the same seed should produce identical random tensors.
        let mut interp1 = Interpreter::new(42);
        let mut interp2 = Interpreter::new(42);

        let create_randn = call(
            field_expr(ident_expr("Tensor"), "randn"),
            vec![array_expr(vec![int_expr(3), int_expr(4)])],
        );

        let t1 = interp1.eval_expr(&create_randn).unwrap();
        let t2 = interp2.eval_expr(&create_randn).unwrap();

        match (&t1, &t2) {
            (Value::Tensor(a), Value::Tensor(b)) => {
                assert_eq!(a.to_vec(), b.to_vec());
            }
            _ => panic!("expected Tensors"),
        }
    }

    #[test]
    fn test_if_else_chain() {
        // if false { 1 } else if false { 2 } else { 3 }  => 3
        let mut interp = Interpreter::new(0);

        let if_stmt = IfStmt {
            condition: bool_expr(false),
            then_block: make_block(vec![], Some(int_expr(1))),
            else_branch: Some(ElseBranch::ElseIf(Box::new(IfStmt {
                condition: bool_expr(false),
                then_block: make_block(vec![], Some(int_expr(2))),
                else_branch: Some(ElseBranch::Else(make_block(vec![], Some(int_expr(3))))),
            }))),
        };

        let result = interp.exec_if(&if_stmt).unwrap();
        assert!(matches!(result, Value::Int(3)));
    }

    #[test]
    fn test_struct_field_assignment() {
        let mut interp = Interpreter::new(0);

        let program = Program {
            declarations: vec![
                make_struct_decl("Point", vec!["x", "y"]),
                make_fn_decl(
                    "main",
                    vec![],
                    make_block(
                        vec![
                            let_mut_stmt(
                                "p",
                                struct_lit(
                                    "Point",
                                    vec![("x", int_expr(10)), ("y", int_expr(20))],
                                ),
                            ),
                            expr_stmt(assign_expr(
                                field_expr(ident_expr("p"), "x"),
                                int_expr(42),
                            )),
                        ],
                        Some(field_expr(ident_expr("p"), "x")),
                    ),
                ),
            ],
        };

        let result = interp.exec(&program).unwrap();
        assert!(matches!(result, Value::Int(42)));
    }

    #[test]
    fn test_mixed_int_float_arithmetic() {
        let mut interp = Interpreter::new(0);

        // Int + Float should promote to Float.
        let result = interp
            .eval_expr(&binary(BinOp::Add, int_expr(2), float_expr(3.5)))
            .unwrap();
        match result {
            Value::Float(v) => assert!((v - 5.5).abs() < 1e-12),
            _ => panic!("expected Float"),
        }
    }
}
