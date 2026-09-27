//! AST Visitor — mutating traversal
//!
//! [`AstVisitorMut`] mirrors [`crate::visit::AstVisitor`] with `&mut`
//! access: same `visit_*` hooks, same `walk_*` functions, same left-to-right,
//! depth-first order over every node kind. It is a separate trait so the
//! read-only visitor and its implementors are unchanged. Used by
//! `cjc-module` to rewrite names during module resolution.

use crate::{
    Block, CallArg, ConstDecl, Decl, DeclKind, Decorator, ElseBranch, EnumDecl, Expr, ExprKind,
    FieldDecl, FieldInit, FnDecl, FnSig, ForIter, ForStmt, Ident, IfStmt, ImplDecl, ImportDecl,
    LetStmt, MatchArm, Param, Pattern, PatternKind, PatternField, Program, RecordDecl, ShapeDim,
    Stmt, StmtKind, StructDecl, TraitDecl, TypeArg, TypeExpr, TypeExprKind, TypeParam,
    VariantDecl, WhileStmt, ClassDecl,
};

// ---------------------------------------------------------------------------
// Visitor trait
// ---------------------------------------------------------------------------

/// Mutating AST visitor.  Override specific `visit_*` methods; call the
/// corresponding `walk_*` function inside your override to continue the
/// default traversal into children.
pub trait AstVisitorMut: Sized {
    /// Visit a top-level [`Program`]. Default: delegates to [`walk_program`].
    fn visit_program(&mut self, program: &mut Program) {
        walk_program(self, program);
    }
    /// Visit a [`Decl`] node. Default: delegates to [`walk_decl`].
    fn visit_decl(&mut self, decl: &mut Decl) {
        walk_decl(self, decl);
    }
    /// Visit a [`Stmt`] node. Default: delegates to [`walk_stmt`].
    fn visit_stmt(&mut self, stmt: &mut Stmt) {
        walk_stmt(self, stmt);
    }
    /// Visit an [`Expr`] node. Default: delegates to [`walk_expr`].
    fn visit_expr(&mut self, expr: &mut Expr) {
        walk_expr(self, expr);
    }
    /// Visit a [`Pattern`] node. Default: delegates to [`walk_pattern`].
    fn visit_pattern(&mut self, pattern: &mut Pattern) {
        walk_pattern(self, pattern);
    }
    /// Visit a [`TypeExpr`] node. Default: delegates to [`walk_type_expr`].
    fn visit_type_expr(&mut self, ty: &mut TypeExpr) {
        walk_type_expr(self, ty);
    }
    /// Visit a [`Block`]. Default: delegates to [`walk_block`].
    fn visit_block(&mut self, block: &mut Block) {
        walk_block(self, block);
    }
    /// Visit a [`FnDecl`] node. Default: delegates to [`walk_fn_decl`].
    fn visit_fn_decl(&mut self, f: &mut FnDecl) {
        walk_fn_decl(self, f);
    }
    /// Visit an [`Ident`]. Default: no-op (leaf node).
    fn visit_ident(&mut self, _ident: &mut Ident) {}
    /// Visit a function [`Param`]. Default: delegates to [`walk_param`].
    fn visit_param(&mut self, param: &mut Param) {
        walk_param(self, param);
    }
    /// Visit a [`MatchArm`]. Default: delegates to [`walk_match_arm`].
    fn visit_match_arm(&mut self, arm: &mut MatchArm) {
        walk_match_arm(self, arm);
    }
}

// ---------------------------------------------------------------------------
// Walk functions — default traversals
// ---------------------------------------------------------------------------

/// Walk a [`Program`] by visiting each declaration in order.
pub fn walk_program<V: AstVisitorMut>(v: &mut V, program: &mut Program) {
    for decl in &mut program.declarations {
        v.visit_decl(decl);
    }
}

/// Walk a [`Decl`] by dispatching on its [`DeclKind`] variant.
pub fn walk_decl<V: AstVisitorMut>(v: &mut V, decl: &mut Decl) {
    match &mut decl.kind {
        DeclKind::Struct(s) => walk_struct_decl(v, s),
        DeclKind::Class(c) => walk_class_decl(v, c),
        DeclKind::Record(r) => walk_record_decl(v, r),
        DeclKind::Fn(f) => v.visit_fn_decl(f),
        DeclKind::Trait(t) => walk_trait_decl(v, t),
        DeclKind::Impl(i) => walk_impl_decl(v, i),
        DeclKind::Enum(e) => walk_enum_decl(v, e),
        DeclKind::Let(l) => walk_let_stmt(v, l),
        DeclKind::Import(i) => walk_import_decl(v, i),
        DeclKind::Const(c) => walk_const_decl(v, c),
        DeclKind::Stmt(s) => v.visit_stmt(s),
    }
}

/// Walk a [`StructDecl`]: visit name, type params, and fields.
pub fn walk_struct_decl<V: AstVisitorMut>(v: &mut V, s: &mut StructDecl) {
    v.visit_ident(&mut s.name);
    for tp in &mut s.type_params {
        walk_type_param(v, tp);
    }
    for field in &mut s.fields {
        walk_field_decl(v, field);
    }
}

/// Walk a [`ClassDecl`]: visit name, type params, and fields.
pub fn walk_class_decl<V: AstVisitorMut>(v: &mut V, c: &mut ClassDecl) {
    v.visit_ident(&mut c.name);
    for tp in &mut c.type_params {
        walk_type_param(v, tp);
    }
    for field in &mut c.fields {
        walk_field_decl(v, field);
    }
}

/// Walk a [`RecordDecl`]: visit name, type params, and fields.
pub fn walk_record_decl<V: AstVisitorMut>(v: &mut V, r: &mut RecordDecl) {
    v.visit_ident(&mut r.name);
    for tp in &mut r.type_params {
        walk_type_param(v, tp);
    }
    for field in &mut r.fields {
        walk_field_decl(v, field);
    }
}

/// Walk an [`EnumDecl`]: visit name, type params, and each variant.
pub fn walk_enum_decl<V: AstVisitorMut>(v: &mut V, e: &mut EnumDecl) {
    v.visit_ident(&mut e.name);
    for tp in &mut e.type_params {
        walk_type_param(v, tp);
    }
    for variant in &mut e.variants {
        walk_variant_decl(v, variant);
    }
}

/// Walk a [`VariantDecl`]: visit name and payload type expressions.
pub fn walk_variant_decl<V: AstVisitorMut>(v: &mut V, variant: &mut VariantDecl) {
    v.visit_ident(&mut variant.name);
    for ty in &mut variant.fields {
        v.visit_type_expr(ty);
    }
}

/// Walk a [`FieldDecl`]: visit name, type, and optional default expression.
pub fn walk_field_decl<V: AstVisitorMut>(v: &mut V, field: &mut FieldDecl) {
    v.visit_ident(&mut field.name);
    v.visit_type_expr(&mut field.ty);
    if let Some(ref mut default) = field.default {
        v.visit_expr(default);
    }
}

/// Walk a [`TraitDecl`]: visit name, type params, super-traits, and method signatures.
pub fn walk_trait_decl<V: AstVisitorMut>(v: &mut V, t: &mut TraitDecl) {
    v.visit_ident(&mut t.name);
    for tp in &mut t.type_params {
        walk_type_param(v, tp);
    }
    for st in &mut t.super_traits {
        v.visit_type_expr(st);
    }
    for method in &mut t.methods {
        walk_fn_sig(v, method);
    }
}

/// Walk an [`ImplDecl`]: visit type params, target type, optional trait ref, and methods.
pub fn walk_impl_decl<V: AstVisitorMut>(v: &mut V, i: &mut ImplDecl) {
    for tp in &mut i.type_params {
        walk_type_param(v, tp);
    }
    v.visit_type_expr(&mut i.target);
    if let Some(ref mut tr) = i.trait_ref {
        v.visit_type_expr(tr);
    }
    for method in &mut i.methods {
        v.visit_fn_decl(method);
    }
}

/// Walk an [`ImportDecl`]: visit path segments and optional alias.
pub fn walk_import_decl<V: AstVisitorMut>(v: &mut V, i: &mut ImportDecl) {
    for ident in &mut i.path {
        v.visit_ident(ident);
    }
    if let Some(ref mut alias) = i.alias {
        v.visit_ident(alias);
    }
}

/// Walk a [`ConstDecl`]: visit name, type, and value expression.
pub fn walk_const_decl<V: AstVisitorMut>(v: &mut V, c: &mut ConstDecl) {
    v.visit_ident(&mut c.name);
    v.visit_type_expr(&mut c.ty);
    v.visit_expr(&mut c.value);
}

/// Walk a [`FnDecl`]: visit name, type params, params, return type, decorators, and body.
pub fn walk_fn_decl<V: AstVisitorMut>(v: &mut V, f: &mut FnDecl) {
    v.visit_ident(&mut f.name);
    for tp in &mut f.type_params {
        walk_type_param(v, tp);
    }
    for param in &mut f.params {
        v.visit_param(param);
    }
    if let Some(ref mut ret) = f.return_type {
        v.visit_type_expr(ret);
    }
    for dec in &mut f.decorators {
        walk_decorator(v, dec);
    }
    v.visit_block(&mut f.body);
}

/// Walk a [`FnSig`]: visit name, type params, params, and return type.
pub fn walk_fn_sig<V: AstVisitorMut>(v: &mut V, sig: &mut FnSig) {
    v.visit_ident(&mut sig.name);
    for tp in &mut sig.type_params {
        walk_type_param(v, tp);
    }
    for param in &mut sig.params {
        v.visit_param(param);
    }
    if let Some(ref mut ret) = sig.return_type {
        v.visit_type_expr(ret);
    }
}

/// Walk a [`Param`]: visit name, type, and optional default expression.
pub fn walk_param<V: AstVisitorMut>(v: &mut V, param: &mut Param) {
    v.visit_ident(&mut param.name);
    v.visit_type_expr(&mut param.ty);
    if let Some(ref mut default) = param.default {
        v.visit_expr(default);
    }
}

/// Walk a [`Decorator`]: visit name and argument expressions.
pub fn walk_decorator<V: AstVisitorMut>(v: &mut V, dec: &mut Decorator) {
    v.visit_ident(&mut dec.name);
    for arg in &mut dec.args {
        v.visit_expr(arg);
    }
}

/// Walk a [`TypeParam`]: visit name and trait bounds.
pub fn walk_type_param<V: AstVisitorMut>(v: &mut V, tp: &mut TypeParam) {
    v.visit_ident(&mut tp.name);
    for bound in &mut tp.bounds {
        v.visit_type_expr(bound);
    }
}

// ---------------------------------------------------------------------------
// Block & Statement walkers
// ---------------------------------------------------------------------------

/// Walk a [`Block`]: visit each statement and the optional trailing expression.
pub fn walk_block<V: AstVisitorMut>(v: &mut V, block: &mut Block) {
    for stmt in &mut block.stmts {
        v.visit_stmt(stmt);
    }
    if let Some(ref mut expr) = block.expr {
        v.visit_expr(expr);
    }
}

/// Walk a [`Stmt`] by dispatching on its [`StmtKind`] variant.
pub fn walk_stmt<V: AstVisitorMut>(v: &mut V, stmt: &mut Stmt) {
    match &mut stmt.kind {
        StmtKind::Let(l) => walk_let_stmt(v, l),
        StmtKind::Expr(e) => v.visit_expr(e),
        StmtKind::Return(Some(e)) => v.visit_expr(e),
        StmtKind::Return(None) => {}
        StmtKind::Break => {}
        StmtKind::Continue => {}
        StmtKind::If(i) => walk_if_stmt(v, i),
        StmtKind::While(w) => walk_while_stmt(v, w),
        StmtKind::For(f) => walk_for_stmt(v, f),
        StmtKind::NoGcBlock(b) => v.visit_block(b),
    }
}

/// Walk a [`LetStmt`]: visit name, optional type annotation, and initializer.
pub fn walk_let_stmt<V: AstVisitorMut>(v: &mut V, l: &mut LetStmt) {
    v.visit_ident(&mut l.name);
    if let Some(ref mut ty) = l.ty {
        v.visit_type_expr(ty);
    }
    v.visit_expr(&mut l.init);
}

/// Walk an [`IfStmt`]: visit condition, then-block, and optional else branch.
pub fn walk_if_stmt<V: AstVisitorMut>(v: &mut V, i: &mut IfStmt) {
    v.visit_expr(&mut i.condition);
    v.visit_block(&mut i.then_block);
    if let Some(ref mut else_branch) = i.else_branch {
        walk_else_branch(v, else_branch);
    }
}

/// Walk an [`ElseBranch`]: dispatch to either an else-if or a terminal else block.
pub fn walk_else_branch<V: AstVisitorMut>(v: &mut V, eb: &mut ElseBranch) {
    match eb {
        ElseBranch::ElseIf(elif) => walk_if_stmt(v, elif),
        ElseBranch::Else(block) => v.visit_block(block),
    }
}

/// Walk a [`WhileStmt`]: visit condition and body block.
pub fn walk_while_stmt<V: AstVisitorMut>(v: &mut V, w: &mut WhileStmt) {
    v.visit_expr(&mut w.condition);
    v.visit_block(&mut w.body);
}

/// Walk a [`ForStmt`]: visit loop variable, iterator (range or expr), and body.
pub fn walk_for_stmt<V: AstVisitorMut>(v: &mut V, f: &mut ForStmt) {
    v.visit_ident(&mut f.ident);
    match &mut f.iter {
        ForIter::Range { start, end } => {
            v.visit_expr(start);
            v.visit_expr(end);
        }
        ForIter::Expr(e) => v.visit_expr(e),
    }
    v.visit_block(&mut f.body);
}

// ---------------------------------------------------------------------------
// Expression walker — covers all 35 ExprKind variants
// ---------------------------------------------------------------------------

/// Walk an [`Expr`] by dispatching on all [`ExprKind`] variants.
///
/// Visits child expressions, identifiers, blocks, and patterns in
/// left-to-right, depth-first order.
pub fn walk_expr<V: AstVisitorMut>(v: &mut V, expr: &mut Expr) {
    match &mut expr.kind {
        ExprKind::IntLit(_) => {}
        ExprKind::FloatLit(_) => {}
        ExprKind::StringLit(_) => {}
        ExprKind::ByteStringLit(_) => {}
        ExprKind::ByteCharLit(_) => {}
        ExprKind::RawStringLit(_) => {}
        ExprKind::RawByteStringLit(_) => {}
        ExprKind::FStringLit(segments) => {
            for (_lit, maybe_expr) in segments {
                if let Some(ref mut e) = maybe_expr {
                    v.visit_expr(e);
                }
            }
        }
        ExprKind::RegexLit { .. } => {}
        ExprKind::TensorLit { rows } => {
            for row in rows {
                for elem in row {
                    v.visit_expr(elem);
                }
            }
        }
        ExprKind::BoolLit(_) => {}
        ExprKind::NaLit => {}
        ExprKind::Ident(ident) => v.visit_ident(ident),
        ExprKind::Binary { left, right, .. } => {
            v.visit_expr(left);
            v.visit_expr(right);
        }
        ExprKind::Unary { operand, .. } => {
            v.visit_expr(operand);
        }
        ExprKind::Call { callee, args } => {
            v.visit_expr(callee);
            for arg in args {
                walk_call_arg(v, arg);
            }
        }
        ExprKind::Field { object, name } => {
            v.visit_expr(object);
            v.visit_ident(name);
        }
        ExprKind::Index { object, index } => {
            v.visit_expr(object);
            v.visit_expr(index);
        }
        ExprKind::MultiIndex { object, indices } => {
            v.visit_expr(object);
            for idx in indices {
                v.visit_expr(idx);
            }
        }
        ExprKind::Assign { target, value } => {
            v.visit_expr(target);
            v.visit_expr(value);
        }
        ExprKind::CompoundAssign { target, value, .. } => {
            v.visit_expr(target);
            v.visit_expr(value);
        }
        ExprKind::IfExpr {
            condition,
            then_block,
            else_branch,
        } => {
            v.visit_expr(condition);
            v.visit_block(then_block);
            if let Some(ref mut eb) = else_branch {
                walk_else_branch(v, eb);
            }
        }
        ExprKind::Pipe { left, right } => {
            v.visit_expr(left);
            v.visit_expr(right);
        }
        ExprKind::Block(block) => v.visit_block(block),
        ExprKind::StructLit { name, fields } => {
            v.visit_ident(name);
            for field in fields {
                walk_field_init(v, field);
            }
        }
        ExprKind::ArrayLit(elems) => {
            for elem in elems {
                v.visit_expr(elem);
            }
        }
        ExprKind::Col(_) => {}
        ExprKind::Lambda { params, body } => {
            for param in params {
                v.visit_param(param);
            }
            v.visit_expr(body);
        }
        ExprKind::Match { scrutinee, arms } => {
            v.visit_expr(scrutinee);
            for arm in arms {
                v.visit_match_arm(arm);
            }
        }
        ExprKind::TupleLit(elems) => {
            for elem in elems {
                v.visit_expr(elem);
            }
        }
        ExprKind::Try(e) => v.visit_expr(e),
        ExprKind::VariantLit {
            enum_name,
            variant,
            fields,
        } => {
            if let Some(ref mut en) = enum_name {
                v.visit_ident(en);
            }
            v.visit_ident(variant);
            for field in fields {
                v.visit_expr(field);
            }
        }
        ExprKind::Cast { expr, target_type } => {
            v.visit_expr(expr);
            v.visit_ident(target_type);
        }
    }
}

/// Walk a [`CallArg`]: visit optional name and value expression.
pub fn walk_call_arg<V: AstVisitorMut>(v: &mut V, arg: &mut CallArg) {
    if let Some(ref mut name) = arg.name {
        v.visit_ident(name);
    }
    v.visit_expr(&mut arg.value);
}

/// Walk a [`FieldInit`]: visit field name and value expression.
pub fn walk_field_init<V: AstVisitorMut>(v: &mut V, fi: &mut FieldInit) {
    v.visit_ident(&mut fi.name);
    v.visit_expr(&mut fi.value);
}

/// Walk a [`MatchArm`]: visit pattern and body expression.
pub fn walk_match_arm<V: AstVisitorMut>(v: &mut V, arm: &mut MatchArm) {
    v.visit_pattern(&mut arm.pattern);
    v.visit_expr(&mut arm.body);
}

// ---------------------------------------------------------------------------
// Pattern walker — covers all 9 PatternKind variants
// ---------------------------------------------------------------------------

/// Walk a [`Pattern`] by dispatching on all [`PatternKind`] variants.
///
/// Visits nested patterns, identifiers, and fields in declaration order.
pub fn walk_pattern<V: AstVisitorMut>(v: &mut V, pattern: &mut Pattern) {
    match &mut pattern.kind {
        PatternKind::Wildcard => {}
        PatternKind::Binding(ident) => v.visit_ident(ident),
        PatternKind::LitInt(_) => {}
        PatternKind::LitFloat(_) => {}
        PatternKind::LitBool(_) => {}
        PatternKind::LitString(_) => {}
        PatternKind::Tuple(pats) => {
            for pat in pats {
                v.visit_pattern(pat);
            }
        }
        PatternKind::Struct { name, fields } => {
            v.visit_ident(name);
            for field in fields {
                walk_pattern_field(v, field);
            }
        }
        PatternKind::Variant {
            enum_name,
            variant,
            fields,
        } => {
            if let Some(ref mut en) = enum_name {
                v.visit_ident(en);
            }
            v.visit_ident(variant);
            for field in fields {
                v.visit_pattern(field);
            }
        }
    }
}

/// Walk a [`PatternField`]: visit field name and optional nested pattern.
pub fn walk_pattern_field<V: AstVisitorMut>(v: &mut V, field: &mut PatternField) {
    v.visit_ident(&mut field.name);
    if let Some(ref mut pat) = field.pattern {
        v.visit_pattern(pat);
    }
}

// ---------------------------------------------------------------------------
// Type expression walker — covers all 5 TypeExprKind variants
// ---------------------------------------------------------------------------

/// Walk a [`TypeExpr`] by dispatching on all [`TypeExprKind`] variants.
///
/// Visits named types, array elements/sizes, tuple members, function
/// parameter/return types, and shape dimensions.
pub fn walk_type_expr<V: AstVisitorMut>(v: &mut V, ty: &mut TypeExpr) {
    match &mut ty.kind {
        TypeExprKind::Named { name, args } => {
            v.visit_ident(name);
            for arg in args {
                walk_type_arg(v, arg);
            }
        }
        TypeExprKind::Array { elem, size } => {
            v.visit_type_expr(elem);
            v.visit_expr(size);
        }
        TypeExprKind::Tuple(tys) => {
            for t in tys {
                v.visit_type_expr(t);
            }
        }
        TypeExprKind::Fn { params, ret } => {
            for p in params {
                v.visit_type_expr(p);
            }
            v.visit_type_expr(ret);
        }
        TypeExprKind::ShapeLit(dims) => {
            for dim in dims {
                if let ShapeDim::Name(ref mut ident) = dim {
                    v.visit_ident(ident);
                }
            }
        }
    }
}

/// Walk a [`TypeArg`]: dispatch to type, expression, or shape variant.
pub fn walk_type_arg<V: AstVisitorMut>(v: &mut V, arg: &mut TypeArg) {
    match arg {
        TypeArg::Type(ty) => v.visit_type_expr(ty),
        TypeArg::Expr(e) => v.visit_expr(e),
        TypeArg::Shape(dims) => {
            for dim in dims {
                if let ShapeDim::Name(ref mut ident) = dim {
                    v.visit_ident(ident);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

