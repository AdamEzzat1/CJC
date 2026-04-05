//! AST Visitor — Read-only traversal infrastructure
//!
//! Provides an `AstVisitor` trait with default walk methods that traverse the
//! full AST node taxonomy.  Downstream crates can implement specific `visit_*`
//! methods and let the rest fall through to the default recursive walk.
//!
//! ## Design decisions
//!
//! - **Read-only** — no `MutVisitor`; that would risk downstream breakage
//! - **Additive overlay** — does not modify any existing AST types
//! - **Complete coverage** — all 35 `ExprKind`, 9 `StmtKind`, 11 `DeclKind`,
//!   and 9 `PatternKind` variants are traversed
//! - **Deterministic** — traversal order matches AST structure (left-to-right,
//!   depth-first)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use cjc_ast::visit::{AstVisitor, walk_program};
//!
//! struct Counter { count: u32 }
//!
//! impl AstVisitor for Counter {
//!     fn visit_expr(&mut self, expr: &cjc_ast::Expr) {
//!         self.count += 1;
//!         walk_expr(self, expr);  // continue traversal
//!     }
//! }
//! ```

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

/// Read-only AST visitor.  Override specific `visit_*` methods; call the
/// corresponding `walk_*` function inside your override to continue the
/// default traversal into children.
pub trait AstVisitor: Sized {
    /// Visit a top-level [`Program`]. Default: delegates to [`walk_program`].
    fn visit_program(&mut self, program: &Program) {
        walk_program(self, program);
    }
    /// Visit a [`Decl`] node. Default: delegates to [`walk_decl`].
    fn visit_decl(&mut self, decl: &Decl) {
        walk_decl(self, decl);
    }
    /// Visit a [`Stmt`] node. Default: delegates to [`walk_stmt`].
    fn visit_stmt(&mut self, stmt: &Stmt) {
        walk_stmt(self, stmt);
    }
    /// Visit an [`Expr`] node. Default: delegates to [`walk_expr`].
    fn visit_expr(&mut self, expr: &Expr) {
        walk_expr(self, expr);
    }
    /// Visit a [`Pattern`] node. Default: delegates to [`walk_pattern`].
    fn visit_pattern(&mut self, pattern: &Pattern) {
        walk_pattern(self, pattern);
    }
    /// Visit a [`TypeExpr`] node. Default: delegates to [`walk_type_expr`].
    fn visit_type_expr(&mut self, ty: &TypeExpr) {
        walk_type_expr(self, ty);
    }
    /// Visit a [`Block`]. Default: delegates to [`walk_block`].
    fn visit_block(&mut self, block: &Block) {
        walk_block(self, block);
    }
    /// Visit a [`FnDecl`] node. Default: delegates to [`walk_fn_decl`].
    fn visit_fn_decl(&mut self, f: &FnDecl) {
        walk_fn_decl(self, f);
    }
    /// Visit an [`Ident`]. Default: no-op (leaf node).
    fn visit_ident(&mut self, _ident: &Ident) {}
    /// Visit a function [`Param`]. Default: delegates to [`walk_param`].
    fn visit_param(&mut self, param: &Param) {
        walk_param(self, param);
    }
    /// Visit a [`MatchArm`]. Default: delegates to [`walk_match_arm`].
    fn visit_match_arm(&mut self, arm: &MatchArm) {
        walk_match_arm(self, arm);
    }
}

// ---------------------------------------------------------------------------
// Walk functions — default traversals
// ---------------------------------------------------------------------------

/// Walk a [`Program`] by visiting each declaration in order.
pub fn walk_program<V: AstVisitor>(v: &mut V, program: &Program) {
    for decl in &program.declarations {
        v.visit_decl(decl);
    }
}

/// Walk a [`Decl`] by dispatching on its [`DeclKind`] variant.
pub fn walk_decl<V: AstVisitor>(v: &mut V, decl: &Decl) {
    match &decl.kind {
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
pub fn walk_struct_decl<V: AstVisitor>(v: &mut V, s: &StructDecl) {
    v.visit_ident(&s.name);
    for tp in &s.type_params {
        walk_type_param(v, tp);
    }
    for field in &s.fields {
        walk_field_decl(v, field);
    }
}

/// Walk a [`ClassDecl`]: visit name, type params, and fields.
pub fn walk_class_decl<V: AstVisitor>(v: &mut V, c: &ClassDecl) {
    v.visit_ident(&c.name);
    for tp in &c.type_params {
        walk_type_param(v, tp);
    }
    for field in &c.fields {
        walk_field_decl(v, field);
    }
}

/// Walk a [`RecordDecl`]: visit name, type params, and fields.
pub fn walk_record_decl<V: AstVisitor>(v: &mut V, r: &RecordDecl) {
    v.visit_ident(&r.name);
    for tp in &r.type_params {
        walk_type_param(v, tp);
    }
    for field in &r.fields {
        walk_field_decl(v, field);
    }
}

/// Walk an [`EnumDecl`]: visit name, type params, and each variant.
pub fn walk_enum_decl<V: AstVisitor>(v: &mut V, e: &EnumDecl) {
    v.visit_ident(&e.name);
    for tp in &e.type_params {
        walk_type_param(v, tp);
    }
    for variant in &e.variants {
        walk_variant_decl(v, variant);
    }
}

/// Walk a [`VariantDecl`]: visit name and payload type expressions.
pub fn walk_variant_decl<V: AstVisitor>(v: &mut V, variant: &VariantDecl) {
    v.visit_ident(&variant.name);
    for ty in &variant.fields {
        v.visit_type_expr(ty);
    }
}

/// Walk a [`FieldDecl`]: visit name, type, and optional default expression.
pub fn walk_field_decl<V: AstVisitor>(v: &mut V, field: &FieldDecl) {
    v.visit_ident(&field.name);
    v.visit_type_expr(&field.ty);
    if let Some(ref default) = field.default {
        v.visit_expr(default);
    }
}

/// Walk a [`TraitDecl`]: visit name, type params, super-traits, and method signatures.
pub fn walk_trait_decl<V: AstVisitor>(v: &mut V, t: &TraitDecl) {
    v.visit_ident(&t.name);
    for tp in &t.type_params {
        walk_type_param(v, tp);
    }
    for st in &t.super_traits {
        v.visit_type_expr(st);
    }
    for method in &t.methods {
        walk_fn_sig(v, method);
    }
}

/// Walk an [`ImplDecl`]: visit type params, target type, optional trait ref, and methods.
pub fn walk_impl_decl<V: AstVisitor>(v: &mut V, i: &ImplDecl) {
    for tp in &i.type_params {
        walk_type_param(v, tp);
    }
    v.visit_type_expr(&i.target);
    if let Some(ref tr) = i.trait_ref {
        v.visit_type_expr(tr);
    }
    for method in &i.methods {
        v.visit_fn_decl(method);
    }
}

/// Walk an [`ImportDecl`]: visit path segments and optional alias.
pub fn walk_import_decl<V: AstVisitor>(v: &mut V, i: &ImportDecl) {
    for ident in &i.path {
        v.visit_ident(ident);
    }
    if let Some(ref alias) = i.alias {
        v.visit_ident(alias);
    }
}

/// Walk a [`ConstDecl`]: visit name, type, and value expression.
pub fn walk_const_decl<V: AstVisitor>(v: &mut V, c: &ConstDecl) {
    v.visit_ident(&c.name);
    v.visit_type_expr(&c.ty);
    v.visit_expr(&c.value);
}

/// Walk a [`FnDecl`]: visit name, type params, params, return type, decorators, and body.
pub fn walk_fn_decl<V: AstVisitor>(v: &mut V, f: &FnDecl) {
    v.visit_ident(&f.name);
    for tp in &f.type_params {
        walk_type_param(v, tp);
    }
    for param in &f.params {
        v.visit_param(param);
    }
    if let Some(ref ret) = f.return_type {
        v.visit_type_expr(ret);
    }
    for dec in &f.decorators {
        walk_decorator(v, dec);
    }
    v.visit_block(&f.body);
}

/// Walk a [`FnSig`]: visit name, type params, params, and return type.
pub fn walk_fn_sig<V: AstVisitor>(v: &mut V, sig: &FnSig) {
    v.visit_ident(&sig.name);
    for tp in &sig.type_params {
        walk_type_param(v, tp);
    }
    for param in &sig.params {
        v.visit_param(param);
    }
    if let Some(ref ret) = sig.return_type {
        v.visit_type_expr(ret);
    }
}

/// Walk a [`Param`]: visit name, type, and optional default expression.
pub fn walk_param<V: AstVisitor>(v: &mut V, param: &Param) {
    v.visit_ident(&param.name);
    v.visit_type_expr(&param.ty);
    if let Some(ref default) = param.default {
        v.visit_expr(default);
    }
}

/// Walk a [`Decorator`]: visit name and argument expressions.
pub fn walk_decorator<V: AstVisitor>(v: &mut V, dec: &Decorator) {
    v.visit_ident(&dec.name);
    for arg in &dec.args {
        v.visit_expr(arg);
    }
}

/// Walk a [`TypeParam`]: visit name and trait bounds.
pub fn walk_type_param<V: AstVisitor>(v: &mut V, tp: &TypeParam) {
    v.visit_ident(&tp.name);
    for bound in &tp.bounds {
        v.visit_type_expr(bound);
    }
}

// ---------------------------------------------------------------------------
// Block & Statement walkers
// ---------------------------------------------------------------------------

/// Walk a [`Block`]: visit each statement and the optional trailing expression.
pub fn walk_block<V: AstVisitor>(v: &mut V, block: &Block) {
    for stmt in &block.stmts {
        v.visit_stmt(stmt);
    }
    if let Some(ref expr) = block.expr {
        v.visit_expr(expr);
    }
}

/// Walk a [`Stmt`] by dispatching on its [`StmtKind`] variant.
pub fn walk_stmt<V: AstVisitor>(v: &mut V, stmt: &Stmt) {
    match &stmt.kind {
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
pub fn walk_let_stmt<V: AstVisitor>(v: &mut V, l: &LetStmt) {
    v.visit_ident(&l.name);
    if let Some(ref ty) = l.ty {
        v.visit_type_expr(ty);
    }
    v.visit_expr(&l.init);
}

/// Walk an [`IfStmt`]: visit condition, then-block, and optional else branch.
pub fn walk_if_stmt<V: AstVisitor>(v: &mut V, i: &IfStmt) {
    v.visit_expr(&i.condition);
    v.visit_block(&i.then_block);
    if let Some(ref else_branch) = i.else_branch {
        walk_else_branch(v, else_branch);
    }
}

/// Walk an [`ElseBranch`]: dispatch to either an else-if or a terminal else block.
pub fn walk_else_branch<V: AstVisitor>(v: &mut V, eb: &ElseBranch) {
    match eb {
        ElseBranch::ElseIf(elif) => walk_if_stmt(v, elif),
        ElseBranch::Else(block) => v.visit_block(block),
    }
}

/// Walk a [`WhileStmt`]: visit condition and body block.
pub fn walk_while_stmt<V: AstVisitor>(v: &mut V, w: &WhileStmt) {
    v.visit_expr(&w.condition);
    v.visit_block(&w.body);
}

/// Walk a [`ForStmt`]: visit loop variable, iterator (range or expr), and body.
pub fn walk_for_stmt<V: AstVisitor>(v: &mut V, f: &ForStmt) {
    v.visit_ident(&f.ident);
    match &f.iter {
        ForIter::Range { start, end } => {
            v.visit_expr(start);
            v.visit_expr(end);
        }
        ForIter::Expr(e) => v.visit_expr(e),
    }
    v.visit_block(&f.body);
}

// ---------------------------------------------------------------------------
// Expression walker — covers all 35 ExprKind variants
// ---------------------------------------------------------------------------

/// Walk an [`Expr`] by dispatching on all [`ExprKind`] variants.
///
/// Visits child expressions, identifiers, blocks, and patterns in
/// left-to-right, depth-first order.
pub fn walk_expr<V: AstVisitor>(v: &mut V, expr: &Expr) {
    match &expr.kind {
        ExprKind::IntLit(_) => {}
        ExprKind::FloatLit(_) => {}
        ExprKind::StringLit(_) => {}
        ExprKind::ByteStringLit(_) => {}
        ExprKind::ByteCharLit(_) => {}
        ExprKind::RawStringLit(_) => {}
        ExprKind::RawByteStringLit(_) => {}
        ExprKind::FStringLit(segments) => {
            for (_lit, maybe_expr) in segments {
                if let Some(ref e) = maybe_expr {
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
            if let Some(ref eb) = else_branch {
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
            if let Some(ref en) = enum_name {
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
pub fn walk_call_arg<V: AstVisitor>(v: &mut V, arg: &CallArg) {
    if let Some(ref name) = arg.name {
        v.visit_ident(name);
    }
    v.visit_expr(&arg.value);
}

/// Walk a [`FieldInit`]: visit field name and value expression.
pub fn walk_field_init<V: AstVisitor>(v: &mut V, fi: &FieldInit) {
    v.visit_ident(&fi.name);
    v.visit_expr(&fi.value);
}

/// Walk a [`MatchArm`]: visit pattern and body expression.
pub fn walk_match_arm<V: AstVisitor>(v: &mut V, arm: &MatchArm) {
    v.visit_pattern(&arm.pattern);
    v.visit_expr(&arm.body);
}

// ---------------------------------------------------------------------------
// Pattern walker — covers all 9 PatternKind variants
// ---------------------------------------------------------------------------

/// Walk a [`Pattern`] by dispatching on all [`PatternKind`] variants.
///
/// Visits nested patterns, identifiers, and fields in declaration order.
pub fn walk_pattern<V: AstVisitor>(v: &mut V, pattern: &Pattern) {
    match &pattern.kind {
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
            if let Some(ref en) = enum_name {
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
pub fn walk_pattern_field<V: AstVisitor>(v: &mut V, field: &PatternField) {
    v.visit_ident(&field.name);
    if let Some(ref pat) = field.pattern {
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
pub fn walk_type_expr<V: AstVisitor>(v: &mut V, ty: &TypeExpr) {
    match &ty.kind {
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
                if let ShapeDim::Name(ref ident) = dim {
                    v.visit_ident(ident);
                }
            }
        }
    }
}

/// Walk a [`TypeArg`]: dispatch to type, expression, or shape variant.
pub fn walk_type_arg<V: AstVisitor>(v: &mut V, arg: &TypeArg) {
    match arg {
        TypeArg::Type(ty) => v.visit_type_expr(ty),
        TypeArg::Expr(e) => v.visit_expr(e),
        TypeArg::Shape(dims) => {
            for dim in dims {
                if let ShapeDim::Name(ref ident) = dim {
                    v.visit_ident(ident);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    fn parse_program(src: &str) -> Program {
        // Minimal hand-built ASTs for unit testing the visitor.
        // Integration tests use the real parser.
        let _ = src; // suppress unused
        Program {
            declarations: Vec::new(),
        }
    }

    /// A simple counter visitor that counts expressions and statements.
    struct NodeCounter {
        exprs: u32,
        stmts: u32,
        decls: u32,
        patterns: u32,
        idents: u32,
    }

    impl NodeCounter {
        fn new() -> Self {
            Self {
                exprs: 0,
                stmts: 0,
                decls: 0,
                patterns: 0,
                idents: 0,
            }
        }
    }

    impl AstVisitor for NodeCounter {
        fn visit_expr(&mut self, expr: &Expr) {
            self.exprs += 1;
            walk_expr(self, expr);
        }
        fn visit_stmt(&mut self, stmt: &Stmt) {
            self.stmts += 1;
            walk_stmt(self, stmt);
        }
        fn visit_decl(&mut self, decl: &Decl) {
            self.decls += 1;
            walk_decl(self, decl);
        }
        fn visit_pattern(&mut self, pattern: &Pattern) {
            self.patterns += 1;
            walk_pattern(self, pattern);
        }
        fn visit_ident(&mut self, _ident: &Ident) {
            self.idents += 1;
        }
    }

    fn dummy_expr(kind: ExprKind) -> Expr {
        Expr {
            kind,
            span: Span::dummy(),
        }
    }

    #[test]
    fn test_visitor_empty_program() {
        let program = Program {
            declarations: Vec::new(),
        };
        let mut counter = NodeCounter::new();
        counter.visit_program(&program);
        assert_eq!(counter.exprs, 0);
        assert_eq!(counter.stmts, 0);
        assert_eq!(counter.decls, 0);
    }

    #[test]
    fn test_visitor_binary_expr() {
        let expr = dummy_expr(ExprKind::Binary {
            op: BinOp::Add,
            left: Box::new(dummy_expr(ExprKind::IntLit(1))),
            right: Box::new(dummy_expr(ExprKind::IntLit(2))),
        });
        let mut counter = NodeCounter::new();
        counter.visit_expr(&expr);
        assert_eq!(counter.exprs, 3, "binary + 2 children = 3 expr nodes");
    }

    #[test]
    fn test_visitor_let_stmt() {
        let let_stmt = LetStmt {
            name: Ident::dummy("x"),
            mutable: false,
            ty: None,
            init: Box::new(dummy_expr(ExprKind::IntLit(42))),
        };
        let decl = Decl {
            kind: DeclKind::Let(let_stmt),
            span: Span::dummy(),
        };
        let program = Program {
            declarations: vec![decl],
        };
        let mut counter = NodeCounter::new();
        counter.visit_program(&program);
        assert_eq!(counter.decls, 1);
        assert_eq!(counter.exprs, 1); // the init expr
        assert_eq!(counter.idents, 1); // "x"
    }

    #[test]
    fn test_visitor_fn_decl() {
        let f = FnDecl {
            name: Ident::dummy("foo"),
            type_params: vec![],
            params: vec![Param {
                name: Ident::dummy("x"),
                ty: TypeExpr {
                    kind: TypeExprKind::Named {
                        name: Ident::dummy("i64"),
                        args: vec![],
                    },
                    span: Span::dummy(),
                },
                default: None,
                is_variadic: false,
                span: Span::dummy(),
            }],
            return_type: None,
            body: Block {
                stmts: vec![],
                expr: Some(Box::new(dummy_expr(ExprKind::IntLit(1)))),
                span: Span::dummy(),
            },
            is_nogc: false,
            effect_annotation: None,
            decorators: vec![],
            vis: Visibility::Private,
        };
        let decl = Decl {
            kind: DeclKind::Fn(f),
            span: Span::dummy(),
        };
        let program = Program {
            declarations: vec![decl],
        };
        let mut counter = NodeCounter::new();
        counter.visit_program(&program);
        assert_eq!(counter.decls, 1);
        assert_eq!(counter.exprs, 1); // body result expr
        assert!(counter.idents >= 3); // "foo", "x", "i64"
    }

    #[test]
    fn test_visitor_match_pattern() {
        let arm = MatchArm {
            pattern: Pattern {
                kind: PatternKind::Tuple(vec![
                    Pattern {
                        kind: PatternKind::Binding(Ident::dummy("a")),
                        span: Span::dummy(),
                    },
                    Pattern {
                        kind: PatternKind::Wildcard,
                        span: Span::dummy(),
                    },
                ]),
                span: Span::dummy(),
            },
            body: dummy_expr(ExprKind::IntLit(1)),
            span: Span::dummy(),
        };
        let match_expr = dummy_expr(ExprKind::Match {
            scrutinee: Box::new(dummy_expr(ExprKind::IntLit(0))),
            arms: vec![arm],
        });
        let mut counter = NodeCounter::new();
        counter.visit_expr(&match_expr);
        assert_eq!(counter.patterns, 3); // Tuple + Binding + Wildcard
        assert_eq!(counter.exprs, 3); // match + scrutinee + arm body
    }

    #[test]
    fn test_visitor_determinism() {
        let expr = dummy_expr(ExprKind::Call {
            callee: Box::new(dummy_expr(ExprKind::Ident(Ident::dummy("f")))),
            args: vec![
                CallArg {
                    name: None,
                    value: dummy_expr(ExprKind::IntLit(1)),
                    span: Span::dummy(),
                },
                CallArg {
                    name: None,
                    value: dummy_expr(ExprKind::IntLit(2)),
                    span: Span::dummy(),
                },
            ],
        });
        let mut c1 = NodeCounter::new();
        let mut c2 = NodeCounter::new();
        c1.visit_expr(&expr);
        c2.visit_expr(&expr);
        assert_eq!(c1.exprs, c2.exprs, "visitor must be deterministic");
        assert_eq!(c1.idents, c2.idents);
    }
}
