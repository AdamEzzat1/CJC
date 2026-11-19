//! Module-scoped name resolution (ADR-0047).
//!
//! [`resolve_modules`] returns a copy of every module's AST in which each
//! function reference is replaced by the function's fully qualified name:
//!
//! - a module's own top-level functions become `module::name` (the entry
//!   module's stay bare, so `main` is still `main`);
//! - a bare name imported from another module (`import m` exposes `m`'s
//!   `pub` functions; `import m.f [as g]` exposes `f` if it is `pub`)
//!   becomes that module's qualified name;
//! - every other name is left alone, so builtins, enum variants and
//!   undefined names resolve exactly as in a single-file program.
//!
//! Both executors run the resolved ASTs, so they agree on what every name
//! means. That removes the old per-executor strategies: `merge_programs`
//! aliasing every imported function under its bare name (which made private
//! functions callable and shadowed builtins), and the AST evaluator's shared
//! namespace (where two modules' private helpers of the same name collided).
//!
//! Resolution is lexically scoped: parameters, `let`, `for`, lambda
//! parameters and match-pattern bindings shadow function names in the
//! region where they are visible. Module-level `let`/`const` names are
//! treated as visible throughout their module.

use std::collections::{BTreeMap, BTreeSet};

use cjc_ast::visit_mut::{self as vm, AstVisitorMut};
use cjc_ast::{
    Block, Decl, DeclKind, Expr, ExprKind, FnDecl, MatchArm, Pattern, PatternKind, Program, Stmt,
    StmtKind, Visibility,
};

use crate::{ModuleGraph, ModuleId};

/// The output of [`resolve_modules`].
#[derive(Debug, Clone)]
pub struct ResolvedModules {
    /// Every module's resolved AST, in dependency (topological) order; the
    /// entry module is last.
    pub modules: Vec<(ModuleId, Program)>,
    /// Names that some module referenced, that did not resolve to a module
    /// function, and that another module declares as a private `fn`:
    /// name → declaring modules. Used by [`explain_undefined`] to say why a
    /// call failed. Such a name may still be a builtin, which is why this is
    /// a hint and not an error.
    pub private_hints: BTreeMap<String, BTreeSet<ModuleId>>,
}

/// Resolve every module in `graph`. See the module docs for the rules.
pub fn resolve_modules(graph: &ModuleGraph) -> Result<ResolvedModules, crate::ModuleError> {
    let order = graph.topological_order()?;

    // Top-level functions per module: name → is_pub.
    let mut fns: BTreeMap<&ModuleId, BTreeMap<String, bool>> = BTreeMap::new();
    for (id, module) in &graph.modules {
        let mut m = BTreeMap::new();
        if let Some(ast) = &module.ast {
            for decl in &ast.declarations {
                if let DeclKind::Fn(f) = &decl.kind {
                    m.insert(f.name.name.clone(), f.vis != Visibility::Private);
                }
            }
        }
        fns.insert(id, m);
    }
    let mut private_owners: BTreeMap<&str, BTreeSet<ModuleId>> = BTreeMap::new();
    for (id, m) in &fns {
        for (name, is_pub) in m {
            if !is_pub {
                private_owners
                    .entry(name.as_str())
                    .or_default()
                    .insert((*id).clone());
            }
        }
    }

    let mut out = Vec::with_capacity(order.len());
    let mut private_hints: BTreeMap<String, BTreeSet<ModuleId>> = BTreeMap::new();

    for id in &order {
        let module = &graph.modules[id];
        let Some(ast) = &module.ast else { continue };
        let own_prefix = if module.is_entry { String::new() } else { id.symbol_prefix() };

        // Bare name → qualified name, for everything this module can call.
        let mut names: BTreeMap<String, String> = BTreeMap::new();
        for name in fns[id].keys() {
            names.insert(name.clone(), format!("{own_prefix}{name}"));
        }
        // Imports, in source order; the module's own functions and earlier
        // imports take precedence.
        for import in &module.imports {
            let Some(target) = &import.resolved_module else { continue };
            let Some(target_fns) = fns.get(target) else { continue };
            let prefix = target.symbol_prefix();
            match &import.symbol {
                None => {
                    for (name, is_pub) in target_fns {
                        if *is_pub {
                            names
                                .entry(name.clone())
                                .or_insert_with(|| format!("{prefix}{name}"));
                        }
                    }
                }
                Some(sym) => {
                    // A private symbol import is reported by `check_visibility`.
                    if target_fns.get(sym) == Some(&true) {
                        let local = import.alias.clone().unwrap_or_else(|| sym.clone());
                        names.entry(local).or_insert_with(|| format!("{prefix}{sym}"));
                    }
                }
            }
        }

        let mut program = ast.clone();
        let mut globals = BTreeSet::new();
        for decl in &program.declarations {
            match &decl.kind {
                DeclKind::Let(l) => {
                    globals.insert(l.name.name.clone());
                }
                DeclKind::Const(c) => {
                    globals.insert(c.name.name.clone());
                }
                _ => {}
            }
        }
        let mut r = Resolver {
            names: &names,
            scopes: vec![globals],
            unresolved: BTreeSet::new(),
        };
        vm::walk_program(&mut r, &mut program);
        // Rename this module's top-level function declarations.
        for decl in &mut program.declarations {
            if let DeclKind::Fn(f) = &mut decl.kind {
                f.name.name = format!("{own_prefix}{}", f.name.name);
            }
        }

        for name in r.unresolved {
            if let Some(owners) = private_owners.get(name.as_str()) {
                let others: BTreeSet<ModuleId> =
                    owners.iter().filter(|o| *o != id).cloned().collect();
                if !others.is_empty() {
                    private_hints.entry(name).or_default().extend(others);
                }
            }
        }
        out.push((id.clone(), program));
    }

    Ok(ResolvedModules { modules: out, private_hints })
}

/// Rewrites free identifiers using `names`, respecting lexical scope.
struct Resolver<'a> {
    names: &'a BTreeMap<String, String>,
    /// Innermost scope last. Scope 0 holds the module's top-level `let`s.
    scopes: Vec<BTreeSet<String>>,
    /// Free names that did not resolve to a module function.
    unresolved: BTreeSet<String>,
}

impl Resolver<'_> {
    fn is_bound(&self, name: &str) -> bool {
        self.scopes.iter().any(|s| s.contains(name))
    }

    fn bind(&mut self, name: &str) {
        self.scopes
            .last_mut()
            .expect("resolver always has a scope")
            .insert(name.to_string());
    }

    fn bind_pattern(&mut self, pattern: &Pattern) {
        struct Bindings<'b>(&'b mut Vec<String>);
        impl cjc_ast::visit::AstVisitor for Bindings<'_> {
            fn visit_pattern(&mut self, p: &Pattern) {
                if let PatternKind::Binding(id) = &p.kind {
                    self.0.push(id.name.clone());
                }
                cjc_ast::visit::walk_pattern(self, p);
            }
        }
        let mut names = Vec::new();
        cjc_ast::visit::AstVisitor::visit_pattern(&mut Bindings(&mut names), pattern);
        for n in names {
            self.bind(&n);
        }
    }

    /// Resolve a free name, or record it as unresolved.
    fn resolve(&mut self, name: &mut String) {
        if self.is_bound(name) {
            return;
        }
        match self.names.get(name.as_str()) {
            Some(qualified) => *name = qualified.clone(),
            None => {
                self.unresolved.insert(name.clone());
            }
        }
    }
}

impl AstVisitorMut for Resolver<'_> {
    fn visit_fn_decl(&mut self, f: &mut FnDecl) {
        // Decorators name functions (`@log fn f`) and are resolved in the
        // enclosing scope, before the function's parameters exist.
        for dec in &mut f.decorators {
            self.resolve(&mut dec.name.name);
            for arg in &mut dec.args {
                self.visit_expr(arg);
            }
        }
        self.scopes.push(BTreeSet::new());
        for p in &mut f.params {
            if let Some(d) = &mut p.default {
                self.visit_expr(d);
            }
            self.bind(&p.name.name);
        }
        self.visit_block(&mut f.body);
        self.scopes.pop();
    }

    fn visit_block(&mut self, block: &mut Block) {
        self.scopes.push(BTreeSet::new());
        vm::walk_block(self, block);
        self.scopes.pop();
    }

    fn visit_decl(&mut self, decl: &mut Decl) {
        match &mut decl.kind {
            // Top-level `let`: the name is already in scope 0.
            DeclKind::Let(l) => self.visit_expr(&mut l.init),
            _ => vm::walk_decl(self, decl),
        }
    }

    fn visit_stmt(&mut self, stmt: &mut Stmt) {
        match &mut stmt.kind {
            StmtKind::Let(l) => {
                // The initializer cannot see the name it defines.
                self.visit_expr(&mut l.init);
                self.bind(&l.name.name);
            }
            StmtKind::For(f) => {
                match &mut f.iter {
                    cjc_ast::ForIter::Range { start, end } => {
                        self.visit_expr(start);
                        self.visit_expr(end);
                    }
                    cjc_ast::ForIter::Expr(e) => self.visit_expr(e),
                }
                self.scopes.push(BTreeSet::new());
                self.bind(&f.ident.name);
                self.visit_block(&mut f.body);
                self.scopes.pop();
            }
            _ => vm::walk_stmt(self, stmt),
        }
    }

    fn visit_match_arm(&mut self, arm: &mut MatchArm) {
        self.scopes.push(BTreeSet::new());
        self.bind_pattern(&arm.pattern);
        self.visit_expr(&mut arm.body);
        self.scopes.pop();
    }

    fn visit_expr(&mut self, expr: &mut Expr) {
        match &mut expr.kind {
            ExprKind::Ident(id) => self.resolve(&mut id.name),
            ExprKind::Lambda { params, body } => {
                self.scopes.push(BTreeSet::new());
                for p in params.iter_mut() {
                    if let Some(d) = &mut p.default {
                        self.visit_expr(d);
                    }
                    self.bind(&p.name.name);
                }
                self.visit_expr(body);
                self.scopes.pop();
            }
            _ => vm::walk_expr(self, expr),
        }
    }
}

/// Append a note to an executor's "undefined function/variable `name`"
/// error when `name` is a private function of another module, which is the
/// usual reason such a call fails in a multi-file program. Both executors
/// call this, so they report the same text.
pub fn explain_undefined(message: String, hints: &BTreeMap<String, BTreeSet<ModuleId>>) -> String {
    for (name, owners) in hints {
        let quoted = format!("`{name}`");
        let undefined = message.contains(&format!("undefined function {quoted}"))
            || message.contains(&format!("undefined variable {quoted}"));
        if undefined {
            let owners: Vec<String> = owners.iter().map(|o| format!("`{o}`")).collect();
            return format!(
                "{message}\nnote: {quoted} is a private function of module {}; \
                 mark it `pub` to use it from other modules",
                owners.join(", ")
            );
        }
    }
    message
}
