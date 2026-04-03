//! `cjc audit` — Numerical hygiene analysis.
//!
//! Static analysis pass over the AST looking for floating-point anti-patterns:
//! naive summation in loops, float equality comparison, unguarded division,
//! and potential catastrophic cancellation.
//!
//! Never executes the program. All findings are deterministic.

use std::fs;
use std::path::Path;
use std::process;
use crate::output::{self, OutputMode};

// ── Finding representation ──────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Severity {
    Warn,
    Info,
}

impl Severity {
    fn label(&self) -> &'static str {
        match self {
            Severity::Warn => "WARN",
            Severity::Info => "INFO",
        }
    }
}

#[derive(Debug, Clone)]
struct Finding {
    severity: Severity,
    line: usize,
    message: String,
    suggestion: String,
}

// ── Args ────────────────────────────────────────────────────────────

pub struct AuditArgs {
    pub file: String,
    pub output: OutputMode,
    pub verbose: bool,
}

impl Default for AuditArgs {
    fn default() -> Self {
        Self {
            file: String::new(),
            output: OutputMode::Color,
            verbose: false,
        }
    }
}

pub fn parse_args(args: &[String]) -> AuditArgs {
    let mut aa = AuditArgs::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-v" | "--verbose" => aa.verbose = true,
            "--plain" => aa.output = OutputMode::Plain,
            "--json" => aa.output = OutputMode::Json,
            "--color" => aa.output = OutputMode::Color,
            other if !other.starts_with('-') => aa.file = other.to_string(),
            other => {
                eprintln!("error: unknown flag `{}` for `cjc audit`", other);
                process::exit(1);
            }
        }
        i += 1;
    }
    if aa.file.is_empty() {
        eprintln!("error: `cjc audit` requires a .cjc file argument");
        process::exit(1);
    }
    aa
}

// ── Line number computation ─────────────────────────────────────────

/// Convert a byte offset into a 1-based line number by counting newlines.
fn offset_to_line(source: &str, offset: usize) -> usize {
    let capped = offset.min(source.len());
    source[..capped].bytes().filter(|&b| b == b'\n').count() + 1
}

// ── AST visitor ─────────────────────────────────────────────────────

struct AuditVisitor<'s> {
    source: &'s str,
    findings: Vec<Finding>,
    /// Stack of loop variable names that are used as accumulators (targets of += with Add).
    loop_depth: u32,
    /// Variables assigned inside loops (potential accumulators).
    loop_assign_targets: Vec<String>,
    verbose: bool,
}

impl<'s> AuditVisitor<'s> {
    fn new(source: &'s str, verbose: bool) -> Self {
        Self {
            source,
            findings: Vec::new(),
            loop_depth: 0,
            loop_assign_targets: Vec::new(),
            verbose,
        }
    }

    fn line(&self, span: &cjc_ast::Span) -> usize {
        offset_to_line(self.source, span.start)
    }

    fn add(&mut self, severity: Severity, line: usize, message: String, suggestion: String) {
        self.findings.push(Finding { severity, line, message, suggestion });
    }

    // ── Top-level walk ──────────────────────────────────────────────

    fn visit_program(&mut self, program: &cjc_ast::Program) {
        for decl in &program.declarations {
            self.visit_decl(decl);
        }
    }

    fn visit_decl(&mut self, decl: &cjc_ast::Decl) {
        match &decl.kind {
            cjc_ast::DeclKind::Fn(f) => self.visit_block(&f.body),
            cjc_ast::DeclKind::Let(l) => self.visit_expr(&l.init),
            cjc_ast::DeclKind::Struct(_) => {}
            cjc_ast::DeclKind::Stmt(s) => self.visit_stmt(s),
            _ => {}
        }
    }

    fn visit_block(&mut self, block: &cjc_ast::Block) {
        for stmt in &block.stmts {
            self.visit_stmt(stmt);
        }
        if let Some(ref expr) = block.expr {
            self.visit_expr(expr);
        }
    }

    fn visit_stmt(&mut self, stmt: &cjc_ast::Stmt) {
        match &stmt.kind {
            cjc_ast::StmtKind::Let(l) => self.visit_expr(&l.init),
            cjc_ast::StmtKind::Expr(e) => self.visit_expr(e),
            cjc_ast::StmtKind::Return(Some(e)) => self.visit_expr(e),
            cjc_ast::StmtKind::Return(None) | cjc_ast::StmtKind::Break | cjc_ast::StmtKind::Continue => {}
            cjc_ast::StmtKind::If(if_stmt) => self.visit_if(if_stmt),
            cjc_ast::StmtKind::While(w) => {
                self.loop_depth += 1;
                let prev_targets = self.loop_assign_targets.clone();
                self.visit_expr(&w.condition);
                self.visit_block(&w.body);
                self.check_loop_accumulators(&w.body, stmt.span.start);
                self.loop_assign_targets = prev_targets;
                self.loop_depth -= 1;
            }
            cjc_ast::StmtKind::For(f) => {
                self.loop_depth += 1;
                let prev_targets = self.loop_assign_targets.clone();
                self.visit_block(&f.body);
                self.check_loop_accumulators(&f.body, stmt.span.start);
                self.loop_assign_targets = prev_targets;
                self.loop_depth -= 1;
            }
            cjc_ast::StmtKind::NoGcBlock(b) => self.visit_block(b),
        }
    }

    fn visit_if(&mut self, if_stmt: &cjc_ast::IfStmt) {
        self.visit_expr(&if_stmt.condition);
        self.visit_block(&if_stmt.then_block);
        if let Some(ref else_br) = if_stmt.else_branch {
            match else_br {
                cjc_ast::ElseBranch::ElseIf(nested) => self.visit_if(nested),
                cjc_ast::ElseBranch::Else(block) => self.visit_block(block),
            }
        }
    }

    fn visit_expr(&mut self, expr: &cjc_ast::Expr) {
        match &expr.kind {
            cjc_ast::ExprKind::Binary { op, left, right } => {
                self.check_binary(expr, *op, left, right);
                self.visit_expr(left);
                self.visit_expr(right);
            }
            cjc_ast::ExprKind::Unary { operand, .. } => self.visit_expr(operand),
            cjc_ast::ExprKind::Call { callee, args } => {
                self.visit_expr(callee);
                for arg in args {
                    self.visit_expr(&arg.value);
                }
            }
            cjc_ast::ExprKind::Field { object, .. } => self.visit_expr(object),
            cjc_ast::ExprKind::Index { object, index } => {
                self.visit_expr(object);
                self.visit_expr(index);
            }
            cjc_ast::ExprKind::MultiIndex { object, indices } => {
                self.visit_expr(object);
                for idx in indices { self.visit_expr(idx); }
            }
            cjc_ast::ExprKind::Assign { target, value } => {
                // Track accumulator targets inside loops
                if self.loop_depth > 0 {
                    if let cjc_ast::ExprKind::Ident(ident) = &target.kind {
                        self.loop_assign_targets.push(ident.name.clone());
                    }
                }
                self.visit_expr(target);
                self.visit_expr(value);
            }
            cjc_ast::ExprKind::CompoundAssign { op, target, value } => {
                // Compound assign += in a loop with Add is a summation accumulator
                if self.loop_depth > 0 && *op == cjc_ast::BinOp::Add {
                    if let cjc_ast::ExprKind::Ident(ident) = &target.kind {
                        let line = self.line(&expr.span);
                        self.add(
                            Severity::Warn,
                            line,
                            format!("naive summation: `{}` accumulated with `+=` in loop", ident.name),
                            "use `kahan_sum()` or `binned_sum()` for numerically stable accumulation".into(),
                        );
                    }
                }
                self.visit_expr(target);
                self.visit_expr(value);
            }
            cjc_ast::ExprKind::IfExpr { condition, then_block, else_branch } => {
                self.visit_expr(condition);
                self.visit_block(then_block);
                if let Some(ref eb) = else_branch {
                    match eb {
                        cjc_ast::ElseBranch::ElseIf(nested) => self.visit_if(nested),
                        cjc_ast::ElseBranch::Else(block) => self.visit_block(block),
                    }
                }
            }
            cjc_ast::ExprKind::Pipe { left, right } => {
                self.visit_expr(left);
                self.visit_expr(right);
            }
            cjc_ast::ExprKind::Block(b) => self.visit_block(b),
            cjc_ast::ExprKind::ArrayLit(elems) => {
                for e in elems { self.visit_expr(e); }
            }
            cjc_ast::ExprKind::TupleLit(elems) => {
                for e in elems { self.visit_expr(e); }
            }
            cjc_ast::ExprKind::StructLit { fields, .. } => {
                for f in fields { self.visit_expr(&f.value); }
            }
            cjc_ast::ExprKind::Lambda { body, .. } => self.visit_expr(body),
            cjc_ast::ExprKind::Match { scrutinee, arms } => {
                self.visit_expr(scrutinee);
                for arm in arms {
                    self.visit_expr(&arm.body);
                }
            }
            cjc_ast::ExprKind::TensorLit { rows } => {
                for row in rows {
                    for e in row { self.visit_expr(e); }
                }
            }
            cjc_ast::ExprKind::Try(inner) => self.visit_expr(inner),
            cjc_ast::ExprKind::VariantLit { fields, .. } => {
                for f in fields { self.visit_expr(f); }
            }
            cjc_ast::ExprKind::FStringLit(segments) => {
                for (_, maybe_expr) in segments {
                    if let Some(ref e) = maybe_expr { self.visit_expr(e); }
                }
            }
            // Terminals: nothing to visit
            cjc_ast::ExprKind::IntLit(_) | cjc_ast::ExprKind::FloatLit(_)
            | cjc_ast::ExprKind::BoolLit(_) | cjc_ast::ExprKind::StringLit(_)
            | cjc_ast::ExprKind::ByteStringLit(_) | cjc_ast::ExprKind::ByteCharLit(_)
            | cjc_ast::ExprKind::RawStringLit(_) | cjc_ast::ExprKind::RawByteStringLit(_)
            | cjc_ast::ExprKind::RegexLit { .. } | cjc_ast::ExprKind::Ident(_)
            | cjc_ast::ExprKind::Col(_) => {}
        }
    }

    // ── Check helpers ───────────────────────────────────────────────

    fn check_binary(&mut self, expr: &cjc_ast::Expr, op: cjc_ast::BinOp, left: &cjc_ast::Expr, right: &cjc_ast::Expr) {
        let line = self.line(&expr.span);

        match op {
            // Check 1: Naive summation in loops (x = x + val pattern)
            cjc_ast::BinOp::Add if self.loop_depth > 0 => {
                // This catches `sum = sum + val` patterns.
                // CompoundAssign `+=` is caught separately above.
                if self.is_accumulator_pattern(left, right) {
                    let name = self.extract_ident_name(left).or_else(|| self.extract_ident_name(right))
                        .unwrap_or_else(|| "accumulator".into());
                    self.add(
                        Severity::Warn,
                        line,
                        format!("naive summation: `{}` accumulated with `+` in loop", name),
                        "use `kahan_sum()` or `binned_sum()` for numerically stable accumulation".into(),
                    );
                }
            }

            // Check 2: Float equality comparison
            cjc_ast::BinOp::Eq | cjc_ast::BinOp::Ne => {
                if self.might_be_float(left) || self.might_be_float(right) {
                    let cmp = if op == cjc_ast::BinOp::Eq { "==" } else { "!=" };
                    self.add(
                        Severity::Warn,
                        line,
                        format!("float equality: comparing with `{}` may fail due to rounding", cmp),
                        "use `approx_eq(a, b, tol)` for tolerance-based comparison".into(),
                    );
                }
            }

            // Check 3: Division without zero-guard
            cjc_ast::BinOp::Div => {
                if self.is_variable(right) {
                    let name = self.extract_ident_name(right).unwrap_or_else(|| "divisor".into());
                    self.add(
                        Severity::Warn,
                        line,
                        format!("unguarded division: `{}` could be zero", name),
                        "add a zero-check before dividing, or use a safe division helper".into(),
                    );
                } else if self.verbose {
                    self.add(
                        Severity::Info,
                        line,
                        "division operation".into(),
                        "ensure divisor cannot be zero at runtime".into(),
                    );
                }
            }

            // Check 4: Catastrophic cancellation
            cjc_ast::BinOp::Sub => {
                if self.is_complex_expr(left) && self.is_complex_expr(right) {
                    self.add(
                        Severity::Info,
                        line,
                        "potential catastrophic cancellation: subtraction of two complex expressions".into(),
                        "if operands are of similar magnitude, consider algebraic reformulation".into(),
                    );
                }
            }

            _ => {}
        }
    }

    /// Check for `x = x + val` or `x = val + x` accumulator pattern inside
    /// the body of a loop. We look at Assign statements in the block.
    fn check_loop_accumulators(&mut self, block: &cjc_ast::Block, _loop_start: usize) {
        for stmt in &block.stmts {
            if let cjc_ast::StmtKind::Expr(ref expr) = stmt.kind {
                if let cjc_ast::ExprKind::Assign { target, value } = &expr.kind {
                    if let cjc_ast::ExprKind::Ident(ref tgt_name) = target.kind {
                        if self.expr_adds_ident(value, &tgt_name.name) {
                            let line = self.line(&expr.span);
                            self.add(
                                Severity::Warn,
                                line,
                                format!("naive summation: `{} = {} + ...` in loop", tgt_name.name, tgt_name.name),
                                "use `kahan_sum()` or `binned_sum()` for numerically stable accumulation".into(),
                            );
                        }
                    }
                }
            }
        }
    }

    /// Returns true if `expr` is of the form `name + something` or `something + name`.
    fn expr_adds_ident(&self, expr: &cjc_ast::Expr, name: &str) -> bool {
        if let cjc_ast::ExprKind::Binary { op: cjc_ast::BinOp::Add, left, right } = &expr.kind {
            let left_is = self.extract_ident_name(left).map(|n| n == name).unwrap_or(false);
            let right_is = self.extract_ident_name(right).map(|n| n == name).unwrap_or(false);
            return left_is || right_is;
        }
        false
    }

    /// True if `left` and `right` form an `x = x + val` style accumulator pattern.
    /// We detect this when one side is a plain identifier and the other side
    /// also references that identifier (common loop accumulator shape).
    fn is_accumulator_pattern(&self, left: &cjc_ast::Expr, right: &cjc_ast::Expr) -> bool {
        // Check if one side is an identifier that appears in the loop_assign_targets
        if let Some(name) = self.extract_ident_name(left) {
            if self.loop_assign_targets.contains(&name) {
                return true;
            }
        }
        if let Some(name) = self.extract_ident_name(right) {
            if self.loop_assign_targets.contains(&name) {
                return true;
            }
        }
        false
    }

    fn extract_ident_name(&self, expr: &cjc_ast::Expr) -> Option<String> {
        if let cjc_ast::ExprKind::Ident(ident) = &expr.kind {
            Some(ident.name.clone())
        } else {
            None
        }
    }

    /// Heuristic: might this expression evaluate to a float?
    fn might_be_float(&self, expr: &cjc_ast::Expr) -> bool {
        match &expr.kind {
            cjc_ast::ExprKind::FloatLit(_) => true,
            cjc_ast::ExprKind::Binary { op, .. } => matches!(op,
                cjc_ast::BinOp::Div | cjc_ast::BinOp::Pow),
            cjc_ast::ExprKind::Call { callee, .. } => {
                // Common math builtins that return float
                if let cjc_ast::ExprKind::Ident(name) = &callee.kind {
                    matches!(name.name.as_str(),
                        "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "atan2"
                        | "sqrt" | "cbrt" | "exp" | "log" | "log2" | "log10"
                        | "abs" | "floor" | "ceil" | "round" | "pow"
                        | "mean" | "std_dev" | "variance" | "sum"
                        | "dot" | "norm" | "det" | "trace"
                        | "random" | "random_normal" | "random_uniform")
                } else {
                    false
                }
            }
            // Variables could be floats, but we only flag if there's stronger evidence
            _ => false,
        }
    }

    /// True if expr is a variable (identifier, field access, or index).
    fn is_variable(&self, expr: &cjc_ast::Expr) -> bool {
        matches!(&expr.kind,
            cjc_ast::ExprKind::Ident(_) | cjc_ast::ExprKind::Field { .. }
            | cjc_ast::ExprKind::Index { .. })
    }

    /// Heuristic: is this a "complex expression" (function call, binary op, etc.)?
    fn is_complex_expr(&self, expr: &cjc_ast::Expr) -> bool {
        matches!(&expr.kind,
            cjc_ast::ExprKind::Call { .. } | cjc_ast::ExprKind::Binary { .. })
    }
}

// ── Entry point ─────────────────────────────────────────────────────

pub fn run(args: &[String]) {
    let aa = parse_args(args);
    let path = Path::new(&aa.file);

    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => { eprintln!("error: could not read `{}`: {}", aa.file, e); process::exit(1); }
    };

    let filename = aa.file.replace('\\', "/");

    let (program, parse_diags) = cjc_parser::parse_source(&source);

    if parse_diags.has_errors() {
        let rendered = parse_diags.render_all_color(&source, &filename, aa.output.use_color());
        eprint!("{}", rendered);
        process::exit(1);
    }

    let mut visitor = AuditVisitor::new(&source, aa.verbose);
    visitor.visit_program(&program);

    // Sort findings by line number for deterministic output
    visitor.findings.sort_by_key(|f| (f.line, f.severity as u8));

    let findings = visitor.findings;
    let warn_count = findings.iter().filter(|f| f.severity == Severity::Warn).count();
    let info_count = findings.iter().filter(|f| f.severity == Severity::Info).count();

    match aa.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"file\": \"{}\",", filename);
            println!("  \"warnings\": {},", warn_count);
            println!("  \"info\": {},", info_count);
            println!("  \"findings\": [");
            for (i, f) in findings.iter().enumerate() {
                print!("    {{\"severity\": \"{}\", \"line\": {}, \"message\": \"{}\", \"suggestion\": \"{}\"}}",
                    f.severity.label(), f.line,
                    f.message.replace('"', "\\\""),
                    f.suggestion.replace('"', "\\\""));
                if i + 1 < findings.len() { print!(","); }
                println!();
            }
            println!("  ]");
            println!("}}");
        }
        _ => {
            eprintln!("{} Auditing `{}`...",
                output::colorize(aa.output, output::BOLD_CYAN, "[audit]"), filename);
            eprintln!();

            if findings.is_empty() {
                eprintln!("{} — no numerical hygiene issues found",
                    output::colorize(aa.output, output::BOLD_GREEN, "CLEAN"));
            } else {
                for f in &findings {
                    let sev_str = match f.severity {
                        Severity::Warn => output::colorize(aa.output, output::BOLD_YELLOW, "[WARN]"),
                        Severity::Info => output::colorize(aa.output, output::BOLD_BLUE, "[INFO]"),
                    };
                    eprintln!("  {} line {}: {}", sev_str, f.line, f.message);
                    eprintln!("         suggestion: {}", f.suggestion);
                }

                eprintln!();
                let mut t = crate::table::Table::new(vec!["Metric", "Count"]);
                t.add_row_owned(vec!["Warnings".into(), format!("{}", warn_count)]);
                t.add_row_owned(vec!["Info".into(), format!("{}", info_count)]);
                t.add_row_owned(vec!["Total findings".into(), format!("{}", findings.len())]);
                eprint!("{}", t.render());
            }
        }
    }
}

pub fn print_help() {
    eprintln!("cjc audit — Numerical hygiene analysis");
    eprintln!();
    eprintln!("Usage: cjc audit <file.cjc> [flags]");
    eprintln!();
    eprintln!("Performs static analysis of CJC source for floating-point anti-patterns:");
    eprintln!("  - Naive summation in loops (suggest kahan_sum / binned_sum)");
    eprintln!("  - Float equality comparison (suggest approx_eq)");
    eprintln!("  - Division without zero-guard");
    eprintln!("  - Catastrophic cancellation (subtraction of complex expressions)");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  -v, --verbose    Show additional info-level findings");
    eprintln!("  --plain          Plain text output");
    eprintln!("  --json           JSON output");
    eprintln!("  --color          Color output (default)");
}
