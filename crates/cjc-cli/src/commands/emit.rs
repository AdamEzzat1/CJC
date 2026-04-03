//! `cjc emit` -- Dump intermediate representations at any pipeline stage.
//!
//! Emits AST, HIR, or MIR for a CJC source file. Useful for debugging
//! compiler internals, understanding lowering transformations, and
//! inspecting optimization effects.
//!
//! All output is deterministic. Never mutates source files.

use std::fs;
use std::process;
use crate::output::{self, OutputMode};

/// Which pipeline stage to emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    Ast,
    Hir,
    Mir,
}

impl Stage {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "ast" => Some(Stage::Ast),
            "hir" => Some(Stage::Hir),
            "mir" => Some(Stage::Mir),
            _ => None,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Stage::Ast => "AST",
            Stage::Hir => "HIR",
            Stage::Mir => "MIR",
        }
    }
}

/// Parsed arguments for `cjc emit`.
pub struct EmitArgs {
    pub file: String,
    pub stage: Stage,
    pub output: OutputMode,
    pub opt: bool,
    pub diff: bool,
}

impl Default for EmitArgs {
    fn default() -> Self {
        Self {
            file: String::new(),
            stage: Stage::Mir,
            output: OutputMode::Color,
            opt: false,
            diff: false,
        }
    }
}

pub fn parse_args(args: &[String]) -> EmitArgs {
    let mut ea = EmitArgs::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--stage" | "-s" => {
                i += 1;
                if i < args.len() {
                    ea.stage = Stage::from_str(&args[i]).unwrap_or_else(|| {
                        eprintln!("error: unknown stage `{}` (expected: ast, hir, mir)", args[i]);
                        process::exit(1);
                    });
                } else {
                    eprintln!("error: --stage requires an argument (ast, hir, mir)");
                    process::exit(1);
                }
            }
            "--opt" => ea.opt = true,
            "--diff" => ea.diff = true,
            "--plain" => ea.output = OutputMode::Plain,
            "--json" => ea.output = OutputMode::Json,
            "--color" => ea.output = OutputMode::Color,
            other if !other.starts_with('-') => ea.file = other.to_string(),
            other => {
                eprintln!("error: unknown flag `{}` for `cjc emit`", other);
                process::exit(1);
            }
        }
        i += 1;
    }
    if ea.file.is_empty() {
        eprintln!("error: `cjc emit` requires a .cjc file argument");
        process::exit(1);
    }
    ea
}

/// Entry point for `cjc emit`.
pub fn run(args: &[String]) {
    let ea = parse_args(args);

    let source = match fs::read_to_string(&ea.file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: could not read `{}`: {}", ea.file, e);
            process::exit(1);
        }
    };

    let filename = ea.file.clone();

    let (program, diags) = cjc_parser::parse_source(&source);
    if diags.has_errors() {
        eprintln!("error: parse errors in `{}`", filename);
        let rendered = diags.render_all_color(&source, &filename, ea.output.use_color());
        eprint!("{}", rendered);
        process::exit(1);
    }

    match ea.stage {
        Stage::Ast => emit_ast(&ea, &program),
        Stage::Hir => emit_hir(&ea, &program),
        Stage::Mir => emit_mir(&ea, &program),
    }
}

// ---------------------------------------------------------------------------
// AST emission
// ---------------------------------------------------------------------------

fn emit_ast(ea: &EmitArgs, program: &cjc_ast::Program) {
    let pretty = cjc_ast::PrettyPrinter::new().print_program(program);

    match ea.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"stage\": \"AST\",");
            println!("  \"file\": \"{}\",", ea.file.replace('\\', "/"));
            println!("  \"declarations\": {},", program.declarations.len());
            // Escape the pretty-printed output for JSON
            let escaped = pretty.replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('\n', "\\n")
                .replace('\t', "\\t");
            println!("  \"output\": \"{}\"", escaped);
            println!("}}");
        }
        _ => {
            let header = format!("=== {} (AST) ===", ea.file.replace('\\', "/"));
            eprintln!("{}", output::colorize(ea.output, output::BOLD_CYAN, &header));
            eprintln!();
            println!("{}", pretty);
        }
    }
}

// ---------------------------------------------------------------------------
// HIR emission
// ---------------------------------------------------------------------------

fn emit_hir(ea: &EmitArgs, program: &cjc_ast::Program) {
    let mut lowering = cjc_hir::AstLowering::new();
    let hir = lowering.lower_program(program);

    match ea.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"stage\": \"HIR\",");
            println!("  \"file\": \"{}\",", ea.file.replace('\\', "/"));
            println!("  \"items\": {},", hir.items.len());
            let debug = format!("{:#?}", hir);
            let escaped = debug.replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('\n', "\\n")
                .replace('\t', "\\t");
            println!("  \"output\": \"{}\"", escaped);
            println!("}}");
        }
        _ => {
            let header = format!("=== {} (HIR) ===", ea.file.replace('\\', "/"));
            eprintln!("{}", output::colorize(ea.output, output::BOLD_CYAN, &header));
            eprintln!();
            println!("{:#?}", hir);
        }
    }
}

// ---------------------------------------------------------------------------
// MIR emission
// ---------------------------------------------------------------------------

fn emit_mir(ea: &EmitArgs, program: &cjc_ast::Program) {
    let mir = cjc_mir_exec::lower_to_mir(program);

    if ea.diff {
        // Show both unoptimized and a note about optimization
        let header = format!("=== {} (MIR — unoptimized) ===", ea.file.replace('\\', "/"));
        eprintln!("{}", output::colorize(ea.output, output::BOLD_CYAN, &header));
        eprintln!();
        let unopt = format_mir_program(&mir);
        println!("{}", unopt);

        eprintln!();
        let opt_header = format!("=== {} (MIR — optimized) ===", ea.file.replace('\\', "/"));
        eprintln!("{}", output::colorize(ea.output, output::BOLD_CYAN, &opt_header));
        eprintln!();
        // Lower again for a fresh copy, apply optimizer
        let mut opt_mir = cjc_mir_exec::lower_to_mir(program);
        cjc_mir::optimize::optimize_program(&mut opt_mir);
        let optimized = format_mir_program(&opt_mir);
        println!("{}", optimized);
        return;
    }

    if ea.opt {
        let mut opt_mir = cjc_mir_exec::lower_to_mir(program);
        cjc_mir::optimize::optimize_program(&mut opt_mir);

        match ea.output {
            OutputMode::Json => {
                emit_mir_json(ea, &opt_mir, true);
            }
            _ => {
                let header = format!("=== {} (MIR — optimized) ===", ea.file.replace('\\', "/"));
                eprintln!("{}", output::colorize(ea.output, output::BOLD_CYAN, &header));
                eprintln!();
                println!("{}", format_mir_program(&opt_mir));
            }
        }
    } else {
        match ea.output {
            OutputMode::Json => {
                emit_mir_json(ea, &mir, false);
            }
            _ => {
                let header = format!("=== {} (MIR) ===", ea.file.replace('\\', "/"));
                eprintln!("{}", output::colorize(ea.output, output::BOLD_CYAN, &header));
                eprintln!();
                println!("{}", format_mir_program(&mir));
            }
        }
    }
}

fn emit_mir_json(ea: &EmitArgs, mir: &cjc_mir::MirProgram, optimized: bool) {
    println!("{{");
    println!("  \"stage\": \"MIR\",");
    println!("  \"file\": \"{}\",", ea.file.replace('\\', "/"));
    println!("  \"optimized\": {},", optimized);
    println!("  \"functions\": {},", mir.functions.len());
    println!("  \"structs\": {},", mir.struct_defs.len());
    println!("  \"enums\": {},", mir.enum_defs.len());
    let formatted = format_mir_program(mir);
    let escaped = formatted.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\t', "\\t");
    println!("  \"output\": \"{}\"", escaped);
    println!("}}");
}

// ---------------------------------------------------------------------------
// MIR pretty-printer
// ---------------------------------------------------------------------------

/// Format an entire MIR program as readable pseudocode.
pub fn format_mir_program(program: &cjc_mir::MirProgram) -> String {
    let mut out = String::new();

    // Struct definitions
    for sd in &program.struct_defs {
        out.push_str(&format!("struct {} {{\n", sd.name));
        for (fname, ftype) in &sd.fields {
            out.push_str(&format!("  {}: {},\n", fname, ftype));
        }
        out.push_str("}\n\n");
    }

    // Enum definitions
    for ed in &program.enum_defs {
        out.push_str(&format!("enum {} {{\n", ed.name));
        for v in &ed.variants {
            if v.fields.is_empty() {
                out.push_str(&format!("  {},\n", v.name));
            } else {
                out.push_str(&format!("  {}({}),\n", v.name, v.fields.join(", ")));
            }
        }
        out.push_str("}\n\n");
    }

    // Functions
    for func in &program.functions {
        out.push_str(&format_mir_function(func));
        out.push('\n');
    }

    out
}

fn format_mir_function(func: &cjc_mir::MirFunction) -> String {
    let mut out = String::new();

    // Decorators
    for dec in &func.decorators {
        out.push_str(&format!("@{}\n", dec));
    }

    // Function signature
    let params: Vec<String> = func.params.iter().map(|p| {
        let mut s = String::new();
        if p.is_variadic { s.push_str("..."); }
        s.push_str(&p.name);
        s.push_str(": ");
        s.push_str(&p.ty_name);
        if let Some(ref def) = p.default {
            s.push_str(&format!(" = {}", format_mir_expr(def)));
        }
        s
    }).collect();

    let ret = func.return_type.as_deref().unwrap_or("void");
    let nogc = if func.is_nogc { " [nogc]" } else { "" };
    out.push_str(&format!("fn {}({}){} -> {}:\n", func.name, params.join(", "), nogc, ret));

    // Body
    format_mir_body(&func.body, 1, &mut out);

    out
}

fn format_mir_body(body: &cjc_mir::MirBody, indent: usize, out: &mut String) {
    let pad = "  ".repeat(indent);
    for stmt in &body.stmts {
        format_mir_stmt(stmt, indent, out);
    }
    if let Some(ref result) = body.result {
        out.push_str(&format!("{}=> {}\n", pad, format_mir_expr(result)));
    }
}

fn format_mir_stmt(stmt: &cjc_mir::MirStmt, indent: usize, out: &mut String) {
    let pad = "  ".repeat(indent);
    match stmt {
        cjc_mir::MirStmt::Let { name, mutable, init, alloc_hint } => {
            let mut_str = if *mutable { "mut " } else { "" };
            let hint_str = match alloc_hint {
                Some(cjc_mir::AllocHint::Stack) => " /* stack */",
                Some(cjc_mir::AllocHint::Arena) => " /* arena */",
                Some(cjc_mir::AllocHint::Rc) => " /* rc */",
                None => "",
            };
            out.push_str(&format!("{}let {}{} = {}{}\n", pad, mut_str, name, format_mir_expr(init), hint_str));
        }
        cjc_mir::MirStmt::Expr(e) => {
            out.push_str(&format!("{}{}\n", pad, format_mir_expr(e)));
        }
        cjc_mir::MirStmt::If { cond, then_body, else_body } => {
            out.push_str(&format!("{}if {}:\n", pad, format_mir_expr(cond)));
            format_mir_body(then_body, indent + 1, out);
            if let Some(ref eb) = else_body {
                out.push_str(&format!("{}else:\n", pad));
                format_mir_body(eb, indent + 1, out);
            }
        }
        cjc_mir::MirStmt::While { cond, body } => {
            out.push_str(&format!("{}while {}:\n", pad, format_mir_expr(cond)));
            format_mir_body(body, indent + 1, out);
        }
        cjc_mir::MirStmt::Return(Some(e)) => {
            out.push_str(&format!("{}return {}\n", pad, format_mir_expr(e)));
        }
        cjc_mir::MirStmt::Return(None) => {
            out.push_str(&format!("{}return\n", pad));
        }
        cjc_mir::MirStmt::Break => {
            out.push_str(&format!("{}break\n", pad));
        }
        cjc_mir::MirStmt::Continue => {
            out.push_str(&format!("{}continue\n", pad));
        }
        cjc_mir::MirStmt::NoGcBlock(body) => {
            out.push_str(&format!("{}nogc {{\n", pad));
            format_mir_body(body, indent + 1, out);
            out.push_str(&format!("{}}}\n", pad));
        }
    }
}

fn format_mir_expr(expr: &cjc_mir::MirExpr) -> String {
    match &expr.kind {
        cjc_mir::MirExprKind::IntLit(v) => format!("{}", v),
        cjc_mir::MirExprKind::FloatLit(v) => output::format_f64(*v, 6),
        cjc_mir::MirExprKind::BoolLit(v) => format!("{}", v),
        cjc_mir::MirExprKind::StringLit(s) => format!("{:?}", s),
        cjc_mir::MirExprKind::ByteStringLit(b) => format!("b{:?}", b),
        cjc_mir::MirExprKind::ByteCharLit(c) => format!("b'{}'", *c as char),
        cjc_mir::MirExprKind::RawStringLit(s) => format!("r{:?}", s),
        cjc_mir::MirExprKind::RawByteStringLit(b) => format!("rb{:?}", b),
        cjc_mir::MirExprKind::RegexLit { pattern, flags } => format!("/{}/{}", pattern, flags),
        cjc_mir::MirExprKind::TensorLit { rows } => {
            let row_strs: Vec<String> = rows.iter().map(|row| {
                let vals: Vec<String> = row.iter().map(format_mir_expr).collect();
                format!("[{}]", vals.join(", "))
            }).collect();
            format!("Tensor[{}]", row_strs.join(", "))
        }
        cjc_mir::MirExprKind::Var(name) => name.clone(),
        cjc_mir::MirExprKind::Binary { op, left, right } => {
            format!("({} {} {})", format_mir_expr(left), op, format_mir_expr(right))
        }
        cjc_mir::MirExprKind::Unary { op, operand } => {
            format!("({}{})", op, format_mir_expr(operand))
        }
        cjc_mir::MirExprKind::Call { callee, args } => {
            let arg_strs: Vec<String> = args.iter().map(format_mir_expr).collect();
            format!("{}({})", format_mir_expr(callee), arg_strs.join(", "))
        }
        cjc_mir::MirExprKind::Field { object, name } => {
            format!("{}.{}", format_mir_expr(object), name)
        }
        cjc_mir::MirExprKind::Index { object, index } => {
            format!("{}[{}]", format_mir_expr(object), format_mir_expr(index))
        }
        cjc_mir::MirExprKind::MultiIndex { object, indices } => {
            let idx_strs: Vec<String> = indices.iter().map(format_mir_expr).collect();
            format!("{}[{}]", format_mir_expr(object), idx_strs.join(", "))
        }
        cjc_mir::MirExprKind::Assign { target, value } => {
            format!("{} = {}", format_mir_expr(target), format_mir_expr(value))
        }
        cjc_mir::MirExprKind::Block(body) => {
            let mut s = String::from("{\n");
            format_mir_body(body, 2, &mut s);
            s.push('}');
            s
        }
        cjc_mir::MirExprKind::StructLit { name, fields } => {
            let field_strs: Vec<String> = fields.iter().map(|(n, e)| {
                format!("{}: {}", n, format_mir_expr(e))
            }).collect();
            format!("{} {{ {} }}", name, field_strs.join(", "))
        }
        cjc_mir::MirExprKind::ArrayLit(elems) => {
            let elem_strs: Vec<String> = elems.iter().map(format_mir_expr).collect();
            format!("[{}]", elem_strs.join(", "))
        }
        cjc_mir::MirExprKind::Col(name) => format!("${}", name),
        cjc_mir::MirExprKind::Lambda { params, body } => {
            let ps: Vec<String> = params.iter().map(|p| {
                format!("{}: {}", p.name, p.ty_name)
            }).collect();
            format!("|{}| {}", ps.join(", "), format_mir_expr(body))
        }
        cjc_mir::MirExprKind::MakeClosure { fn_name, captures } => {
            let cap_strs: Vec<String> = captures.iter().map(format_mir_expr).collect();
            format!("closure({}, [{}])", fn_name, cap_strs.join(", "))
        }
        cjc_mir::MirExprKind::If { cond, then_body, else_body } => {
            let mut s = format!("if {} {{ ", format_mir_expr(cond));
            let mut then_str = String::new();
            format_mir_body(then_body, 0, &mut then_str);
            s.push_str(then_str.trim());
            s.push_str(" }");
            if let Some(ref eb) = else_body {
                s.push_str(" else { ");
                let mut else_str = String::new();
                format_mir_body(eb, 0, &mut else_str);
                s.push_str(else_str.trim());
                s.push_str(" }");
            }
            s
        }
        cjc_mir::MirExprKind::Match { scrutinee, arms } => {
            let mut s = format!("match {} {{ ", format_mir_expr(scrutinee));
            for (i, arm) in arms.iter().enumerate() {
                let body_str = if let Some(ref result) = arm.body.result {
                    format_mir_expr(result)
                } else if !arm.body.stmts.is_empty() {
                    "{ ... }".to_string()
                } else {
                    "()".to_string()
                };
                s.push_str(&format!("{:?} => {}", arm.pattern, body_str));
                if i + 1 < arms.len() { s.push_str(", "); }
            }
            s.push_str(" }");
            s
        }
        cjc_mir::MirExprKind::VariantLit { enum_name, variant, fields } => {
            if fields.is_empty() {
                format!("{}::{}", enum_name, variant)
            } else {
                let f_strs: Vec<String> = fields.iter().map(format_mir_expr).collect();
                format!("{}::{}({})", enum_name, variant, f_strs.join(", "))
            }
        }
        cjc_mir::MirExprKind::TupleLit(elems) => {
            let elem_strs: Vec<String> = elems.iter().map(format_mir_expr).collect();
            format!("({})", elem_strs.join(", "))
        }
        // Linalg opcodes
        cjc_mir::MirExprKind::LinalgLU { operand } => {
            format!("linalg_lu({})", format_mir_expr(operand))
        }
        cjc_mir::MirExprKind::LinalgQR { operand } => {
            format!("linalg_qr({})", format_mir_expr(operand))
        }
        cjc_mir::MirExprKind::LinalgCholesky { operand } => {
            format!("linalg_cholesky({})", format_mir_expr(operand))
        }
        cjc_mir::MirExprKind::LinalgInv { operand } => {
            format!("linalg_inv({})", format_mir_expr(operand))
        }
        cjc_mir::MirExprKind::Broadcast { operand, target_shape } => {
            let shape_strs: Vec<String> = target_shape.iter().map(format_mir_expr).collect();
            format!("broadcast({}, [{}])", format_mir_expr(operand), shape_strs.join(", "))
        }
        // Fallback for any variants we haven't explicitly handled
        #[allow(unreachable_patterns)]
        _ => format!("{:?}", expr.kind),
    }
}

pub fn print_help() {
    eprintln!("cjc emit -- Dump intermediate representations at any pipeline stage");
    eprintln!();
    eprintln!("Usage: cjc emit <file.cjc> [flags]");
    eprintln!();
    eprintln!("Stages:");
    eprintln!("  --stage ast        Pretty-print the AST");
    eprintln!("  --stage hir        Print the HIR (desugared form)");
    eprintln!("  --stage mir        Print the MIR (default)");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  -s, --stage <S>    Pipeline stage: ast, hir, mir (default: mir)");
    eprintln!("  --opt              Show optimized MIR (CF + DCE applied)");
    eprintln!("  --diff             Show both unoptimized and optimized MIR");
    eprintln!("  --plain            Plain text output");
    eprintln!("  --json             JSON output");
    eprintln!("  --color            Color output (default)");
}
