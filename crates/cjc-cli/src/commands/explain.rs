//! `cjcl explain` — Show desugared/lowered form of a CJC program.
//!
//! Lowers the source to HIR and presents the desugared forms:
//! for→while, match→branches, closure captures. Useful for understanding
//! what the compiler actually sees after desugaring.

use std::fs;
use std::process;
use crate::output::{self, OutputMode};

pub struct ExplainArgs {
    pub file: String,
    pub output: OutputMode,
    pub verbose: bool,
}

impl Default for ExplainArgs {
    fn default() -> Self {
        Self {
            file: String::new(),
            output: OutputMode::Color,
            verbose: false,
        }
    }
}

pub fn parse_args(args: &[String]) -> ExplainArgs {
    let mut ea = ExplainArgs::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-v" | "--verbose" => ea.verbose = true,
            "--plain" => ea.output = OutputMode::Plain,
            "--json" => ea.output = OutputMode::Json,
            "--color" => ea.output = OutputMode::Color,
            other if !other.starts_with('-') => ea.file = other.to_string(),
            other => {
                eprintln!("error: unknown flag `{}` for `cjcl explain`", other);
                process::exit(1);
            }
        }
        i += 1;
    }
    if ea.file.is_empty() {
        eprintln!("error: `cjcl explain` requires a .cjcl file argument");
        process::exit(1);
    }
    ea
}

pub fn run(args: &[String]) {
    let ea = parse_args(args);

    let source = match fs::read_to_string(&ea.file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: could not read `{}`: {}", ea.file, e);
            process::exit(1);
        }
    };

    let filename = ea.file.replace('\\', "/");

    let (program, diags) = cjc_parser::parse_source(&source);
    if diags.has_errors() {
        let rendered = diags.render_all_color(&source, &filename, ea.output.use_color());
        eprint!("{}", rendered);
        process::exit(1);
    }

    // Lower to HIR
    let mut lowering = cjc_hir::AstLowering::new();
    let hir = lowering.lower_program(&program);

    // Extract functions and structs from HirItem list
    let mut functions: Vec<&cjc_hir::HirFn> = Vec::new();
    let mut structs: Vec<&cjc_hir::HirStructDef> = Vec::new();
    for item in &hir.items {
        match item {
            cjc_hir::HirItem::Fn(f) => functions.push(f),
            cjc_hir::HirItem::Struct(s) => structs.push(s),
            _ => {}
        }
    }

    match ea.output {
        OutputMode::Json => {
            println!("{{");
            println!("  \"file\": \"{}\",", filename);
            println!("  \"functions\": [");
            for (i, func) in functions.iter().enumerate() {
                println!("    {{");
                println!("      \"name\": \"{}\",", func.name);
                println!("      \"params\": [{}],",
                    func.params.iter()
                        .map(|p| format!("\"{}\"", p.name))
                        .collect::<Vec<_>>().join(", "));
                println!("      \"is_nogc\": {}", func.is_nogc);
                print!("    }}");
                if i + 1 < functions.len() { print!(","); }
                println!();
            }
            println!("  ]");
            println!("}}");
        }
        _ => {
            eprintln!("{} Explaining `{}`...",
                output::colorize(ea.output, output::BOLD_CYAN, "[explain]"),
                filename);
            eprintln!();

            if functions.is_empty() {
                eprintln!("No functions found.");
                return;
            }

            for func in &functions {
                // Function header
                let params_str: Vec<String> = func.params.iter()
                    .map(|p| format!("{}: {}", p.name, p.ty_name))
                    .collect();
                let sig = format!("fn {}({})", func.name, params_str.join(", "));
                eprintln!("{}", output::colorize(ea.output, output::BOLD, &sig));

                // Annotations
                if func.is_nogc {
                    eprintln!("  {} #[nogc]",
                        output::colorize(ea.output, output::CYAN, "annotation:"));
                }
                if !func.decorators.is_empty() {
                    eprintln!("  {} @{}",
                        output::colorize(ea.output, output::CYAN, "decorators:"),
                        func.decorators.join(", @"));
                }

                // Body summary
                let stmt_count = func.body.stmts.len();
                let has_result = func.body.expr.is_some();
                eprintln!("  {} {} statements{}",
                    output::colorize(ea.output, output::CYAN, "body:"),
                    stmt_count,
                    if has_result { " + tail expression" } else { "" });

                // Verbose: print HIR body
                if ea.verbose {
                    eprintln!("  {} {{", output::colorize(ea.output, output::DIM, "lowered"));
                    for stmt in &func.body.stmts {
                        eprintln!("    {:#?}", stmt);
                    }
                    if let Some(ref result) = func.body.expr {
                        eprintln!("    => {:#?}", result);
                    }
                    eprintln!("  }}");
                }

                eprintln!();
            }

            // Struct summary
            if !structs.is_empty() {
                eprintln!("{}", output::colorize(ea.output, output::BOLD, "Structs:"));
                for s in &structs {
                    let fields: Vec<String> = s.fields.iter()
                        .map(|(name, ty)| format!("{}: {}", name, ty))
                        .collect();
                    eprintln!("  struct {} {{ {} }}", s.name, fields.join(", "));
                }
                eprintln!();
            }

            // Summary table
            let mut t = crate::table::Table::new(vec!["Element", "Count"]);
            t.add_row_owned(vec!["Functions".into(), format!("{}", functions.len())]);
            t.add_row_owned(vec!["Structs".into(), format!("{}", structs.len())]);
            let nogc_count = functions.iter().filter(|f| f.is_nogc).count();
            t.add_row_owned(vec!["NoGC functions".into(), format!("{}", nogc_count)]);
            t.add_row_owned(vec!["Total items".into(), format!("{}", hir.items.len())]);
            eprint!("{}", t.render());
        }
    }
}

pub fn print_help() {
    eprintln!("cjcl explain — Show desugared/lowered form of a CJC program");
    eprintln!();
    eprintln!("Usage: cjcl explain <file.cjcl> [flags]");
    eprintln!();
    eprintln!("Shows:");
    eprintln!("  - Function signatures after lowering");
    eprintln!("  - Desugared loop/match forms (with --verbose)");
    eprintln!("  - NoGC annotations and decorators");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  -v, --verbose   Show full HIR body for each function");
    eprintln!("  --plain         Plain text output");
    eprintln!("  --json          JSON output");
    eprintln!("  --color         Color output (default)");
}
