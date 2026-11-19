//! Fuzz-driven generator of small, well-formed CJC-Lang programs.
//!
//! Raw-byte fuzzing almost never produces a program that parses cleanly, so
//! targets that only act on parseable programs (MIR verifier, AST validator,
//! AST metrics, optimizer and executor parity) were in practice checked only
//! on the empty program. `gen_program` decodes arbitrary bytes into a program
//! that is well-formed by construction:
//!
//! - typed `let mut` bindings for `i64` / `f64` / `bool` state
//! - helper `fn`s whose bodies only call *earlier* helpers (no recursion)
//! - `if`/`else` statements and expressions, bounded `for k in 0..N` loops
//!   (N ≤ 3, no `while`), so every program terminates quickly
//! - every binary expression parenthesised, so precedence never matters
//! - integer edge values (`i64::MAX`, `i64::MIN`, 0, -1) and `/ % **`, so
//!   wrapping, division-by-zero errors and constant folding all get exercised
//!
//! Decoding is total and deterministic: once the bytes run out every choice
//! reads 0, which always selects a terminal (literal / variable) form.

pub fn gen_program(data: &[u8]) -> String {
    let mut g = Gen { data, pos: 0, out: String::new(), loop_vars: Vec::new(), in_g0: false };
    g.program();
    g.out
}

const INT_VARS: [&str; 2] = ["i0", "i1"];
const MAX_EXPR_DEPTH: usize = 3;
const MAX_STMT_DEPTH: usize = 3;
const MAX_BLOCK_STMTS: usize = 4;
const MAX_TOP_STMTS: usize = 12;

struct Gen<'a> {
    data: &'a [u8],
    pos: usize,
    out: String,
    /// `for` loop variables in scope (usable as `i64` operands).
    loop_vars: Vec<String>,
    /// Generating `g0`'s own body: calls to `g0` are forbidden.
    in_g0: bool,
}

impl Gen<'_> {
    fn byte(&mut self) -> u8 {
        let b = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b
    }

    fn pick(&mut self, n: usize) -> usize {
        self.byte() as usize % n
    }

    fn exhausted(&self) -> bool {
        self.pos >= self.data.len()
    }

    fn program(&mut self) {
        // Helpers. `f1` may call `f0`; `g0` is the float helper.
        let body = self.int_expr_in(0, &["a", "b"], 0);
        self.out.push_str(&format!("fn f0(a: i64, b: i64) -> i64 {{\n    {body}\n}}\n"));
        let body = self.int_expr_in(0, &["a", "b"], 1);
        self.out.push_str(&format!("fn f1(a: i64, b: i64) -> i64 {{\n    {body}\n}}\n"));
        // `g0`'s body must not call `g0` (unbounded recursion).
        self.in_g0 = true;
        let body = self.float_expr(0);
        self.in_g0 = false;
        self.out.push_str(&format!("fn g0(x: f64) -> f64 {{\n    {}\n}}\n", body.replace("x0", "x")));

        let i0 = self.int_lit();
        let i1 = self.int_lit();
        let x0 = self.float_lit();
        let b0 = self.bool_lit();
        self.out.push_str(&format!(
            "let mut i0: i64 = {i0};\nlet mut i1: i64 = {i1};\nlet mut x0: f64 = {x0};\nlet mut b0: bool = {b0};\n"
        ));

        for _ in 0..MAX_TOP_STMTS {
            if self.exhausted() {
                break;
            }
            self.stmt(0, 0);
        }
        self.out.push_str("print(i0);\nprint(i1);\nprint(x0);\nprint(b0);\n");
    }

    fn indent(&mut self, level: usize) {
        for _ in 0..level {
            self.out.push_str("    ");
        }
    }

    fn block(&mut self, depth: usize, level: usize) {
        self.out.push_str("{\n");
        let n = 1 + self.pick(MAX_BLOCK_STMTS);
        for _ in 0..n {
            self.stmt(depth + 1, level + 1);
        }
        self.indent(level);
        self.out.push('}');
    }

    fn stmt(&mut self, depth: usize, level: usize) {
        self.indent(level);
        let choices = if depth >= MAX_STMT_DEPTH { 5 } else { 7 };
        match self.pick(choices) {
            0 => {
                let v = INT_VARS[self.pick(2)];
                let e = self.int_expr(0);
                self.out.push_str(&format!("{v} = {e};"));
            }
            1 => {
                let e = self.float_expr(0);
                self.out.push_str(&format!("x0 = {e};"));
            }
            2 => {
                let e = self.bool_expr(0);
                self.out.push_str(&format!("b0 = {e};"));
            }
            3 => {
                let e = self.int_expr(0);
                self.out.push_str(&format!("print({e});"));
            }
            4 => {
                let e = self.float_expr(0);
                self.out.push_str(&format!("print({e});"));
            }
            5 => {
                let c = self.bool_expr(0);
                self.out.push_str(&format!("if {c} "));
                self.block(depth, level);
                self.out.push_str(" else ");
                self.block(depth, level);
            }
            _ => {
                let k = format!("k{}", self.loop_vars.len());
                let n = 1 + self.pick(3);
                self.out.push_str(&format!("for {k} in 0..{n} "));
                self.loop_vars.push(k);
                self.block(depth, level);
                self.loop_vars.pop();
            }
        }
        self.out.push('\n');
    }

    fn int_lit(&mut self) -> String {
        const LITS: [&str; 10] = [
            "0", "1", "2", "7", "100", "(-1)", "(-13)",
            "9223372036854775807", "(-9223372036854775807 - 1)", "65536",
        ];
        LITS[self.pick(LITS.len())].to_string()
    }

    fn float_lit(&mut self) -> String {
        const LITS: [&str; 8] = ["0.0", "1.0", "0.5", "(-2.25)", "3.75", "1000000.0", "0.001", "(-0.0)"];
        LITS[self.pick(LITS.len())].to_string()
    }

    fn bool_lit(&mut self) -> &'static str {
        if self.pick(2) == 0 { "false" } else { "true" }
    }

    fn int_expr(&mut self, depth: usize) -> String {
        let mut vars: Vec<String> = INT_VARS.iter().map(|s| s.to_string()).collect();
        vars.extend(self.loop_vars.iter().cloned());
        let vars: Vec<&str> = vars.iter().map(|s| s.as_str()).collect();
        self.int_expr_in(depth, &vars, 2)
    }

    /// Integer expression over `vars`, calling only helpers `f0..f{callable}`.
    fn int_expr_in(&mut self, depth: usize, vars: &[&str], callable: usize) -> String {
        let choices = if depth >= MAX_EXPR_DEPTH { 2 } else { 12 };
        match self.pick(choices) {
            0 => self.int_lit(),
            1 => vars[self.pick(vars.len())].to_string(),
            2..=4 => {
                let op = ["+", "-", "*"][self.pick(3)];
                let (l, r) = (self.int_expr_in(depth + 1, vars, callable), self.int_expr_in(depth + 1, vars, callable));
                format!("({l} {op} {r})")
            }
            5 => {
                let op = ["/", "%"][self.pick(2)];
                let (l, r) = (self.int_expr_in(depth + 1, vars, callable), self.int_expr_in(depth + 1, vars, callable));
                format!("({l} {op} {r})")
            }
            6 => {
                let base = self.int_expr_in(depth + 1, vars, callable);
                let exp = self.pick(6);
                format!("({base} ** {exp})")
            }
            7 => format!("(-{})", self.int_expr_in(depth + 1, vars, callable)),
            8 if callable > 0 => {
                let f = self.pick(callable);
                let (a, b) = (self.int_expr_in(depth + 1, vars, callable), self.int_expr_in(depth + 1, vars, callable));
                format!("f{f}({a}, {b})")
            }
            9 => {
                let c = self.int_cmp(depth + 1, vars, callable);
                let (a, b) = (self.int_expr_in(depth + 1, vars, callable), self.int_expr_in(depth + 1, vars, callable));
                format!("(if {c} {{ {a} }} else {{ {b} }})")
            }
            _ => vars[self.pick(vars.len())].to_string(),
        }
    }

    fn int_cmp(&mut self, depth: usize, vars: &[&str], callable: usize) -> String {
        let op = ["<", "<=", "==", "!=", ">", ">="][self.pick(6)];
        let (l, r) = (self.int_expr_in(depth, vars, callable), self.int_expr_in(depth, vars, callable));
        format!("({l} {op} {r})")
    }

    fn float_expr(&mut self, depth: usize) -> String {
        let choices = if depth >= MAX_EXPR_DEPTH { 2 } else { 7 };
        match self.pick(choices) {
            0 => self.float_lit(),
            1 => "x0".to_string(),
            2..=4 => {
                let op = ["+", "-", "*", "/"][self.pick(4)];
                let (l, r) = (self.float_expr(depth + 1), self.float_expr(depth + 1));
                format!("({l} {op} {r})")
            }
            5 => format!("(-{})", self.float_expr(depth + 1)),
            _ if self.in_g0 => "x0".to_string(),
            _ => format!("g0({})", self.float_expr(depth + 1)),
        }
    }

    fn bool_expr(&mut self, depth: usize) -> String {
        let choices = if depth >= MAX_EXPR_DEPTH { 2 } else { 7 };
        match self.pick(choices) {
            0 => self.bool_lit().to_string(),
            1 => "b0".to_string(),
            2 => {
                let mut vars: Vec<String> = INT_VARS.iter().map(|s| s.to_string()).collect();
                vars.extend(self.loop_vars.iter().cloned());
                let vars: Vec<&str> = vars.iter().map(|s| s.as_str()).collect();
                self.int_cmp(depth + 1, &vars, 2)
            }
            3 => {
                let op = ["<", "<=", "==", "!=", ">", ">="][self.pick(6)];
                let (l, r) = (self.float_expr(depth + 1), self.float_expr(depth + 1));
                format!("({l} {op} {r})")
            }
            4 => format!("(!{})", self.bool_expr(depth + 1)),
            _ => {
                let op = ["&&", "||"][self.pick(2)];
                let (l, r) = (self.bool_expr(depth + 1), self.bool_expr(depth + 1));
                format!("({l} {op} {r})")
            }
        }
    }
}
