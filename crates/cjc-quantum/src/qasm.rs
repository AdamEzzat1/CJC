//! OpenQASM 2.0 import and export for circuits.
//!
//! This is interchange with other classical simulators (Qiskit Aer, Cirq,
//! Qulacs, …): the same circuit can be run everywhere and the results
//! compared. It is not a full OpenQASM implementation.
//!
//! # Supported subset
//!
//! Only gates that CJC represents **exactly** are accepted, so an import
//! never changes amplitudes (not even by a global phase):
//!
//! | QASM | CJC |
//! |---|---|
//! | `h x y z s t` | `H X Y Z S T` |
//! | `rx(θ) ry(θ) rz(θ)` | `Rx Ry Rz` (same conventions as qelib1: `rz(θ) = diag(e^{-iθ/2}, e^{iθ/2})`) |
//! | `cx` / `CX`, `cz`, `swap`, `ccx` | `CNOT CZ SWAP Toffoli` |
//! | `id`, `barrier` | ignored (no effect on the state) |
//! | `creg`, trailing `measure` | accepted and ignored: CJC circuits measure at the end anyway |
//!
//! Everything else is an error naming the construct: `u1/u2/u3/u/p`,
//! `sdg/tdg` and the other qelib1 gates, `gate`/`opaque` definitions, `reset`,
//! `if`, and a gate after a `measure` (mid-circuit measurement).
//!
//! Qubit `k` of the flattened registers (registers in declaration order) is
//! CJC qubit `k`, which is also Qiskit's little-endian basis ordering.
//!
//! # Exactness
//!
//! Export prints angles with Rust's shortest round-trip formatting, and import
//! parses number literals with `str::parse::<f64>` (correctly rounded), so
//! `from_qasm(to_qasm(c))` reproduces every gate and angle bit for bit.
//! Parameter expressions (`pi/2`, `-3*pi/4`, `cos(0.1)`, …) are evaluated left to
//! right in binary64; functions use `cjc_repro::dmath`, so the result is the
//! same on every platform.

use crate::circuit::Circuit;
use crate::gates::Gate;
use crate::pure::{PureCircuit, PureGate};
use cjc_repro::dmath;

/// Largest circuit accepted from QASM (the dense statevector cap).
pub const MAX_QASM_QUBITS: usize = 26;

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

/// A binary64 value as a QASM real literal that parses back to the same bits.
fn fmt_angle(x: f64) -> String {
    let s = format!("{:?}", x); // shortest round-trip representation
    // QASM 2.0 reals need a decimal point before any exponent ("1e-5" → "1.0e-5").
    if s.contains('.') || s.contains("inf") || s.contains("NaN") {
        s
    } else if let Some(i) = s.find('e') {
        format!("{}.0{}", &s[..i], &s[i..])
    } else {
        format!("{}.0", s)
    }
}

fn header(n_qubits: usize) -> String {
    format!("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[{}];\n", n_qubits)
}

fn gate_line(g: &Gate) -> String {
    match *g {
        Gate::H(q) => format!("h q[{}];", q),
        Gate::X(q) => format!("x q[{}];", q),
        Gate::Y(q) => format!("y q[{}];", q),
        Gate::Z(q) => format!("z q[{}];", q),
        Gate::S(q) => format!("s q[{}];", q),
        Gate::T(q) => format!("t q[{}];", q),
        Gate::Rx(q, t) => format!("rx({}) q[{}];", fmt_angle(t), q),
        Gate::Ry(q, t) => format!("ry({}) q[{}];", fmt_angle(t), q),
        Gate::Rz(q, t) => format!("rz({}) q[{}];", fmt_angle(t), q),
        Gate::CNOT(c, t) => format!("cx q[{}],q[{}];", c, t),
        Gate::CZ(a, b) => format!("cz q[{}],q[{}];", a, b),
        Gate::SWAP(a, b) => format!("swap q[{}],q[{}];", a, b),
        Gate::Toffoli(a, b, c) => format!("ccx q[{}],q[{}],q[{}];", a, b, c),
    }
}

/// OpenQASM 2.0 text for a circuit.
pub fn to_qasm(c: &Circuit) -> String {
    let mut s = header(c.n_qubits());
    for g in c.gates() {
        s.push_str(&gate_line(g));
        s.push('\n');
    }
    s
}

/// OpenQASM 2.0 text for a pure-backend circuit.
pub fn pure_to_qasm(c: &PureCircuit) -> String {
    let mut s = header(c.n_qubits);
    for g in &c.gates {
        s.push_str(&gate_line(&pure_to_gate(g)));
        s.push('\n');
    }
    s
}

fn pure_to_gate(g: &PureGate) -> Gate {
    match *g {
        PureGate::H(q) => Gate::H(q),
        PureGate::X(q) => Gate::X(q),
        PureGate::Y(q) => Gate::Y(q),
        PureGate::Z(q) => Gate::Z(q),
        PureGate::S(q) => Gate::S(q),
        PureGate::T(q) => Gate::T(q),
        PureGate::Rx(q, t) => Gate::Rx(q, t),
        PureGate::Ry(q, t) => Gate::Ry(q, t),
        PureGate::Rz(q, t) => Gate::Rz(q, t),
        PureGate::CNOT(a, b) => Gate::CNOT(a, b),
        PureGate::CZ(a, b) => Gate::CZ(a, b),
        PureGate::SWAP(a, b) => Gate::SWAP(a, b),
        PureGate::Toffoli(a, b, c) => Gate::Toffoli(a, b, c),
    }
}

fn gate_to_pure(g: &Gate) -> PureGate {
    match *g {
        Gate::H(q) => PureGate::H(q),
        Gate::X(q) => PureGate::X(q),
        Gate::Y(q) => PureGate::Y(q),
        Gate::Z(q) => PureGate::Z(q),
        Gate::S(q) => PureGate::S(q),
        Gate::T(q) => PureGate::T(q),
        Gate::Rx(q, t) => PureGate::Rx(q, t),
        Gate::Ry(q, t) => PureGate::Ry(q, t),
        Gate::Rz(q, t) => PureGate::Rz(q, t),
        Gate::CNOT(a, b) => PureGate::CNOT(a, b),
        Gate::CZ(a, b) => PureGate::CZ(a, b),
        Gate::SWAP(a, b) => PureGate::SWAP(a, b),
        Gate::Toffoli(a, b, c) => PureGate::Toffoli(a, b, c),
    }
}

// ---------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------

/// Parse OpenQASM 2.0 into a circuit.
pub fn from_qasm(src: &str) -> Result<Circuit, String> {
    let (n, gates) = parse(src)?;
    let mut c = Circuit::new(n);
    for g in gates {
        c.add(g);
    }
    Ok(c)
}

/// Parse OpenQASM 2.0 into a pure-backend circuit.
pub fn pure_from_qasm(src: &str) -> Result<PureCircuit, String> {
    let (n, gates) = parse(src)?;
    let mut c = PureCircuit::new(n);
    for g in &gates {
        c.add(gate_to_pure(g));
    }
    Ok(c)
}

struct Register {
    name: String,
    offset: usize,
    size: usize,
}

/// A statement with the 1-based line it starts on.
struct Stmt {
    text: String,
    line: usize,
}

fn statements(src: &str) -> Vec<Stmt> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut start_line = 1;
    let mut line = 1;
    for raw in src.split('\n') {
        let code = match raw.find("//") {
            Some(i) => &raw[..i],
            None => raw,
        };
        for ch in code.chars() {
            if ch == ';' {
                let t = cur.trim().to_string();
                if !t.is_empty() {
                    out.push(Stmt { text: t, line: start_line });
                }
                cur.clear();
            } else {
                if cur.trim().is_empty() && !ch.is_whitespace() {
                    start_line = line;
                }
                cur.push(ch);
            }
        }
        cur.push(' ');
        line += 1;
    }
    if !cur.trim().is_empty() {
        out.push(Stmt {
            text: cur.trim().to_string(),
            line: start_line,
        });
    }
    out
}

fn parse(src: &str) -> Result<(usize, Vec<Gate>), String> {
    let stmts = statements(src);
    if stmts.is_empty() {
        return Err("QASM: empty program (expected \"OPENQASM 2.0;\")".into());
    }
    let first = &stmts[0];
    let version = first.text.strip_prefix("OPENQASM").map(str::trim);
    match version {
        Some(v) if v.starts_with("2.") || v == "2" => {}
        Some(v) => {
            return Err(format!(
                "QASM line {}: OpenQASM {} is not supported (only 2.0)",
                first.line, v
            ))
        }
        None => {
            return Err(format!(
                "QASM line {}: program must start with \"OPENQASM 2.0;\"",
                first.line
            ))
        }
    }

    let mut regs: Vec<Register> = Vec::new();
    let mut n_qubits = 0usize;
    let mut gates = Vec::new();
    let mut measured = false;

    for st in &stmts[1..] {
        let err = |msg: String| format!("QASM line {}: {}", st.line, msg);
        let (head, rest) = split_head(&st.text);
        match head {
            "include" => {
                let file = rest.trim().trim_matches('"');
                if file != "qelib1.inc" {
                    return Err(err(format!("only \"qelib1.inc\" can be included, got \"{}\"", file)));
                }
            }
            "qreg" => {
                let (name, size) = parse_decl(rest).map_err(err)?;
                if regs.iter().any(|r| r.name == name) {
                    return Err(err(format!("register '{}' declared twice", name)));
                }
                if size == 0 {
                    return Err(err(format!("register '{}' has size 0", name)));
                }
                n_qubits = n_qubits.saturating_add(size);
                if n_qubits > MAX_QASM_QUBITS {
                    return Err(err(format!(
                        "{} qubits declared; the dense simulator supports at most {}",
                        n_qubits, MAX_QASM_QUBITS
                    )));
                }
                regs.push(Register {
                    name,
                    offset: n_qubits - size,
                    size,
                });
            }
            "creg" => {
                parse_decl(rest).map_err(err)?;
            }
            "barrier" => {}
            "measure" => measured = true,
            "gate" | "opaque" => {
                return Err(err("custom gate definitions are not supported".into()))
            }
            "reset" => return Err(err("reset is not supported".into())),
            "if" => return Err(err("classically controlled gates (if) are not supported".into())),
            _ => {
                let (name, params, operands) = parse_gate_call(&st.text).map_err(err)?;
                if measured {
                    return Err(err(format!(
                        "gate '{}' after a measurement: mid-circuit measurement is not supported",
                        name
                    )));
                }
                let (arity, n_params) = match name.as_str() {
                    "h" | "x" | "y" | "z" | "s" | "t" | "id" => (1, 0),
                    "rx" | "ry" | "rz" => (1, 1),
                    "cx" | "CX" | "cz" | "swap" => (2, 0),
                    "ccx" => (3, 0),
                    other => {
                        return Err(err(format!(
                            "unsupported gate '{}'; supported: h x y z s t rx ry rz cx cz swap ccx id",
                            other
                        )))
                    }
                };
                if params.len() != n_params {
                    return Err(err(format!(
                        "gate '{}' takes {} parameter(s), got {}",
                        name,
                        n_params,
                        params.len()
                    )));
                }
                if operands.len() != arity {
                    return Err(err(format!(
                        "gate '{}' takes {} qubit operand(s), got {}",
                        name,
                        arity,
                        operands.len()
                    )));
                }
                let angle = match params.first() {
                    Some(p) => {
                        let v = eval_expr(p).map_err(err)?;
                        if !v.is_finite() {
                            return Err(err(format!("parameter '{}' is not finite", p)));
                        }
                        Some(v)
                    }
                    None => None,
                };
                // Resolve operands; a whole register broadcasts.
                let resolved: Vec<Vec<usize>> = operands
                    .iter()
                    .map(|o| resolve_operand(o, &regs))
                    .collect::<Result<_, _>>()
                    .map_err(err)?;
                let width = resolved.iter().map(Vec::len).max().unwrap_or(1);
                if resolved.iter().any(|r| r.len() != 1 && r.len() != width) {
                    return Err(err("broadcast registers must have equal sizes".into()));
                }
                for i in 0..width {
                    let qs: Vec<usize> = resolved
                        .iter()
                        .map(|r| if r.len() == 1 { r[0] } else { r[i] })
                        .collect();
                    for a in 0..qs.len() {
                        for b in (a + 1)..qs.len() {
                            if qs[a] == qs[b] {
                                return Err(err(format!(
                                    "gate '{}' has duplicate qubit operand {}",
                                    name, qs[a]
                                )));
                            }
                        }
                    }
                    let g = match name.as_str() {
                        "h" => Gate::H(qs[0]),
                        "x" => Gate::X(qs[0]),
                        "y" => Gate::Y(qs[0]),
                        "z" => Gate::Z(qs[0]),
                        "s" => Gate::S(qs[0]),
                        "t" => Gate::T(qs[0]),
                        "id" => continue,
                        "rx" => Gate::Rx(qs[0], angle.unwrap()),
                        "ry" => Gate::Ry(qs[0], angle.unwrap()),
                        "rz" => Gate::Rz(qs[0], angle.unwrap()),
                        "cx" | "CX" => Gate::CNOT(qs[0], qs[1]),
                        "cz" => Gate::CZ(qs[0], qs[1]),
                        "swap" => Gate::SWAP(qs[0], qs[1]),
                        _ => Gate::Toffoli(qs[0], qs[1], qs[2]),
                    };
                    gates.push(g);
                }
            }
        }
    }
    if n_qubits == 0 {
        return Err("QASM: no qreg declared".into());
    }
    Ok((n_qubits, gates))
}

/// First word and the remainder (`"qreg q[3]"` → `("qreg", " q[3]")`).
fn split_head(s: &str) -> (&str, &str) {
    let end = s
        .find(|c: char| c.is_whitespace() || c == '(')
        .unwrap_or(s.len());
    (&s[..end], &s[end..])
}

fn is_ident(s: &str) -> bool {
    let mut cs = s.chars();
    matches!(cs.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && cs.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// `name[size]` in a register declaration.
fn parse_decl(rest: &str) -> Result<(String, usize), String> {
    let r = rest.trim();
    let open = r.find('[').ok_or_else(|| format!("expected name[size], got '{}'", r))?;
    let close = r.rfind(']').ok_or_else(|| format!("expected name[size], got '{}'", r))?;
    if close < open || !r[close + 1..].trim().is_empty() {
        return Err(format!("expected name[size], got '{}'", r));
    }
    let name = r[..open].trim();
    if !is_ident(name) {
        return Err(format!("invalid register name '{}'", name));
    }
    let size: usize = r[open + 1..close]
        .trim()
        .parse()
        .map_err(|_| format!("invalid register size in '{}'", r))?;
    Ok((name.to_string(), size))
}

/// `name(p1, p2) a, b[1]` → (name, params, operands).
fn parse_gate_call(s: &str) -> Result<(String, Vec<String>, Vec<String>), String> {
    let (name, mut rest) = split_head(s);
    if !is_ident(name) {
        return Err(format!("cannot parse statement '{}'", s));
    }
    let mut params = Vec::new();
    rest = rest.trim_start();
    if let Some(r) = rest.strip_prefix('(') {
        let mut depth = 1usize;
        let mut end = None;
        for (i, ch) in r.char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let end = end.ok_or_else(|| format!("unbalanced parentheses in '{}'", s))?;
        params = split_top_level(&r[..end]);
        rest = &r[end + 1..];
    }
    let operands: Vec<String> = rest
        .split(',')
        .map(|o| o.trim().to_string())
        .filter(|o| !o.is_empty())
        .collect();
    Ok((name.to_string(), params, operands))
}

fn split_top_level(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for ch in s.chars() {
        match ch {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                out.push(cur.trim().to_string());
                cur.clear();
                continue;
            }
            _ => {}
        }
        cur.push(ch);
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

/// `q[2]` → [offset+2]; `q` → every qubit of q.
fn resolve_operand(o: &str, regs: &[Register]) -> Result<Vec<usize>, String> {
    let (name, idx) = match o.find('[') {
        Some(open) => {
            let close = o
                .rfind(']')
                .ok_or_else(|| format!("invalid qubit operand '{}'", o))?;
            if close < open || !o[close + 1..].trim().is_empty() {
                return Err(format!("invalid qubit operand '{}'", o));
            }
            let i: usize = o[open + 1..close]
                .trim()
                .parse()
                .map_err(|_| format!("invalid qubit index in '{}'", o))?;
            (o[..open].trim(), Some(i))
        }
        None => (o.trim(), None),
    };
    let reg = regs
        .iter()
        .find(|r| r.name == name)
        .ok_or_else(|| format!("unknown quantum register '{}'", name))?;
    match idx {
        Some(i) if i < reg.size => Ok(vec![reg.offset + i]),
        Some(i) => Err(format!(
            "qubit {}[{}] out of range (register size {})",
            name, i, reg.size
        )),
        None => Ok((reg.offset..reg.offset + reg.size).collect()),
    }
}

// ---------------------------------------------------------------------------
// Parameter expressions: + - * / ^, unary minus, parentheses, pi, numbers,
// and sin cos tan exp ln sqrt.
// ---------------------------------------------------------------------------

fn eval_expr(s: &str) -> Result<f64, String> {
    let toks = tokenize(s)?;
    let mut p = ExprParser { toks: &toks, pos: 0 };
    let v = p.expr()?;
    if p.pos != toks.len() {
        return Err(format!("unexpected input in parameter '{}'", s));
    }
    Ok(v)
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Ident(String),
    Op(char),
}

fn tokenize(s: &str) -> Result<Vec<Tok>, String> {
    let b: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() || c == '.' {
            let start = i;
            while i < b.len() && (b[i].is_ascii_digit() || b[i] == '.') {
                i += 1;
            }
            if i < b.len() && (b[i] == 'e' || b[i] == 'E') {
                let save = i;
                i += 1;
                if i < b.len() && (b[i] == '+' || b[i] == '-') {
                    i += 1;
                }
                if i < b.len() && b[i].is_ascii_digit() {
                    while i < b.len() && b[i].is_ascii_digit() {
                        i += 1;
                    }
                } else {
                    i = save;
                }
            }
            let lit: String = b[start..i].iter().collect();
            let v: f64 = lit
                .parse()
                .map_err(|_| format!("invalid number '{}' in parameter '{}'", lit, s))?;
            out.push(Tok::Num(v));
        } else if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == '_') {
                i += 1;
            }
            out.push(Tok::Ident(b[start..i].iter().collect()));
        } else if "+-*/^()".contains(c) {
            out.push(Tok::Op(c));
            i += 1;
        } else {
            return Err(format!("unexpected character '{}' in parameter '{}'", c, s));
        }
    }
    Ok(out)
}

struct ExprParser<'a> {
    toks: &'a [Tok],
    pos: usize,
}

impl<'a> ExprParser<'a> {
    fn peek_op(&self, c: char) -> bool {
        matches!(self.toks.get(self.pos), Some(Tok::Op(o)) if *o == c)
    }

    fn expr(&mut self) -> Result<f64, String> {
        let mut v = self.term()?;
        loop {
            if self.peek_op('+') {
                self.pos += 1;
                v += self.term()?;
            } else if self.peek_op('-') {
                self.pos += 1;
                v -= self.term()?;
            } else {
                return Ok(v);
            }
        }
    }

    fn term(&mut self) -> Result<f64, String> {
        let mut v = self.unary()?;
        loop {
            if self.peek_op('*') {
                self.pos += 1;
                v *= self.unary()?;
            } else if self.peek_op('/') {
                self.pos += 1;
                v /= self.unary()?;
            } else {
                return Ok(v);
            }
        }
    }

    fn unary(&mut self) -> Result<f64, String> {
        if self.peek_op('-') {
            self.pos += 1;
            return Ok(-self.unary()?);
        }
        if self.peek_op('+') {
            self.pos += 1;
            return self.unary();
        }
        self.power()
    }

    /// Right-associative `^`, binding tighter than unary minus: -2^2 = -4.
    fn power(&mut self) -> Result<f64, String> {
        let base = self.atom()?;
        if self.peek_op('^') {
            self.pos += 1;
            let exp = self.unary()?;
            return Ok(dmath::pow(base, exp));
        }
        Ok(base)
    }

    fn atom(&mut self) -> Result<f64, String> {
        match self.toks.get(self.pos).cloned() {
            Some(Tok::Num(v)) => {
                self.pos += 1;
                Ok(v)
            }
            Some(Tok::Op('(')) => {
                self.pos += 1;
                let v = self.expr()?;
                if !self.peek_op(')') {
                    return Err("missing ')' in parameter".into());
                }
                self.pos += 1;
                Ok(v)
            }
            Some(Tok::Ident(id)) => {
                self.pos += 1;
                if id == "pi" {
                    return Ok(std::f64::consts::PI);
                }
                let f: fn(f64) -> f64 = match id.as_str() {
                    "sin" => dmath::sin,
                    "cos" => dmath::cos,
                    "tan" => |x| {
                        let (s, c) = dmath::sin_cos(x);
                        s / c
                    },
                    "exp" => dmath::exp,
                    "ln" => dmath::ln,
                    "sqrt" => f64::sqrt, // IEEE correctly rounded
                    other => return Err(format!("unknown identifier '{}' in parameter", other)),
                };
                if !self.peek_op('(') {
                    return Err(format!("'{}' must be followed by '('", id));
                }
                self.pos += 1;
                let v = self.expr()?;
                if !self.peek_op(')') {
                    return Err("missing ')' in parameter".into());
                }
                self.pos += 1;
                Ok(f(v))
            }
            Some(Tok::Op(c)) => Err(format!("unexpected '{}' in parameter", c)),
            None => Err("empty parameter expression".into()),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    fn all_gates_circuit() -> Circuit {
        let mut c = Circuit::new(4);
        for g in [
            Gate::H(0),
            Gate::X(1),
            Gate::Y(2),
            Gate::Z(3),
            Gate::S(0),
            Gate::T(1),
            Gate::Rx(2, 0.1),
            Gate::Ry(3, -1e-7),
            Gate::Rz(0, 123456.789),
            Gate::Rx(1, 5e-324),
            Gate::Ry(2, -0.0),
            Gate::CNOT(0, 1),
            Gate::CZ(2, 3),
            Gate::SWAP(1, 3),
            Gate::Toffoli(0, 1, 2),
        ] {
            c.add(g);
        }
        c
    }

    fn same_gates(a: &Circuit, b: &Circuit) -> bool {
        a.n_qubits() == b.n_qubits()
            && a.gates().len() == b.gates().len()
            && a.gates().iter().zip(b.gates()).all(|(x, y)| format!("{:?}", x) == format!("{:?}", y)
                && angle_bits(x) == angle_bits(y))
    }

    fn angle_bits(g: &Gate) -> Option<u64> {
        match *g {
            Gate::Rx(_, t) | Gate::Ry(_, t) | Gate::Rz(_, t) => Some(t.to_bits()),
            _ => None,
        }
    }

    #[test]
    fn round_trip_is_exact() {
        let c = all_gates_circuit();
        let text = to_qasm(&c);
        let back = from_qasm(&text).unwrap();
        assert!(same_gates(&c, &back), "{}", text);
        // And the statevectors are bit-identical.
        let (a, b) = (c.execute().unwrap(), back.execute().unwrap());
        for (x, y) in a.amplitudes.iter().zip(&b.amplitudes) {
            assert_eq!((x.re.to_bits(), x.im.to_bits()), (y.re.to_bits(), y.im.to_bits()));
        }
    }

    #[test]
    fn export_format() {
        let mut c = Circuit::new(2);
        c.add(Gate::H(0));
        c.add(Gate::Rz(1, 1e-5));
        c.add(Gate::CNOT(0, 1));
        assert_eq!(
            to_qasm(&c),
            "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\nh q[0];\nrz(1.0e-5) q[1];\ncx q[0],q[1];\n"
        );
    }

    #[test]
    fn parses_qiskit_style_program() {
        let src = "OPENQASM 2.0;\n\
                   include \"qelib1.inc\";\n\
                   qreg q[3];\n\
                   creg c[3];\n\
                   h q[0];\n\
                   cx q[0],q[1];\n\
                   rz(pi/2) q[2];\n\
                   ry(-3*pi/4) q[1];\n\
                   barrier q[0],q[1],q[2];\n\
                   measure q[0] -> c[0];\n\
                   measure q[1] -> c[1];\n";
        let c = from_qasm(src).unwrap();
        assert_eq!(c.n_qubits(), 3);
        assert_eq!(c.gates().len(), 4);
        assert_eq!(angle_bits(&c.gates()[2]), Some(FRAC_PI_2.to_bits()));
        assert_eq!(angle_bits(&c.gates()[3]), Some((-(3.0 * PI) / 4.0).to_bits()));
    }

    #[test]
    fn parameter_expressions() {
        for (src, want) in [
            ("pi", PI),
            ("pi/4", FRAC_PI_4),
            ("-pi/2", -FRAC_PI_2),
            ("2*pi - 1", 2.0 * PI - 1.0),
            ("(1+2)*3", 9.0),
            ("2^3^2", 512.0),
            ("-2^2", -4.0),
            ("1.5e-3", 1.5e-3),
            ("sqrt(2)", std::f64::consts::SQRT_2),
            ("cos(0)", 1.0),
            (".5", 0.5),
        ] {
            assert_eq!(eval_expr(src).unwrap(), want, "{}", src);
        }
        for bad in ["", "pi pi", "foo", "sin 1", "(1", "1)", "2 $ 3", "1e"] {
            assert!(eval_expr(bad).is_err(), "{:?} should be rejected", bad);
        }
    }

    #[test]
    fn registers_flatten_in_declaration_order_and_broadcast() {
        let src = "OPENQASM 2.0; include \"qelib1.inc\"; qreg a[2]; qreg b[2]; h a; cx a, b; x b[1];";
        let c = from_qasm(src).unwrap();
        assert_eq!(c.n_qubits(), 4);
        let got: Vec<String> = c.gates().iter().map(|g| format!("{:?}", g)).collect();
        assert_eq!(
            got,
            ["H(0)", "H(1)", "CNOT(0, 2)", "CNOT(1, 3)", "X(3)"].map(String::from)
        );
    }

    #[test]
    fn comments_and_multiline_statements() {
        let src = "// header comment\nOPENQASM 2.0; // version\ninclude \"qelib1.inc\";\nqreg q[2];\ncx q[0],\n   q[1]; // split\n";
        let c = from_qasm(src).unwrap();
        assert_eq!(c.gates().len(), 1);
    }

    #[test]
    fn unsupported_constructs_are_errors() {
        let pre = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[3];\ncreg c[3];\n";
        for (body, needle) in [
            ("u1(0.3) q[0];", "unsupported gate 'u1'"),
            ("sdg q[0];", "unsupported gate 'sdg'"),
            ("gate my q { h q; }", "custom gate definitions"),
            ("reset q[0];", "reset"),
            ("if(c==1) x q[0];", "classically controlled"),
            ("measure q[0] -> c[0];\nh q[1];", "mid-circuit measurement"),
            ("cx q[0],q[0];", "duplicate qubit operand"),
            ("h q[3];", "out of range"),
            ("h r[0];", "unknown quantum register"),
            ("rx q[0];", "takes 1 parameter"),
            ("cx q[0];", "takes 2 qubit operand"),
            ("rx(1/0) q[0];", "not finite"),
        ] {
            let e = from_qasm(&format!("{}{}", pre, body)).unwrap_err();
            assert!(e.contains(needle), "{:?} gave {:?}", body, e);
            assert!(e.starts_with("QASM line "), "{:?}", e);
        }
        assert!(from_qasm("").unwrap_err().contains("empty"));
        assert!(from_qasm("OPENQASM 3.0; qubit[2] q;").unwrap_err().contains("only 2.0"));
        assert!(from_qasm("OPENQASM 2.0; include \"stdgates.inc\";").unwrap_err().contains("qelib1.inc"));
        assert!(from_qasm("OPENQASM 2.0; qreg q[27];").unwrap_err().contains("at most 26"));
        assert!(from_qasm("OPENQASM 2.0;").unwrap_err().contains("no qreg"));
    }

    #[test]
    fn error_line_numbers_point_at_the_statement() {
        let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\n\nh q[0];\nfoo q[0];\n";
        assert!(from_qasm(src).unwrap_err().starts_with("QASM line 6:"));
    }

    #[test]
    fn pure_backend_round_trip() {
        let c = all_gates_circuit();
        let pc = pure_from_qasm(&to_qasm(&c)).unwrap();
        assert_eq!(pure_to_qasm(&pc), to_qasm(&c));
    }
}
