//! Phase 0.5 Item 3 — minimal TOML parser for `cjcl abng train --config`.
//!
//! cjc-cli's package contract forbids external dependencies (zero-deps
//! in the published surface, see project memory). The full `toml` crate
//! would break that, so this module hand-rolls a parser for the subset
//! of TOML the `train` command's config files actually need.
//!
//! ## Subset supported
//!
//! * **Comments** — `#` to end of line (anywhere outside a string).
//! * **Tables** — `[name]` headers; the next key=value pairs belong
//!   to the most recently declared table. The empty/root table holds
//!   any pairs declared before the first `[name]`.
//! * **Array of tables** — `[[name]]` headers (v0.7+ A3 addition).
//!   Each occurrence appends a new entry to the named array of tables.
//!   Unlike `[name]`, repeated `[[name]]` headers are NOT duplicate-key
//!   errors. Access via [`TomlDoc::array_tables`].
//! * **Keys** — bare-keys only: `[A-Za-z0-9_-]+`. No dotted keys, no
//!   quoted keys, no whitespace inside a key.
//! * **Values:**
//!   * Booleans `true` / `false`.
//!   * Integers (decimal, optional sign): `42`, `-7`. Underscore
//!     separators inside the digit run are accepted (`1_000_000`).
//!   * Floats: `3.14`, `-1e3`, `1.5e-3`, `1.7976931348623157e308`.
//!     Exponent and decimal point both optional individually but at
//!     least one must be present (otherwise the value parses as int).
//!   * Strings: double-quoted `"..."`. Escapes: `\\`, `\"`, `\n`,
//!     `\r`, `\t`. No multiline `"""..."""`. No literal-strings (`'...'`).
//!   * Arrays: `[v1, v2, v3]`. Trailing comma allowed. Elements
//!     must be homogeneous (all the same `TomlValue` variant) — this
//!     is stricter than real TOML but matches what `train` consumes.
//!     Nested arrays of integers and arrays of arrays are supported
//!     (e.g., `[[0, 1], [0, 2]]` for `add_nodes` pairs).
//!
//! ## Determinism
//!
//! Parser visits keys in declaration order; the resulting `TomlTable`
//! is a `Vec<(String, TomlValue)>` (NOT a `HashMap`). Iteration
//! order = declaration order = byte-stable across runs.
//!
//! ## NOT supported
//!
//! * Multiline strings (basic or literal).
//! * Inline tables (`{ a = 1, b = 2 }`).
//! * Dotted keys (`a.b = 1`).
//! * `[[array_of_tables]]` syntax — use arrays of arrays instead.
//! * Datetimes / dates / times.
//! * Hex / octal / binary integer literals.
//! * Underscore separators in floats. (Integers only.)
//!
//! Any unsupported construct surfaces a `TomlError` rather than
//! silently misparsing.

use std::fmt;

/// Runtime TOML value.
#[derive(Debug, Clone, PartialEq)]
pub enum TomlValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Array(Vec<TomlValue>),
}

impl TomlValue {
    pub fn type_name(&self) -> &'static str {
        match self {
            TomlValue::Bool(_) => "bool",
            TomlValue::Int(_) => "integer",
            TomlValue::Float(_) => "float",
            TomlValue::String(_) => "string",
            TomlValue::Array(_) => "array",
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        if let TomlValue::Bool(b) = self {
            Some(*b)
        } else {
            None
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        if let TomlValue::Int(i) = self {
            Some(*i)
        } else {
            None
        }
    }

    /// Permissive numeric accessor: accepts both integer and float
    /// forms and returns f64. `42` and `42.0` both yield `Some(42.0)`.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            TomlValue::Float(f) => Some(*f),
            TomlValue::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        if let TomlValue::String(s) = self {
            Some(s.as_str())
        } else {
            None
        }
    }

    pub fn as_array(&self) -> Option<&[TomlValue]> {
        if let TomlValue::Array(arr) = self {
            Some(arr.as_slice())
        } else {
            None
        }
    }
}

/// Ordered table: `Vec<(key, value)>` to keep declaration order
/// deterministic for diagnostics and for `cjcl abng diff`.
pub type TomlTable = Vec<(String, TomlValue)>;

/// A parsed TOML document.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TomlDoc {
    /// Tables, in declaration order. The root table (pre-first-`[name]`
    /// key=value pairs) is named `""` if present.
    pub tables: Vec<(String, TomlTable)>,
    /// Array-of-tables entries, in declaration order. Each `[[name]]`
    /// header appends one entry here under `name`. Lookup via
    /// [`TomlDoc::array_tables`].
    pub array_tables: Vec<(String, TomlTable)>,
}

impl TomlDoc {
    /// Look up a table by name. Returns `None` if no `[name]` header
    /// declared this table. (The root table's name is `""`.)
    pub fn table(&self, name: &str) -> Option<&TomlTable> {
        self.tables
            .iter()
            .find_map(|(n, t)| if n == name { Some(t) } else { None })
    }

    /// Look up a key in the named table. Returns `None` if the table
    /// is absent or the key isn't in it.
    pub fn get(&self, table: &str, key: &str) -> Option<&TomlValue> {
        self.table(table).and_then(|t| {
            t.iter()
                .find_map(|(k, v)| if k == key { Some(v) } else { None })
        })
    }

    /// Return every array-of-tables entry whose header matches `name`,
    /// in declaration order. v0.7+ A3 addition for the Locke policy DSL.
    pub fn array_tables(&self, name: &str) -> Vec<&TomlTable> {
        self.array_tables
            .iter()
            .filter_map(|(n, t)| if n == name { Some(t) } else { None })
            .collect()
    }
}

/// Parser errors. Each variant includes a 1-based line number for
/// diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub enum TomlError {
    UnexpectedChar { line: usize, ch: char, ctx: &'static str },
    UnterminatedString { line: usize },
    UnexpectedEof { line: usize, ctx: &'static str },
    BadEscape { line: usize, ch: char },
    BadInt { line: usize, raw: String },
    BadFloat { line: usize, raw: String },
    HeterogeneousArray { line: usize, first: &'static str, second: &'static str },
    DuplicateKey { line: usize, key: String },
    DuplicateTable { line: usize, name: String },
    EmptyKey { line: usize },
    UnclosedTableHeader { line: usize },
    UnclosedArray { line: usize },
}

impl fmt::Display for TomlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TomlError::UnexpectedChar { line, ch, ctx } => {
                write!(f, "toml: line {line}: unexpected `{ch}` while {ctx}")
            }
            TomlError::UnterminatedString { line } => {
                write!(f, "toml: line {line}: unterminated string literal")
            }
            TomlError::UnexpectedEof { line, ctx } => {
                write!(f, "toml: line {line}: unexpected end of input while {ctx}")
            }
            TomlError::BadEscape { line, ch } => {
                write!(f, "toml: line {line}: unsupported escape sequence `\\{ch}`")
            }
            TomlError::BadInt { line, raw } => {
                write!(f, "toml: line {line}: invalid integer literal `{raw}`")
            }
            TomlError::BadFloat { line, raw } => {
                write!(f, "toml: line {line}: invalid float literal `{raw}`")
            }
            TomlError::HeterogeneousArray { line, first, second } => {
                write!(
                    f,
                    "toml: line {line}: heterogeneous array — first element \
                     was `{first}`, later element is `{second}`"
                )
            }
            TomlError::DuplicateKey { line, key } => {
                write!(f, "toml: line {line}: duplicate key `{key}` in current table")
            }
            TomlError::DuplicateTable { line, name } => {
                write!(f, "toml: line {line}: duplicate table header `[{name}]`")
            }
            TomlError::EmptyKey { line } => {
                write!(f, "toml: line {line}: empty key (expected identifier before `=`)")
            }
            TomlError::UnclosedTableHeader { line } => {
                write!(f, "toml: line {line}: unclosed `[...]` table header")
            }
            TomlError::UnclosedArray { line } => {
                write!(f, "toml: line {line}: unclosed `[...]` array")
            }
        }
    }
}

impl std::error::Error for TomlError {}

/// Parse a TOML document into a [`TomlDoc`].
pub fn parse(input: &str) -> Result<TomlDoc, TomlError> {
    let mut p = Parser::new(input);
    p.parse_doc()
}

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
    line: usize,
}

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Self {
        Self {
            src: s.as_bytes(),
            pos: 0,
            line: 1,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.pos += 1;
        if b == b'\n' {
            self.line += 1;
        }
        Some(b)
    }

    /// Skip ASCII whitespace (spaces and tabs). Newlines are kept —
    /// callers explicitly consume them when they're meaningful.
    fn skip_inline_ws(&mut self) {
        while let Some(b) = self.peek() {
            if b == b' ' || b == b'\t' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    /// Skip whitespace, newlines, and `#`-comments.
    fn skip_ws_and_comments(&mut self) {
        loop {
            match self.peek() {
                Some(b' ') | Some(b'\t') => {
                    self.pos += 1;
                }
                Some(b'\r') => {
                    self.pos += 1;
                }
                Some(b'\n') => {
                    self.pos += 1;
                    self.line += 1;
                }
                Some(b'#') => {
                    // Comment to end of line.
                    while let Some(b) = self.peek() {
                        if b == b'\n' {
                            break;
                        }
                        self.pos += 1;
                    }
                }
                _ => break,
            }
        }
    }

    fn parse_doc(&mut self) -> Result<TomlDoc, TomlError> {
        let mut doc = TomlDoc::default();
        let mut current_table: TomlTable = Vec::new();
        let mut current_name = String::new();
        // The kind of header that opened the current table: false for
        // `[name]`, true for `[[name]]`. Flushed table is routed to the
        // right field on TomlDoc.
        let mut current_is_array_entry = false;
        let mut declared_table_names: Vec<String> = Vec::new();

        // Helper closure to flush current_table to the right field.
        // Inlined because closures + mutable borrows make this ugly otherwise.
        macro_rules! flush_current {
            () => {
                if !current_table.is_empty() || current_name != *"" {
                    if current_is_array_entry {
                        doc.array_tables.push((current_name.clone(), current_table));
                    } else {
                        doc.tables.push((current_name.clone(), current_table));
                    }
                }
            };
        }

        loop {
            self.skip_ws_and_comments();
            match self.peek() {
                None => break,
                Some(b'[') => {
                    // Flush current table.
                    flush_current!();
                    let header_line = self.line;
                    self.pos += 1; // consume first '['
                    // v0.7+ A3: detect `[[name]]` (array-of-tables) vs `[name]`.
                    let is_array_header = self.peek() == Some(b'[');
                    if is_array_header {
                        self.pos += 1; // consume second '['
                    }
                    self.skip_inline_ws();
                    let name = self.parse_bare_key()?;
                    self.skip_inline_ws();
                    match self.peek() {
                        Some(b']') => self.pos += 1,
                        _ => return Err(TomlError::UnclosedTableHeader { line: header_line }),
                    }
                    if is_array_header {
                        match self.peek() {
                            Some(b']') => self.pos += 1,
                            _ => return Err(TomlError::UnclosedTableHeader { line: header_line }),
                        }
                    }
                    if !is_array_header {
                        // Standard table: still enforce uniqueness vs prior `[name]`.
                        if declared_table_names.contains(&name) {
                            return Err(TomlError::DuplicateTable {
                                line: header_line,
                                name,
                            });
                        }
                        declared_table_names.push(name.clone());
                    }
                    current_name = name;
                    current_table = Vec::new();
                    current_is_array_entry = is_array_header;
                }
                Some(_) => {
                    let pair_line = self.line;
                    let key = self.parse_bare_key()?;
                    if key.is_empty() {
                        return Err(TomlError::EmptyKey { line: pair_line });
                    }
                    self.skip_inline_ws();
                    match self.peek() {
                        Some(b'=') => self.pos += 1,
                        Some(other) => {
                            return Err(TomlError::UnexpectedChar {
                                line: pair_line,
                                ch: other as char,
                                ctx: "expecting `=` after key",
                            })
                        }
                        None => {
                            return Err(TomlError::UnexpectedEof {
                                line: pair_line,
                                ctx: "expecting `=` after key",
                            })
                        }
                    }
                    self.skip_inline_ws();
                    let value = self.parse_value()?;
                    if current_table.iter().any(|(k, _)| k == &key) {
                        return Err(TomlError::DuplicateKey {
                            line: pair_line,
                            key,
                        });
                    }
                    current_table.push((key, value));
                }
            }
        }
        // Flush the trailing table.
        flush_current!();
        Ok(doc)
    }

    fn parse_bare_key(&mut self) -> Result<String, TomlError> {
        let start = self.pos;
        while let Some(b) = self.peek() {
            if b.is_ascii_alphanumeric() || b == b'_' || b == b'-' {
                self.pos += 1;
            } else {
                break;
            }
        }
        let s = std::str::from_utf8(&self.src[start..self.pos]).unwrap_or("");
        Ok(s.to_string())
    }

    fn parse_value(&mut self) -> Result<TomlValue, TomlError> {
        self.skip_inline_ws();
        let line = self.line;
        let b = self.peek().ok_or(TomlError::UnexpectedEof {
            line,
            ctx: "expecting value",
        })?;
        match b {
            b'"' => self.parse_string().map(TomlValue::String),
            b'[' => self.parse_array(),
            b't' | b'f' => self.parse_bool(),
            b'-' | b'+' => self.parse_number(),
            b'0'..=b'9' => self.parse_number(),
            other => Err(TomlError::UnexpectedChar {
                line,
                ch: other as char,
                ctx: "expecting value",
            }),
        }
    }

    fn parse_string(&mut self) -> Result<String, TomlError> {
        let line = self.line;
        debug_assert_eq!(self.peek(), Some(b'"'));
        self.pos += 1; // opening quote
        let mut out = String::new();
        loop {
            match self.peek() {
                None => return Err(TomlError::UnterminatedString { line }),
                Some(b'\n') => return Err(TomlError::UnterminatedString { line }),
                Some(b'"') => {
                    self.pos += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.pos += 1;
                    let esc = self.peek().ok_or(TomlError::UnterminatedString { line })?;
                    let ch = match esc {
                        b'\\' => '\\',
                        b'"' => '"',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        other => {
                            return Err(TomlError::BadEscape {
                                line,
                                ch: other as char,
                            })
                        }
                    };
                    out.push(ch);
                    self.pos += 1;
                }
                Some(b) => {
                    out.push(b as char);
                    self.pos += 1;
                }
            }
        }
    }

    fn parse_bool(&mut self) -> Result<TomlValue, TomlError> {
        if self.src[self.pos..].starts_with(b"true") {
            self.pos += 4;
            Ok(TomlValue::Bool(true))
        } else if self.src[self.pos..].starts_with(b"false") {
            self.pos += 5;
            Ok(TomlValue::Bool(false))
        } else {
            Err(TomlError::UnexpectedChar {
                line: self.line,
                ch: self.peek().unwrap_or(b'?') as char,
                ctx: "expecting `true` or `false`",
            })
        }
    }

    fn parse_number(&mut self) -> Result<TomlValue, TomlError> {
        let line = self.line;
        let start = self.pos;
        // Optional sign.
        if matches!(self.peek(), Some(b'-') | Some(b'+')) {
            self.pos += 1;
        }
        // Digit run + optional underscores.
        let mut saw_digit = false;
        while let Some(b) = self.peek() {
            if b.is_ascii_digit() || b == b'_' {
                if b.is_ascii_digit() {
                    saw_digit = true;
                }
                self.pos += 1;
            } else {
                break;
            }
        }
        if !saw_digit {
            return Err(TomlError::BadInt {
                line,
                raw: String::from_utf8_lossy(&self.src[start..self.pos]).to_string(),
            });
        }
        let mut is_float = false;
        // Optional fractional part.
        if self.peek() == Some(b'.') {
            is_float = true;
            self.pos += 1;
            while let Some(b) = self.peek() {
                if b.is_ascii_digit() {
                    self.pos += 1;
                } else {
                    break;
                }
            }
        }
        // Optional exponent.
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            is_float = true;
            self.pos += 1;
            if matches!(self.peek(), Some(b'-') | Some(b'+')) {
                self.pos += 1;
            }
            while let Some(b) = self.peek() {
                if b.is_ascii_digit() {
                    self.pos += 1;
                } else {
                    break;
                }
            }
        }
        let raw = std::str::from_utf8(&self.src[start..self.pos]).unwrap_or("");
        if is_float {
            let cleaned: String = raw.chars().filter(|c| *c != '_').collect();
            cleaned
                .parse::<f64>()
                .map(TomlValue::Float)
                .map_err(|_| TomlError::BadFloat {
                    line,
                    raw: raw.to_string(),
                })
        } else {
            let cleaned: String = raw.chars().filter(|c| *c != '_').collect();
            cleaned
                .parse::<i64>()
                .map(TomlValue::Int)
                .map_err(|_| TomlError::BadInt {
                    line,
                    raw: raw.to_string(),
                })
        }
    }

    fn parse_array(&mut self) -> Result<TomlValue, TomlError> {
        let open_line = self.line;
        debug_assert_eq!(self.peek(), Some(b'['));
        self.pos += 1; // opening bracket
        let mut out: Vec<TomlValue> = Vec::new();
        let mut first_kind: Option<&'static str> = None;
        loop {
            self.skip_ws_and_comments();
            match self.peek() {
                Some(b']') => {
                    self.pos += 1;
                    return Ok(TomlValue::Array(out));
                }
                None => return Err(TomlError::UnclosedArray { line: open_line }),
                _ => {}
            }
            let v = self.parse_value()?;
            if let Some(first) = first_kind {
                if first != v.type_name() {
                    return Err(TomlError::HeterogeneousArray {
                        line: self.line,
                        first,
                        second: v.type_name(),
                    });
                }
            } else {
                first_kind = Some(v.type_name());
            }
            out.push(v);
            self.skip_ws_and_comments();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    return Ok(TomlValue::Array(out));
                }
                Some(other) => {
                    return Err(TomlError::UnexpectedChar {
                        line: self.line,
                        ch: other as char,
                        ctx: "expecting `,` or `]` in array",
                    });
                }
                None => return Err(TomlError::UnclosedArray { line: open_line }),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_ok(input: &str) -> TomlDoc {
        parse(input).unwrap()
    }

    fn parse_err(input: &str) -> TomlError {
        parse(input).unwrap_err()
    }

    // ── primitives ─────────────────────────────────────────────────

    #[test]
    fn empty_input() {
        let d = parse_ok("");
        assert!(d.tables.is_empty());
    }

    #[test]
    fn pure_comment() {
        let d = parse_ok("# nothing here\n# another comment\n");
        assert!(d.tables.is_empty());
    }

    #[test]
    fn single_int_in_root() {
        let d = parse_ok("seed = 42\n");
        assert_eq!(d.get("", "seed"), Some(&TomlValue::Int(42)));
    }

    #[test]
    fn negative_int() {
        let d = parse_ok("x = -7\n");
        assert_eq!(d.get("", "x"), Some(&TomlValue::Int(-7)));
    }

    #[test]
    fn underscored_int() {
        let d = parse_ok("big = 1_000_000\n");
        assert_eq!(d.get("", "big"), Some(&TomlValue::Int(1_000_000)));
    }

    #[test]
    fn simple_float() {
        let d = parse_ok("pi = 3.14\n");
        assert_eq!(d.get("", "pi"), Some(&TomlValue::Float(3.14)));
    }

    #[test]
    fn exponential_float() {
        let d = parse_ok("eps = 1.5e-3\n");
        assert_eq!(d.get("", "eps"), Some(&TomlValue::Float(1.5e-3)));
    }

    #[test]
    fn float_max() {
        let d = parse_ok("inf_like = 1.7976931348623157e308\n");
        assert_eq!(
            d.get("", "inf_like"),
            Some(&TomlValue::Float(1.7976931348623157e308))
        );
    }

    #[test]
    fn bool_true_false() {
        let d = parse_ok("a = true\nb = false\n");
        assert_eq!(d.get("", "a"), Some(&TomlValue::Bool(true)));
        assert_eq!(d.get("", "b"), Some(&TomlValue::Bool(false)));
    }

    #[test]
    fn simple_string() {
        let d = parse_ok("name = \"hello world\"\n");
        assert_eq!(
            d.get("", "name"),
            Some(&TomlValue::String("hello world".to_string()))
        );
    }

    #[test]
    fn string_with_escapes() {
        let d = parse_ok("s = \"a\\nb\\tc\\\"d\\\\e\"\n");
        assert_eq!(
            d.get("", "s"),
            Some(&TomlValue::String("a\nb\tc\"d\\e".to_string()))
        );
    }

    // ── tables ────────────────────────────────────────────────────

    #[test]
    fn one_table() {
        let d = parse_ok("[graph]\nseed = 42\n");
        assert_eq!(d.get("graph", "seed"), Some(&TomlValue::Int(42)));
        assert_eq!(d.tables.len(), 1);
    }

    #[test]
    fn root_then_table() {
        let d = parse_ok("debug = true\n[graph]\nseed = 0\n");
        assert_eq!(d.get("", "debug"), Some(&TomlValue::Bool(true)));
        assert_eq!(d.get("graph", "seed"), Some(&TomlValue::Int(0)));
    }

    #[test]
    fn multiple_tables_in_declaration_order() {
        let d = parse_ok("[a]\nx = 1\n[b]\ny = 2\n[c]\nz = 3\n");
        let names: Vec<&str> = d.tables.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    // ── arrays ────────────────────────────────────────────────────

    #[test]
    fn empty_array() {
        let d = parse_ok("xs = []\n");
        assert_eq!(d.get("", "xs"), Some(&TomlValue::Array(vec![])));
    }

    #[test]
    fn array_of_ints() {
        let d = parse_ok("xs = [1, 2, 3]\n");
        let arr = d.get("", "xs").unwrap().as_array().unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0], TomlValue::Int(1));
    }

    #[test]
    fn array_of_floats() {
        let d = parse_ok("xs = [-1.0, 0.0, 1.0]\n");
        let arr = d.get("", "xs").unwrap().as_array().unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0], TomlValue::Float(-1.0));
    }

    #[test]
    fn array_of_arrays() {
        let d = parse_ok("pairs = [[0, 1], [0, 2]]\n");
        let outer = d.get("", "pairs").unwrap().as_array().unwrap();
        assert_eq!(outer.len(), 2);
        let inner0 = outer[0].as_array().unwrap();
        assert_eq!(inner0[0], TomlValue::Int(0));
        assert_eq!(inner0[1], TomlValue::Int(1));
    }

    #[test]
    fn array_with_trailing_comma() {
        let d = parse_ok("xs = [1, 2, 3,]\n");
        let arr = d.get("", "xs").unwrap().as_array().unwrap();
        assert_eq!(arr.len(), 3);
    }

    #[test]
    fn array_with_newlines() {
        let d = parse_ok("xs = [\n  1,\n  2,\n  3,\n]\n");
        let arr = d.get("", "xs").unwrap().as_array().unwrap();
        assert_eq!(arr.len(), 3);
    }

    // ── error variants ────────────────────────────────────────────

    #[test]
    fn err_unclosed_string() {
        let e = parse_err("name = \"hello\n");
        assert!(matches!(e, TomlError::UnterminatedString { line: 1 }));
    }

    #[test]
    fn err_bad_escape() {
        let e = parse_err("s = \"\\q\"\n");
        assert!(matches!(e, TomlError::BadEscape { ch: 'q', .. }));
    }

    #[test]
    fn err_unclosed_array() {
        let e = parse_err("xs = [1, 2,\n");
        assert!(matches!(e, TomlError::UnclosedArray { .. }));
    }

    #[test]
    fn err_heterogeneous_array() {
        let e = parse_err("xs = [1, \"two\"]\n");
        assert!(matches!(
            e,
            TomlError::HeterogeneousArray {
                first: "integer",
                second: "string",
                ..
            }
        ));
    }

    #[test]
    fn err_duplicate_key() {
        let e = parse_err("a = 1\na = 2\n");
        assert!(matches!(e, TomlError::DuplicateKey { .. }));
    }

    #[test]
    fn err_duplicate_table() {
        let e = parse_err("[graph]\n[graph]\n");
        assert!(matches!(e, TomlError::DuplicateTable { .. }));
    }

    #[test]
    fn err_unclosed_table_header() {
        let e = parse_err("[graph\nseed = 1\n");
        assert!(matches!(e, TomlError::UnclosedTableHeader { .. }));
    }

    #[test]
    fn err_unexpected_char_value_position() {
        let e = parse_err("a = ;\n");
        assert!(matches!(e, TomlError::UnexpectedChar { .. }));
    }

    // ── integration: full train config ─────────────────────────────

    #[test]
    fn train_config_full_example() {
        let src = r#"
# cjcl abng train config
[graph]
seed = 42
add_nodes = [[0, 1], [0, 2]]

[codebook]
n_dims = 1
n_bins = 4
boundaries = [-1.0, 0.0, 1.0]

[leaf_head]
input_dim = 1
hidden_dims = [2]
output_dim = 1
activation = "tanh"

[blr_prior]
a = 1.0
b = 1.5
sigma_init = 1.0

[density]
enabled = true

[calibration]
n_bins = 15

[decision_policy]
thresholds = [0.5, 64.0, 128.0, 0.05, 0.02, 4.0, 0.1, 32.0, 10.0, 8.0, 20.0, 1.7976931348623157e308, 0.005, 1.05]

[training]
n_observations = 100
observation_seed = 42
decide_step_every = 25
max_decide_steps = 1000

[output]
path = "model.snap"
"#;
        let d = parse_ok(src);
        assert_eq!(d.get("graph", "seed"), Some(&TomlValue::Int(42)));
        assert_eq!(
            d.get("codebook", "n_bins"),
            Some(&TomlValue::Int(4))
        );
        assert_eq!(
            d.get("blr_prior", "a"),
            Some(&TomlValue::Float(1.0))
        );
        assert_eq!(
            d.get("leaf_head", "activation"),
            Some(&TomlValue::String("tanh".to_string()))
        );
        let thresholds = d.get("decision_policy", "thresholds").unwrap().as_array().unwrap();
        assert_eq!(thresholds.len(), 14);
        let pairs = d.get("graph", "add_nodes").unwrap().as_array().unwrap();
        assert_eq!(pairs.len(), 2);
    }

    // ── v0.7+ A3: array-of-tables ──────────────────────────────────

    #[test]
    fn array_of_tables_collects_repeated_headers() {
        let src = r#"
[[suppress]]
code = "E9082"
column = "weight"

[[suppress]]
code = "E9080"
column = "patient_nbr"

[[suppress]]
code = "E9008"
"#;
        let d = parse_ok(src);
        let entries = d.array_tables("suppress");
        assert_eq!(entries.len(), 3);
        assert_eq!(
            entries[0]
                .iter()
                .find(|(k, _)| k == "code")
                .unwrap()
                .1,
            TomlValue::String("E9082".into())
        );
        assert_eq!(
            entries[1]
                .iter()
                .find(|(k, _)| k == "column")
                .unwrap()
                .1,
            TomlValue::String("patient_nbr".into())
        );
        // Third entry has no `column` key — that's allowed.
        assert!(entries[2].iter().all(|(k, _)| k != "column"));
    }

    #[test]
    fn array_of_tables_preserves_declaration_order() {
        let src = "[[x]]\nv = 1\n[[x]]\nv = 2\n[[x]]\nv = 3\n";
        let d = parse_ok(src);
        let entries = d.array_tables("x");
        assert_eq!(
            entries
                .iter()
                .map(|t| t.iter().find(|(k, _)| k == "v").unwrap().1.as_int().unwrap())
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn array_of_tables_unknown_name_returns_empty() {
        let src = "[[suppress]]\ncode = \"E1\"\n";
        let d = parse_ok(src);
        assert!(d.array_tables("nonexistent").is_empty());
    }

    #[test]
    fn array_of_tables_interleaved_with_named_tables() {
        let src = r#"
[[suppress]]
code = "E1"

[meta]
version = 1

[[suppress]]
code = "E2"

[[owner]]
team = "data"
"#;
        let d = parse_ok(src);
        let suppress = d.array_tables("suppress");
        assert_eq!(suppress.len(), 2);
        let owner = d.array_tables("owner");
        assert_eq!(owner.len(), 1);
        // The named [meta] table is still in `tables`, not `array_tables`.
        assert_eq!(
            d.get("meta", "version"),
            Some(&TomlValue::Int(1))
        );
    }

    #[test]
    fn array_of_tables_unclosed_double_bracket_errs() {
        // `[[x]` should fail — only one closing bracket.
        let e = parse_err("[[x]\nv = 1\n");
        assert!(matches!(e, TomlError::UnclosedTableHeader { .. }));
    }

    #[test]
    fn array_of_tables_allows_repeated_name_no_duplicate_error() {
        // Previously `[graph]\n[graph]` was a DuplicateTable error.
        // `[[suppress]]\n[[suppress]]` must NOT error.
        assert!(parse("[[suppress]]\n[[suppress]]\n").is_ok());
    }
}
