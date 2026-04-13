//! Comprehensive error code taxonomy for CJC diagnostics.
//!
//! Every diagnostic emitted by the CJC compiler carries a typed [`ErrorCode`]
//! that identifies the error category and specific condition. Codes follow a
//! Rust+Elm-inspired numbering scheme:
//!
//! | Range    | Category                                  |
//! |----------|-------------------------------------------|
//! | E0xxx    | Lexer errors                              |
//! | E06xx    | Snap (serialization) errors                |
//! | E1xxx    | Parser errors                             |
//! | E2xxx    | Type errors                               |
//! | E3xxx    | Borrow / ownership errors                 |
//! | E4xxx    | Effect errors (NoGC, purity violations)   |
//! | E5xxx    | Name resolution errors                    |
//! | E6xxx    | Generics / trait errors                   |
//! | E7xxx    | MIR / internal compiler errors            |
//! | E8xxx    | Runtime errors                            |
//! | E9xxx    | Module system errors                      |
//! | W0xxx    | Warnings                                  |
//!
//! Each variant carries a default message template (via
//! [`ErrorCode::message_template`]), a [`Severity`] (via
//! [`ErrorCode::severity`]), and a human-readable category name (via
//! [`ErrorCode::category`]).

use super::Severity;

/// A typed error code covering the entire CJC compiler pipeline.
///
/// Each variant maps to a unique string representation (e.g., `"E0001"`),
/// a default message template, a [`Severity`], and a category name.
/// Use [`DiagnosticBuilder::new`](super::DiagnosticBuilder::new) to construct
/// diagnostics from an `ErrorCode`.
///
/// # Examples
///
/// ```
/// use cjc_diag::ErrorCode;
///
/// let code = ErrorCode::E2001;
/// assert_eq!(code.code_str(), "E2001");
/// assert_eq!(code.message_template(), "type mismatch");
/// assert_eq!(code.category(), "type");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    // ── Lexer errors (E0xxx) ──────────────────────────────────────────
    /// Unexpected character in source
    E0001,
    /// Unterminated string literal
    E0002,
    /// Invalid numeric literal
    E0003,
    /// Invalid escape sequence in string
    E0004,
    /// Unterminated block comment
    E0005,
    /// Invalid regex literal
    E0006,
    /// Missing digits after numeric prefix (0x, 0b, 0o)
    E0007,
    /// Invalid character in numeric literal
    E0008,
    /// Unterminated byte string
    E0009,
    /// Invalid byte literal
    E0010,

    // ── Parser errors (E1xxx) ─────────────────────────────────────────
    /// Unexpected token
    E1000,
    /// Expected specific token, found something else
    E1001,
    /// Expected expression
    E1002,
    /// Missing type annotation on function parameter
    E1003,
    /// Missing closing delimiter (brace, paren, bracket)
    E1004,
    /// Invalid pattern in match arm
    E1005,
    /// Expected function body
    E1006,
    /// Duplicate field in struct literal
    E1007,
    /// Invalid assignment target
    E1008,
    /// Expected type expression
    E1009,
    /// Missing return type after `->`
    E1010,
    /// Invalid declaration at top level
    E1011,
    /// Empty match expression (no arms)
    E1012,
    /// Missing effect annotation after `/`
    E1013,

    // ── Type errors (E2xxx) ───────────────────────────────────────────
    /// Type mismatch
    E2001,
    /// Cannot infer type
    E2002,
    /// Incompatible types in binary operation
    E2003,
    /// Cannot unify types
    E2004,
    /// Wrong number of arguments in function call
    E2005,
    /// Undefined variable
    E2006,
    /// Undefined function
    E2007,
    /// Undefined field on type
    E2008,
    /// Non-exhaustive match patterns
    E2009,
    /// Cannot assign to immutable variable
    E2010,
    /// Return type mismatch
    E2011,
    /// Cannot call non-function type
    E2012,
    /// Cannot index non-indexable type
    E2013,
    /// Duplicate struct/enum definition
    E2014,
    /// Missing field in struct literal
    E2015,

    // ── Borrow/Ownership errors (E3xxx) ───────────────────────────────
    /// Move of borrowed value
    E3001,
    /// Use after move
    E3002,
    /// Cannot borrow as mutable
    E3003,
    /// Cannot assign to immutable variable (ownership)
    E3004,
    /// Value does not live long enough
    E3005,
    /// Cannot move out of borrowed content
    E3006,
    /// Mutable borrow while immutable borrow exists
    E3007,
    /// Double mutable borrow
    E3008,

    // ── Effect errors (E4xxx) ─────────────────────────────────────────
    /// GC operation in nogc context
    E4001,
    /// IO operation in pure context
    E4002,
    /// Nondeterministic operation in deterministic context
    E4003,
    /// Allocation in nogc function
    E4004,
    /// Effect annotation mismatch (declared vs actual)
    E4005,
    /// Mutation in pure context
    E4006,

    // ── Name resolution errors (E5xxx) ────────────────────────────────
    /// Unresolved name
    E5001,
    /// Ambiguous name
    E5002,
    /// Duplicate definition
    E5003,
    /// Import not found
    E5004,

    // ── Generics/Trait errors (E6xxx) ─────────────────────────────────
    /// Trait bound not satisfied
    E6001,
    /// Missing trait implementation
    E6002,
    /// Conflicting trait implementations
    E6003,
    /// Type parameter unused
    E6004,
    /// Wrong number of type arguments
    E6005,
    /// Trait method signature mismatch
    E6006,

    // ── MIR/Internal compiler errors (E7xxx) ──────────────────────────
    /// Internal compiler error in MIR lowering
    E7001,
    /// SSA verification failed
    E7002,
    /// CFG verification failed
    E7003,

    // ── Runtime errors (E8xxx) ────────────────────────────────────────
    /// Index out of bounds
    E8001,
    /// Division by zero
    E8002,
    /// Stack overflow
    E8003,
    /// Shape mismatch in tensor operation
    E8004,
    /// Type assertion failed at runtime
    E8005,

    // ── Module system errors (E9xxx) ──────────────────────────────────
    /// Module not found
    E9001,
    /// Circular dependency
    E9002,

    // ── Snap errors (E0600+) ─────────────────────────────────────────
    /// Snapshot logic hash mismatch
    E0601,
    /// Snapshot type hash mismatch
    E0602,
    /// Snapshot layout hash mismatch
    E0603,
    /// Type not snap-compatible
    E0604,

    // ── Warnings (W0xxx) ─────────────────────────────────────────────
    /// Unused variable
    W0001,
    /// Unused import
    W0002,
    /// Dead code (unreachable)
    W0003,
    /// Variable shadows outer scope
    W0004,
    /// Deprecated feature
    W0005,
}

impl ErrorCode {
    /// Returns the canonical string representation of this error code.
    ///
    /// The format is a one-letter prefix (`E` for errors, `W` for warnings)
    /// followed by a four-digit number (e.g., `"E0001"`, `"W0001"`).
    ///
    /// # Examples
    ///
    /// ```
    /// use cjc_diag::ErrorCode;
    ///
    /// assert_eq!(ErrorCode::E0001.code_str(), "E0001");
    /// assert_eq!(ErrorCode::W0001.code_str(), "W0001");
    /// ```
    pub fn code_str(&self) -> &'static str {
        match self {
            // Lexer
            ErrorCode::E0001 => "E0001",
            ErrorCode::E0002 => "E0002",
            ErrorCode::E0003 => "E0003",
            ErrorCode::E0004 => "E0004",
            ErrorCode::E0005 => "E0005",
            ErrorCode::E0006 => "E0006",
            ErrorCode::E0007 => "E0007",
            ErrorCode::E0008 => "E0008",
            ErrorCode::E0009 => "E0009",
            ErrorCode::E0010 => "E0010",
            // Parser
            ErrorCode::E1000 => "E1000",
            ErrorCode::E1001 => "E1001",
            ErrorCode::E1002 => "E1002",
            ErrorCode::E1003 => "E1003",
            ErrorCode::E1004 => "E1004",
            ErrorCode::E1005 => "E1005",
            ErrorCode::E1006 => "E1006",
            ErrorCode::E1007 => "E1007",
            ErrorCode::E1008 => "E1008",
            ErrorCode::E1009 => "E1009",
            ErrorCode::E1010 => "E1010",
            ErrorCode::E1011 => "E1011",
            ErrorCode::E1012 => "E1012",
            ErrorCode::E1013 => "E1013",
            // Borrow/Ownership
            ErrorCode::E3001 => "E3001",
            ErrorCode::E3002 => "E3002",
            ErrorCode::E3003 => "E3003",
            ErrorCode::E3004 => "E3004",
            ErrorCode::E3005 => "E3005",
            ErrorCode::E3006 => "E3006",
            ErrorCode::E3007 => "E3007",
            ErrorCode::E3008 => "E3008",
            // Type
            ErrorCode::E2001 => "E2001",
            ErrorCode::E2002 => "E2002",
            ErrorCode::E2003 => "E2003",
            ErrorCode::E2004 => "E2004",
            ErrorCode::E2005 => "E2005",
            ErrorCode::E2006 => "E2006",
            ErrorCode::E2007 => "E2007",
            ErrorCode::E2008 => "E2008",
            ErrorCode::E2009 => "E2009",
            ErrorCode::E2010 => "E2010",
            ErrorCode::E2011 => "E2011",
            ErrorCode::E2012 => "E2012",
            ErrorCode::E2013 => "E2013",
            ErrorCode::E2014 => "E2014",
            ErrorCode::E2015 => "E2015",
            // Effect
            ErrorCode::E4001 => "E4001",
            ErrorCode::E4002 => "E4002",
            ErrorCode::E4003 => "E4003",
            ErrorCode::E4004 => "E4004",
            ErrorCode::E4005 => "E4005",
            ErrorCode::E4006 => "E4006",
            // Name resolution
            ErrorCode::E5001 => "E5001",
            ErrorCode::E5002 => "E5002",
            ErrorCode::E5003 => "E5003",
            ErrorCode::E5004 => "E5004",
            // Generics/Trait
            ErrorCode::E6001 => "E6001",
            ErrorCode::E6002 => "E6002",
            ErrorCode::E6003 => "E6003",
            ErrorCode::E6004 => "E6004",
            ErrorCode::E6005 => "E6005",
            ErrorCode::E6006 => "E6006",
            // MIR
            ErrorCode::E7001 => "E7001",
            ErrorCode::E7002 => "E7002",
            ErrorCode::E7003 => "E7003",
            // Runtime
            ErrorCode::E8001 => "E8001",
            ErrorCode::E8002 => "E8002",
            ErrorCode::E8003 => "E8003",
            ErrorCode::E8004 => "E8004",
            ErrorCode::E8005 => "E8005",
            // Module
            ErrorCode::E9001 => "E9001",
            ErrorCode::E9002 => "E9002",
            // Snap
            ErrorCode::E0601 => "E0601",
            ErrorCode::E0602 => "E0602",
            ErrorCode::E0603 => "E0603",
            ErrorCode::E0604 => "E0604",
            // Warnings
            ErrorCode::W0001 => "W0001",
            ErrorCode::W0002 => "W0002",
            ErrorCode::W0003 => "W0003",
            ErrorCode::W0004 => "W0004",
            ErrorCode::W0005 => "W0005",
        }
    }

    /// Returns the default human-readable message template for this error code.
    ///
    /// [`DiagnosticBuilder`](super::DiagnosticBuilder) uses this as the
    /// diagnostic message unless overridden via
    /// [`DiagnosticBuilder::message`](super::DiagnosticBuilder::message).
    ///
    /// # Examples
    ///
    /// ```
    /// use cjc_diag::ErrorCode;
    ///
    /// assert_eq!(ErrorCode::E0002.message_template(), "unterminated string literal");
    /// ```
    pub fn message_template(&self) -> &'static str {
        match self {
            // Lexer
            ErrorCode::E0001 => "unexpected character",
            ErrorCode::E0002 => "unterminated string literal",
            ErrorCode::E0003 => "invalid numeric literal",
            ErrorCode::E0004 => "invalid escape sequence",
            ErrorCode::E0005 => "unterminated block comment",
            ErrorCode::E0006 => "invalid regex literal",
            ErrorCode::E0007 => "missing digits after numeric prefix",
            ErrorCode::E0008 => "invalid character in numeric literal",
            ErrorCode::E0009 => "unterminated byte string",
            ErrorCode::E0010 => "invalid byte literal",
            // Parser
            ErrorCode::E1000 => "unexpected token",
            ErrorCode::E1001 => "expected token not found",
            ErrorCode::E1002 => "expected expression",
            ErrorCode::E1003 => "missing type annotation on function parameter",
            ErrorCode::E1004 => "missing closing delimiter",
            ErrorCode::E1005 => "invalid pattern",
            ErrorCode::E1006 => "expected function body",
            ErrorCode::E1007 => "duplicate field in struct literal",
            ErrorCode::E1008 => "invalid assignment target",
            ErrorCode::E1009 => "expected type expression",
            ErrorCode::E1010 => "missing return type",
            ErrorCode::E1011 => "invalid top-level declaration",
            ErrorCode::E1012 => "empty match expression",
            ErrorCode::E1013 => "missing effect annotation after `/`",
            // Borrow/Ownership
            ErrorCode::E3001 => "move of borrowed value",
            ErrorCode::E3002 => "use after move",
            ErrorCode::E3003 => "cannot borrow as mutable",
            ErrorCode::E3004 => "cannot assign to immutable variable",
            ErrorCode::E3005 => "value does not live long enough",
            ErrorCode::E3006 => "cannot move out of borrowed content",
            ErrorCode::E3007 => "mutable borrow while immutable borrow exists",
            ErrorCode::E3008 => "double mutable borrow",
            // Type
            ErrorCode::E2001 => "type mismatch",
            ErrorCode::E2002 => "cannot infer type",
            ErrorCode::E2003 => "incompatible types in binary operation",
            ErrorCode::E2004 => "cannot unify types",
            ErrorCode::E2005 => "wrong number of arguments",
            ErrorCode::E2006 => "undefined variable",
            ErrorCode::E2007 => "undefined function",
            ErrorCode::E2008 => "undefined field",
            ErrorCode::E2009 => "non-exhaustive match patterns",
            ErrorCode::E2010 => "cannot assign to immutable variable",
            ErrorCode::E2011 => "return type mismatch",
            ErrorCode::E2012 => "cannot call non-function type",
            ErrorCode::E2013 => "cannot index non-indexable type",
            ErrorCode::E2014 => "duplicate type definition",
            ErrorCode::E2015 => "missing field in struct literal",
            // Effect
            ErrorCode::E4001 => "GC operation in nogc context",
            ErrorCode::E4002 => "IO operation in pure context",
            ErrorCode::E4003 => "nondeterministic operation in deterministic context",
            ErrorCode::E4004 => "allocation in nogc function",
            ErrorCode::E4005 => "effect annotation mismatch",
            ErrorCode::E4006 => "mutation in pure context",
            // Name resolution
            ErrorCode::E5001 => "unresolved name",
            ErrorCode::E5002 => "ambiguous name",
            ErrorCode::E5003 => "duplicate definition",
            ErrorCode::E5004 => "import not found",
            // Generics/Trait
            ErrorCode::E6001 => "trait bound not satisfied",
            ErrorCode::E6002 => "missing trait implementation",
            ErrorCode::E6003 => "conflicting trait implementations",
            ErrorCode::E6004 => "unused type parameter",
            ErrorCode::E6005 => "wrong number of type arguments",
            ErrorCode::E6006 => "trait method signature mismatch",
            // MIR
            ErrorCode::E7001 => "internal compiler error in MIR lowering",
            ErrorCode::E7002 => "SSA verification failed",
            ErrorCode::E7003 => "CFG verification failed",
            // Runtime
            ErrorCode::E8001 => "index out of bounds",
            ErrorCode::E8002 => "division by zero",
            ErrorCode::E8003 => "stack overflow",
            ErrorCode::E8004 => "shape mismatch in tensor operation",
            ErrorCode::E8005 => "type assertion failed",
            // Module
            ErrorCode::E9001 => "module not found",
            ErrorCode::E9002 => "circular module dependency",
            // Snap
            ErrorCode::E0601 => "snapshot logic hash mismatch",
            ErrorCode::E0602 => "snapshot type hash mismatch",
            ErrorCode::E0603 => "snapshot layout hash mismatch",
            ErrorCode::E0604 => "type is not snap-compatible",
            // Warnings
            ErrorCode::W0001 => "unused variable",
            ErrorCode::W0002 => "unused import",
            ErrorCode::W0003 => "unreachable code",
            ErrorCode::W0004 => "variable shadows outer scope",
            ErrorCode::W0005 => "deprecated feature",
        }
    }

    /// Returns the [`Severity`] for this error code.
    ///
    /// All `W0xxx` codes map to [`Severity::Warning`]; every other code maps
    /// to [`Severity::Error`].
    ///
    /// # Examples
    ///
    /// ```
    /// use cjc_diag::{ErrorCode, Severity};
    ///
    /// assert_eq!(ErrorCode::E0001.severity(), Severity::Error);
    /// assert_eq!(ErrorCode::W0001.severity(), Severity::Warning);
    /// ```
    pub fn severity(&self) -> Severity {
        match self {
            ErrorCode::W0001
            | ErrorCode::W0002
            | ErrorCode::W0003
            | ErrorCode::W0004
            | ErrorCode::W0005 => Severity::Warning,
            _ => Severity::Error,
        }
    }

    /// Returns a human-readable category name for this error code.
    ///
    /// Categories are derived from the numeric prefix of the code string
    /// (e.g., `"lexer"` for E0xxx, `"parser"` for E1xxx). Snap errors
    /// (E06xx) are distinguished from other E0xxx codes.
    ///
    /// # Examples
    ///
    /// ```
    /// use cjc_diag::ErrorCode;
    ///
    /// assert_eq!(ErrorCode::E0001.category(), "lexer");
    /// assert_eq!(ErrorCode::E0601.category(), "snap");
    /// assert_eq!(ErrorCode::E2001.category(), "type");
    /// assert_eq!(ErrorCode::W0003.category(), "warning");
    /// ```
    pub fn category(&self) -> &'static str {
        let code = self.code_str();
        if code.starts_with("W") {
            return "warning";
        }
        match &code[1..2] {
            "0" => {
                // Check for snap errors (E06xx)
                if code.starts_with("E06") {
                    "snap"
                } else {
                    "lexer"
                }
            }
            "1" => "parser",
            "2" => "type",
            "3" => "ownership",
            "4" => "effect",
            "5" => "name resolution",
            "6" => "generics/trait",
            "7" => "mir",
            "8" => "runtime",
            "9" => "module",
            _ => "unknown",
        }
    }
}

impl std::fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.code_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code_str() {
        assert_eq!(ErrorCode::E0001.code_str(), "E0001");
        assert_eq!(ErrorCode::E1000.code_str(), "E1000");
        assert_eq!(ErrorCode::E2001.code_str(), "E2001");
        assert_eq!(ErrorCode::W0001.code_str(), "W0001");
    }

    #[test]
    fn test_error_code_severity() {
        assert_eq!(ErrorCode::E0001.severity(), Severity::Error);
        assert_eq!(ErrorCode::E1000.severity(), Severity::Error);
        assert_eq!(ErrorCode::W0001.severity(), Severity::Warning);
        assert_eq!(ErrorCode::W0003.severity(), Severity::Warning);
    }

    #[test]
    fn test_error_code_category() {
        assert_eq!(ErrorCode::E0001.category(), "lexer");
        assert_eq!(ErrorCode::E1000.category(), "parser");
        assert_eq!(ErrorCode::E2001.category(), "type");
        assert_eq!(ErrorCode::E3001.category(), "ownership");
        assert_eq!(ErrorCode::E4001.category(), "effect");
        assert_eq!(ErrorCode::E6001.category(), "generics/trait");
        assert_eq!(ErrorCode::E0601.category(), "snap");
        assert_eq!(ErrorCode::W0001.category(), "warning");
    }

    #[test]
    fn test_borrow_ownership_codes() {
        // Verify all E3xxx codes are defined and have correct properties
        let borrow_codes = [
            (ErrorCode::E3001, "E3001", "move of borrowed value"),
            (ErrorCode::E3002, "E3002", "use after move"),
            (ErrorCode::E3003, "E3003", "cannot borrow as mutable"),
            (ErrorCode::E3004, "E3004", "cannot assign to immutable variable"),
            (ErrorCode::E3005, "E3005", "value does not live long enough"),
            (ErrorCode::E3006, "E3006", "cannot move out of borrowed content"),
            (ErrorCode::E3007, "E3007", "mutable borrow while immutable borrow exists"),
            (ErrorCode::E3008, "E3008", "double mutable borrow"),
        ];
        for (code, expected_str, expected_msg) in &borrow_codes {
            assert_eq!(code.code_str(), *expected_str);
            assert_eq!(code.message_template(), *expected_msg);
            assert_eq!(code.severity(), Severity::Error);
            assert_eq!(code.category(), "ownership");
        }
    }

    #[test]
    fn test_error_code_display() {
        assert_eq!(format!("{}", ErrorCode::E0001), "E0001");
        assert_eq!(format!("{}", ErrorCode::W0001), "W0001");
    }

    #[test]
    fn test_message_templates_nonempty() {
        // Verify all codes have non-empty templates
        let codes = [
            ErrorCode::E0001, ErrorCode::E1000, ErrorCode::E2001,
            ErrorCode::E4001, ErrorCode::E6001, ErrorCode::E7001,
            ErrorCode::E8001, ErrorCode::E9001, ErrorCode::E0601,
            ErrorCode::W0001,
        ];
        for code in &codes {
            assert!(!code.message_template().is_empty(), "{} has empty template", code);
        }
    }
}
