# Language Hardening Phase — Stack Role Group Prompt

**Phase:** Language Hardening (Post Mathematics Hardening)
**Objective:** Close all syntactic, operator, ML optimization, and CLI gaps identified in the codebase audit. Deliver compound assignments, if-expressions, bitwise operators, power operator, hex/binary/octal literals, ML AD improvements (Hessian, Jacobian, stop_gradient, gradient checkpointing), REPL, and color diagnostics.

---

## Roles

### 1. Syntax Extension Engineer
**Focus:** Lexer, parser, and AST changes for new operators and expression forms.

**Responsibilities:**
- Add new TokenKind variants for compound assignments (`+=`, `-=`, `*=`, `/=`, `%=`), bitwise operators (`Amp`, `Bar`, `Caret`, `Tilde`, `LtLt`, `GtGt`), and power operator (`StarStar`)
- Extend BinOp/UnaryOp enums with new variants
- Add `ExprKind::CompoundAssign` for compound assignments
- Convert `if` from statement-only to expression form (`ExprKind::IfExpr`)
- Implement hex (`0x`), binary (`0b`), and octal (`0o`) literal lexing
- Update precedence table for new operators (bitwise between logical and comparison)
- Ensure all parser changes preserve backward compatibility

**Key Files:**
- `crates/cjc-lexer/src/lib.rs` — Token definitions and lexing
- `crates/cjc-ast/src/lib.rs` — AST node definitions
- `crates/cjc-parser/src/lib.rs` — Pratt parser and precedence table

### 2. Three-Layer Wiring Engineer
**Focus:** Ensure every new operator/expression evaluates correctly in both engines.

**Responsibilities:**
- Wire compound assignments in eval: desugar `x += e` to `x = x + e`
- Wire bitwise operators for Int values in both eval and MIR-exec
- Wire power operator (`**`) for Int and Float in both eval and MIR-exec
- Wire if-expression evaluation (return value from branches)
- Register all new operations in effect_registry if needed
- Ensure eval/MIR-exec parity for every new feature

**Key Files:**
- `crates/cjc-eval/src/lib.rs` — AST interpreter
- `crates/cjc-mir-exec/src/lib.rs` — MIR executor
- `crates/cjc-types/src/effect_registry.rs` — Effect classification

### 3. ML Optimization Architect
**Focus:** Autodiff improvements for production ML workflows.

**Responsibilities:**
- Implement `stop_gradient()` builtin to prevent gradient flow
- Implement gradient checkpointing for memory-efficient training
- Add Hessian computation (second-order derivatives via double-backprop)
- Add Jacobian computation (multi-output gradient collection)
- Integrate new AD features with existing GradOp infrastructure
- Wire new AD builtins in both execution engines

**Key Files:**
- `crates/cjc-ad/src/lib.rs` — Forward + reverse mode AD
- `crates/cjc-runtime/src/builtins.rs` — Builtin dispatch
- `crates/cjc-types/src/effect_registry.rs` — Effect classification

### 4. CLI & DX Specialist
**Focus:** REPL, color diagnostics, and structured CLI.

**Responsibilities:**
- Implement basic REPL loop with persistent interpreter state
- Add ANSI color output to DiagnosticRenderer (errors=red, warnings=yellow, hints=cyan)
- Add `--color` / `--no-color` flags with TTY auto-detection
- Add `--help` and `--version` flags
- Restructure CLI argument parsing for maintainability

**Key Files:**
- `crates/cjc-cli/src/main.rs` — CLI entry point
- `crates/cjc-diag/src/lib.rs` — Diagnostic rendering

### 5. Regression Gatekeeper
**Focus:** Test every new feature, ensure zero regressions.

**Responsibilities:**
- Write tests for all compound assignments (all types: Int, Float, Tensor)
- Write tests for if-expression (nested, with else-if, in let bindings)
- Write tests for all 6 bitwise operators (Int only)
- Write tests for power operator (Int, Float, negative exponents)
- Write tests for hex/binary/octal literals
- Write tests for ML AD improvements
- Write parity tests (eval vs MIR-exec) for every new feature
- Run full workspace regression suite

**Key Files:**
- `tests/reinforcement_learning_tests/` — New test directory
- `tests/test_reinforcement_learning.rs` — Test harness
- `Cargo.toml` — Test binary registration

---

## Implementation Sprints

### Sprint 1: Syntax Foundations
- Compound assignments (`+=`, `-=`, `*=`, `/=`, `%=`)
- Power operator (`**`)
- Hex/binary/octal integer literals

### Sprint 2: Expression & Operator Expansion
- If-as-expression
- Bitwise operators (`&`, `|`, `^`, `~`, `<<`, `>>`)

### Sprint 3: ML Optimization
- `stop_gradient()` builtin
- Gradient checkpointing
- Hessian and Jacobian computation

### Sprint 4: CLI & DX
- REPL with persistent state
- Color diagnostics
- Structured CLI (--help, --version, --color)

### Sprint 5: Testing & Regression
- Full test suite for all new features
- Parity tests (eval vs MIR-exec)
- Full workspace regression

---

## Constraints
1. Zero external dependencies for syntax/operator changes
2. All new features must work in BOTH eval and MIR-exec
3. Three-layer wiring: lexer/parser/AST + effect_registry + both engines
4. Backward compatibility: all existing CJC programs must continue to work
5. Operator precedence must follow standard conventions
6. REPL may use stdin directly (no external line-editing dependency)

## Success Criteria
- All new tests pass
- Full workspace regression: 0 failures
- Chess RL benchmark: unchanged results
- Math hardening tests: unchanged results
