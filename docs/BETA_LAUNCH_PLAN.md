# CJC Beta Launch Plan — Stack Role Group Prompt

## Verification Summary (2025-03-23)

### Items Already Complete (Removed from Plan)
- `if` as expression — `ExprKind::IfExpr` in AST, eval, mir-exec
- `for x in arr` — `ForIter::Expr` variant supports array iteration
- String interpolation — `f"...{expr}..."` via `FStringLit` token
- README.md — Exists at repo root, well-written
- Parallel accumulator foundation — `BinnedAccumulatorF64::merge()` ready for future threading

### Items Still Needed
- `--multi-file` defaults to `false` (line 181 of main.rs)
- Error message audit not done
- No pre-built binaries / CI release workflow
- No `cjc init` command
- No example `.cjc` programs
- REPL lacks tab completion / history persistence
- No package manager implementation
- LSP at 746 LOC (has completion + hover, needs expansion)

---

## ROLE

You are a beta launch preparation team working inside the CJC compiler repository.

You consist of:

1. **Release Engineer** — owns CI/CD, binary packaging, GitHub Actions, release artifacts
2. **Developer Experience Lead** — owns error messages, REPL, editor integration, first-run experience
3. **Documentation & Examples Writer** — owns tutorials, example programs, API docs
4. **Language Ergonomics Auditor** — identifies friction points in the first 5 minutes of use
5. **QA Gate Keeper** — ensures no regressions, validates all changes against 5,560+ existing tests

---

## PRIME DIRECTIVES

1. **Do not break the compiler pipeline** — Lexer → Parser → AST → HIR → MIR → Exec
2. **Do not break determinism** — Same seed = bit-identical output
3. **Do not break existing tests** — All 5,560+ tests must pass after every change
4. **Minimize invasiveness** — Prefer small targeted changes over refactors
5. **Both executors must agree** — eval and mir-exec produce identical results

---

## PHASE 1: Pre-Beta Blockers (Do First)

### 1.1 Make `--multi-file` the Default

**File:** `crates/cjc-cli/src/main.rs`
**Change:** Line 181: `multi_file: false` → `multi_file: true`
**Also:** Add `--single-file` flag for opting out (backward compat)

**Test:** Existing multi-file tests should still pass. Single-file programs should work unchanged.

### 1.2 Error Message Audit

Audit the top 10 most common user mistakes and ensure clear error messages:

| Mistake | Current Error | Expected Error |
|---------|--------------|----------------|
| `fn f(x)` (no type) | Parse error | "parameter `x` requires a type annotation: `fn f(x: i64)`" |
| Missing `else` in if-expr used as value | Silent Void | "if expression used as value requires an else branch" |
| `let x = 5` then `x = 6` (immutable) | Runtime error | "cannot assign to `x`: variable is immutable. Use `let mut x = 5`" |
| Wrong arg count to builtin | Vague error | "mean() expects 1 argument (array), got 0" |
| Semicolon after `if {}` | Parse error | "unexpected `;` after if block (CJC doesn't use semicolons after block statements)" |
| Unknown function | Runtime error | "unknown function `foo`. Did you mean `floor`?" (fuzzy match) |
| Type mismatch | Vague | "expected f64, got i64 in argument 1 of `mean()`" |
| Array index OOB | Runtime panic | "index 5 out of bounds for array of length 3" |
| Missing return value | Void | "function `f` declared to return i64 but body returns void" |
| `HashMap` used in CJC code | N/A | N/A (CJC has no HashMap — this is a Rust-side invariant) |

**Files to audit:**
- `crates/cjc-parser/src/lib.rs` — Parse error messages
- `crates/cjc-eval/src/lib.rs` — Runtime error messages
- `crates/cjc-mir-exec/src/lib.rs` — MIR runtime error messages
- `crates/cjc-diag/src/` — Diagnostic infrastructure

### 1.3 GitHub Actions CI + Release Workflow

Create `.github/workflows/ci.yml`:
- Build on Windows, macOS, Linux
- Run `cargo test --workspace`
- On tag push (e.g., `v0.1.0-beta.1`), build release binaries
- Upload to GitHub Releases

Create `.github/workflows/release.yml`:
- Triggered on tag push `v*`
- Matrix: `windows-latest`, `macos-latest`, `ubuntu-latest`
- `cargo build --release`
- Upload artifacts: `cjc-windows-x64.exe`, `cjc-macos-x64`, `cjc-linux-x64`

---

## PHASE 2: First Week After Beta

### 2.1 Example Programs

Create `examples/` directory with 5-10 `.cjc` programs:

```
examples/
  01_hello.cjc            — Hello world + basic types
  02_fibonacci.cjc        — Recursion + loops
  03_statistics.cjc       — Mean, sd, correlation on sample data
  04_linear_regression.cjc — Fit a line, print coefficients
  05_dataframe.cjc        — Load CSV, filter, group, summarise
  06_neural_net.cjc       — Simple MLP training loop
  07_autodiff.cjc         — Gradient computation with AD
  08_time_series.cjc      — ARIMA fitting + forecast
  09_monte_carlo.cjc      — Pi estimation with deterministic RNG
  10_chess_demo.cjc       — Minimal chess board + evaluation
```

Each program should:
- Be under 50 lines
- Run with `cjc run examples/01_hello.cjc`
- Include comments explaining what it does
- Demonstrate one key CJC feature

### 2.2 `cjc init` Command

**File:** `crates/cjc-cli/src/commands/init.rs`

```
$ cjc init my_project
Created my_project/
  main.cjc
  cjc.toml (stub)
```

### 2.3 REPL Improvements

**Files:** `crates/cjc-cli/src/line_editor.rs`

- History persistence to `~/.cjc_history`
- Basic tab completion for builtins
- Multi-line input (detect unclosed `{` and continue on next line)

---

## PHASE 3: First Month

### 3.1 Package Manager v0

Per ADR-0013:
- `cjc.toml` manifest parsing
- `cjc install` from git URLs
- Lockfile with SHA-256 checksums
- Local cache at `~/.cjc/cache/`

### 3.2 LSP Expansion

Extend `crates/cjc-analyzer/`:
- Go-to-definition for functions and variables
- Signature help on function calls
- Workspace symbol search
- Better completion (context-aware, not just keyword list)

### 3.3 Jupyter Kernel

Create `cjc-jupyter/` crate:
- Implement Jupyter kernel protocol (ZMQ)
- Execute CJC code cells
- Return output as display_data
- Support `--seed` per notebook

---

## VERIFICATION PROTOCOL

After each phase:

1. `cargo check --workspace` — Zero errors
2. `cargo test --workspace` — Zero failures (5,560+ tests)
3. `cargo build --release` — Binary builds cleanly
4. Manual smoke test: `cjc run examples/03_statistics.cjc`
5. Binary size check: Should remain under 10 MB

---

## DEVELOPMENT WORKFLOW

For each task:

### STEP 1: Identify affected files
### STEP 2: Make minimal changes
### STEP 3: Run tests
### STEP 4: Document changes
### STEP 5: Commit with descriptive message

---

## OUTPUT FORMAT

```
FILE: path/to/file.rs
<code>
```

**Test Summary:**
```
New tests:      X
Existing tests: Y (all passing)
Failures:       0
```
