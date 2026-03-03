# Language Hardening Phase — Changes

**Date:** 2026-03-02
**Commit:** (pending)
**Tests:** 83 new, 3118 total workspace (0 failures, 20 ignored)

---

## Summary

This phase closes the major syntactic, operator, ML optimization, and CLI gaps identified in the codebase audit. Three categories of changes were made:

1. **Syntax & Operator Extensions** — 6 new language features
2. **ML Autodiff Builtins** — 4 new gradient utilities
3. **CLI & Developer Experience** — REPL, color diagnostics, structured flags

---

## 1. Syntax & Operator Extensions

### 1.1 Compound Assignments (`+=`, `-=`, `*=`, `/=`, `%=`, `**=`)

**Before:** `x = x + 1;`
**After:** `x += 1;`

- New tokens: `PlusEq`, `MinusEq`, `StarEq`, `SlashEq`, `PercentEq`, `StarStarEq`
- New AST node: `ExprKind::CompoundAssign { op, target, value }`
- Desugared in HIR to `target = target op value`
- Works with Int, Float, and Tensor values
- Also supports bitwise compound: `&=`, `|=`, `^=`, `<<=`, `>>=`

### 1.2 If-as-Expression

**Before:** `if` was statement-only; couldn't use as a value
**After:** `let x = if cond { a } else { b };`

- New AST node: `ExprKind::IfExpr { condition, then_block, else_branch }`
- Parsed in expression (prefix) context when `if` appears in expression position
- Supports `else if` chains
- Returns `Value::Void` if condition is false and no else branch

### 1.3 Bitwise Operators (`&`, `|`, `^`, `~`, `<<`, `>>`)

- New tokens: `Amp`, `Caret`, `LtLt`, `GtGt` (and compound variants)
- New BinOp variants: `BitAnd`, `BitOr`, `BitXor`, `Shl`, `Shr`
- New UnaryOp variant: `BitNot`
- `Pipe` (`|`) now serves dual purpose: lambda params in prefix, bitwise OR in infix
- `Tilde` (`~`) now serves dual purpose: regex match in `~=`, bitwise NOT as prefix
- `&` is now a valid single-character token (no longer errors with "did you mean `&&`?")
- Int-only operations; runtime error on Float/other types

**Precedence (high to low):**
```
Postfix (. [ ()     26
Unary   (- ! ~)     24
Power   (**)        22  (right-associative)
Mul     (* / %)     20
Add     (+ -)       18
Shift   (<< >>)     16
Compare (< > <= >=) 14
Equal   (== !=)     12
BitAnd  (&)         11
BitXor  (^)         10
BitOr   (|)          9
And     (&&)         8
Or      (||)         6
Pipe    (|>)         4
Assign  (= += etc)   2  (right-associative)
```

### 1.4 Power Operator (`**`)

- New token: `StarStar`
- New BinOp variant: `Pow`
- Right-associative: `2 ** 3 ** 2` = `2 ** 9` = 512
- Higher precedence than multiplication: `2 * 3 ** 2` = `2 * 9` = 18
- Int: `(a as f64).powf(b as f64) as i64`
- Float: `a.powf(b)`
- Constant folding: `a.wrapping_pow(b as u32)` for integers

### 1.5 Hex/Binary/Octal Integer Literals

- `0xFF`, `0XFF` — Hexadecimal (case-insensitive prefix)
- `0b1010`, `0B1010` — Binary
- `0o777`, `0O777` — Octal
- Underscore separators: `0xFF_FF`, `0b1111_0000`
- All produce `IntLit` tokens; `int_value()` handles base conversion
- Error diagnostics for missing digits after prefix

### 1.6 Optimizer Updates

All new operators are handled in the MIR optimizer:
- **Constant folding:** `Pow`, `BitAnd`, `BitOr`, `BitXor`, `Shl`, `Shr`, `BitNot`
- **Strength reduction:** unchanged (existing passes)
- **Dead code elimination:** unchanged
- **CSE:** works with new operators via `expr_key()`
- **LICM:** unchanged (new ops are pure, correctly hoisted)

---

## 2. ML Autodiff Builtins

| Builtin | Args | Description |
|---------|------|-------------|
| `stop_gradient(x)` | 1 | Returns x unchanged. Semantic marker: AD should not propagate gradients through this value. |
| `grad_checkpoint(x)` | 1 | Returns x unchanged. Semantic marker: memory checkpoint boundary for gradient recomputation. |
| `clip_grad(x, min, max)` | 3 | Clips value to [min, max] range. For gradient clipping in training loops. |
| `grad_scale(x, scale)` | 2 | Scales value by scalar factor. For gradient scaling/accumulation. |

All registered as `PURE` in effect registry. Wired in both eval and MIR-exec.

---

## 3. CLI & Developer Experience

### 3.1 REPL (Interactive Mode)

```bash
cjc repl [--seed N] [--color|--no-color]
```

- Persistent `Interpreter` state across lines
- Variables, functions, and state carry over
- Parse errors shown with full diagnostics
- Type `exit`, `quit`, or Ctrl+C to exit
- Banner: `CJC REPL v0.1.0`

### 3.2 Color Diagnostics

```
error[E0001]: unexpected token     ← bold red
  --> test.cjc:1:14                ← bold blue
   |
 1 | let x = 42 +;
   |              ^ expected expr  ← bold red underline
   |
   = hint: remove trailing `+`    ← bold cyan
```

- New `DiagnosticRenderer::new_with_color(source, filename, use_color)`
- ANSI codes: bold red (errors), bold yellow (warnings), bold cyan (hints), bold blue (locations), bold green (secondary labels)
- Backward compatible: `new()` defaults to no color
- New `DiagnosticBag::render_all_color(source, filename, use_color)`

### 3.3 CLI Flags

| Flag | Description |
|------|-------------|
| `--help`, `-h` | Print usage and exit |
| `--version`, `-V` | Print version and exit |
| `--color` | Force color output |
| `--no-color` | Disable color output |
| `--seed N` | Set RNG seed (default: 42) |
| `--time` | Print execution time |
| `--mir-opt` | Enable MIR optimizations |
| `--mir-mono` | Enable monomorphization |
| `--multi-file` | Enable module resolution |

---

## Files Modified

### Core Crates (6 files)
| File | Changes |
|------|---------|
| `crates/cjc-lexer/src/lib.rs` | 15 new token types, hex/bin/oct lexing, compound operator lexing |
| `crates/cjc-ast/src/lib.rs` | 6 new BinOp, 1 UnaryOp, 2 new ExprKind, Display impls |
| `crates/cjc-parser/src/lib.rs` | New precedence table (13 levels), compound assign parsing, if-expr prefix, bitwise infix, `**` right-assoc |
| `crates/cjc-types/src/lib.rs` | Type checking for new ops + CompoundAssign + IfExpr |
| `crates/cjc-eval/src/lib.rs` | Pow, bitwise, BitNot eval; CompoundAssign desugar; IfExpr; `eval_binary_values()` helper |
| `crates/cjc-mir-exec/src/lib.rs` | Pow, bitwise, BitNot eval in MIR layer |

### IR & Optimizer (2 files)
| File | Changes |
|------|---------|
| `crates/cjc-hir/src/lib.rs` | CompoundAssign desugar, IfExpr lowering, var ref collection |
| `crates/cjc-mir/src/optimize.rs` | Constant folding for Pow, bitwise, BitNot |

### Runtime & Effects (2 files)
| File | Changes |
|------|---------|
| `crates/cjc-runtime/src/builtins.rs` | 4 new ML builtins |
| `crates/cjc-types/src/effect_registry.rs` | 4 new PURE registrations |

### CLI & Diagnostics (2 files)
| File | Changes |
|------|---------|
| `crates/cjc-diag/src/lib.rs` | ANSI color support, `new_with_color()`, `render_all_color()` |
| `crates/cjc-cli/src/main.rs` | REPL command, --help/--version/--color flags, structured arg parsing |

### Tests (10 new files)
| File | Tests | Coverage |
|------|-------|----------|
| `test_01_compound_assignments.rs` | 11 | +=, -=, *=, /=, %=, chaining, loops |
| `test_02_if_expression.rs` | 7 | Basic, let binding, nested, else-if, float |
| `test_03_bitwise_operators.rs` | 13 | &, \|, ^, ~, <<, >>, popcount, mask extraction |
| `test_04_power_operator.rs` | 11 | Int, float, negative exp, right-assoc, precedence |
| `test_05_hex_bin_oct_literals.rs` | 15 | Hex, binary, octal, underscores, mixed arithmetic |
| `test_06_ml_ad_builtins.rs` | 12 | stop_gradient, grad_checkpoint, clip_grad, grad_scale |
| `test_07_cli_features.rs` | 4 | Color/plain diagnostic rendering |
| `test_08_parity.rs` | 11 | Eval vs MIR-exec parity for all new features |

### Documentation (2 new files)
| File | Description |
|------|-------------|
| `docs/language_hardening_phase/STACK_ROLE_GROUP.md` | 5-role stack role group prompt |
| `docs/language_hardening_phase/CHANGES.md` | This document |

---

## Test Results

| Suite | Passed | Failed | Ignored |
|-------|--------|--------|---------|
| Language Hardening (new) | 83 | 0 | 0 |
| Math Hardening | 120 | 0 | 0 |
| Chess RL Benchmark | 66 | 0 | 3 |
| Full Workspace | **3118** | **0** | **20** |

**Zero regressions.** All previous tests continue to pass.
