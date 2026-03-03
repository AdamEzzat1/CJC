# CLI and REPL Guide

## Installation

```bash
cargo install --path crates/cjc-cli
```

This installs the `cjc` binary to `~/.cargo/bin/` (added to PATH by rustup).

## Commands

| Command | Description |
|---------|-------------|
| `cjc run <file.cjc>` | Run a CJC program |
| `cjc repl` | Start interactive REPL |
| `cjc check <file.cjc>` | Type-check without execution |
| `cjc parse <file.cjc>` | Parse and pretty-print the AST |
| `cjc lex <file.cjc>` | Tokenize and print all tokens |

## Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--help`, `-h` | Print usage and exit | — |
| `--version`, `-V` | Print version (`cjc 0.1.0`) | — |
| `--seed <N>` | Set RNG seed | 42 |
| `--time` | Print execution time to stderr | Off |
| `--mir-opt` | Enable MIR optimizations (constant folding + DCE) | Off |
| `--mir-mono` | Enable MIR monomorphization | Off |
| `--multi-file` | Enable multi-file module resolution | Off |
| `--color` | Force ANSI color output | Auto |
| `--no-color` | Disable color output | Auto |
| `--reproducible` | Enable reproducibility mode | Off |

Flags can appear anywhere on the command line.

## Usage Examples

```bash
# Run a program
cjc run myprogram.cjc

# Run with custom seed and timing
cjc run simulation.cjc --seed 12345 --time

# Run with MIR optimizations
cjc run compute.cjc --mir-opt

# Type-check without running
cjc check myprogram.cjc

# See the parsed AST
cjc parse myprogram.cjc

# See the token stream
cjc lex myprogram.cjc
```

## Execution Engines

CJC has two execution backends:

1. **AST Interpreter** (default) — tree-walk interpreter via `cjc_eval`
2. **MIR Executor** (`--mir-opt` or `--mir-mono`) — lowered to HIR then MIR,
   with optional constant folding, dead code elimination, and monomorphization

Both engines produce identical results for valid programs (verified by 3,495+
parity tests).

## Interactive REPL

Start with:

```bash
cjc repl
cjc repl --seed 999    # custom seed
```

### Prompt

```
CJC REPL v0.1.0  (type :help for commands, :quit to exit)
cjc>
```

### Features

- **Persistent state** — variables and functions persist across lines
- **Multi-line input** — lines ending with `{` or `\` auto-continue;
  unbalanced braces prompt for more input
- **History** — up/down arrows cycle through history; saved to `~/.cjc_history`
- **Keyboard shortcuts**:
  - Left/Right arrows: move cursor within line
  - Home/End: jump to start/end of line
  - Ctrl+A: go to start, Ctrl+E: go to end
  - Ctrl+U: clear line, Ctrl+K: kill to end of line
  - Ctrl+L: clear screen
  - Ctrl+C: cancel current input
  - Ctrl+D: exit REPL (EOF)
  - Backspace/Delete: delete characters

### Meta-Commands

| Command | Description |
|---------|-------------|
| `:help`, `:h` | Show available commands |
| `:quit`, `:q` | Exit the REPL |
| `:reset` | Clear all variable bindings |
| `:type <expr>` | Type-check an expression |
| `:ast <expr>` | Display the AST of an expression |
| `:mir <expr>` | Display the MIR of an expression |
| `:env` | Show environment info |
| `:seed` | Show the current RNG seed |

### Example Session

```
$ cjc repl
CJC REPL v0.1.0  (type :help for commands, :quit to exit)
cjc> let x: i64 = 42;
cjc> print(x * 2);
84
cjc> fn square(n: i64) -> i64 {
...     n * n
... }
cjc> print(square(7));
49
cjc> :seed
RNG seed: 42
cjc> :reset
Environment reset.
cjc> :quit
```

### Multi-Line Example

```
cjc> fn fibonacci(n: i64) -> i64 {
...     if n <= 1 {
...         return n;
...     }
...     fibonacci(n - 1) + fibonacci(n - 2)
... }
cjc> print(fibonacci(10));
55
```

## Color Diagnostics

CJC produces colored error messages with structured error codes:

```
error[E0160]: Cannot assign to field of record type
  --> myfile.cjc:5:5
   |
 5 |     c.width = 1024;
   |     ^^^^^^^^^^^^^^^ record fields are immutable
   |
   hint: Use `struct` instead of `record` for mutable fields
```

Error code ranges:
- **E0xxx**: Type errors
- **E1xxx**: Parse errors
- **E2xxx**: Name resolution errors
- **E3xxx**: Bound violations
- **E4xxx**: Effect violations
- **E5xxx**: NoGC violations
- **E6xxx**: Trait errors
- **E7xxx**: Pattern matching errors
- **E8xxx**: Module errors
