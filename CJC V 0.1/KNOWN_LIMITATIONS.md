# Known Limitations and Roadmap

## Current Limitations (v0.1)

### Type System

- **Function parameters require explicit type annotations.** You cannot write
  `fn f(x)` — it must be `fn f(x: i64)`. Use `Any` for dynamic/polymorphic
  parameters: `fn f(x: Any)`.

- **Generics are parsed but not polymorphically instantiated.** Generic type
  parameters (`<T>`) are recognized by the parser and used for trait bound
  checking, but the runtime uses dynamic dispatch via `Any` rather than
  monomorphic specialization.

- **No type inference for function return types.** Return types must be
  explicitly annotated when not void.

### REPL

- **No autocomplete/tab completion.** The REPL supports history and keyboard
  shortcuts but not intelligent completion.
- **No syntax highlighting** in the REPL (output diagnostics are colored).
- **Single-file only** — the REPL doesn't support `import` statements.

### Performance

- **CPU-only tensor operations.** No GPU acceleration (CUDA/ROCm/Metal).
- **No SIMD vectorization.** Tensor ops are scalar loops in Rust.
- **No parallel execution.** All computation is single-threaded.
- **MIR optimizer is basic.** Only constant folding and dead code elimination.
  No common subexpression elimination, loop-invariant code motion, or inlining.

### Statistics API

- **`mean()` is a tensor dot method only.** There is no free function `mean()`.
  Use `Tensor.from_vec(data, shape).mean()` for tensor mean, or use `median()`,
  `sd()`, `variance()`, `quantile()`, `iqr()` which are free functions.

- **`lm()` auto-adds an intercept column.** The call `lm(X, y, n, p)` internally
  prepends a column of ones to the design matrix. Output has `p+1` coefficients
  (intercept + p slopes). Requires `n > p+1` or rank-deficiency error.

### Broadcasting

- **No MIR-level broadcast fusion.** `broadcast("sin", t)` allocates a new
  tensor per call. Chaining `broadcast("sin", broadcast("exp", t))` creates an
  intermediate. Fused broadcast pipelines are planned for v0.2.

- **No dot-operator syntax (`.+`, `.*`).** Broadcasting is explicit via the
  `broadcast()` and `broadcast2()` builtins, not via operator overloading.

### Language Features

- **No async/await.** CJC is designed for batch scientific computing.
- **No networking.** No HTTP, sockets, or other network I/O.
- **No package manager.** All 300+ builtins are built-in; no third-party
  packages or library ecosystem.
- **No string interpolation in all contexts.** Format strings (`f"..."`) work
  but have limited expression support inside `{}`.
- **`if` statement vs expression.** `if` works as an expression
  (`let x = if cond { a } else { b }`) but returns Void when there's no
  `else` branch.

### Data

- **TidyView operations require the eval engine.** Some tidy verbs may not
  work with `--mir-opt`.
- **No streaming/lazy DataFrames.** TidyView is a bitmask projection but
  all data must fit in memory.

### Platform

- **Windows-focused development.** Tested primarily on Windows; Unix/macOS
  should work but has less testing.
- **REPL history path** uses `~/.cjc_history` which may need adjustment
  on some systems.

## Workarounds

| Limitation | Workaround |
|-----------|------------|
| No type inference for params | Use `Any` for polymorphic code |
| No generics specialization | Use `Any` + runtime dispatch |
| No GPU | Use optimized CPU matmul (already uses efficient algorithms) |
| No packages | All common operations are built-in (300+ functions) |
| No networking | Use `file_read`/`file_write` + external tools |
| No string interpolation | Use `to_string()` + string concatenation |

## Roadmap (v0.2)

Planned improvements:

1. **Full type inference** — Hindley-Milner style inference for all bindings
2. **Polymorphic generics** — True monomorphization with specialization
3. **Complete MIR optimizer** — CSE, LICM, inlining, strength reduction
4. **LSP server** — Language Server Protocol for IDE integration
5. **Module system** — `import` with package resolution
6. **REPL autocomplete** — Tab completion for builtins and variables
7. **Documentation comments** — `///` doc comments with generation
8. **Error recovery** — Parser continues after errors for better diagnostics
9. **Parallel matmul** — Multi-threaded matrix multiplication
10. **WebAssembly target** — Compile CJC to WASM for browser execution
