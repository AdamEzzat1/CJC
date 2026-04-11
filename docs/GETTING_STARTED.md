# Getting Started with CJC-Lang

CJC-Lang is a deterministic numerical programming language designed for reproducible computation, statistical computing, and machine learning pipelines.

## Installation

CJC-Lang is a single binary. Install from crates.io:

```bash
cargo install cjc-lang
```

Or build from source:

```bash
git clone https://github.com/your-org/cjc.git
cd cjc
cargo build --release
```

The binary will be at `target/release/cjcl` (or `target\release\cjcl.exe` on Windows).

## Your First Program

Create a file called `hello.cjcl`:

```
print("Hello, CJC-Lang!");

let x = 42;
let y = x * 2 + 1;
print("The answer is: " + to_string(y));
```

Run it:

```bash
cjcl run hello.cjcl
```

## The REPL

Start an interactive session:

```bash
cjcl repl
```

Type expressions and see results immediately:

```
cjcl> 2 + 3
5
cjcl> let name = "world";
cjcl> print("hello, " + name);
hello, world
```

## Running with MIR Execution

CJC-Lang has two execution backends. The MIR backend is faster and supports optimization:

```bash
cjcl run hello.cjcl --mir
cjcl run hello.cjcl --mir-opt   # with optimizations enabled
```

## Deterministic Execution

Every CJC-Lang program is deterministic by default. Supply a seed for reproducible random operations:

```bash
cjcl run simulation.cjcl --seed 42
```

Running the same program with the same seed always produces bit-identical output, regardless of platform or thread count.

## Multi-file Programs

CJC-Lang supports multi-file programs via the `--multi-file` flag. Use `import` declarations to pull in other modules:

```
// main.cjcl
import math.linalg

fn main() -> i64 {
    print(math::linalg::add(1, 2));
    0
}
```

Run with:

```bash
cjcl run --multi-file main.cjcl
```

Modules are resolved relative to the directory of the entry file. Cyclic imports are detected and reported.

## What Next?

- Read the [Tutorial](TUTORIAL.md) for a progressive 10-lesson introduction
- See [Syntax Reference](SYNTAX.md) for the full language grammar
- Explore the `examples/` directory for example programs

---

**Version:** v0.1.4 — CJC was renamed to CJC-Lang in this release. The CLI binary is `cjcl` (was `cjc`), and the source extension is `.cjcl` (was `.cjc`). Internal crate names remain `cjc-*`.
