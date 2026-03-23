# Getting Started with CJC

CJC is a deterministic numerical programming language designed for reproducible computation, statistical computing, and machine learning pipelines.

## Installation

CJC is a single binary. Build from source:

```bash
git clone https://github.com/your-org/cjc.git
cd cjc
cargo build --release
```

The binary will be at `target/release/cjc` (or `target\release\cjc.exe` on Windows).

## Your First Program

Create a file called `hello.cjc`:

```
print("Hello, CJC!");

let x = 42;
let y = x * 2 + 1;
print("The answer is: " + to_string(y));
```

Run it:

```bash
cjc run hello.cjc
```

## The REPL

Start an interactive session:

```bash
cjc repl
```

Type expressions and see results immediately:

```
cjc> 2 + 3
5
cjc> let name = "world";
cjc> print("hello, " + name);
hello, world
```

## Running with MIR Execution

CJC has two execution backends. The MIR backend is faster and supports optimization:

```bash
cjc run hello.cjc --mir
cjc run hello.cjc --mir-opt   # with optimizations enabled
```

## Deterministic Execution

Every CJC program is deterministic by default. Supply a seed for reproducible random operations:

```bash
cjc run simulation.cjc --seed 42
```

Running the same program with the same seed always produces bit-identical output, regardless of platform or thread count.

## What Next?

- Read the [Tutorial](TUTORIAL.md) for a progressive 10-lesson introduction
- See [Syntax Reference](SYNTAX.md) for the full language grammar
- Explore the `demo/` directory for example programs
