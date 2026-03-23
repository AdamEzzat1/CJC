# CJC Tutorial

A progressive 10-lesson introduction to the CJC language. Each lesson builds on the previous one. All examples are complete, runnable programs.

---

## Lesson 1: Variables, Types, and Printing

CJC has four primitive types: `i64`, `f64`, `bool`, and `String`. Variables are declared with `let` and are immutable by default. Use `let mut` for mutable bindings.

```
let x: i64 = 42;
let pi: f64 = 3.14159;
let active: bool = true;
let name: String = "CJC";

print("x = " + to_string(x));
print("pi = " + to_string(pi));
print("active = " + to_string(active));
print("Hello from " + name + "!");

let mut counter: i64 = 0;
counter = counter + 1;
print("counter = " + to_string(counter));
```

---

## Lesson 2: Functions and Closures

Functions are declared with `fn`. All parameters require type annotations. Closures use `|args| body` syntax and capture variables from their enclosing scope.

```
fn add(a: f64, b: f64) -> f64 {
    a + b
}

fn greet(name: String) -> String {
    "Hello, " + name + "!"
}

print(to_string(add(2.5, 3.7)));
print(greet("world"));

let factor = 10.0;
let scale = |x: f64| x * factor;
print("scaled: " + to_string(scale(4.2)));

fn apply(f: Any, x: f64) -> f64 {
    f(x)
}
print("applied: " + to_string(apply(scale, 5.0)));
```

---

## Lesson 3: Control Flow

CJC supports `if`/`else` (usable as expressions), `while` loops, `for` loops over ranges, `match` expressions, and `break`/`continue`.

```
let score = 85;

let grade = if score >= 90 {
    "A"
} else {
    if score >= 80 {
        "B"
    } else {
        "C"
    }
};
print("Grade: " + grade);

let mut sum = 0;
let mut i = 1;
while i <= 10 {
    sum = sum + i;
    i = i + 1;
}
print("Sum 1..10 = " + to_string(sum));

let mut total = 0;
for j in 0..5 {
    total = total + j;
}
print("Sum 0..4 = " + to_string(total));

let val = 2;
match val {
    1 => print("one"),
    2 => print("two"),
    3 => print("three"),
    _ => print("other"),
}
```

---

## Lesson 4: Arrays, Tensors, and Basic Math

Arrays are dynamically sized collections. Tensors are fixed-shape numerical arrays optimized for linear algebra. Use `array_push` to append (it returns a new array).

```
let mut arr = [10, 20, 30];
arr = array_push(arr, 40);
print("length: " + to_string(len(arr)));
print("first: " + to_string(arr[0]));

let a = tensor([[1.0, 2.0], [3.0, 4.0]]);
let b = tensor([[5.0, 6.0], [7.0, 8.0]]);
let c = matmul(a, b);
print("matmul result:");
print(to_string(c));

let t = tensor([1.0, 2.0, 3.0, 4.0, 5.0]);
print("sum: " + to_string(sum(t)));
print("mean: " + to_string(mean(t)));

let sq = tensor([4.0, 9.0, 16.0]);
print("sqrt: " + to_string(sqrt(sq)));
```

---

## Lesson 5: Statistics and Distributions

CJC provides built-in statistical functions and probability distributions with deterministic RNG.

```
let data = tensor([2.3, 4.1, 3.8, 5.2, 4.7, 3.1, 4.9]);

print("mean:   " + to_string(mean(data)));
print("sd:     " + to_string(sd(data)));
print("min:    " + to_string(min(data)));
print("max:    " + to_string(max(data)));
print("median: " + to_string(median(data)));

let x = 0.0;
print("normal_pdf(0): " + to_string(normal_pdf(x, 0.0, 1.0)));
print("normal_cdf(0): " + to_string(normal_cdf(x, 0.0, 1.0)));

let sample1 = tensor([5.1, 4.9, 5.3, 5.0, 4.8]);
let sample2 = tensor([4.2, 4.5, 4.1, 4.4, 4.3]);
let result = t_test(sample1, sample2);
print("t-test result: " + to_string(result));
```

---

## Lesson 6: DataFrames and Tidy Operations

CJC includes a tidyverse-inspired data DSL for tabular data manipulation.

```
let df = dataframe({
    "name":  ["Alice", "Bob", "Carol", "Dave"],
    "score": [92.0, 85.0, 91.0, 78.0],
    "group": ["A", "B", "A", "B"]
});

print("Full data:");
view(df);

let high = filter(df, |row: Any| row.score > 85.0);
print("High scorers:");
view(high);

let grouped = group_by(df, "group");
let summary = summarize(grouped, "score", "mean");
print("Mean by group:");
view(summary);
```

---

## Lesson 7: Linear Algebra

CJC provides dense and sparse linear algebra primitives with deterministic floating-point reductions.

```
let a = tensor([[4.0, 1.0], [1.0, 3.0]]);
let b = tensor([1.0, 2.0]);

let x = solve(a, b);
print("Solution Ax = b:");
print(to_string(x));

let m = tensor([[1.0, 2.0], [3.0, 4.0]]);
let svd_result = svd(m);
print("SVD singular values: " + to_string(svd_result.s));

let sym = tensor([[2.0, 1.0], [1.0, 3.0]]);
let eig = eigen(sym);
print("Eigenvalues: " + to_string(eig.values));

print("Transpose:");
print(to_string(m.transpose()));
```

---

## Lesson 8: Automatic Differentiation

CJC supports forward-mode (dual numbers) and reverse-mode (tape-based) automatic differentiation.

```
fn quadratic(x: f64) -> f64 {
    3.0 * x * x + 2.0 * x + 1.0
}

let x = 2.0;
let g = grad(quadratic, x);
print("f(2)  = " + to_string(quadratic(x)));
print("f'(2) = " + to_string(g));

fn rosenbrock(x: f64, y: f64) -> f64 {
    let a = 1.0 - x;
    let b = y - x * x;
    a * a + 100.0 * b * b
}

let gx = grad(|x: f64| rosenbrock(x, 1.0), 1.0);
print("d/dx rosenbrock(1,1) = " + to_string(gx));
```

---

## Lesson 9: Machine Learning

Build training loops with tensors, autodiff, and the Adam optimizer.

```
fn mse_loss(pred: Any, target: Any) -> f64 {
    let diff = pred - target;
    mean(diff * diff)
}

let x_train = tensor([[1.0], [2.0], [3.0], [4.0]]);
let y_train = tensor([2.0, 4.0, 6.0, 8.0]);

let mut w = tensor([0.1]);
let mut b = 0.0;
let lr = 0.01;

let mut epoch = 0;
while epoch < 100 {
    let pred = matmul(x_train, w) + b;
    let loss = mse_loss(pred, y_train);
    let grad_w = grad(|w: Any| mse_loss(matmul(x_train, w) + b, y_train), w);
    w = w - lr * grad_w;
    if epoch % 25 == 0 {
        print("epoch " + to_string(epoch) + " loss: " + to_string(loss));
    }
    epoch = epoch + 1;
}
print("Learned weight: " + to_string(w));
```

---

## Lesson 10: Determinism Guarantees and Snap Serialization

CJC guarantees bit-identical results across runs and platforms. The snap system lets you serialize and verify computation state.

```
fn simulate(seed: i64, n: i64) -> f64 {
    let mut total = 0.0;
    let mut i = 0;
    while i < n {
        let r = rand_f64();
        total = total + r;
        i = i + 1;
    }
    total / to_f64(n)
}

let result1 = simulate(42, 1000);
let result2 = simulate(42, 1000);
assert(result1 == result2, "Determinism violation!");
print("Both runs: " + to_string(result1));

let state = tensor([1.0, 2.0, 3.0]);
snap_save("checkpoint.snap", state);
let restored = snap_load("checkpoint.snap");
assert(state == restored, "Snap round-trip failed!");
print("Snap verified: state preserved exactly.");
```

**Key determinism guarantees:**
- Same seed produces bit-identical RNG sequences (SplitMix64)
- Floating-point reductions use Kahan summation (no accumulation drift)
- No `HashMap` iteration (BTreeMap everywhere)
- SIMD kernels avoid FMA for cross-platform consistency
- Parallel operations produce identical results regardless of thread count
