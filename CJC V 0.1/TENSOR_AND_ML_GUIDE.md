# Tensors, Machine Learning, and Automatic Differentiation

## Tensor Fundamentals

Tensors are CJC's core numerical type — n-dimensional arrays with
copy-on-write storage for efficient memory sharing.

### Creating Tensors

```
// From data
let v = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2]);

// Factory constructors
let zeros = Tensor.zeros([3, 3]);
let ones = Tensor.ones([4, 4]);
let eye = Tensor.eye(3);                      // 3x3 identity
let full = Tensor.full([2, 3], 7.0);          // 2x3 filled with 7.0

// Ranges
let lin = Tensor.linspace(0.0, 1.0, 11);      // 11 points from 0 to 1
let rng = Tensor.arange(0.0, 10.0, 0.5);      // 0, 0.5, 1, ... 9.5

// Random (seeded by --seed flag)
let normal = Tensor.randn([100, 50]);          // standard normal
let uniform = Tensor.uniform([100, 50]);       // uniform [0, 1)

// Tensor literal syntax
let mat = [| 1.0, 2.0; 3.0, 4.0 |];          // 2x2 matrix
```

### Shape Operations

```
let t = Tensor.randn([2, 3, 4]);
print(Tensor.shape(t));                // [2, 3, 4]
print(Tensor.len(t));                  // 24

let flat = Tensor.reshape(t, [6, 4]);  // reshape to 6x4
let tr = Tensor.transpose(t);         // general transpose
let bc = Tensor.broadcast_to(Tensor.ones([1, 4]), [3, 4]);
```

### Element Access

```
let m = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2]);
let val = Tensor.get(m, [0, 1]);       // 2.0
let m2 = Tensor.set(m, [0, 1], 99.0); // COW: m unchanged, m2 updated
```

### Arithmetic

```
let a = Tensor.ones([3, 3]);
let b = Tensor.full([3, 3], 2.0);

let c = Tensor.add(a, b);             // element-wise addition
let d = Tensor.mul(a, b);             // element-wise multiplication
let e = Tensor.scalar_mul(a, 5.0);    // scalar multiplication
let f = Tensor.neg(a);                // negation
let s = Tensor.sum_axis(c, 0);        // sum along axis 0
```

### Reductions

```
let data = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0], [5]);
print(sum(data));                      // 15 (Kahan-stable)
print(mean(data));                     // 3
print(argmax(data));                   // 4
print(argmin(data));                   // 0
```

## Matrix Operations

### Matrix Multiplication

```
let A = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
let B = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]);
let C = matmul(A, B);                 // 2x2 result
```

### Batched Matrix Multiply

```
// Batch of 4 matrix multiplies: [4, 3, 5] x [4, 5, 2] -> [4, 3, 2]
let Q = Tensor.randn([4, 3, 5]);
let K = Tensor.randn([4, 5, 2]);
let out = Tensor.bmm(Q, K);
```

### Linear Layer

```
// y = x @ W^T + b
let x = Tensor.randn([8, 64]);        // batch=8, features=64
let W = Tensor.randn([32, 64]);       // out=32, in=64
let b = Tensor.zeros([32]);
let y = Tensor.linear(x, W, b);       // [8, 32]
```

## Linear Algebra

```
let A = Tensor.from_vec([4.0, 7.0, 2.0, 6.0], [2, 2]);

// Decompositions
let lu_result = linalg.lu(A);
let qr_result = linalg.qr(A);
let chol = linalg.cholesky(A);        // requires positive definite

// Properties
print(det(A));                         // determinant
print(trace(A));                       // sum of diagonal
print(matrix_rank(A));                 // rank
print(cond(A));                        // condition number

// Solve Ax = b
let b = Tensor.from_vec([1.0, 2.0], [2]);
let x = solve(A, b);

// Inverse
let A_inv = linalg.inv(A);

// Eigenvalues (symmetric)
let sym = Tensor.from_vec([2.0, 1.0, 1.0, 3.0], [2, 2]);
let eigvals = eigh(sym);
```

## Neural Network Building Blocks

### Activations

```
let x = Tensor.randn([8, 64]);

let r = Tensor.relu(x);               // max(0, x)
let s = Tensor.sigmoid(x);            // 1 / (1 + exp(-x))
let g = Tensor.gelu(x);               // GELU activation
let t = Tensor.tanh_activation(x);    // tanh
let p = Tensor.softmax(x);            // softmax over last dim
```

### Layer Normalization

```
let x = Tensor.randn([8, 64]);
let gamma = Tensor.ones([64]);
let beta = Tensor.zeros([64]);
let normalized = Tensor.layer_norm(x, gamma, beta, 1e-5);
```

### Multi-Head Attention

```
let Q = Tensor.randn([8, 64]);        // [seq_len, model_dim]
let K = Tensor.randn([8, 64]);
let V = Tensor.randn([8, 64]);

// Split into heads
let num_heads: i64 = 4;
let Q_h = Tensor.split_heads(Q, num_heads);  // [seq, heads, head_dim]
let K_h = Tensor.split_heads(K, num_heads);
let V_h = Tensor.split_heads(V, num_heads);

// Compute attention
let attn = attention(Q_h, K_h, V_h, Tensor.zeros([1]));

// Merge heads back
let out = Tensor.merge_heads(attn);           // [seq, model_dim]
```

### Convolution and Pooling

```
// 1D convolution
let signal = Tensor.randn([1, 16, 100]);  // [batch, channels, length]
let kernel = Tensor.randn([32, 16, 3]);   // [out_ch, in_ch, kernel_size]
let conv1 = Tensor.conv1d(signal, kernel, 1, 1);

// 2D convolution
let image = Tensor.randn([1, 3, 28, 28]); // [batch, channels, H, W]
let filter = Tensor.randn([16, 3, 3, 3]); // [out_ch, in_ch, kH, kW]
let conv2 = Tensor.conv2d(image, filter, 1, 1);

// Max pooling
let pooled = Tensor.maxpool2d(conv2, 2, 2); // kernel=2, stride=2
```

## Automatic Differentiation

### Forward-Mode AD (Dual Numbers)

```
// Compute f(x) = x^2 + 3x at x = 2
// f'(x) = 2x + 3, so f'(2) = 7
let x = Dual.new(2.0, 1.0);           // value=2, derivative seed=1
// Use grad() to compute derivatives
let df = grad(fn(x: f64) -> f64 { x * x + 3.0 * x }, 0);
```

### Reverse-Mode AD (GradGraph)

```
let g = GradGraph.new();

// Create parameters
let w = GradGraph.parameter(g, 3.0);
let x = GradGraph.input(g, 2.0);

// Build computation: y = w * x + 1
let wx = GradGraph.mul(g, w, x);
let one = GradGraph.input(g, 1.0);
let y = GradGraph.add(g, wx, one);

// Backward pass
GradGraph.backward(g);

// Read gradients
print(GradGraph.value(y));             // 7.0
print(GradGraph.grad(w));             // dy/dw = x = 2.0
```

### GradGraph Operations

Supports: add, sub, mul, div, neg, matmul, sum, mean, relu, sigmoid,
tanh, sin, cos, sqrt, pow, exp, ln, scalar_mul.

## Loss Functions

```
let predictions = Tensor.from_vec([0.9, 0.1, 0.8], [3]);
let targets = Tensor.from_vec([1.0, 0.0, 1.0], [3]);

let mse = mse_loss(predictions, targets);
let bce = binary_cross_entropy(predictions, targets);

// For multi-class
let logits = Tensor.randn([4, 10]);     // 4 samples, 10 classes
let labels = [3, 7, 1, 5];
let ce = cross_entropy_loss(logits, labels);
```

## Optimizers

```
// Adam optimizer
let optimizer = Adam.new(0.001, 0.9, 0.999, 1e-8);

// SGD optimizer
let sgd = Sgd.new(0.01);

// Update step
let updated = OptimizerState.step(optimizer, params, grads);
```

## Learning Rate Schedules

```
let initial_lr: f64 = 0.01;
let step: i64 = 100;

// Step decay
let lr1 = lr_step_decay(initial_lr, step, 0.1, 30);

// Cosine annealing
let lr2 = lr_cosine(initial_lr, step, 1000);

// Linear warmup
let lr3 = lr_linear_warmup(initial_lr, step, 500);
```

## Gradient Utilities

```
// Clip gradients to prevent explosion
let clipped = clip_grad(grad_tensor, 1.0);

// Scale gradients
let scaled = grad_scale(grad_tensor, 0.5);

// Stop gradient (mark as non-differentiable)
let detached = stop_gradient(tensor);
```

## Complex Numbers

```
let z1 = Complex(3.0, 4.0);           // 3 + 4i
let z2 = Complex(1.0, 2.0);

let sum = Complex.add(z1, z2);        // 4 + 6i
let prod = Complex.mul(z1, z2);       // -5 + 10i
print(Complex.abs(z1));               // 5.0 (magnitude)
print(Complex.conj(z1));              // 3 - 4i
```

## FFT and Signal Processing

```
let signal = Tensor.from_vec([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [8]);

// Real FFT
let spectrum = rfft(signal);

// Power spectral density
let power = psd(signal);

// Window functions
let w_hann = hann(256);
let w_hamming = hamming(256);
let w_blackman = blackman(256);
```

## Half-Precision Types

```
// f16 (IEEE 754 half-precision)
let h = f64_to_f16(3.14);
let back = f16_to_f64(h);

// bf16 (brain float)
let bf = f32_to_bf16(3.14);
let back32 = bf16_to_f32(bf);
```
