"""
Python Autodiff Throughput Benchmark
=====================================
Equivalent to bench/ad_bench/main.rs.

Implements both forward-mode (Dual numbers) and reverse-mode (tape-based)
AD from scratch in pure Python — no PyTorch, no JAX, no autograd.

This is a fair comparison: both CJC and Python implement the same algorithms
with zero external library help.

Output: JSON for the runner to parse.
"""

import time
import math
import json
import random

random.seed(42)

# ── Forward-Mode AD (Dual Numbers) ───────────────────────────

class Dual:
    __slots__ = ('value', 'deriv')

    def __init__(self, value, deriv=0.0):
        self.value = value
        self.deriv = deriv

    @staticmethod
    def variable(v):
        return Dual(v, 1.0)

    @staticmethod
    def constant(v):
        return Dual(v, 0.0)

    def __add__(self, other):
        return Dual(self.value + other.value, self.deriv + other.deriv)

    def __sub__(self, other):
        return Dual(self.value - other.value, self.deriv - other.deriv)

    def __mul__(self, other):
        return Dual(
            self.value * other.value,
            self.value * other.deriv + self.deriv * other.value
        )

    def __truediv__(self, other):
        denom = other.value * other.value
        return Dual(
            self.value / other.value,
            (self.deriv * other.value - self.value * other.deriv) / denom
        )

    def __neg__(self):
        return Dual(-self.value, -self.deriv)

    def sin(self):
        return Dual(math.sin(self.value), self.deriv * math.cos(self.value))

    def cos(self):
        return Dual(math.cos(self.value), -self.deriv * math.sin(self.value))

    def exp(self):
        e = math.exp(self.value)
        return Dual(e, self.deriv * e)

    def ln(self):
        return Dual(math.log(self.value), self.deriv / self.value)

    def sqrt(self):
        s = math.sqrt(self.value)
        return Dual(s, self.deriv / (2.0 * s))


# ── Reverse-Mode AD (Tape-based) ─────────────────────────────

class TapeNode:
    __slots__ = ('value', 'grad', 'deps')
    def __init__(self, value, deps=None):
        self.value = value
        self.grad = 0.0
        self.deps = deps or []  # list of (node, local_gradient)

class Tape:
    def __init__(self):
        self.nodes = []

    def parameter(self, value):
        node = TapeNode(value)
        self.nodes.append(node)
        return node

    def add(self, a, b):
        node = TapeNode(a.value + b.value, [(a, 1.0), (b, 1.0)])
        self.nodes.append(node)
        return node

    def sub(self, a, b):
        node = TapeNode(a.value - b.value, [(a, 1.0), (b, -1.0)])
        self.nodes.append(node)
        return node

    def mul(self, a, b):
        node = TapeNode(a.value * b.value, [(a, b.value), (b, a.value)])
        self.nodes.append(node)
        return node

    def div(self, a, b):
        node = TapeNode(a.value / b.value, [
            (a, 1.0 / b.value),
            (b, -a.value / (b.value * b.value))
        ])
        self.nodes.append(node)
        return node

    def backward(self, loss_node):
        loss_node.grad = 1.0
        for node in reversed(self.nodes):
            if node.grad == 0.0:
                continue
            for (dep, local_grad) in node.deps:
                dep.grad += node.grad * local_grad


# ── Benchmark: Many-to-One ───────────────────────────────────

def bench_forward_many_to_one(params):
    """Forward mode: f(x) = sum(x_i^2), compute all N gradients."""
    n = len(params)
    start = time.perf_counter()

    value = sum(x * x for x in params)

    grads = []
    for i in range(n):
        result = Dual(0.0, 0.0)
        for j in range(n):
            xj = Dual.variable(params[j]) if j == i else Dual.constant(params[j])
            result = result + xj * Dual(params[j], 1.0 if j == i else 0.0)
        grads.append(result.deriv)

    elapsed = time.perf_counter() - start
    return value, grads, elapsed


def bench_reverse_many_to_one(params):
    """Reverse mode: f(x) = sum(x_i^2), single backward pass."""
    n = len(params)
    start = time.perf_counter()

    tape = Tape()
    param_nodes = [tape.parameter(x) for x in params]

    # Compute x_i^2 for each param, then sum via tree reduction
    sq_nodes = [tape.mul(p, p) for p in param_nodes]

    current = sq_nodes
    while len(current) > 1:
        nxt = []
        for k in range(0, len(current), 2):
            if k + 1 < len(current):
                nxt.append(tape.add(current[k], current[k + 1]))
            else:
                nxt.append(current[k])
        current = nxt

    loss = current[0]
    value = loss.value

    tape.backward(loss)

    grads = [p.grad for p in param_nodes]

    elapsed = time.perf_counter() - start
    return value, grads, elapsed


# ── Benchmark: One-to-Many ───────────────────────────────────

def bench_forward_one_to_many(t_val, num_outputs):
    """Forward mode: 1 input -> N outputs, single pass."""
    start = time.perf_counter()
    t = Dual.variable(t_val)

    values = []
    derivs = []
    for i in range(num_outputs):
        mod = i % 5
        if mod == 0:
            r = t.sin()
        elif mod == 1:
            r = t.cos()
        elif mod == 2:
            r = t.exp()
        elif mod == 3:
            r = t * t
        else:
            r = (t * Dual.constant(0.5)).ln()
        values.append(r.value)
        derivs.append(r.deriv)

    elapsed = time.perf_counter() - start
    return values, derivs, elapsed


def bench_reverse_one_to_many(t_val, num_outputs):
    """Reverse mode: N backward passes (one per output)."""
    start = time.perf_counter()

    values = []
    derivs = []
    for i in range(num_outputs):
        tape = Tape()
        t = tape.parameter(t_val)

        mod = i % 5
        if mod == 0:
            out = tape.mul(t, t)  # x^2
        elif mod == 1:
            t2 = tape.mul(t, t)
            out = tape.mul(t2, t)  # x^3
        elif mod == 2:
            t2 = tape.mul(t, t)
            out = tape.mul(t2, t2)  # x^4
        elif mod == 3:
            out = tape.mul(t, t)  # x^2
        else:
            t2 = tape.mul(t, t)
            t3 = tape.mul(t2, t)
            out = tape.mul(t3, t2)  # x^5

        values.append(out.value)
        tape.backward(out)
        derivs.append(t.grad)

    elapsed = time.perf_counter() - start
    return values, derivs, elapsed


# ── Run benchmarks ───────────────────────────────────────────

results = {"many_to_one": [], "one_to_many": []}

# Part 1: Many-to-One
sizes_m2o = [10, 50, 100, 500, 1000, 2000, 5000, 10000]

for n in sizes_m2o:
    params = [random.gauss(0, 1) for _ in range(n)]

    if n <= 2000:
        fwd_val, fwd_grads, fwd_time = bench_forward_many_to_one(params)
    else:
        fwd_val, fwd_grads, fwd_time = 0.0, [], 0.0

    rev_val, rev_grads, rev_time = bench_reverse_many_to_one(params)

    # Verify correctness
    correct = True
    for i in range(n):
        expected = 2.0 * params[i]
        if abs(rev_grads[i] - expected) > 1e-6:
            correct = False
            break
    if fwd_grads:
        for i in range(n):
            expected = 2.0 * params[i]
            if abs(fwd_grads[i] - expected) > 1e-6:
                correct = False
                break

    results["many_to_one"].append({
        "n": n,
        "fwd_us": fwd_time * 1e6,
        "rev_us": rev_time * 1e6,
        "fwd_gps": n / fwd_time if fwd_time > 0 else 0,
        "rev_gps": n / rev_time if rev_time > 0 else 0,
        "correct": correct,
    })

# Part 2: One-to-Many
sizes_o2m = [10, 50, 100, 500, 1000, 2000, 5000]

for n in sizes_o2m:
    t_val = 1.5
    _, _, fwd_time = bench_forward_one_to_many(t_val, n)

    if n <= 2000:
        _, _, rev_time = bench_reverse_one_to_many(t_val, n)
    else:
        rev_time = 0.0

    results["one_to_many"].append({
        "n": n,
        "fwd_us": fwd_time * 1e6,
        "rev_us": rev_time * 1e6,
        "fwd_gps": n / fwd_time if fwd_time > 0 else 0,
        "rev_gps": n / rev_time if rev_time > 0 else 0,
    })

print(json.dumps(results))
