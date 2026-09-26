# ADR-0044 Quantum Value Semantics — Circuits Are Values, Simulator States Are Handles

- **Status:** Accepted (2026-09-24)
- **Crates:** `cjc-quantum` (`dispatch.rs`), `cjc-types` (signatures)
- **Evidence:** `docs/quantum_simulation_research_stack/verification/copy_cost/` (benchmark), `copy_cost.out.txt`, `refcount_probe.cjcl`
- **Related:** [[ADR-0009 Vec COW Array]], [[ADR-0045 Quantum State Interop Between Builtins]], [[ADR-0046 Deterministic Elementary Functions (dmath)]]

## Context

Every quantum value is `Value::QuantumState(Rc<RefCell<dyn Any>>)`. Before
this decision, gate builtins borrowed the state mutably, changed it, and
returned `args[0].clone()`: the same `Rc`. So every name bound to a quantum
value shared one object:

```cjc
let a = qubits(2);
a = q_x(a, 0);
let b = a;
b = q_x(b, 1);        // also added the X to `a`
let c = with_h(a);    // a function that "returns a new circuit" changed a
```

Running exactly that program printed `q_n_gates(a) == 4`. It broke the most
common variational pattern, building variants from a shared base circuit, and
it contradicted the rest of the language, where arrays and tuples behave as
values (ADR-0009).

The audit recommended copy-on-write "matching `array_push`". Before deciding,
two facts were measured.

**1. Reference counts inside the builtin.** Both executors were instrumented
with a temporary probe (since removed), running `refcount_probe.cjcl`:

| Call | `Rc::strong_count` in `dispatch_quantum` (eval = MIR) |
|---|---|
| `a = q_x(a, 0)`, `a` not aliased | 2 (binding + argument vector) |
| `b = q_x(b, 1)` after `let b = a` | 3 |
| gate inside a function on a parameter | 5 |

Copy-on-write means "copy if `strong_count > 1`", so it copies on **every**
call, including the unaliased reassign. A threshold of "> 2" is unsound: an
array element passed as `xs[0]` also arrives with count 2, and would be
mutated inside the array.

**2. `array_push` is not the O(1) precedent it claims to be.** Its comment
says reassigning to the same name gives refcount 1 and a zero-copy push. A
timed loop shows quadratic behaviour: 10k pushes take 1.9 s, 20k take 10.3 s,
and 40k take 46.9 s (MIR executor). Its `Rc::make_mut` always copies.

**3. Cost of one copy relative to one gate** (`copy_cost`, release build):

| State | gate | deep copy | copy / gate |
|---|---|---|---|
| stabilizer n=1,000 | 7.6 µs | 456 µs | 60× |
| stabilizer n=10,000 | 368 µs | 26.6 ms | 72× |
| MPS n=100, χ=32 (bond-saturated) | 0.007 µs | 8.8 µs | ~1,250× |
| density n=10 | 3.6 ms | 3.8 ms | 1.0× |
| circuit, 1,000 gates (append vs copy gate list) | 0.09 µs | 1.7 µs | 18× |
| circuit, 10,000 gates | 0.09 µs | 39 µs | 440× |

## Decision

Split by what the value **is**, not by one rule for everything.

1. **Circuits are values.** A gate returns a new circuit: a copy of the gate
   list plus one gate. The argument is never modified. This applies to
   `qubits`/`q_*` on both backends. Circuits are descriptions: small,
   compositional, and reused across parameter sweeps. Copying the gate list
   is cheap next to executing the circuit, which costs O(2ⁿ · gates).
2. **Simulator states are mutable handles, and this is documented.** MPS,
   stabilizer, and density operations change the state in place and return
   the same handle, as before. This is kept because:
   - copying per gate would cost 60–1,250× for stabilizer and MPS, the two
     backends whose purpose is scale;
   - measurement (`stabilizer_measure`) *must* change the state: the collapse
     is the physics. Value semantics would need an API break that returns an
     `(outcome, state)` pair;
   - this matches the established simulators (Stim `TableauSimulator`, Qiskit
     Aer simulators, qsim), where the circuit is a value and the simulator is
     a stateful engine.
3. **`q_copy(value)`** is the one way to fork any quantum value. It works on
   circuits, statevectors, MPS, stabilizer, density, graphs, codes, and
   Hamiltonians, on both backends.
4. **Other descriptions follow the circuit rule.** `q_fermion_add_term`
   (ADR-0045) returns a new Hamiltonian.

## Alternatives rejected

- **Refcount copy-on-write for everything** (the audit's recommendation). In
  this runtime it means copying on every call (fact 1). That is a 60–1,250×
  regression on stabilizer and MPS workloads, for no semantic gain on states
  that measurement must mutate anyway.
- **Documented handles for everything, including circuits.** Honest, but it
  keeps the parameter-sweep bug as "documented behaviour". It also leaves
  circuits unlike arrays, the other value type users build up incrementally.
- **A persistent (shared-prefix) gate list, O(1) per gate.** Correct and
  faster for 10k+ gate circuits. But `Circuit` is a public type whose
  `gates() -> &[Gate]` slice is read by adjoint differentiation and is
  exposed to downstream crates, so it would change a public API for a cost
  that only shows past ~10k gates. It stays available if profiling asks for
  it.

## Consequences

- **Breaking, silently for one pattern:** a discarded gate result
  (`q_x(c, 0);` as a statement) is now a no-op. Neither the repo's `.cjcl`
  demos nor its tests contain that pattern: the grep found only two audit
  probes, which already error on missing arguments. A lint for "discarded
  result of a pure builtin" would catch it and is noted as a follow-up.
- Code that reassigns (`c = q_h(c, 0)`) gets bit-identical results. The
  11 tests in `test_quantum_value_semantics.rs` pin both halves of the
  decision in both executors.
- Building a circuit gate-by-gate is O(gates²) in copies: about 0.2 s total
  for 10,000 gates, invisible below ~1,000.
- **Root cause left open.** The real fix for both this cost and
  `array_push`'s quadratic behaviour is executor-level *move on last use*:
  when evaluating `x = f(x, …)`, pass `x`'s value by move so the builtin sees
  refcount 1. That touches both executors and every COW builtin, so it needs
  its own ADR.
