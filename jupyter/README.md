# CJC Jupyter Kernel

A sidecar-based Jupyter kernel that delegates execution to the `cjc` binary.

## Architecture

```
Jupyter Notebook  ──(ZMQ)──>  cjc_kernel.py  ──(subprocess)──>  cjc run --format json
```

The kernel inherits from `ipykernel.kernelbase.Kernel` and calls `cjc run --format json`
as a subprocess for each cell execution. This preserves CJC's determinism guarantees
(same seed = bit-identical output) since no Rust runtime is embedded in Python.

## Installation

```bash
# 1. Build CJC (ensure cjc binary is on PATH or set CJC_HOME)
cargo build --release -p cjc-cli

# 2. Install Python dependencies
pip install ipykernel jupyter

# 3. Install the kernel
cd jupyter
python install.py            # for current user
python install.py --sys-prefix  # for virtualenv/conda

# 4. Launch Jupyter
jupyter notebook
# Select "CJC" kernel from the New menu
```

## Protocol

- **Input**: Cell code string
- **Process**: Written to temp `.cjc` file, executed via `cjc run --format json --seed <N>`
- **Output**: JSON `{"ok":true,"output":[...]}` or `{"ok":false,"error":"..."}`
- **Exit codes**: 0=success, 1=runtime, 2=parse, 3=type, 4=parity

## Magic Commands

| Command      | Description                                    |
|-------------|------------------------------------------------|
| `%seed N`   | Set the deterministic RNG seed                 |
| `%time`     | Show execution timing statistics               |
| `%latency`  | Benchmark sidecar round-trip overhead          |
| `%cjc_path` | Show the resolved path to the cjc binary       |

## Example Programs

### TFIM Ground State (`examples/tfim_ground_state.cjc`)
Transverse Field Ising Model for N=6 spins. Builds the full 64x64 Hamiltonian
via Kronecker products and sweeps the transverse field to find the quantum
phase transition near h/J ≈ 1.0.

### PINN Burgers Equation (`examples/pinn_burgers.cjc`)
Physics-Informed Neural Network solving the 1D viscous Burgers' equation.
2-layer MLP (2→16→16→1) with tanh activations, trained with physics-informed
loss (IC + BC + PDE residual via finite differences).
