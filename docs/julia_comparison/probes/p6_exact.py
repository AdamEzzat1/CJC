"""Exact ground-state energy of H = sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1}),
open boundary, n sites -- the Hamiltonian in crates/cjc-quantum/src/dmrg.rs:77-85.
Reference for probe p6_dmrg_heisenberg.cjcl. Needs numpy only."""
import numpy as np
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0, -1.0]).astype(complex)
def op(o, i, n):
    m = np.array([[1.0]], dtype=complex)
    for k in range(n):
        m = np.kron(m, o if k == i else np.eye(2))
    return m
for n in (4, 6, 8):
    H = sum(op(P, i, n) @ op(P, i + 1, n) for i in range(n - 1) for P in (X, Y, Z))
    print(n, repr(float(np.linalg.eigvalsh(H).min())))
