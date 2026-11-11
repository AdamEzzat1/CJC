"""Exact-diagonalisation oracle for tests/test_dmrg_exact.rs and the
cjc-quantum DMRG exact-value tests.

Hamiltonians (open boundary, n sites), matching crates/cjc-quantum/src/dmrg.rs
`local_hamiltonian` / `hamiltonian_ops`:

  Heisenberg: H = sum_{i=0}^{n-2} (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
  Ising:      H = sum_{i=0}^{n-2} Z_i Z_{i+1}      (no transverse field)

Ground energies come from dense numpy.linalg.eigvalsh (needs numpy only).
Heisenberg for n >= 12 uses the S^z_total = 0 sector (the ground state of the
open AFM chain with an even number of sites lives there); for n <= 10 the full
dense spectrum is also computed and checked against the sector value.

Run: python tests/refs/dmrg_ed.py
"""
import itertools
import numpy as np

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0, -1.0]).astype(complex)


def op(o, i, n):
    m = np.array([[1.0]], dtype=complex)
    for k in range(n):
        m = np.kron(m, o if k == i else np.eye(2))
    return m


def dense(n, paulis):
    return sum(op(P, i, n) @ op(P, i + 1, n) for i in range(n - 1) for P in paulis)


def heis_sector(n):
    # basis states with n/2 up spins; XX+YY = 2(S+S- + S-S+) -> hops with amplitude 2
    states = [s for s in range(1 << n) if bin(s).count("1") == n // 2]
    idx = {s: k for k, s in enumerate(states)}
    H = np.zeros((len(states), len(states)))
    for s in states:
        k = idx[s]
        for i in range(n - 1):
            bi = (s >> i) & 1
            bj = (s >> (i + 1)) & 1
            H[k, k] += 1.0 if bi == bj else -1.0
            if bi != bj:
                t = s ^ (1 << i) ^ (1 << (i + 1))
                H[idx[t], k] += 2.0
    return float(np.linalg.eigvalsh(H).min())


for n in (2, 4, 6, 8, 10, 12):
    if n <= 10:
        eh = float(np.linalg.eigvalsh(dense(n, (X, Y, Z))).min())
        assert abs(eh - heis_sector(n)) < 1e-11, (n, eh, heis_sector(n))
    else:
        eh = heis_sector(n)
    # Ising ZZ is diagonal: ground energy is exactly -(n-1) (Neel state).
    ei = float(np.diag(dense(n, (Z,)).real).min()) if n <= 10 else -(n - 1.0)
    print(f"n={n:2d} heisenberg={eh!r} ising={ei!r}")
