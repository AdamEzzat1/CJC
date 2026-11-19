"""Independent check of cjc-quantum's H2 Hamiltonian (fermion.rs::h2_hamiltonian).

Dependency-light: numpy + scipy only (PySCF/OpenFermion were not installed; see
VERIFY_FOLLOWUPS.md). Computes STO-3G integrals for H2 from closed-form
s-Gaussian formulas (Szabo & Ostlund, "Modern Quantum Chemistry", App. A),
does RHF (symmetry-determined MOs), builds the 4-spin-orbital second-quantized
Hamiltonian, maps it with Jordan-Wigner to a 16x16 matrix, and diagonalizes.

Then compares against the 2-qubit matrix shipped in cjc-quantum.

Run: python h2_sto3g_check.py
"""
import itertools
import numpy as np
from scipy.special import erf

BOHR_PER_ANGSTROM = 1.0 / 0.529177210903
R_ANG = 0.7414
R = R_ANG * BOHR_PER_ANGSTROM

# STO-3G for H (zeta = 1.24 already folded into exponents)
ALPHA = np.array([3.42525091, 0.62391373, 0.16885540])
COEF = np.array([0.15432897, 0.53532814, 0.44463454])
CENTERS = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, R])]


def norm(a):
    return (2.0 * a / np.pi) ** 0.75


def f0(t):
    return 1.0 if t < 1e-12 else 0.5 * np.sqrt(np.pi / t) * erf(np.sqrt(t))


def prim_S(a, A, b, B):
    p = a + b
    return (np.pi / p) ** 1.5 * np.exp(-a * b / p * np.dot(A - B, A - B))


def prim_T(a, A, b, B):
    p = a + b
    r2 = np.dot(A - B, A - B)
    return a * b / p * (3.0 - 2.0 * a * b / p * r2) * prim_S(a, A, b, B)


def prim_V(a, A, b, B, C, Z=1.0):
    p = a + b
    P = (a * A + b * B) / p
    return (-2.0 * np.pi / p * Z * np.exp(-a * b / p * np.dot(A - B, A - B))
            * f0(p * np.dot(P - C, P - C)))


def prim_ERI(a, A, b, B, c, C, d, D):
    p, q = a + b, c + d
    P, Q = (a * A + b * B) / p, (c * C + d * D) / q
    pref = 2.0 * np.pi ** 2.5 / (p * q * np.sqrt(p + q))
    return (pref * np.exp(-a * b / p * np.dot(A - B, A - B) - c * d / q * np.dot(C - D, C - D))
            * f0(p * q / (p + q) * np.dot(P - Q, P - Q)))


def contract(fn, idx):
    """Contract primitive integral fn over the contracted basis functions idx."""
    tot = 0.0
    for prims in itertools.product(range(3), repeat=len(idx)):
        w = 1.0
        args = []
        for mu, k in zip(idx, prims):
            w *= COEF[k] * norm(ALPHA[k])
            args += [ALPHA[k], CENTERS[mu]]
        tot += w * fn(*args)
    return tot


n = 2
S = np.array([[contract(prim_S, (i, j)) for j in range(n)] for i in range(n)])
T = np.array([[contract(prim_T, (i, j)) for j in range(n)] for i in range(n)])
V = np.zeros((n, n))
for C in CENTERS:
    V += np.array([[contract(lambda a, A, b, B: prim_V(a, A, b, B, C), (i, j))
                    for j in range(n)] for i in range(n)])
Hcore = T + V
eri = np.zeros((n, n, n, n))  # chemists' notation (ij|kl)
for i, j, k, l in itertools.product(range(n), repeat=4):
    eri[i, j, k, l] = contract(prim_ERI, (i, j, k, l))

# RHF MOs for homonuclear H2 are fixed by symmetry
s12 = S[0, 1]
Cmo = np.array([[1, 1], [1, -1]], dtype=float)
Cmo[:, 0] /= np.sqrt(2 * (1 + s12))
Cmo[:, 1] /= np.sqrt(2 * (1 - s12))
h_mo = Cmo.T @ Hcore @ Cmo
eri_mo = np.einsum("pi,qj,rk,sl,pqrs->ijkl", Cmo, Cmo, Cmo, Cmo, eri)
e_nuc = 1.0 / R

# Spin-orbital integrals; ordering: 0=g↑ 1=g↓ 2=u↑ 3=u↓
nso = 4
spat = lambda p: p // 2
spin = lambda p: p % 2
h_so = np.zeros((nso, nso))
g_so = np.zeros((nso,) * 4)  # physicists' <pq|rs>
for p, q in itertools.product(range(nso), repeat=2):
    if spin(p) == spin(q):
        h_so[p, q] = h_mo[spat(p), spat(q)]
for p, q, r, s in itertools.product(range(nso), repeat=4):
    if spin(p) == spin(r) and spin(q) == spin(s):
        g_so[p, q, r, s] = eri_mo[spat(p), spat(r), spat(q), spat(s)]

# Jordan-Wigner annihilation operators as 16x16 matrices
I2 = np.eye(2)
Zm = np.diag([1.0, -1.0])
a1 = np.array([[0.0, 1.0], [0.0, 0.0]])  # |0><1|: annihilate occupied (1) -> 0


def kron_all(ms):
    out = np.array([[1.0]])
    for m in ms:
        out = np.kron(out, m)
    return out


ann = [kron_all([Zm] * p + [a1] + [I2] * (nso - p - 1)) for p in range(nso)]
cre = [a.T for a in ann]
H = np.zeros((16, 16))
for p, q in itertools.product(range(nso), repeat=2):
    H += h_so[p, q] * cre[p] @ ann[q]
for p, q, r, s in itertools.product(range(nso), repeat=4):
    if g_so[p, q, r, s] != 0.0:
        H += 0.5 * g_so[p, q, r, s] * cre[p] @ cre[q] @ ann[s] @ ann[r]

Nop = sum(c @ a for c, a in zip(cre, ann))
Szop = 0.5 * sum((1 if spin(p) == 0 else -1) * cre[p] @ ann[p] for p in range(nso))
occ = np.round(np.diag(Nop)).astype(int)
sz = np.round(2 * np.diag(Szop)).astype(int)
sector = [i for i in range(16) if occ[i] == 2 and sz[i] == 0]
Hsec = H[np.ix_(sector, sector)]
e_sec = np.sort(np.linalg.eigvalsh(Hsec))

# cjc-quantum's shipped matrix (fermion.rs:373-420)
g = dict(g0=-0.4804, g1=0.3435, g2=-0.4347, g3=0.5716, g4=0.0910, g5=0.0910)
X = np.array([[0, 1], [1, 0]]); Y = np.array([[0, -1j], [1j, 0]]); Z2 = np.diag([1, -1])
Hc = (g["g0"] * np.eye(4) + g["g1"] * np.kron(I2, Z2) + g["g2"] * np.kron(Z2, I2)
      + g["g3"] * np.kron(Z2, Z2) + g["g4"] * np.kron(X, X) + g["g5"] * np.kron(Y, Y))
e_cjc = np.sort(np.linalg.eigvalsh(Hc).real)

print(f"R = {R_ANG} Å = {R:.6f} bohr")
print(f"S12 = {s12:.6f}")
print(f"h_mo diag = {np.diag(h_mo)}")
print(f"(gg|gg)={eri_mo[0,0,0,0]:.6f} (uu|uu)={eri_mo[1,1,1,1]:.6f} "
      f"(gg|uu)={eri_mo[0,0,1,1]:.6f} (gu|gu)={eri_mo[0,1,0,1]:.6f}")
print(f"E_nuc = 1/R = {e_nuc:.6f} Ha")
e_hf = 2 * h_mo[0, 0] + eri_mo[0, 0, 0, 0]
print(f"E_HF electronic = {e_hf:.6f}  total = {e_hf + e_nuc:.6f}")
print(f"FCI electronic ground (N=2, Sz=0) = {e_sec[0]:.6f}  total = {e_sec[0] + e_nuc:.6f}")
print(f"N=2, Sz=0 electronic spectrum = {np.round(e_sec, 6)}")
print(f"CJC 2-qubit matrix spectrum    = {np.round(e_cjc, 6)}")
print(f"CJC min + E_nuc = {e_cjc[0] + e_nuc:.6f}")
print(f"|CJC min - FCI electronic| = {abs(e_cjc[0] - e_sec[0]):.2e} Ha")

# Which physical states do CJC's 4 eigenvalues correspond to?
print("\nFull Fock-space spectrum by (N, 2Sz):")
parity_even = []
for N in range(5):
    for s2 in sorted(set(sz[occ == N])):
        idx = [i for i in range(16) if occ[i] == N and sz[i] == s2]
        ev = np.sort(np.linalg.eigvalsh(H[np.ix_(idx, idx)]))
        print(f"  N={N} 2Sz={s2:+d}: {np.round(ev, 6)}")
for e in e_cjc:
    allev = []
    for N in range(5):
        for s2 in sorted(set(sz[occ == N])):
            idx = [i for i in range(16) if occ[i] == N and sz[i] == s2]
            for v in np.linalg.eigvalsh(H[np.ix_(idx, idx)]):
                allev.append((abs(v - e), N, s2, v))
    d, N, s2, v = min(allev)
    print(f"CJC eigenvalue {e:+.6f} -> nearest physical state N={N} 2Sz={s2:+d} E={v:+.6f} (diff {d:.1e})")
