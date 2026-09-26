"""Spectrum of cjc-quantum's LiH matrix by particle number (JW: occupation = popcount of basis index)."""
import numpy as np
rows = [l.split() for l in open("cjc_hamiltonians.txt") if l.startswith("lih ")]
H = np.zeros((16, 16), dtype=complex)
for _, _, i, j, re, im in rows:
    H[int(i), int(j)] = float(re) + 1j * float(im)
N = np.array([bin(k).count("1") for k in range(16)])
coupling = max(abs(H[a, b]) for a in range(16) for b in range(16) if N[a] != N[b])
print(f"max |H_ij| between different particle numbers: {coupling:.2e}")
for n in range(5):
    idx = np.where(N == n)[0]
    print(f"N={n}: {np.round(np.sort(np.linalg.eigvalsh(H[np.ix_(idx, idx)])), 6)}")
