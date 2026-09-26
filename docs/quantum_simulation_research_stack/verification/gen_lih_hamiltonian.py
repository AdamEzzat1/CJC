"""Generate cjc-quantum's 4-qubit LiH Hamiltonian from first principles.

Model: LiH, STO-3G, R = 1.546 Å, RHF orbitals, frozen Li 1s core, active space
of 2 electrons in 2 spatial orbitals (HOMO, LUMO) → 4 spin orbitals → 4 qubits
under Jordan–Wigner (qubit k = spin orbital k, interleaved alpha/beta:
0 = HOMO↑, 1 = HOMO↓, 2 = LUMO↑, 3 = LUMO↓). The constant term includes the
nuclear repulsion and the frozen-core energy, so eigenvalues are total energies.

Run (Linux; PySCF has no Windows wheels):
    pip install pyscf openfermion && python gen_lih_hamiltonian.py
Output: a Rust table for fermion.rs::lih_hamiltonian() and reference energies.
"""
import numpy as np, pyscf, openfermion as of
from pyscf import gto, scf, fci, ao2mo
from openfermion.chem.molecular_data import spinorb_from_spatial

mol = gto.M(atom="Li 0 0 0; H 0 0 1.546", basis="sto-3g", unit="Angstrom")
mf = scf.RHF(mol).run(verbose=0)
e_fci_full = fci.FCI(mf).kernel()[0]

ncas, nelecas = 2, 2
ncore = (mol.nelectron - nelecas) // 2
C = mf.mo_coeff
core, act = C[:, :ncore], C[:, ncore:ncore + ncas]
hcore = mf.get_hcore()
dm_core = 2 * core @ core.T
vj, vk = mf.get_jk(mol, dm_core)
e_core = mol.energy_nuc() + np.einsum("ij,ji", dm_core, hcore + 0.5 * (vj - 0.5 * vk))
h1 = act.T @ (hcore + vj - 0.5 * vk) @ act
eri = ao2mo.restore(1, ao2mo.kernel(mol, act), ncas)
two = np.asarray(eri.transpose(0, 2, 3, 1), order="C")  # openfermionpyscf convention
ob, tb = spinorb_from_spatial(h1, two)
iop = of.InteractionOperator(e_core, ob, 0.5 * tb)
qop = of.jordan_wigner(iop)
qop.compress(1e-14)

terms = []
for term, coeff in qop.terms.items():
    assert abs(coeff.imag) < 1e-12, (term, coeff)
    s = ["I"] * 4
    for q, p in term:
        s[q] = p
    terms.append(("".join(s), float(coeff.real)))
terms.sort()

H = of.get_sparse_operator(qop, n_qubits=4).toarray()
num = of.get_sparse_operator(of.number_operator(4), n_qubits=4).diagonal().real
idx2 = [k for k in range(16) if round(num[k]) == 2]
e_cas = np.sort(np.linalg.eigvalsh(H[np.ix_(idx2, idx2)]))[0]
# HF determinant: spin orbitals 0 and 1 occupied
hf = of.jw_configuration_state([0, 1], 4)
e_hf = float(np.real(hf.conj() @ H @ hf))

print(f"// pyscf {pyscf.__version__}, openfermion {of.__version__}")
print(f"// E_nuc = {mol.energy_nuc():.10f}  E_core(const incl. E_nuc) = {e_core:.10f}")
print(f"// E_RHF = {mf.e_tot:.10f}  <HF|H|HF> = {e_hf:.10f}")
print(f"// CASCI(2,2) ground (N=2) = {e_cas:.10f}  full-space FCI = {e_fci_full:.10f}")
print(f"// {len(terms)} Pauli terms (qubit 0 first)")
print("const LIH_TERMS: [(&str, f64); %d] = [" % len(terms))
for s, c in terms:
    print(f'    ("{s}", {c!r}),')
print("];")
