"""Reference check of cjc-quantum's H2 and LiH Hamiltonians with PySCF + OpenFermion.

Runs in Linux (PySCF has no Windows wheels). Reads cjc_hamiltonians.txt, which
ham_dump/ extracts from CJC's own FermionicHamiltonian::expectation.
"""
import numpy as np, pyscf, openfermion as of
from pyscf import gto, scf, fci, ao2mo
from openfermion.chem.molecular_data import spinorb_from_spatial

print(f"pyscf {pyscf.__version__}  openfermion {of.__version__}  numpy {np.__version__}")

def load(name):
    rows = [l.split() for l in open("cjc_hamiltonians.txt") if l.startswith(name + " ")]
    n = int(rows[0][1]); d = 2 ** n
    H = np.zeros((d, d), dtype=complex)
    for _, _, i, j, re, im in rows:
        H[int(i), int(j)] = float(re) + 1j * float(im)
    herm = np.max(np.abs(H - H.conj().T))
    return n, H, herm

def reference(atom, label, ncas=None, nelecas=None):
    mol = gto.M(atom=atom, basis="sto-3g", unit="Angstrom")
    mf = scf.RHF(mol).run(verbose=0)
    e_fci = fci.FCI(mf).kernel()[0]
    C = mf.mo_coeff
    if ncas is not None:  # frozen-core active space
        ncore = (mol.nelectron - nelecas) // 2
        core = C[:, :ncore]; act = C[:, ncore:ncore + ncas]
        hcore = mf.get_hcore()
        dm_core = 2 * core @ core.T
        vj, vk = mf.get_jk(mol, dm_core)
        e_core = mol.energy_nuc() + np.einsum("ij,ji", dm_core, hcore + 0.5 * (vj - 0.5 * vk))
        h1 = act.T @ (hcore + vj - 0.5 * vk) @ act
        eri = ao2mo.restore(1, ao2mo.kernel(mol, act), ncas)
        const, nel = e_core, nelecas
    else:
        h1 = C.T @ mf.get_hcore() @ C
        eri = ao2mo.restore(1, ao2mo.kernel(mol, C), C.shape[1])
        const, nel = mol.energy_nuc(), mol.nelectron
    two = np.asarray(eri.transpose(0, 2, 3, 1), order="C")  # openfermionpyscf convention
    ob, tb = spinorb_from_spatial(h1, two)
    iop = of.InteractionOperator(0.0, ob, 0.5 * tb)
    Hq = of.get_sparse_operator(of.jordan_wigner(iop)).toarray()
    nq = ob.shape[0]
    num = of.get_sparse_operator(of.number_operator(nq), n_qubits=nq).diagonal().real
    idx = [k for k in range(2 ** nq) if round(num[k]) == nel]
    e_sector = np.sort(np.linalg.eigvalsh(Hq[np.ix_(idx, idx)]))
    print(f"\n== {label}: E_nuc={mol.energy_nuc():.6f}  E_RHF={mf.e_tot:.6f}  E_FCI(total)={e_fci:.6f}")
    print(f"   JW qubits={nq}; N={nel} sector ground (electronic, excl. const)={e_sector[0]:.6f}; +const={e_sector[0] + const:.6f}")
    return mol, e_fci, e_sector, const

# H2 at the geometry the CJC docstring states
mol, e_fci, e_sec, const = reference("H 0 0 0; H 0 0 0.7414", "H2 R=0.7414 A")
n, H, herm = load("h2")
ev = np.sort(np.linalg.eigvalsh(H))
print(f"   CJC h2 (from CJC code): hermiticity err={herm:.1e}; spectrum={np.round(ev, 6)}")
print(f"   CJC min={ev[0]:.6f}; min + E_nuc={ev[0] + mol.energy_nuc():.6f}; vs FCI total diff={ev[0] + mol.energy_nuc() - e_fci:+.2e} Ha")
print(f"   PySCF/OpenFermion N=2 spectrum={np.round(e_sec, 6)}")

# LiH at the geometry the CJC docstring states: full FCI + a 2-orbital (4-qubit) frozen-core active space
molL, e_fciL, e_secL, _ = reference("Li 0 0 0; H 0 0 1.546", "LiH R=1.546 A (full space)")
_, _, e_actL, constL = reference("Li 0 0 0; H 0 0 1.546", "LiH R=1.546 A (4-qubit active space: 2e in 2 orbitals, frozen core)", ncas=2, nelecas=2)
n, HL, hermL = load("lih")
evL = np.sort(np.linalg.eigvalsh(HL))
print(f"\n   CJC lih (from CJC code): hermiticity err={hermL:.1e}; lowest 4={np.round(evL[:4], 6)}")
print(f"   CJC docstring claims ground ~ -7.8825 Ha; CJC matrix min = {evL[0]:.6f}")
print(f"   diff vs full FCI total = {evL[0] - e_fciL:+.4f} Ha; vs 4-qubit active-space total = {evL[0] - (e_actL[0] + constL):+.4f} Ha")
