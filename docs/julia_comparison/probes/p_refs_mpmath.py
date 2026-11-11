"""50-digit references for probes P3 (svd) and P4 (erf/erfc). Needs mpmath only.
Run: python -I p_refs_mpmath.py"""
import mpmath as mp
mp.mp.dps = 50
print("P4 erf/erfc references and the CJC values printed by p4_erf.cjcl:")
cjc = {("erf", 0.5): 0.5205000163047472, ("erf", 1.0): 0.8427006897475899,
       ("erfc", 3.0): 2.210514889775622e-05, ("erfc", 5.0): 1.547817043333678e-12}
for (fn, x), got in cjc.items():
    ref = (mp.erf if fn == "erf" else mp.erfc)(mp.mpf(x))
    print(f"  {fn}({x}) ref={mp.nstr(ref, 17)} cjc={got!r} rel_err={mp.nstr(abs(got - ref) / ref, 3)}")
print("P3 singular values of [[1,1],[1,c]], c = fl(1 + 1e-10) (the literal as parsed):")
c = mp.mpf(1.0 + 1e-10)        # exact binary value of the f64 literal
A = mp.matrix([[1, 1], [1, c]])
s = mp.svd_r(A, compute_uv=False)
print("  sigma =", [mp.nstr(v, 17) for v in s], " CJC svd: [2.00000000005, 0.0]; CJC eigh: 5.000000413701855e-11")
