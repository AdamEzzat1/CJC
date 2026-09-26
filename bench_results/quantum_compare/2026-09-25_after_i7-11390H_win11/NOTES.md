# Notes on this run

- W1–W3 and W6 rows: one run of `quantum_compare run --suite baseline --reps 5`, then the
  Aer and Stim drivers, at the working tree after item 14 and the MPS SVD sign fix.
- W4 rows (CJC and Aer) were re-run afterwards, in `../2026-09-25_after-mps_i7-11390H_win11/`,
  after one more MPS change (canonicalize only when truncation drops more than round-off,
  `TRUNC_REL_TOL` in `mps.rs`), and merged here. W4 outputs from that change differ in low
  bits from the first run; the W4 `work/` dumps are from the re-run, so the Aer comparison
  matches them.
- Timings on this laptop varied up to 2x between runs on unchanged code (W3_n250 rust:
  0.38 s, 0.90 s, 0.65 s in three runs). Compare execute times across runs only for
  effects much larger than that. The controlled old-vs-new numbers are the interleaved A/B
  in `../2026-09-25_kernels_i7-11390H_win11/kernels.jsonl`.
- The baseline directory's Aer *totals* include `qiskit.quantum_info.Statevector.sample_memory`
  (~54 s at 22 qubits); this run's Aer driver samples with numpy instead. Compare Aer
  *execute* columns across the two runs, not totals.
