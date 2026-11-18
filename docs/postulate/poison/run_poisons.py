# Run from anywhere: python -I docs/postulate/poison/run_poisons.py [name ...]
"""Apply each poison to the real source, run the gate it targets, require a
failure, revert. Writes the patch and the failing output under docs/postulate/poison."""
import difflib, pathlib, subprocess, sys
root = pathlib.Path(__file__).resolve().parents[3]
out_dir = root / "docs/postulate/poison"
POISONS = [
    ("01_span_shifted_one_byte",
     "crates/cjc-ad/src/tracker.rs",
     "        SourceSpan::new(0, start, end)\n    }",
     "        SourceSpan::new(0, start + 1, end)\n    }",
     ["--test", "postulate_phase1", "gate_spans_cover", "gate_executor_parity_xor"]),
    ("02_mir_skips_the_receiver_span_reset",
     "crates/cjc-mir-exec/src/lib.rs",
     "                self.ad.span = ad_span;\n                self.dispatch_method(obj_val, name, arg_vals)",
     "                self.dispatch_method(obj_val, name, arg_vals)",
     ["--test", "postulate_phase1", "span_is_the_method_call"]),
    ("02b_mir_stamps_no_spans",
     "crates/cjc-mir-exec/src/lib.rs",
     "let ad_span = span.map(|s| cjc_ad::tracker::Tracker::call_span(s.start, s.end));",
     "let ad_span: Option<cjc_ad::export::SourceSpan> = { let _ = span; None };",
     ["--test", "postulate_phase1", "gate_"]),
    ("03_telemetry_perturbs_a_gradient",
     "crates/cjc-ad/src/tracker.rs",
     "                if loss < g.len() {\n                    t.on_backward(g, loss);",
     "                if loss < g.len() {\n"
     "                    if let Some(Some(gr)) = g.param_grads.get_mut(0) {\n"
     "                        let mut v = gr.to_vec();\n"
     "                        v[0] = f64::from_bits(v[0].to_bits() ^ 1);\n"
     "                        *gr = cjc_runtime::Tensor::from_vec(v, &gr.shape().to_vec()).unwrap();\n"
     "                    }\n"
     "                    t.on_backward(g, loss);",
     ["--test", "postulate_phase1", "gate_telemetry_parity"]),
    ("04_analyzer_answers_with_a_nondeterministic_field",
     "crates/cjc-analyzer/src/postulate.rs",
     '        "executor": "cjc-eval",',
     '        "executor": "cjc-eval",\n        "nonce": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64,',
     ["-p", "cjc-analyzer", "--test", "replay", "gate_lsp_replay"]),
    ("05_telemetry_ignores_its_sampling_rate",
     "crates/cjc-ad/src/telemetry.rs",
     "        let step = self.step;\n        self.step += 1;\n        if step % self.every != 0 {",
     "        let step = self.step;\n        self.step += 1;\n        let line = record(g, loss_idx, step, self.seed);\n        if step % self.every != 0 {\n            drop(line);",
     ["cmd:", "cargo", "run", "--release", "-q", "-p", "postulate-overhead", "--", "101", "--check", "15"]),
]
only = sys.argv[1:]
summary = []
for name, rel, old, new, args in POISONS:
    if only and name not in only:
        continue
    p = root / rel
    orig = p.read_bytes()
    text = orig.decode("utf-8")
    nl = "\r\n" if "\r\n" in text else "\n"
    o, n = old.replace("\n", nl), new.replace("\n", nl)
    assert text.count(o) == 1, (name, text.count(o))
    poisoned = text.replace(o, n)
    patch = "".join(difflib.unified_diff(text.replace("\r\n", "\n").splitlines(True),
                                         poisoned.replace("\r\n", "\n").splitlines(True),
                                         "a/" + rel, "b/" + rel))
    (out_dir / f"{name}.patch").write_text(patch, encoding="utf-8", newline="\n")
    try:
        p.write_bytes(poisoned.encode("utf-8"))
        cmd = ["cargo", "test"] + args[:-1] + ["--", args[-1]] if args[0] == "-p" else None
        if args[0] == "cmd:":
            cmd = args[1:]
        if args[0] == "--test":
            cmd = ["cargo", "test", "--test", args[1], "--"] + args[2:]
        r = subprocess.run(cmd, cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
    finally:
        p.write_bytes(orig)
    lines = (r.stdout + r.stderr).splitlines()
    keep = [l for l in lines if l.startswith("test ") or "panicked at" in l or l.startswith("test result")
            or "left:" in l or "right:" in l or l.startswith("error")
            or "gate:" in l or l.startswith(("fixture", "mlp_xor", "linreg", "tanh_fit"))]
    failed = r.returncode != 0 and any("FAILED" in l or "gate: FAIL" in l for l in lines)
    (out_dir / f"{name}.out.txt").write_text(
        f"$ {' '.join(cmd)}\nexit code: {r.returncode}\n\n" + "\n".join(keep[:60]) + "\n", encoding="utf-8")
    summary.append((name, failed, r.returncode))
    print(name, "FAILED-AS-REQUIRED" if failed else "!!! DID NOT FAIL", r.returncode, flush=True)
    for l in keep[:25]:
        print("   ", l[:220])
sys.exit(0 if all(f for _, f, _ in summary) else 1)
