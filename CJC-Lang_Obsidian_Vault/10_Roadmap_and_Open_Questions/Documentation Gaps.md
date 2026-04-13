---
title: Documentation Gaps
tags: [roadmap, docs]
status: Audit result
---

# Documentation Gaps

Places where either the code moved ahead of the docs, the docs moved ahead of the code, or the two disagree. Found during vault construction.

## Doc says "incomplete", code says otherwise

### `cjc-module`

- **Docs:** `README.md` and `CLAUDE.md` both call the module system "incomplete" or "experimental".
- **Code:** `crates/cjc-module/src/lib.rs` is ~1,183 lines with `ModuleId`, `ModuleGraph`, DFS cycle detection, symbol mangling, and topological `merge_programs`.
- **Gap:** What is actually missing — CLI wiring? Parser recognition of `mod`/`import`? Path resolution? The docs don't say.

See [[Module System]].

### `cjc-analyzer` (LSP)

- **Docs:** Listed as "experimental".
- **Code:** The crate exists; the workspace compiles it.
- **Gap:** Unclear what LSP features are implemented vs. stubs. No editor integration docs.

See [[Language Server]].

## Doc count vs. code count

### Builtins

- `README.md`: 221+ builtins.
- Code survey: ~334 registered handlers in `cjc-runtime/src/builtins.rs`.
- **Gap:** No canonical list. The number in the README has likely grown without being updated.

See [[Builtins Catalog]].

### Tests

- `README.md`: 3,700+.
- `CLAUDE.md` memory: 5,320.
- **Gap:** No authoritative "current test count" file that is updated on every release.

### Chess RL tests

- `README.md`: "150 per README" (paraphrased).
- `CLAUDE.md`: 49 in `chess_rl_project/`.
- Showcase note totals to **216** across four suites.
- **Gap:** The number depends on which suites you count. A single total belongs in [[Chess RL Demo]].

## Version drift

### v0.1.4 rebrand

- The rebrand from **CJC** to **CJC-Lang** (and `.cjc` → `.cjcl`, `cjc` CLI → `cjcl`) happened in v0.1.4.
- Many older docs (everything under `docs/spec/*.md`, most of `docs/`, and many example filenames) still use the old naming.
- **Gap:** A documentation sweep is pending — this is acceptable because internal crate names (`cjc-*`) were intentionally not renamed, but user-facing doc text should reflect v0.1.4.

See [[Version History]].

## Spec-only features

Some features are specified in `docs/spec/` but have no implementation yet. These are not gaps so much as **spec-not-yet-code**:

- **Bastion** library — `docs/bastion/` has a detailed 15-primitive statistical-computing spec. The library itself is not built. See [[Bastion]].
- **Stage 3 tasks** (S3-P0-05 through S3-P2-04) — roadmap exists, code lives partly on feature branches or in TODO comments.
- **LLVM backend** — `cjc-codegen` crate not yet present.

## Performance claims without a fresh baseline

`docs/spec/CJC_PERFORMANCE_MANIFESTO.md` has concrete numbers (RNN, transformer, binary size). They are **not timestamped with a commit SHA**. This means the numbers could be stale after any optimizer or runtime change.

**Recommendation:** Every future perf manifesto update should carry a commit SHA and a machine description.

See [[Performance Profile]].

## Examples directory as a spec

The `examples/` directory is (de facto) a spec: the chess RL demo, the 9 ML examples, the vizor gallery, and the NLP examples are all what people will use to learn the language. But there is no central "examples index" doc that connects them to the language reference or the determinism contract.

**Recommendation:** Either (a) keep [[Deterministic Workflow Examples]] as the de-facto index in this vault, or (b) add an `examples/README.md` with the same role.

## ADR coverage

`docs/adr/` contains 12 ADRs (ADR-0001 through ADR-0012 per the roadmap task registry). ADRs referenced by roadmap tasks include:

- ADR-0003 (fixture runner)
- ADR-0008 (runtime submodule split)
- ADR-0009 (Vec COW)
- ADR-0010 (scope stack SmallVec)
- ADR-0011 (parallel matmul)
- ADR-0012 (CFG/SSA/dominator tree)

**Gap:** The vault doesn't yet have individual notes for each ADR. This would be a good next pass.

## Related

- [[Open Questions]]
- [[Roadmap]]
- [[Version History]]
- [[Current State of CJC-Lang]]
