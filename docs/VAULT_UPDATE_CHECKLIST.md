# Vault Update Checklist

When landing a change that touches user-visible language, runtime, or workflow, also update the Obsidian vault at `CJC-Lang_Obsidian_Vault/` so future readers can trust it.

Keep the rules short enough to do from memory during a release PR.

## What triggers a vault update

| Change | Update |
|---|---|
| New or removed crate | `12_Source_Maps/Crate Index.md` + crate alias stub (see below) |
| New builtin (wiring-pattern three-place registration) | `06_Tensors_ML_AD/Builtins Catalog.md` or `07_Data_and_CLI/CLI Surfaces.md` depending on surface |
| New CLI subcommand | `07_Data_and_CLI/CLI Surfaces.md` |
| New ADR merged in `docs/adr/` | New `13_ADRs/ADR-XXXX Title.md` note + row in `13_ADRs/ADR Index.md` |
| New language feature (if/match/for/closures/decorators) | Concept note under `02_Language/`, plus update `01_Foundation/Current State of CJC-Lang.md` status table |
| New determinism primitive | Note under `05_Determinism_and_Numerics/`, plus `Determinism Contract.md` rationale |
| New optimizer pass or MIR analysis | Note under `03_Compiler/`, plus link from `MIR Optimizer.md` |
| Rebrand, rename, or naming sweep | Grep the whole vault for the old name, update every hit, then update `MEMORY.md` |
| Any feature marked "incomplete" that becomes shipped | Flip the status frontmatter from `Proposed` / `Partial` → `Implemented`, remove from `Open Questions.md` |

## Release-process checklist

Copy this block into the release PR description:

```
Vault update checklist (v0.x.y):
- [ ] MEMORY.md updated with the version entry and any new completed milestones
- [ ] Current State of CJC-Lang.md reflects new feature statuses
- [ ] Crate Index.md lists every workspace crate
- [ ] 13_ADRs/ADR Index.md has a row for every new ADR
- [ ] Any renamed CLI command / extension is reflected in Getting Started + CLI Surfaces
- [ ] `python scripts/vault_audit.py` reports zero broken wikilinks
- [ ] Numbers (test counts, builtin counts, LOC) still match reality, or marked `Needs verification`
```

## Crate alias stubs

When a new crate lands in `crates/`, create a stub note under `12_Source_Maps/<crate-name>.md`:

```markdown
---
title: <crate-name>
tags: [crate, alias]
status: Crate alias
---

# `<crate-name>`

Crate alias note. The full concept lives in **[[Concept Name]]**.

<one-sentence description>

## Related

- [[Concept Name]]
- [[Crate Index]]
```

This lets wikilinks like `[[cjc-foo]]` resolve without duplicating content — the alias points at the concept hub.

## Wiring Pattern for builtins (rule of three)

Every new builtin must appear in **three** places (see `CLAUDE.md`):

1. `cjc-runtime/src/builtins.rs` — shared stateless dispatch
2. `cjc-eval/src/lib.rs` — AST interpreter call handling
3. `cjc-mir-exec/src/lib.rs` — MIR executor call handling

…and, if user-visible, a fourth: a row in `06_Tensors_ML_AD/Builtins Catalog.md` (or the appropriate catalog). The catalog is the only place where the total builtin count is authoritative.

## Stale number convention

Any number that came from a one-time measurement (test count, LOC, builtin count, benchmark) must either be:

- **Timestamped** — "as of 2026-03-21" in the note body, *and* in `status:` frontmatter, OR
- **Marked** — `status: Needs verification` and link to `Open Questions.md`

Do not silently carry forward old numbers without one of these annotations.

## Archive, don't delete

When a note becomes obsolete (superseded design, abandoned feature, etc.):

1. Do **not** delete the file.
2. Change `status:` to `Historical / superseded`.
3. Add a header line explaining what replaced it and why.
4. Keep inbound links working so the history stays navigable.

## Broken-link audit

Run the audit before every release:

```bash
python scripts/vault_audit.py
python scripts/vault_audit.py --show-orphans   # optional: find orphan notes
```

Exits non-zero on broken links, suitable for `pre-commit` or CI.

## Why these rules

Past pain points this checklist is designed to prevent:

- **Silent drift.** The README said "cjc-module is incomplete" long after the crate shipped with 1,183 LOC of implementation. A single status flip would have avoided this.
- **Number rot.** "3,700+ tests" and "5,320 tests" both lived in different docs simultaneously. Timestamp + `Needs verification` would have caught it.
- **Orphan rebrands.** After v0.1.4 (CJC → CJC-Lang, `cjc` → `cjcl`, `.cjc` → `.cjcl`), several docs still referenced the old names because no one grep-swept the vault.
- **Broken links.** The vault has ~100+ wikilinks. Manual audit is impossible — hence `vault_audit.py`.
