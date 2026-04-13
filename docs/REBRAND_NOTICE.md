# v0.1.4 Rebrand Notice (2026-04-06)

CJC was renamed to **CJC-Lang** in v0.1.4. Any document in this repo that was written before that release uses the old naming and **has not been rewritten**.

## What changed

| Area | Old (≤ v0.1.3) | New (v0.1.4+) |
|---|---|---|
| Project name | CJC | CJC-Lang |
| CLI binary | `cjc` | `cjcl` |
| File extension | `.cjc` | `.cjcl` |
| Cargo crate (install) | `cjc` | `cjc-lang` |

Internal crate names remain `cjc-*` (e.g., `cjc-lexer`, `cjc-runtime`). Only user-facing names were touched.

## What did *not* change

- Every internal workspace crate keeps its `cjc-*` name.
- Rust import paths (`use cjc_runtime::...`) are unchanged.
- The language semantics are identical across the rebrand — it was a cosmetic change.
- Stored `.cjc` files continue to work if passed to `cjcl run` (the parser doesn't gate on extension).

## Why legacy docs aren't rewritten

Historical documents (release notes, progress reports, audits, benchmark logs, phase reports) describe *past* activity. Rewriting them would make the history unreliable — a reader couldn't tell which commit the document was describing. Instead, each stale doc carries a one-line banner pointing here.

## What this means when you read a pre-v0.1.4 doc

Whenever you see any of these in an old document, mentally substitute the v0.1.4 name:

- "CJC" → "CJC-Lang"
- `cjc run foo.cjc` → `cjcl run foo.cjcl`
- `cjc.exe` → `cjcl.exe`
- `cargo install cjc` → `cargo install cjc-lang`

The semantics described in the document still apply — only the name has moved.

## Live documentation

For current naming and usage, always consult:

- `docs/GETTING_STARTED.md` — current install, CLI, multi-file workflow
- `README.md` (repo root) — current feature matrix
- `CJC-Lang_Obsidian_Vault/01_Foundation/Current State of CJC-Lang.md` — current implementation status
- `CJC-Lang_Obsidian_Vault/01_Foundation/Version History.md` — versioning timeline

## Related

- Commit `cb7c76c` — "v0.1.4: Rebrand CJC to CJC-Lang"
- `CLAUDE.md` — the project instructions at the repo root use the new names
