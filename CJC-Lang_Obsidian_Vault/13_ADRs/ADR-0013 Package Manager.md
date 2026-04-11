---
title: ADR-0013 Package Manager
tags: [adr, proposed, packaging, cli]
status: Proposed
date: 2026-03-22
source: docs/adr/ADR-0013-package-manager.md
---

# ADR-0013 — Minimal Package Manager for CJC-Lang v1.0

**Status:** Proposed · **Date:** 2026-03-22

## The decision (proposed)

Introduce a minimal package manager with three parts:

1. **Manifest** — `cjc.toml` with `[package]`, `[dependencies]`, `[dev-dependencies]`, and a `deterministic = true` metadata flag.
2. **Lockfile** — `cjc.lock` pinning every dependency by exact git commit hash + SHA-256 checksum of the fetched tree.
3. **Resolver** — deterministic by construction: exact version pins only, lexicographic topological order, no semver range solving.

**Registry:** Git-only in v1. A `pkg.cjc-lang.org`-style central registry is explicitly deferred to a future ADR.

**New crate:** `cjc-pkg` (resolution, fetching, lockfile). `cjc-cli` grows `init`, `lock`, `fetch`, `build`, `build --verify-lock`.

## Why this matters

- **Reproducibility all the way down.** The whole CJC-Lang determinism story (same seed → same bits) extends to *the source tree itself*. Same `cjc.toml` + same `cjc.lock` → identical dependency trees on every machine.
- **No SAT solver, no diamonds.** Range resolution is inherently non-deterministic across time — a new library release can change yesterday's solution. Exact pinning removes the entire class of problem.
- **Minimal blast radius.** Git is universal and already content-addressed. No registry infrastructure needed for a v1.0 launch.

## Deliberately deferred

- Version range resolution / `cjc update` with automatic dedup
- Diamond conflict auto-resolution (v1 hard-errors on mismatch)
- Binary/pre-compiled packages
- Build scripts and install hooks (intentional — they are a non-determinism source)
- Workspaces and monorepo layouts

## What this constrains

- [[Module System]] must be extended so imports can resolve against packages fetched into `~/.cjc/cache/git/`.
- Lockfile format freezes early: `source` / `commit` / `checksum` fields are permanent.
- Determinism auditor must verify resolution produces identical lockfiles across Linux/macOS/Windows.

## Related

- [[Module System]]
- [[CLI Surfaces]]
- [[Determinism Contract]]
- [[ADR Index]]
