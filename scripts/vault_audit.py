#!/usr/bin/env python3
"""
vault_audit.py — Audit the CJC-Lang Obsidian vault for broken wikilinks.

Walks CJC-Lang_Obsidian_Vault/ and reports any [[Target]] that does not
resolve to a .md file anywhere in the vault. Exits non-zero if broken
links are found, so it can be wired into pre-commit or CI.

Usage:
    python scripts/vault_audit.py
    python scripts/vault_audit.py --vault path/to/vault
    python scripts/vault_audit.py --show-orphans     # also print notes with no inbound links
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")

DEFAULT_VAULT = Path(__file__).resolve().parent.parent / "CJC-Lang_Obsidian_Vault"


def collect_notes(vault: Path) -> dict[str, Path]:
    """Return a map from note stem (filename without .md) to path."""
    notes: dict[str, Path] = {}
    for md in vault.rglob("*.md"):
        stem = md.stem
        if stem in notes:
            # Duplicate — keep the first, warn later
            continue
        notes[stem] = md
    return notes


def scan_links(path: Path) -> list[tuple[int, str]]:
    """Yield (line_no, target) for every wikilink in the file."""
    results: list[tuple[int, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    for lineno, line in enumerate(text.splitlines(), start=1):
        for match in WIKILINK_RE.finditer(line):
            target = match.group(1).strip()
            results.append((lineno, target))
    return results


def audit(vault: Path, show_orphans: bool) -> int:
    if not vault.exists():
        print(f"error: vault not found at {vault}", file=sys.stderr)
        return 2

    notes = collect_notes(vault)
    print(f"Scanning {len(notes)} notes in {vault}")

    broken: list[tuple[Path, int, str]] = []
    inbound: dict[str, int] = {stem: 0 for stem in notes}

    for stem, path in notes.items():
        for lineno, target in scan_links(path):
            if target in notes:
                inbound[target] += 1
            else:
                broken.append((path, lineno, target))

    if broken:
        print(f"\nBROKEN LINKS ({len(broken)}):")
        for path, lineno, target in broken:
            rel = path.relative_to(vault)
            print(f"  {rel}:{lineno}  -> [[{target}]]")

    if show_orphans:
        orphans = [stem for stem, n in inbound.items() if n == 0 and stem not in ("MEMORY", "README")]
        if orphans:
            print(f"\nORPHAN NOTES (no inbound links, {len(orphans)}):")
            for stem in sorted(orphans):
                rel = notes[stem].relative_to(vault)
                print(f"  {rel}")

    if broken:
        print(f"\nFAIL: {len(broken)} broken wikilinks")
        return 1

    print(f"\nOK: all {sum(len(scan_links(p)) for p in notes.values())} wikilinks resolve")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Obsidian vault wikilinks.")
    parser.add_argument("--vault", type=Path, default=DEFAULT_VAULT, help="Path to the vault directory.")
    parser.add_argument("--show-orphans", action="store_true", help="Also print orphan notes.")
    args = parser.parse_args()
    return audit(args.vault, args.show_orphans)


if __name__ == "__main__":
    sys.exit(main())
