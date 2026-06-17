"""``python -m seshat run <script.py> [--out trace.seshat]``

Records a Python script's execution into a ``.seshat`` trace, which the Rust
``seshat analyze`` CLI then turns into a cross-language report.
"""

import sys

from .recorder import run_path

USAGE = "usage: python -m seshat run <script.py> [--out <trace.seshat>] [--thermal]"


def main(argv):
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    if argv[0] != "run":
        print(f"seshat: unknown subcommand {argv[0]!r}\n{USAGE}", file=sys.stderr)
        return 1
    rest = argv[1:]
    script = None
    out = "trace.seshat"
    thermal = False
    i = 0
    while i < len(rest):
        a = rest[i]
        if a == "--out":
            i += 1
            if i >= len(rest):
                print("seshat: --out needs a path", file=sys.stderr)
                return 1
            out = rest[i]
        elif a == "--thermal":
            thermal = True  # sample CPU frequency (needs the `seshat[thermal]` extra → psutil)
        elif not a.startswith("--"):
            script = a
        else:
            print(f"seshat: unknown flag {a!r}", file=sys.stderr)
            return 1
        i += 1
    if script is None:
        print(f"seshat: run needs a <script.py>\n{USAGE}", file=sys.stderr)
        return 1
    n = run_path(script, out, trace_thermal=thermal)
    print(f"seshat: recorded {n} events -> {out}")
    print(f"seshat: now run  `seshat analyze {out}`  (the Rust CLI) for the report")
    return 0


def _console():
    """Entry point for the installed ``seshat-record`` script. Allows
    ``seshat-record script.py`` as shorthand for ``seshat-record run script.py``."""
    argv = sys.argv[1:]
    if argv and argv[0] not in ("run", "help", "-h", "--help"):
        argv = ["run"] + argv
    raise SystemExit(main(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
