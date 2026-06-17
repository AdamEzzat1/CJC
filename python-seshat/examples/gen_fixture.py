"""Generate the committed cross-language test fixture.

    python examples/gen_fixture.py <out.seshat>

Produces a tiny `.seshat` with both Python frames (`work`) and a native boundary
(json's C scanner), used by the Rust test `tests/python_bridge.rs` to prove the
analyzer reads Python-produced traces.
"""

import json
import sys

import seshat


def work():
    total = 0
    for i in range(5):
        obj = json.loads('{"n": %d}' % i)
        total += obj["n"]
    return total


def main():
    out = sys.argv[1]
    # calls mode → deterministic committed fixture (sampling mode is timing-based)
    with seshat.record(out, trace_memory=False, mode="calls"):
        with seshat.zone("work"):
            work()
    print("wrote", out)


if __name__ == "__main__":
    main()
