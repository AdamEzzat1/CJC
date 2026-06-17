"""A tiny mixed Python + native data pipeline, to demonstrate Seshat's
cross-language capture.

Each `json.*`, `re.*`, `sorted`, and `"".join` call is a *native* (C) call —
exactly the kind of Python→native boundary that a PyO3 Rust extension also
produces. Recording this with Seshat shows Python frames and the native boundary
together in one trace.

Run:
    python -m seshat run examples/demo.py --out demo_py.seshat
    seshat analyze demo_py.seshat        # the Rust CLI
"""

import json
import re

import seshat

_WORD = re.compile(r"[a-z]+")


def parse(raw):
    # json.loads is a C call -> boundary crossing
    return [json.loads(line) for line in raw]


def tokenize(text):
    # re.findall is a C call -> boundary crossing
    return _WORD.findall(text.lower())


def transform(records):
    out = []
    for rec in records:
        toks = tokenize(rec["text"])
        out.append({"id": rec["id"], "n": len(toks), "first": toks[0] if toks else ""})
    return out


def aggregate(rows):
    # sorted() is a C call -> boundary crossing
    return sorted(rows, key=lambda r: r["n"], reverse=True)


def serialize(rows):
    # json.dumps is a C call -> boundary crossing
    return "\n".join(json.dumps(r) for r in rows)


def main():
    # sized so the run lasts long enough for the sampler to collect a few
    # hundred time-weighted samples (sampling mode); calls mode needs far less.
    n = 12000
    with seshat.zone("setup"):
        raw = [
            json.dumps({"id": i, "text": f"the quick brown fox number {i} jumps"})
            for i in range(n)
        ]
    with seshat.zone("parse"):
        records = parse(raw)
    with seshat.zone("transform"):
        rows = transform(records)
    with seshat.zone("aggregate"):
        ranked = aggregate(rows)
    with seshat.zone("serialize"):
        blob = serialize(ranked)
        # The manual way: you *tell* Seshat about a cross-domain copy.
        seshat.mark_copy("rustheap", "numpy", len(blob) * 8)
    with seshat.zone("handoff"):
        # The automatic way: Seshat *finds* copy-inducing native calls on its
        # own (no mark_copy). If NumPy is installed, these calls emit Copy edges
        # that show up in `seshat analyze`'s copy section next to the marked one.
        # Guarded so the demo still runs with a pure-stdlib environment.
        try:
            import numpy as np

            buf = np.frombuffer(blob.encode("utf-8"), dtype=np.uint8)  # zero-copy view
            strided = np.ascontiguousarray(buf[::2])  # non-contiguous → real copy
            _ = strided.copy()                        # bound method → exact nbytes
        except Exception:
            pass
    return len(blob)


if __name__ == "__main__":
    total = main()
    print(f"pipeline produced {total} bytes")
