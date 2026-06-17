# seshat (Python recorder)

Pure-stdlib Python recorder for the cross-language [`cjc-seshat`](../crates/cjc-seshat)
profiler. It captures **real Python frames** and the **Python↔native (Rust/C)
boundary** via `sys.setprofile`, writes the `.seshat` binary format, and the Rust
`seshat` CLI analyzes it.

## Why there is no PyO3 / maturin here

The `.seshat` *file* is the cross-language interface. The Python side only needs
to **write** that format — so this package is pure stdlib Python: no PyO3, no
maturin, no Python headers, no compilation. It works with any CPython 3.x.

`sys.setprofile` is the whole trick: its callback fires on every Python
`call`/`return` **and** every `c_call`/`c_return` — i.e. every time Python
crosses into a native function. A PyO3 Rust extension is one kind of native
callee, so that single hook captures the Py↔Rust seam.

## Install

```bash
pip install ./python-seshat          # zero dependencies, pure stdlib
```

## Two modes

| Mode | Weighting | Overhead | Deterministic? | Use for |
|---|---|---|---|---|
| **`sampling`** (default) | **time** | low (`~py-spy`) | no | "where is my wall-clock going?" |
| `calls` | call count | high (`~cProfile`) | **yes** | CI `diff`/`variance` gates |

`sampling` mode uses `sys.setprofile` only to *maintain* the call stack + Py↔native
boundary cheaply, while a background thread samples that stack at a fixed interval.
So a slow function gets more samples (accurate), and the event count is tiny — the
demo dropped from **24,000 events (calls) to ~400 (sampling)**, and its measured
boundary share dropped from a call-weighted 72% to a **time-weighted 40%** (the
honest number). `calls` mode emits one sample per call: reproducible, for CI.

## Usage

```python
import seshat

# default = sampling (time-weighted)
with seshat.record("run.seshat"):
    with seshat.zone("parse"):
        data = parse(raw)          # json.loads etc. show up as native crossings
    with seshat.zone("compute"):
        result = my_rust_ext.process(data)   # PyO3 call = a boundary crossing

# deterministic trace for CI diffing:
with seshat.record("ci.seshat", mode="calls"):
    ...
```

Then analyze with the Rust CLI:

```bash
seshat analyze run.seshat            # text report
seshat analyze run.seshat --json
seshat analyze run.seshat --svg fg.svg
```

Or record a whole script without editing it:

```bash
python -m seshat run myscript.py --out run.seshat
seshat analyze run.seshat
```

## What it captures today

| Capability | Status |
|---|---|
| Real Python call frames (name/file/line) | ✅ |
| Python↔native boundary crossings | ✅ automatic; **time-weighted** in sampling mode |
| **Time-weighted flamegraph** (vs call-weighted) | ✅ sampling mode (default) |
| **Low overhead** (~py-spy class) | ✅ sampling mode — 60× fewer events than calls mode |
| Pipeline stages via `seshat.zone(...)` | ✅ |
| Python-heap memory (`tracemalloc`) | ✅ temporal (sampling) → true peak; snapshot (calls) → by site |
| Cross-domain copies — manual `seshat.mark_copy(...)` | ✅ |
| Cross-domain copies — **auto-discovered** (numpy/torch/arrow copy calls) | ✅ heuristic registry; exact bytes for bound methods (`ndarray.copy`), `0`=unknown for free fns (`np.array`) |
| Async: coroutine frames tagged + **await stalls measured** | ✅ resume counts + max-await via `setprofile` suspend/resume pairing |
| Deterministic trace for CI | ✅ `mode="calls"` |
| **Multi-thread capture** | ✅ `threading.setprofile` + per-thread stacks; one Sample per live thread per tick, stable logical thread ids (main = 0) |
| **GIL-wait detection** | ✅ *heuristic* (sampling) — a thread frozen at one frame while another progresses → `GilWait`; **approximate**, not an exact GIL signal |
| Native Rust-frame unwinding inside the extension | ⬜ needs a stack unwinder (the Rust extension can self-instrument with `cjc-seshat`'s collector) |
| Thermal / perf counters | ⬜ no portable stdlib source |

Weighting note: a Sample is emitted per `call`/`c_call`, so the flamegraph is
inclusive-call-count weighted (like `cProfile`'s structure). Because a
deterministic Python program has a fixed call sequence, the resulting trace is
itself deterministic.

## API

- `seshat.record(path, trace_memory=True)` — context manager; writes on exit.
- `seshat.zone(name)` — context manager; marks a pipeline stage.
- `seshat.mark_copy(from, to, bytes)` — record a cross-domain copy.
- `seshat.mark_boundary(name)` — record a named boundary crossing.
- `seshat.run_path(script, out)` — record a script file programmatically.
- `seshat.Recorder` — low-level `start(trace_memory=...)` / `stop`.
- `seshat.TraceWriter` — the `.seshat` binary writer (emit events directly).

Analyze N runs for nondeterministic cost (the Rust CLI):

```bash
seshat variance run1.seshat run2.seshat run3.seshat
```

## Tests

```bash
cd python-seshat
python tests/test_format.py          # or: python -m pytest tests/
python examples/gen_fixture.py /tmp/x.seshat   # regenerate the Rust test fixture
```
