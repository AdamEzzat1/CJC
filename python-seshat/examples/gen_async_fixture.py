"""Generate the committed async test fixture.

    python examples/gen_async_fixture.py <out.seshat>

Records a tiny asyncio program so the Rust test `tests/python_bridge.rs` can
prove the analyzer sees measured await stalls (AwaitResume edges) from real
Python coroutines.
"""

import asyncio
import sys

import seshat


async def fetch(i):
    await asyncio.sleep(0)  # a real await suspend/resume
    return i * i


async def worker(n):
    total = 0
    for i in range(n):
        total += await fetch(i)
    return total


def main():
    out = sys.argv[1]
    # calls mode → deterministic committed fixture
    with seshat.record(out, trace_memory=False, mode="calls"):
        asyncio.run(worker(40))
    print("wrote", out)


if __name__ == "__main__":
    main()
