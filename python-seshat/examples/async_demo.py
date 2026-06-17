"""An asyncio demo: coroutine frames are captured as async tasks.

    python -m seshat run examples/async_demo.py --out async.seshat
    seshat analyze async.seshat        # the Rust CLI — async frames show as `async:`
"""

import asyncio
import json


async def fetch(i):
    # a real await point (yields control to the event loop)
    await asyncio.sleep(0)
    return json.dumps({"id": i, "v": i * i})


async def worker(n):
    out = []
    for i in range(n):
        out.append(await fetch(i))
    return out


def main():
    results = asyncio.run(worker(300))
    return len(results)


if __name__ == "__main__":
    total = main()
    print(f"async pipeline produced {total} results")
