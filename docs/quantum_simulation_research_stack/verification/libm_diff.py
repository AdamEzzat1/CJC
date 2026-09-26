"""Diff two libm_bits_<platform>.txt dumps (same emitter, same angles)."""
import struct, sys
f = lambda h: struct.unpack("<d", struct.pack("<Q", int(h, 16)))[0]
A = [l.split() for l in open(sys.argv[1]) if l.strip()]
B = [l.split() for l in open(sys.argv[2]) if l.strip()]
assert len(A) == len(B) and all(a[0] == b[0] for a, b in zip(A, B))
rows = [(a, b) for a, b in zip(A, B) if a[1:] != b[1:]]
small = sum(1 for a, _ in rows if abs(f(a[0])) <= 4 * 3.141592653589794)
print(f"{sys.argv[1]} vs {sys.argv[2]}")
print(f"angles: {len(A)}; angles with any differing bit: {len(rows)} ({100*len(rows)/len(A):.2f}%), of which |theta|<=4pi: {small}")
print(f"cos differs: {sum(a[1] != b[1] for a, b in rows)}; sin differs: {sum(a[2] != b[2] for a, b in rows)}")
for a, b in rows[:4]:
    print(f"  theta={f(a[0])!r}: cos {f(a[1])!r} vs {f(b[1])!r}; sin {f(a[2])!r} vs {f(b[2])!r}")
