"""Seshat — Python-side recorder for the cross-language ``.seshat`` profiler.

Pure stdlib Python. Records real Python frames and the Python↔native (Rust/C)
boundary via ``sys.setprofile``, writes the ``.seshat`` binary format, and lets
the Rust ``cjc-seshat`` analyzer produce a cross-language report.

::

    import seshat
    with seshat.record("run.seshat"):
        my_program()

    # then:  seshat analyze run.seshat   (the Rust CLI)
"""

from .recorder import Recorder, mark_boundary, mark_copy, record, run_path, zone
from .format import TraceWriter

__all__ = [
    "Recorder",
    "record",
    "run_path",
    "zone",
    "mark_copy",
    "mark_boundary",
    "TraceWriter",
]
__version__ = "0.1.0"
