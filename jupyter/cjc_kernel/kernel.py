"""
CJC Jupyter Kernel — Sidecar implementation.

Delegates execution to the `cjc` binary via subprocess, communicating
through JSON over stdout. This avoids embedding a Rust runtime in
Python and preserves CJC's determinism guarantees.

Protocol:
  - Each cell is written to a temp file and executed via `cjc run --format json`.
  - `cjc` emits `{"ok":true,"output":[...]}` on success.
  - `cjc` emits `{"ok":false,"error":"..."}` on failure.
  - Exit codes: 0=success, 1=runtime, 2=parse, 3=type, 4=parity.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time

from ipykernel.kernelbase import Kernel


class CJCKernel(Kernel):
    implementation = "cjc_kernel"
    implementation_version = "0.1.2"
    language = "cjc"
    language_version = "0.1.2"
    language_info = {
        "name": "cjc",
        "mimetype": "text/x-cjc",
        "file_extension": ".cjc",
        "codemirror_mode": "rust",
    }
    banner = "CJC — Deterministic Numerical Programming Language"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cjc_binary = self._find_cjc()
        self._seed = 42  # default deterministic seed
        self._execution_times = []  # for latency tracking

    # ── Binary discovery ─────────────────────────────────────────────

    def _find_cjc(self) -> str:
        """Locate the cjc binary. Checks PATH, then common build locations."""
        # 1. Check PATH
        found = shutil.which("cjc")
        if found:
            return found

        # 2. Check workspace target directories (dev builds)
        workspace = os.environ.get("CJC_HOME", "")
        if workspace:
            candidates = [
                os.path.join(workspace, "target", "release", "cjc"),
                os.path.join(workspace, "target", "release", "cjc.exe"),
                os.path.join(workspace, "target", "debug", "cjc"),
                os.path.join(workspace, "target", "debug", "cjc.exe"),
            ]
            for c in candidates:
                if os.path.isfile(c):
                    return c

        # 3. Check relative to this file (installed alongside)
        here = os.path.dirname(os.path.abspath(__file__))
        for name in ("cjc", "cjc.exe"):
            p = os.path.join(here, "..", "..", "target", "release", name)
            if os.path.isfile(p):
                return os.path.abspath(p)
            p = os.path.join(here, "..", "..", "target", "debug", name)
            if os.path.isfile(p):
                return os.path.abspath(p)

        return "cjc"  # hope it's on PATH at runtime

    # ── Cell execution ───────────────────────────────────────────────

    def do_execute(
        self, code, silent, store_history=True,
        user_expressions=None, allow_stdin=False,
    ):
        code = code.strip()
        if not code:
            return self._ok_reply()

        # Handle magic commands
        if code.startswith("%"):
            return self._handle_magic(code, silent)

        # Write code to temp file and execute
        start = time.perf_counter()
        result = self._run_cjc(code)
        elapsed = time.perf_counter() - start
        self._execution_times.append(elapsed)

        if result["ok"]:
            if not silent:
                output_text = "\n".join(result.get("output", []))
                if output_text:
                    # Send as execute_result for notebook display
                    self.send_response(
                        self.iopub_socket,
                        "execute_result",
                        {
                            "execution_count": self.execution_count,
                            "data": {"text/plain": output_text},
                            "metadata": {},
                        },
                    )
            return self._ok_reply()
        else:
            error_text = result.get("error", "Unknown error")
            if not silent:
                self.send_response(
                    self.iopub_socket,
                    "stream",
                    {"name": "stderr", "text": error_text},
                )
            return {
                "status": "error",
                "execution_count": self.execution_count,
                "ename": "CJCError",
                "evalue": error_text,
                "traceback": [error_text],
            }

    def _run_cjc(self, code: str) -> dict:
        """Execute CJC code via subprocess, return parsed JSON result."""
        # Determine if this is a full program or a bare expression
        is_program = (
            "fn " in code
            or "let " in code
            or "print(" in code
            or "for " in code
            or "while " in code
            or "struct " in code
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cjc", delete=False, encoding="utf-8"
        ) as f:
            if is_program:
                f.write(code)
            else:
                # Bare expression — wrap in main
                f.write(f"fn main() {{ print({code}); }}")
            f.flush()
            tmp_path = f.name

        try:
            cmd = [
                self._cjc_binary,
                "run",
                "--format", "json",
                "--seed", str(self._seed),
                tmp_path,
            ]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                encoding="utf-8",
            )

            # Try to parse JSON from stdout.
            # CJC's print() emits directly to stdout AND collects in output,
            # so --format json appends a JSON line after the raw output.
            # We scan from the last line backwards to find the JSON object.
            stdout = proc.stdout.strip()
            if stdout:
                lines = stdout.splitlines()
                # Try last line first (the JSON summary)
                for line in reversed(lines):
                    line = line.strip()
                    if line.startswith("{"):
                        try:
                            return json.loads(line)
                        except json.JSONDecodeError:
                            continue
                # No JSON found — treat raw stdout as output
                return {"ok": True, "output": lines}

            # If no stdout, check stderr for errors
            if proc.returncode != 0:
                stderr = proc.stderr.strip()
                return {"ok": False, "error": stderr or f"cjc exited with code {proc.returncode}"}

            return {"ok": True, "output": []}

        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "Execution timed out (60s limit)"}
        except FileNotFoundError:
            return {
                "ok": False,
                "error": (
                    f"CJC binary not found at: {self._cjc_binary}\n"
                    "Set CJC_HOME env var or add cjc to PATH."
                ),
            }
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ── Magic commands ───────────────────────────────────────────────

    def _handle_magic(self, code: str, silent: bool):
        parts = code.split(None, 1)
        magic = parts[0]
        arg = parts[1] if len(parts) > 1 else ""

        if magic == "%seed":
            if arg:
                try:
                    self._seed = int(arg)
                    if not silent:
                        self._stream("stdout", f"Seed set to {self._seed}")
                except ValueError:
                    self._stream("stderr", f"Invalid seed: {arg}")
            else:
                self._stream("stdout", f"Current seed: {self._seed}")
            return self._ok_reply()

        elif magic == "%time":
            if not self._execution_times:
                self._stream("stdout", "No executions recorded yet.")
            else:
                last = self._execution_times[-1]
                avg = sum(self._execution_times) / len(self._execution_times)
                self._stream(
                    "stdout",
                    f"Last: {last*1000:.2f}ms | "
                    f"Avg: {avg*1000:.2f}ms | "
                    f"Count: {len(self._execution_times)}",
                )
            return self._ok_reply()

        elif magic == "%latency":
            # Benchmark: run a trivial program to measure sidecar overhead
            start = time.perf_counter()
            self._run_cjc('fn main() { let x = 1; }')
            overhead = time.perf_counter() - start
            self._stream(
                "stdout",
                f"Sidecar round-trip overhead: {overhead*1000:.2f}ms",
            )
            return self._ok_reply()

        elif magic == "%cjc_path":
            self._stream("stdout", f"CJC binary: {self._cjc_binary}")
            return self._ok_reply()

        else:
            self._stream("stderr", f"Unknown magic command: {magic}")
            return self._ok_reply()

    # ── Helpers ──────────────────────────────────────────────────────

    def _stream(self, name: str, text: str):
        self.send_response(
            self.iopub_socket, "stream", {"name": name, "text": text + "\n"}
        )

    def _ok_reply(self):
        return {
            "status": "ok",
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": {},
        }

    def do_is_complete(self, code):
        """Check if code is a complete input (for multi-line cells)."""
        code = code.strip()
        if not code:
            return {"status": "incomplete", "indent": ""}
        # Count braces
        opens = code.count("{") - code.count("}")
        if opens > 0:
            return {"status": "incomplete", "indent": "    "}
        return {"status": "complete"}

    def do_complete(self, code, cursor_pos):
        """Basic tab completion for CJC builtins."""
        # Extract the token being completed
        text = code[:cursor_pos]
        # Find the start of the current identifier
        start = cursor_pos
        while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "_"):
            start -= 1
        prefix = text[start:cursor_pos]

        if not prefix:
            return {"matches": [], "cursor_start": start, "cursor_end": cursor_pos, "status": "ok"}

        matches = [b for b in _CJC_BUILTINS if b.startswith(prefix)]
        return {
            "matches": matches,
            "cursor_start": start,
            "cursor_end": cursor_pos,
            "metadata": {},
            "status": "ok",
        }

    def do_shutdown(self, restart):
        return {"status": "ok", "restart": restart}


# ── Builtin names for tab completion ─────────────────────────────────
# This is a static list; a future version could query `cjc schema --json`.

_CJC_BUILTINS = sorted([
    # Core
    "print", "len", "type_of", "assert", "panic",
    # Math
    "abs", "sqrt", "pow", "exp", "ln", "log2", "log10",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "floor", "ceil", "round", "min", "max", "clamp",
    # Array
    "array_push", "array_pop", "array_len", "array_slice",
    "array_map", "array_filter", "array_reduce",
    "array_any", "array_all", "array_find",
    "array_enumerate", "array_zip", "array_sort_by", "array_unique",
    "range", "relu", "reshape", "sort", "reverse",
    # Statistics
    "mean", "median", "variance", "std_dev", "covariance", "correlation",
    "jarque_bera", "anderson_darling", "ks_test",
    "cohens_d", "eta_squared", "cramers_v",
    "levene_test", "bartlett_test",
    # Sampling
    "latin_hypercube", "sobol_sequence",
    "train_test_split", "kfold_indices",
    "bootstrap", "permutation_test", "stratified_split",
    # Data
    "is_na", "drop_na", "coalesce",
    "as_factor", "factor_levels", "factor_codes",
    "fct_relevel", "fct_lump", "fct_count",
    # Tensor / Linear Algebra
    "Tensor.zeros", "Tensor.ones", "Tensor.from_vec",
    "matmul", "dot", "cross", "outer", "norm",
    "eigh", "svd", "kron", "det", "trace", "inverse", "solve",
    "lu_decompose", "qr_decompose", "cholesky",
    # String
    "str_len", "str_upper", "str_lower", "str_contains",
    "str_split", "str_trim", "str_replace", "to_string",
    # Tensor utilities
    "tanh", "tensor_slice", "slice",
])
