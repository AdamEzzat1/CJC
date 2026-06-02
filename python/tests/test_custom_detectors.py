"""Pytest suite for the Python custom-detector bridge (ADR-0041).

Run with:

    cd python
    .\\.venv\\Scripts\\python.exe -m pytest tests/test_custom_detectors.py -v

These tests cover the Python-side API mirror of the Rust integration tests:

- A simple Python detector subclass produces findings.
- Namespace enforcement: codes outside E9500..=E9999 raise at registration.
- Axis-name enforcement: unknown axis names raise.
- The sink's `emit()` rejects empty messages and non-Info severities on
  axes-empty detectors (matching the Rust contract).
- Two consecutive runs over the same input produce byte-identical JSON.
- Detector ordering: same set of detectors in different list order
  produces the same report.
- The detector sees the right column names, n_rows, and column types via
  the read-only DataFrame view.
- Determinism survives Python-side use of dict, set (i.e. potentially
  hash-randomized iteration): we shuffle the emission order inside the
  detector and verify the final JSON is unchanged.
"""

from __future__ import annotations

import json
import random
from typing import Any, List

import pytest

import cjc_locke


# ── Sample detectors ──────────────────────────────────────────────────────────


class _NamePatternDetector(cjc_locke.CustomDetector):
    """Flag any column whose name starts with one of the configured prefixes."""

    def __init__(self, code: str, axes: List[str], prefixes: List[str], severity: str = "warning"):
        self._code = code
        self._axes = axes
        self._prefixes = prefixes
        self._severity = severity

    def code(self) -> str:
        return self._code

    def belief_axes(self) -> List[str]:
        return self._axes

    def run(self, df, sink) -> None:
        for name in df.column_names():
            if any(name.startswith(p) for p in self._prefixes):
                sink.emit(
                    self._severity,
                    f"`{name}` matches a configured prefix",
                    column=name,
                    sample_size=df.n_rows,
                )


class _AdvisoryDetector(cjc_locke.CustomDetector):
    """Info-severity detector with no belief axes — passes through the
    'advisory-only' code path in the sink."""

    def code(self) -> str:
        return "E9501"

    def belief_axes(self) -> List[str]:
        return []

    def run(self, df, sink) -> None:
        sink.emit("info", f"n_rows = {df.n_rows}", sample_size=df.n_rows)


class _ShuffleEmissionDetector(cjc_locke.CustomDetector):
    """Emits findings in a randomised order each run. The framework's
    canonical sort on the Rust side must mask the shuffle so the final
    report bytes stay identical across invocations."""

    def __init__(self, seed: int):
        self._rng = random.Random(seed)

    def code(self) -> str:
        return "E9502"

    def belief_axes(self) -> List[str]:
        return ["leakage"]

    def run(self, df, sink) -> None:
        names = list(df.column_names())
        self._rng.shuffle(names)
        for n in names:
            sink.emit(
                "warning",
                f"shuffle finding on `{n}`",
                column=n,
                sample_size=df.n_rows,
            )


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def lc_like_data():
    """A small dict-shaped 'fake LendingClub' frame: 4 columns including
    two that match the post-origination pattern."""
    return {
        "loan_amnt": [1000.0, 2000.0, 3000.0],
        "total_pymnt": [0.5, 1.0, 2.5],
        "total_rec_int": [0.1, 0.2, 0.3],
        "dti": [10.0, 20.0, 30.0],
    }


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_simple_detector_contributes_findings(lc_like_data):
    detector = _NamePatternDetector(
        code="E9500",
        axes=["leakage"],
        prefixes=["total_"],
    )
    report = cjc_locke.validate(
        lc_like_data,
        label="test",
        custom_detectors=[detector],
    )
    findings = [
        f for f in report.to_dict()["findings"] if f["code"] == "E9500"
    ]
    columns = {f["column"] for f in findings}
    assert columns == {"total_pymnt", "total_rec_int"}, columns


def test_axis_assignment_lands_in_report(lc_like_data):
    detector = _NamePatternDetector(
        code="E9500",
        axes=["leakage"],
        prefixes=["total_"],
    )
    report = cjc_locke.validate(
        lc_like_data,
        custom_detectors=[detector],
    )
    # The belief leakage axis should be reduced (not 1.0) by the warning
    # findings on the post-origination columns.
    belief = cjc_locke.belief_report(report)
    assert belief.leakage_score < 1.0
    # Other axes unaffected.
    assert belief.drift_score == 1.0
    assert belief.lineage_score == 1.0


def test_advisory_detector_does_not_affect_belief(lc_like_data):
    detector = _AdvisoryDetector()
    report = cjc_locke.validate(
        lc_like_data,
        custom_detectors=[detector],
    )
    findings = [f for f in report.to_dict()["findings"] if f["code"] == "E9501"]
    assert len(findings) == 1
    assert findings[0]["severity"] == "info"
    belief = cjc_locke.belief_report(report)
    # All belief axes should still be at their baseline because the
    # advisory detector declares no axes.
    assert belief.leakage_score == 1.0
    assert belief.drift_score == 1.0


def test_invalid_code_raises_at_registration(lc_like_data):
    """A detector with a code in the built-in range must be rejected at
    registration time (PyValueError raised from validate())."""

    class _BadCodeDetector(cjc_locke.CustomDetector):
        def code(self):
            return "E9001"  # built-in range

        def belief_axes(self):
            return ["leakage"]

        def run(self, df, sink):
            pass

    with pytest.raises(ValueError) as exc:
        cjc_locke.validate(lc_like_data, custom_detectors=[_BadCodeDetector()])
    assert "E9001" in str(exc.value) or "namespace" in str(exc.value).lower()


def test_malformed_code_raises(lc_like_data):
    class _MalformedCodeDetector(cjc_locke.CustomDetector):
        def code(self):
            return "X9500"  # wrong prefix

        def belief_axes(self):
            return ["leakage"]

        def run(self, df, sink):
            pass

    with pytest.raises(ValueError):
        cjc_locke.validate(lc_like_data, custom_detectors=[_MalformedCodeDetector()])


def test_unknown_axis_raises(lc_like_data):
    class _UnknownAxis(cjc_locke.CustomDetector):
        def code(self):
            return "E9500"

        def belief_axes(self):
            return ["nonsense"]

        def run(self, df, sink):
            pass

    with pytest.raises(ValueError) as exc:
        cjc_locke.validate(lc_like_data, custom_detectors=[_UnknownAxis()])
    assert "nonsense" in str(exc.value)


def test_sink_rejects_empty_message(lc_like_data):
    class _EmptyMessage(cjc_locke.CustomDetector):
        def code(self):
            return "E9500"

        def belief_axes(self):
            return ["leakage"]

        def run(self, df, sink):
            sink.emit("warning", "")  # empty message
            assert sink.n_pending == 0
            assert sink.last_error is not None and "empty" in sink.last_error.lower()
            # The rest of the suite should still get a real finding.
            sink.emit("warning", "this one is fine", column="dti")

    report = cjc_locke.validate(lc_like_data, custom_detectors=[_EmptyMessage()])
    e9500 = [f for f in report.to_dict()["findings"] if f["code"] == "E9500"]
    assert len(e9500) == 1
    assert e9500[0]["column"] == "dti"


def test_sink_rejects_unknown_severity(lc_like_data):
    class _BadSeverity(cjc_locke.CustomDetector):
        def code(self):
            return "E9500"

        def belief_axes(self):
            return ["leakage"]

        def run(self, df, sink):
            # Unknown severity raises PyValueError but the wrapper
            # catches it and surfaces via stderr — the validate call
            # itself does not propagate it (a detector exception does
            # not abort the whole audit).
            try:
                sink.emit("disaster", "nope")
            except ValueError:
                pass

    report = cjc_locke.validate(lc_like_data, custom_detectors=[_BadSeverity()])
    e9500 = [f for f in report.to_dict()["findings"] if f["code"] == "E9500"]
    assert e9500 == []


def test_determinism_two_runs_byte_identical(lc_like_data):
    detector = _NamePatternDetector(
        code="E9500",
        axes=["leakage"],
        prefixes=["total_"],
    )
    r1 = cjc_locke.validate(lc_like_data, custom_detectors=[detector])
    r2 = cjc_locke.validate(lc_like_data, custom_detectors=[detector])
    assert r1.to_json() == r2.to_json()


def test_determinism_under_emission_shuffle(lc_like_data):
    """The Python detector emits findings in a random order each call.
    The Rust framework sorts them canonically after `run()` returns, so
    the final JSON must be the same across invocations even though the
    raw emission order differs."""
    d1 = _ShuffleEmissionDetector(seed=1)
    d2 = _ShuffleEmissionDetector(seed=2)  # different shuffle seed
    r1 = cjc_locke.validate(lc_like_data, custom_detectors=[d1])
    r2 = cjc_locke.validate(lc_like_data, custom_detectors=[d2])
    assert r1.to_json() == r2.to_json()


def test_detector_list_order_does_not_affect_output(lc_like_data):
    d_total = _NamePatternDetector(
        code="E9500",
        axes=["leakage"],
        prefixes=["total_"],
    )
    d_loan = _NamePatternDetector(
        code="E9501",
        axes=["schema"],
        prefixes=["loan_"],
    )
    r_a = cjc_locke.validate(lc_like_data, custom_detectors=[d_total, d_loan])
    r_b = cjc_locke.validate(lc_like_data, custom_detectors=[d_loan, d_total])
    assert r_a.to_json() == r_b.to_json()


def test_dataframe_view_exposes_metadata(lc_like_data):
    seen = {}

    class _Introspect(cjc_locke.CustomDetector):
        def code(self):
            return "E9500"

        def belief_axes(self):
            return ["leakage"]

        def run(self, df, sink):
            seen["n_rows"] = df.n_rows
            seen["n_cols"] = df.n_cols
            seen["names"] = df.column_names()
            seen["loan_amnt_type"] = df.column_type("loan_amnt")
            seen["missing_type"] = df.column_type("does_not_exist")
            seen["loan_amnt_data"] = df.get_float("loan_amnt")
            # Emit one finding so the run is "useful" from the framework's
            # perspective (not strictly required).
            sink.emit("warning", "introspection complete", column="loan_amnt")

    cjc_locke.validate(lc_like_data, custom_detectors=[_Introspect()])
    assert seen["n_rows"] == 3
    assert seen["n_cols"] == 4
    assert "loan_amnt" in seen["names"]
    assert seen["loan_amnt_type"] == "Float"
    assert seen["missing_type"] is None
    assert seen["loan_amnt_data"] == [1000.0, 2000.0, 3000.0]


def test_no_custom_detectors_is_backward_compatible(lc_like_data):
    """Calling validate() without custom_detectors must produce a
    report byte-identical to the pre-v0.8 behaviour (validated by the
    fact that NO E95** findings exist)."""
    report = cjc_locke.validate(lc_like_data)
    for f in report.to_dict()["findings"]:
        code = f["code"]
        if code.startswith("E"):
            n = int(code[1:])
            assert n < 9500, f"unexpected custom-namespace finding {code}"


def test_belief_score_belief_axis_routing(lc_like_data):
    """A detector that declares both schema AND leakage axes should
    affect both scores."""

    class _TwoAxis(cjc_locke.CustomDetector):
        def code(self):
            return "E9500"

        def belief_axes(self):
            return ["leakage", "schema"]

        def run(self, df, sink):
            sink.emit("error", "two-axis finding", column="loan_amnt", sample_size=df.n_rows)

    report = cjc_locke.validate(lc_like_data, custom_detectors=[_TwoAxis()])
    belief = cjc_locke.belief_report(report)
    assert belief.leakage_score < 1.0
    assert belief.schema_score < 1.0
    # Drift and lineage still untouched.
    assert belief.drift_score == 1.0
