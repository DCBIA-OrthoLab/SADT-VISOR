"""What ALI placed reaches the person who started the run.

Every per-scan count was written for the operator alone, so a clinician saw a
run finish without being told which landmarks it could not place.
"""

import json

from sadt_ali_cbct import dispatch, progress


def _report(*records):
    return {"cases": {f"k{i}": r for i, r in enumerate(records)}}


def test_one_line_per_scan_saying_how_many_were_placed():
    lines = dispatch.user_summary(_report(
        {"status": "ok", "landmarks_found": ["A", "B"], "landmarks_failed": {}},
        {"status": "ok", "landmarks_found": ["A"], "landmarks_failed": {"B": "no convergence"}},
        {"status": "failed"},
    ), "scan")
    assert lines == [
        ("info", "scan 1/3: 2 landmarks placed"),
        ("warning", "scan 2/3: 1 landmark placed, 1 not found (B)"),
        ("warning", "scan 3/3: no landmark placed"),
    ]


def test_a_large_cohort_names_only_what_needs_looking_at():
    complete = {"status": "ok", "landmarks_found": ["A"], "landmarks_failed": {}}
    partial = {"status": "ok", "landmarks_found": [], "landmarks_failed": ["A"]}
    lines = dispatch.user_summary(_report(*([complete] * 11 + [partial])), "scan")
    assert lines == [("warning", "scan 12/12: 0 landmarks placed, 1 not found (A)"),
                     ("info", "11 of 12 scans have every landmark")]


def test_the_lines_reach_the_user(tmp_path, monkeypatch):
    stream = tmp_path / "events.jsonl"
    monkeypatch.setenv(progress.VARIABLE, str(stream))
    dispatch._tell_the_user(_report(
        {"status": "ok", "landmarks_found": ["A"], "landmarks_failed": {}}), "scan")
    (record,) = [json.loads(line) for line in stream.read_text().splitlines()]
    assert record["audience"] == "user" and record["level"] == "info"
