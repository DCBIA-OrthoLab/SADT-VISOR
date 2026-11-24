"""run_tool.py driven by another program, the way the 3D Slicer extension drives it.

Every test here runs REAL processes: small stand-in tools, each in a venv of its
own, under a tools folder named by `SADT_TOOLS_DIR`. What is under test is
process plumbing -- a result file written by a child, a failure carried up a
chain, a whole tree of processes stopped by one cancel -- and a mocked
`subprocess` can show none of it.

The venvs are bare (no pip): the stand-ins import nothing but the standard
library, so building them costs a second, not a download.
"""

import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
import venv
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))

import run_tool  # noqa: E402

STAND_INS = {
    # Reports progress on the server's channel and returns its output file.
    "Leaf": '''
        import json, os
        from pathlib import Path

        def run(output_dir: Path, label: str = "leaf") -> Path:
            """Write one file."""
            channel = os.environ.get("SADT_PROGRESS_FILE")
            for step in range(3):
                if channel:
                    with open(channel, "a") as handle:
                        handle.write(json.dumps({"fraction": step / 3, "message": "step %d" % step}) + "\\n")
            output_dir.mkdir(parents=True, exist_ok=True)
            produced = output_dir / (label + ".txt")
            produced.write_text(label)
            return produced
    ''',
    # Fails the way a tool refuses its input: a class of its own, a message for a clinician.
    "Boom": '''
        from pathlib import Path

        class ToolInputError(ValueError):
            pass

        def run(output_dir: Path) -> Path:
            """Refuse the input."""
            raise ToolInputError("the scan has no spacing")
    ''',
    # Calls another tool through its supervisor.
    "Caller": '''
        from pathlib import Path

        def run(output_dir: Path, callee: str = "Leaf", *, sup=None) -> Path:
            """Call another tool and keep what it produced."""
            # The callee writes into the supervisor's scratch, removed once this
            # run returns: what a tool keeps, it copies into its own output_dir.
            produced = sup.run(callee)
            output_dir.mkdir(parents=True, exist_ok=True)
            kept = output_dir / produced.name
            kept.write_text(produced.read_text())
            return kept
    ''',
    # Writes its pid, then never returns: the process a cancel has to reach.
    "Sleeper": '''
        import os, time
        from pathlib import Path

        def run(output_dir: Path, pid_file: str = "") -> Path:
            """Sleep."""
            Path(pid_file).write_text(str(os.getpid()))
            time.sleep(600)
            return output_dir
    ''',
    # Calls Sleeper through its supervisor: a chain with something to cancel at the bottom.
    "SleepCaller": '''
        from pathlib import Path

        def run(output_dir: Path, pid_file: str = "", *, sup=None) -> Path:
            """Call Sleeper."""
            return sup.run("Sleeper", pid_file=pid_file)
    ''',
    # Finds its own model under the DATA root, and takes no supervisor.
    "Owner": '''
        from pathlib import Path

        def run(output_dir: Path, *, data_root=None) -> Path:
            """Read its own model."""
            return Path(data_root) / "Owner" / "models" / "weights.txt"
    ''',
    # Gives its callee the second half of its own bar.
    "SpanCaller": '''
        from pathlib import Path

        def run(output_dir: Path, *, sup=None) -> Path:
            """Call Leaf inside a span."""
            produced = sup.run("Leaf", _progress=(0.5, 1.0))
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir
    ''',
    # Returns its output folder as text, as AREG_CBCT does.
    "TextResult": '''
        from pathlib import Path

        def run(output_dir: Path) -> Path:
            """Return a str."""
            output_dir.mkdir(parents=True, exist_ok=True)
            return str(output_dir)
    ''',
    # Prints a name in another alphabet: on Windows, without UTF-8 forced,
    # the redirected stdout is the locale's code page and this would raise.
    "Unicode": '''
        import os
        from pathlib import Path

        def run(output_dir: Path) -> Path:
            """Print a non-ASCII name and say whether UTF-8 mode is on."""
            output_dir.mkdir(parents=True, exist_ok=True)
            print("scan \u0141\u00f3d\u017a \u5199\u771f")
            (output_dir / "utf8.txt").write_text(os.environ.get("PYTHONUTF8", ""), encoding="utf-8")
            return output_dir
    ''',
    # Calls itself: a cycle the supervisor must refuse by name.
    "Loop": '''
        from pathlib import Path

        def run(output_dir: Path, *, sup=None) -> Path:
            """Call itself."""
            return sup.run("Loop")
    ''',
}


def _make_venv(target):
    """A bare venv for a stand-in. With uv when it is there, as a real tool's is:
    the interpreter DRIVING the run may be an old one (3D Slicer embeds 3.9), and
    a venv made from some standalone 3.9 builds cannot start at all."""
    if shutil.which("uv"):
        subprocess.run(
            ["uv", "venv", "--quiet", "--python", "3.12", str(target)], check=True
        )
    else:
        venv.EnvBuilder(with_pip=False).create(str(target))


@pytest.fixture(scope="module")
def tools_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("tools")
    for name, body in STAND_INS.items():
        tool = root / name
        package = tool / "src" / ("sadt_" + name.lower())
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(textwrap.dedent(body))
        (tool / "pyproject.toml").write_text('[project]\nname = "sadt-{}"\n'.format(name.lower()))
        _make_venv(tool / ".venv")
    return root


@pytest.fixture
def tools(tools_dir, monkeypatch):
    monkeypatch.setattr(run_tool, "TOOLS_DIR", tools_dir)
    monkeypatch.delenv(run_tool.PROGRESS_VARIABLE, raising=False)
    monkeypatch.delenv(run_tool.SUPERVISOR_CHAIN_VARIABLE, raising=False)
    return tools_dir


def _finish(job, timeout=60):
    job.wait(timeout=timeout)
    return job


# ---------------------------------------------------------------------------
# A run started from another program
# ---------------------------------------------------------------------------

def test_a_started_run_returns_its_result(tools, tmp_path):
    job = _finish(run_tool.start("Leaf", {"output_dir": tmp_path / "out"}, tmp_path / "job"))

    assert job.poll() == 0, job.log_file.read_text()
    produced = job.result()
    assert produced == tmp_path / "out" / "leaf.txt"
    assert produced.read_text() == "leaf"


def test_progress_is_read_from_the_job_folder(tools, tmp_path):
    job = _finish(run_tool.start("Leaf", {"output_dir": tmp_path / "out"}, tmp_path / "job"))

    assert [event["message"] for event in job.new_progress()] == ["step 0", "step 1", "step 2"]
    # Each event is handed over once.
    assert job.new_progress() == []


def test_a_failure_is_raised_under_the_tools_own_name(tools, tmp_path):
    job = _finish(run_tool.start("Boom", {"output_dir": tmp_path / "out"}, tmp_path / "job"))

    assert job.poll() == 1
    with pytest.raises(run_tool.ToolFailed) as raised:
        job.result()
    assert type(raised.value).__name__ == "ToolInputError"
    assert str(raised.value) == "the scan has no spacing"


def test_a_nested_failure_reaches_the_top_with_its_type_and_message(tools, tmp_path):
    """Caller -> Boom: the clinician reads Boom's reason, not "Caller failed"."""
    job = _finish(run_tool.start(
        "Caller", {"output_dir": tmp_path / "out", "callee": "Boom"}, tmp_path / "job"
    ))

    with pytest.raises(run_tool.ToolFailed) as raised:
        job.result()
    assert type(raised.value).__name__ == "ToolInputError"
    assert str(raised.value) == "the scan has no spacing"


def test_a_nested_run_returns_the_callees_result(tools, tmp_path):
    job = _finish(run_tool.start("Caller", {"output_dir": tmp_path / "out"}, tmp_path / "job"))

    assert job.poll() == 0, job.log_file.read_text()
    assert job.result().read_text() == "leaf"


def test_a_cycle_is_refused_by_name(tools, tmp_path):
    job = _finish(run_tool.start("Loop", {"output_dir": tmp_path / "out"}, tmp_path / "job"))

    with pytest.raises(run_tool.ToolFailed) as raised:
        job.result()
    assert type(raised.value).__name__ == "RecursionError"
    assert "Loop -> Loop" in str(raised.value)


def test_an_unknown_tool_is_reported_in_the_result_file(tools, tmp_path):
    """Layer 1 fails before any tool runs: the caller still gets a reason."""
    result = tmp_path / "result.json"
    code = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_tool.py"), "NoSuchTool", "--result-file", str(result)],
        env=dict(os.environ, **{run_tool.TOOLS_DIR_VARIABLE: str(tools)}),
        capture_output=True,
    ).returncode

    assert code == 2
    payload = json.loads(result.read_text())
    assert payload["kind"] == "error"
    assert payload["type"] == "ToolNotBuilt"
    assert "NoSuchTool" in payload["message"]


def test_the_variables_of_an_embedding_python_are_not_inherited(tools, tmp_path):
    """3D Slicer sets PYTHONHOME/PYTHONPATH for its own interpreter. Inherited by
    a tool's venv they would load Slicer's standard library into it."""
    environment = dict(os.environ, PYTHONHOME="/nonexistent/python/home", PYTHONPATH="/nonexistent")
    job = _finish(run_tool.start(
        "Leaf", {"output_dir": tmp_path / "out"}, tmp_path / "job", env=environment
    ))

    assert job.poll() == 0, job.log_file.read_text()


def test_the_data_root_is_injected_and_no_supervisor_with_it(tools, tmp_path):
    """A tool that declares only `data_root` is handed it, and is NOT handed a
    supervisor it did not ask for."""
    job = _finish(run_tool.start(
        "Owner", {"output_dir": tmp_path / "out"}, tmp_path / "job", data_dir=tmp_path / "DATA"
    ))

    assert job.poll() == 0, job.log_file.read_text()
    assert job.result() == tmp_path / "DATA" / "Owner" / "models" / "weights.txt"


# ---------------------------------------------------------------------------
# Progress of a chain
# ---------------------------------------------------------------------------

def test_a_fraction_is_folded_into_the_callers_span():
    assert run_tool.fold_fraction(0.0, (0.5, 1.0)) == 0.5
    assert run_tool.fold_fraction(0.5, (0.2, 0.6)) == pytest.approx(0.4)
    assert run_tool.fold_fraction(2.0, (0.2, 0.6)) == pytest.approx(0.6)


def test_without_a_span_a_callee_moves_no_bar():
    assert run_tool.fold_fraction(0.3, None) is None
    assert run_tool.fold_fraction(None, (0.2, 0.6)) is None


def test_a_callees_progress_lands_inside_its_span(tools, tmp_path):
    """The bar a GUI draws must never run backwards because a callee started at 0."""
    job = _finish(run_tool.start("SpanCaller", {"output_dir": tmp_path / "out"}, tmp_path / "job"))

    assert job.poll() == 0, job.log_file.read_text()
    events = [event for event in job.new_progress() if event["message"].startswith("step")]
    assert [event["message"] for event in events] == ["step 0", "step 1", "step 2"]
    assert [event["fraction"] for event in events] == pytest.approx([0.5, 0.6667, 0.8333], abs=1e-3)


def test_a_callee_without_a_span_reports_messages_only(tools, tmp_path):
    job = _finish(run_tool.start("Caller", {"output_dir": tmp_path / "out"}, tmp_path / "job"))

    events = [event for event in job.new_progress() if event["message"].startswith("step")]
    assert [event["message"] for event in events] == ["step 0", "step 1", "step 2"]
    assert all(event["fraction"] is None for event in events)


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------

def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie still answers signal 0 until it is reaped; it is not running.
    stat = Path("/proc/{}/stat".format(pid))
    if stat.is_file():
        return stat.read_text().rsplit(")", 1)[-1].split()[0] != "Z"
    return True


def _wait_for(path, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file() and path.read_text():
            return int(path.read_text())
        time.sleep(0.05)
    raise AssertionError("{} never appeared".format(path))


@pytest.mark.skipif(os.name == "nt", reason="the POSIX process-group path")
def test_cancel_stops_the_whole_chain(tools, tmp_path):
    """SleepCaller -> Sleeper: cancelling the run must not leave Sleeper running.
    It runs in a process of its own, two re-execs below the one `start()` made."""
    pid_file = tmp_path / "sleeper.pid"
    job = run_tool.start(
        "SleepCaller", {"output_dir": tmp_path / "out", "pid_file": str(pid_file)}, tmp_path / "job"
    )
    sleeper = _wait_for(pid_file)
    assert sleeper != job.process.pid

    started = time.monotonic()
    job.cancel(grace=0.5)

    assert job.poll() is not None
    deadline = time.monotonic() + 5
    while _alive(sleeper) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(sleeper)
    assert time.monotonic() - started < 5


def test_a_reused_job_folder_does_not_answer_with_an_old_result(tools, tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    run_tool.write_result(job_dir / "result.json", {"kind": "path", "value": "/stale"})

    job = _finish(run_tool.start("Boom", {"output_dir": tmp_path / "out"}, job_dir))

    with pytest.raises(run_tool.ToolFailed):
        job.result()


def test_a_path_returned_as_text_is_a_path(tools, tmp_path):
    """AREG_CBCT returns its folder as a str: the run did finish and must say so."""
    job = _finish(run_tool.start("TextResult", {"output_dir": tmp_path / "out"}, tmp_path / "job"))

    assert job.poll() == 0, job.log_file.read_text()
    assert job.result() == tmp_path / "out"


def test_a_result_that_is_no_path_is_refused_by_name():
    with pytest.raises(TypeError) as raised:
        run_tool.result_payload(42)
    assert "int" in str(raised.value)


def test_a_tool_runs_in_utf8_mode_and_may_print_any_alphabet(tools, tmp_path):
    job = _finish(run_tool.start("Unicode", {"output_dir": tmp_path / "out"}, tmp_path / "job"))

    assert job.poll() == 0, job.log_file.read_text(encoding="utf-8")
    assert (tmp_path / "out" / "utf8.txt").read_text(encoding="utf-8") == "1"
    assert "\u0141\u00f3d\u017a" in job.log_file.read_text(encoding="utf-8")
