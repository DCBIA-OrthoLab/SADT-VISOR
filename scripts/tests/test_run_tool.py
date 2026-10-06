"""run_tool.py builds a CLI from a signature and chains tools through venvs.

The pure parts are tested here directly. The parts that need a built tool run
`_template` for real -- and skip rather than fail when it has no venv, because a
contributor working on one tool has no reason to have built the others.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
REPO_ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

import run_tool  # noqa: E402


# ---------------------------------------------------------------------------
# Coercion -- the rule an optional path depends on
# ---------------------------------------------------------------------------

def test_an_empty_path_stays_empty():
    """`Path("")` is `PosixPath(".")`, the current directory, and it is TRUTHY.

    Coercing the "not supplied" default of an optional path therefore hands the
    tool a real directory: ASO read `landmarks=""` as a supplied landmark folder
    and walked the whole checkout, `.venv` included. Calling `run()` directly in
    Python keeps the `""` the signature declares, so coercing it here is also
    what makes the two call paths disagree.
    """
    assert run_tool.coerce("", Path) == ""
    assert not run_tool.coerce("", Path)
    # And a real path still becomes one.
    assert run_tool.coerce("out/", Path) == Path("out")


def test_lists_of_paths_are_coerced_elementwise():
    from typing import List

    assert run_tool.coerce(["a", "b"], List[Path]) == [Path("a"), Path("b")]


def test_scalars_are_left_alone():
    assert run_tool.coerce(3, int) == 3
    assert run_tool.coerce("cuda", str) == "cuda"


# ---------------------------------------------------------------------------
# The parser, built from a signature
# ---------------------------------------------------------------------------

def _run_with(body):
    """Compile a `run()` and hand it back, so a test can state its own shape."""
    namespace = {}
    exec(compile(textwrap.dedent(body), "<fixture>", "exec"), namespace)
    return namespace["run"]


SIMPLE = """
    from pathlib import Path
    from typing import Literal

    def run(scans: Path, output_dir: Path, device: Literal["cuda", "cpu"] = "cuda",
            crop: bool = False, keep: bool = True, margin: float = 1.5,
            structures: list[Literal["MAND", "MAX"]] = ["MAND"]) -> Path:
        \"\"\"Segment things.\"\"\"
        return output_dir
"""


def test_options_are_named_after_the_arguments():
    parser = run_tool.build_parser(_run_with(SIMPLE), "Fixture")
    parsed = parser.parse_args(["--scans", "in/", "--output-dir", "out/"])

    assert parsed.scans == "in/"
    assert parsed.output_dir == "out/"
    # Underscores become dashes on the command line, never in the dest.
    assert run_tool.option_name("output_dir") == "--output-dir"


def test_an_argument_without_a_default_is_required():
    parser = run_tool.build_parser(_run_with(SIMPLE), "Fixture")
    with pytest.raises(SystemExit):
        parser.parse_args(["--scans", "in/"])  # no --output-dir


def test_defaults_come_from_the_signature():
    parser = run_tool.build_parser(_run_with(SIMPLE), "Fixture")
    parsed = parser.parse_args(["--scans", "in/", "--output-dir", "out/"])

    assert parsed.device == "cuda"
    assert parsed.margin == 1.5
    assert parsed.structures == ["MAND"]


def test_a_literal_is_validated_as_choices():
    parser = run_tool.build_parser(_run_with(SIMPLE), "Fixture")
    with pytest.raises(SystemExit):
        parser.parse_args(["--scans", "in/", "--output-dir", "out/", "--device", "tpu"])


def test_a_bool_gets_a_flag_and_its_negation():
    """A tool's default may be either, so `--no-keep` has to exist for
    `keep=True` just as `--crop` does for `crop=False`."""
    parser = run_tool.build_parser(_run_with(SIMPLE), "Fixture")
    base = ["--scans", "in/", "--output-dir", "out/"]

    assert parser.parse_args(base).crop is False
    assert parser.parse_args(base + ["--crop"]).crop is True
    assert parser.parse_args(base).keep is True
    assert parser.parse_args(base + ["--no-keep"]).keep is False


def test_a_long_option_list_is_validated_but_not_printed():
    """ALI publishes 119 landmarks and argparse spells every one of them into
    the usage line, which makes `--help` unreadable."""
    many = ", ".join('"L{}"'.format(index) for index in range(30))
    run = _run_with("""
        from pathlib import Path
        from typing import Literal

        def run(output_dir: Path, points: list[Literal[{}]] = []) -> Path:
            \"\"\"Place points.\"\"\"
            return output_dir
    """.format(many))
    parser = run_tool.build_parser(run, "Fixture")

    assert "L0,L1" not in parser.format_usage().replace(" ", "")
    # Still refused, though.
    with pytest.raises(SystemExit):
        parser.parse_args(["--output-dir", "out/", "--points", "NOPE"])


# ---------------------------------------------------------------------------
# The supervisor
# ---------------------------------------------------------------------------

SUPERVISED = """
    from pathlib import Path

    def run(scans: Path, output_dir: Path, *, sup=None) -> Path:
        \"\"\"Orient things.\"\"\"
        return output_dir
"""


def test_a_supervisor_is_detected_and_kept_off_the_command_line():
    run = _run_with(SUPERVISED)
    assert run_tool.takes_supervisor(run) is True

    parser = run_tool.build_parser(run, "Fixture")
    assert "--sup" not in parser.format_usage()
    with pytest.raises(SystemExit):
        parser.parse_args(["--scans", "in/", "--output-dir", "out/", "--sup", "x"])


def test_a_tool_without_one_is_not_given_one():
    assert run_tool.takes_supervisor(_run_with(SIMPLE)) is False


def test_the_supervisor_exposes_exactly_the_five_members(tmp_path):
    """The frozen interface. A tool cannot tell this from the server's."""
    sup = run_tool.LocalSupervisor(out=tmp_path / "out", tmp=tmp_path)

    for member in ("run", "out", "tmp", "progress", "log"):
        assert hasattr(sup, member), member
    assert sup.out.is_absolute() and sup.tmp.is_absolute()


def test_the_supervisor_log_takes_a_level_and_an_audience(tmp_path, capsys):
    """The server's `sup.log(message, level=, user=)`. One argument is still
    an info line, which is what every existing caller writes."""
    sup = run_tool.LocalSupervisor(out=tmp_path / "out", tmp=tmp_path)

    sup.log("plain")
    sup.log("3 of 7 landmarks found", level="warning")
    sup.log("scan 4 skipped", level="warning", user=True)

    lines = capsys.readouterr().err.splitlines()
    assert lines == ["[sup] INFO: plain", "[sup] WARNING: 3 of 7 landmarks found",
                     "[sup] (user) WARNING: scan 4 skipped"]


def test_a_callers_span_is_never_passed_to_the_callee(tmp_path, monkeypatch):
    """`_progress=(start, end)` belongs to the supervisor, as on the server:
    the callee declares no such argument and would refuse it."""
    sent = {}

    def fake_run(command, **_):
        sent.update(json.loads(Path(command[command.index("--params-file") + 1]).read_text()))
        result = Path(command[command.index("--result-file") + 1])
        result.write_text(json.dumps({"kind": "path", "value": str(tmp_path)}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_tool.subprocess, "run", fake_run)
    sup = run_tool.LocalSupervisor(out=tmp_path / "out", tmp=tmp_path)

    sup.run("Leaf", scans="in", _progress=(0.2, 0.6))

    assert sent == {"scans": "in"}


def test_supervisor_paths_are_absolute(tmp_path, monkeypatch):
    """The callee starts from a NEUTRAL working directory -- which is what makes
    "writes only under output_dir" testable -- so a relative path handed across
    that boundary resolves against the wrong place."""
    monkeypatch.chdir(tmp_path)
    sup = run_tool.LocalSupervisor(out="out", tmp=".")

    assert sup.out.is_absolute()
    assert sup.tmp.is_absolute()


def test_an_unknown_tool_says_what_exists():
    with pytest.raises(run_tool.ToolNotBuilt) as raised:
        run_tool.tool_python("NoSuchTool")
    assert "NoSuchTool" in str(raised.value)
    assert "_template" in str(raised.value)


# ---------------------------------------------------------------------------
# End to end, through a real venv
# ---------------------------------------------------------------------------

def _built(name):
    try:
        run_tool.tool_python(name)
    except run_tool.ToolNotBuilt:
        return False
    return True


needs_template = pytest.mark.skipif(
    not _built("_template"), reason="run `cd tools/_template && uv sync` first"
)


@needs_template
def test_a_tool_runs_in_its_own_venv(tmp_path):
    import numpy as np

    scans = tmp_path / "scans"
    scans.mkdir()
    np.save(scans / "a.npy", np.arange(100, dtype=np.float32))

    completed = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_tool.py"), "_template",
         "--scans", str(scans), "--output-dir", str(tmp_path / "out")],
        capture_output=True, text=True,
    )

    assert completed.returncode == 0, completed.stderr
    produced = Path(completed.stdout.strip())
    assert produced.is_file()
    assert "mean=50.0" in produced.read_text()


@needs_template
def test_the_tool_runs_under_its_own_interpreter_not_the_callers(tmp_path):
    """uv's venvs all symlink `bin/python` to the same CPython, so comparing
    resolved binaries makes every tool look like every other one -- and the
    callee would be imported into its CALLER's environment, which is exactly
    the dependency mixing this repository exists to prevent."""
    completed = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_tool.py"), "_template",
         "--scans", str(tmp_path), "--output-dir", str(tmp_path / "out"),
         "--print-prefix"],
        capture_output=True, text=True,
    )
    # The option does not exist; what matters is that the FAILURE came from the
    # tool's own parser, i.e. the re-exec happened.
    assert completed.returncode != 0
    assert "run_tool.py _template" in completed.stderr


# ---------------------------------------------------------------------------
# Progress -- the channel the server uses, exercised from a checkout
# ---------------------------------------------------------------------------

def test_progress_is_a_no_op_when_no_one_asked_for_it(monkeypatch):
    monkeypatch.delenv(run_tool.PROGRESS_VARIABLE, raising=False)

    assert run_tool.append_progress(0.5, "anything") is False


def test_an_event_is_one_json_line(tmp_path, monkeypatch):
    events = tmp_path / "events.jsonl"
    monkeypatch.setenv(run_tool.PROGRESS_VARIABLE, str(events))

    assert run_tool.append_progress(0.25, "scan 1 of 4") is True
    assert json.loads(events.read_text()) == {"fraction": 0.25, "message": "scan 1 of 4"}


def test_a_failure_to_report_is_swallowed(tmp_path, monkeypatch):
    """Telemetry must never be the reason a run fails."""
    monkeypatch.setenv(run_tool.PROGRESS_VARIABLE, str(tmp_path / "no" / "such" / "dir"))

    assert run_tool.append_progress(0.5, "into a directory that is not there") is False


def test_the_supervisor_reports_on_the_same_file_a_tool_does(tmp_path, monkeypatch):
    """`sup.progress` is one of the supervisor's five members and four tools
    already call it. It has to land where a tool's own `progress.py` lands, or
    a chain would report half of itself on one channel and half on the other."""
    events = tmp_path / "events.jsonl"
    monkeypatch.setenv(run_tool.PROGRESS_VARIABLE, str(events))
    sup = run_tool.LocalSupervisor(out=tmp_path / "out", tmp=tmp_path)

    sup.progress(0.2, "predicting landmarks with ALI_CBCT")

    assert json.loads(events.read_text()) == {
        "fraction": 0.2, "message": "predicting landmarks with ALI_CBCT",
    }


def test_a_nested_run_inherits_the_file_rather_than_making_its_own(tmp_path, monkeypatch):
    """A supervised child re-enters this script with the parent's file in its
    environment -- which is the whole implementation of chain progress. A second
    echo would print every event once per level of the chain."""
    monkeypatch.setenv(run_tool.PROGRESS_VARIABLE, str(tmp_path / "parent.jsonl"))

    stop = run_tool.watch_progress()

    assert os.environ[run_tool.PROGRESS_VARIABLE] == str(tmp_path / "parent.jsonl")
    stop()


@needs_template
def test_a_developer_sees_what_the_server_would(tmp_path):
    """Set by run_tool.py itself, so a tool's progress reporting is exercised
    from a checkout instead of being discovered to be silent once deployed."""
    import numpy as np

    scans = tmp_path / "scans"
    scans.mkdir()
    for name in ("a", "b", "c"):
        np.save(scans / f"{name}.npy", np.arange(10, dtype=np.float32))

    completed = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_tool.py"), "_template",
         "--scans", str(scans), "--output-dir", str(tmp_path / "out")],
        capture_output=True, text=True,
    )

    assert completed.returncode == 0, completed.stderr
    reported = [line for line in completed.stderr.splitlines() if "[progress]" in line]
    assert reported == [
        "[progress]  0% scan 1 of 3",
        "[progress] 33% scan 2 of 3",
        "[progress] 67% scan 3 of 3",
    ]
