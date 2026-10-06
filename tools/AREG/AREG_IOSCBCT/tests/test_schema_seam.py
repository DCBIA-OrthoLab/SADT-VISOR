"""The arguments this tool sends are the arguments the callees publish.

`tools.py` names four tools by string and sends each a dict of parameters. A
tool cannot import another -- separate virtualenvs are the whole reason the
split exists -- so nothing in this repository type-checks that seam. A renamed
argument on the far side breaks a chain silently, an hour into a job, inside a
child process.

This reads all four schemas OUT OF PROCESS, the way the server does: each
tool's own `describe.py`, run by that tool's own interpreter. It skips per tool
when the tool is not built, because CI builds each one in its own job and a
contributor working here has no reason to have built the others.

`sadt_testkit.tool_schema` is not used: it looks a tool up at
`tools/<name>`, and ALI_CBCT and ALI_IOS live under the `ALI/` grouping folder.
Resolving the directory is four lines, and having them here keeps this test
running for exactly the tools it is about.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from sadt_areg_ioscbct import tools


def repo_root():
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "tools").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError("could not find the checkout holding tools/ and scripts/")


def tool_directory(name):
    """`tools/<name>` or `tools/<group>/<name>`, whichever exists."""
    root = repo_root() / "tools"
    for candidate in [root / name] + sorted(root.glob("*/" + name)):
        if candidate.is_dir():
            return candidate
    return None


def schema_of(name):
    """What `GET /tools` would publish for `name`, or a skip."""
    directory = tool_directory(name)
    if directory is None:
        pytest.skip(f"there is no tool called {name} under tools/")
    interpreter = directory / ".venv" / "bin" / "python"
    if not interpreter.is_file():
        pytest.skip(f"run `uv sync` in {directory.relative_to(repo_root())}")

    completed = subprocess.run(
        [str(interpreter), str(repo_root() / "scripts" / "describe.py"), str(directory)],
        capture_output=True, text=True, timeout=300,
    )
    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr)
        pytest.fail(f"describe.py refused {name}")
    return json.loads(completed.stdout)


class RecordingSup:
    """Records the parameters of one call and produces a directory."""

    def __init__(self, tmp_path):
        self.tmp = tmp_path / "tmp"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.calls = []
        self.spans = []

    def run(self, tool, **params):
        # `_progress` is the supervisor's, removed before the callee sees it,
        # so it is never among the arguments checked against a schema.
        self.spans.append(params.pop("_progress", None))
        self.calls.append((tool, params))
        produced = self.tmp / "produced"
        produced.mkdir(exist_ok=True)
        return produced

    def progress(self, fraction, message):
        pass

    def log(self, message, level="info", user=False):
        pass


def assert_sends_only_declared_arguments(name, params):
    schema = schema_of(name)
    declared = set(schema["arguments"])
    unknown = sorted(set(params) - declared)
    assert not unknown, (
        f"AREG_IOSCBCT sends {unknown} to {name}, which declares {sorted(declared)}"
    )
    required = {
        argument for argument, spec in schema["arguments"].items() if spec.get("required")
    }
    # `output_dir` is filled in by the server at dispatch, not by a caller.
    missing = sorted(required - set(params) - {"output_dir"})
    assert not missing, f"AREG_IOSCBCT does not send {name} its required {missing}"


def test_the_crown_segmentation_request_matches_crown_segs_schema(tmp_path):
    sup = RecordingSup(tmp_path)
    tools.label_crowns(sup, str(tmp_path / "ios"), "/models/crown")
    assert_sends_only_declared_arguments("Crown_Seg", sup.calls[0][1])


def test_the_cbct_landmark_request_matches_ali_cbcts_schema(tmp_path):
    sup = RecordingSup(tmp_path)
    tools.predict_cbct_landmarks(sup, str(tmp_path / "cbct"), "/models/ali")
    assert_sends_only_declared_arguments("ALI_CBCT", sup.calls[0][1])


def test_the_intraoral_landmark_request_matches_ali_ioss_schema(tmp_path):
    sup = RecordingSup(tmp_path)
    tools.predict_ios_landmarks(sup, str(tmp_path / "ios"), "/models/ali")
    assert_sends_only_declared_arguments("ALI_IOS", sup.calls[0][1])


def test_the_orientation_request_matches_asos_schema(tmp_path):
    sup = RecordingSup(tmp_path)
    tools.orient_cbct(sup, str(tmp_path / "cbct"), "/models/gold", "/models/ali")
    assert_sends_only_declared_arguments("ASO", sup.calls[0][1])


def test_occlusal_is_a_network_ali_ios_really_offers():
    """The one value in the whole seam that is neither an argument name nor a
    path: it names one of ALI_IOS's landmark families, and a family renamed
    there would be a 422 three tools down."""
    schema = schema_of("ALI_IOS")
    assert "Occlusal" in schema["arguments"]["networks"]["choices"]


def test_every_cbct_landmark_it_asks_for_is_one_ali_cbct_can_predict():
    """Asked for BY NAME rather than by region, so a landmark ALI no longer
    catalogs would be silently absent from the result instead of refused."""
    schema = schema_of("ALI_CBCT")
    offered = set(schema["arguments"]["landmarks"]["choices"])
    assert set(tools.CBCT_LANDMARKS) <= offered


def test_the_modes_it_asks_aso_for_are_modes_aso_has():
    schema = schema_of("ASO")
    assert "CBCT" in schema["arguments"]["modality"]["choices"]
    assert "Fully-Automated" in schema["arguments"]["automation"]["choices"]


def test_every_tool_it_names_is_declared_in_its_own_published_calls():
    """`describe.py` reads `tools.py` to publish the `calls` list the server
    checks at startup. A tool called but not published is a chain the server
    cannot validate."""
    schema = schema_of("AREG_IOSCBCT")
    assert set(schema["calls"]) == {"Crown_Seg", "ALI_CBCT", "ALI_IOS", "ASO"}
