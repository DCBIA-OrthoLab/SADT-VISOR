"""The arguments this tool sends are the arguments the callees publish.

`tools.py` names SIX tools by string and sends each a dict of parameters --
the widest call graph in this repository. A tool cannot import another --
separate virtualenvs are the whole reason the split exists -- so nothing here
type-checks that seam. A renamed argument on the far side breaks a chain
silently, an hour into a job, inside a child process.

This reads all six schemas OUT OF PROCESS, the way the server does: each tool's
own `describe.py`, run by that tool's own interpreter. It skips per tool when
the tool is not built, because CI builds each one in its own job and a
contributor working here has no reason to have built the others.

`sadt_testkit.tool_schema` is not used: it looks a tool up at `tools/<name>`,
and ALI_CBCT and AREG_CBCT live under grouping folders. Resolving the directory
is four lines, and having them here keeps this test running for exactly the
tools it is about.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from sadt_vface import catalogs, tools


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

    def run(self, tool, **params):
        self.calls.append((tool, params))
        produced = self.tmp / "produced"
        produced.mkdir(exist_ok=True)
        return produced

    def progress(self, fraction, message):
        pass


def assert_sends_only_declared_arguments(name, params):
    schema = schema_of(name)
    declared = set(schema["arguments"])
    unknown = sorted(set(params) - declared)
    assert not unknown, (
        f"VFACE sends {unknown} to {name}, which declares {sorted(declared)}"
    )
    required = {
        argument for argument, spec in schema["arguments"].items() if spec.get("required")
    }
    # `output_dir` is filled in by the server at dispatch, not by a caller.
    missing = sorted(required - set(params) - {"output_dir"})
    assert not missing, f"VFACE does not send {name} its required {missing}"


def test_the_orientation_request_matches_asos_schema(tmp_path):
    sup = RecordingSup(tmp_path)
    frame = catalogs.FRAMES[catalogs.FRAME_CRANIAL_BASE]
    tools.orient_scans(sup, str(tmp_path / "scans"), "/models/gold",
                       frame["landmarks"], frame["suffix"], "/models/ali")
    assert_sends_only_declared_arguments("ASO", sup.calls[0][1])


def test_the_mask_request_matches_amasss_schema(tmp_path):
    sup = RecordingSup(tmp_path)
    tools.segment_masks(sup, str(tmp_path / "oriented"), "/models/amasss", ["CB", "MAND"])
    assert_sends_only_declared_arguments("AMASSS", sup.calls[0][1])


def test_the_mirror_request_matches_automatrixs_schema(tmp_path):
    sup = RecordingSup(tmp_path)
    tools.mirror(sup, str(tmp_path / "oriented"), "/models/mirror.tfm")
    assert_sends_only_declared_arguments("AutoMatrix", sup.calls[0][1])


def test_the_registration_request_matches_areg_cbcts_schema(tmp_path):
    sup = RecordingSup(tmp_path)
    tools.register(sup, str(tmp_path / "t1"), str(tmp_path / "t2"),
                   catalogs.REGION_MANDIBLE, str(tmp_path / "masks"))
    assert_sends_only_declared_arguments("AREG_CBCT", sup.calls[0][1])


def test_the_landmark_request_matches_ali_cbcts_schema(tmp_path):
    sup = RecordingSup(tmp_path)
    tools.predict_landmarks(sup, str(tmp_path / "registered"), ["Ba", "S", "N"],
                            "/models/ali")
    assert_sends_only_declared_arguments("ALI_CBCT", sup.calls[0][1])


def test_the_surface_request_matches_batch_dental_segs_schema(tmp_path):
    sup = RecordingSup(tmp_path)
    tools.segment_surfaces(sup, str(tmp_path / "registered"), "/models/bds")
    assert_sends_only_declared_arguments("Batch_Dental_Seg", sup.calls[0][1])


# ---------------------------------------------------------------------------
# The values that are neither argument names nor paths
# ---------------------------------------------------------------------------

def test_every_frames_landmarks_are_ones_aso_can_register_on():
    """Each frame names its own points, and a point ASO no longer catalogs
    would be a 422 one tool down -- on a list this tool declares, not on
    anything the caller sent."""
    offered = set(schema_of("ASO")["arguments"]["cbct_landmarks"]["choices"])
    for name, frame in catalogs.FRAMES.items():
        missing = sorted(set(frame["landmarks"]) - offered)
        assert not missing, f"the {name} frame asks ASO for {missing}"


def test_every_region_names_an_areg_region_that_exists():
    offered = set(schema_of("AREG_CBCT")["arguments"]["regions"]["choices"])
    asked = {entry["areg"] for entry in catalogs.REGION_TABLE.values()}
    assert asked <= offered


def test_every_structure_names_an_amasss_structure_that_exists():
    offered = set(schema_of("AMASSS")["arguments"]["structures"]["choices"])
    asked = {entry["structure"] for entry in catalogs.REGION_TABLE.values()}
    assert asked <= offered


def test_the_modes_it_asks_for_are_modes_those_tools_have():
    aso = schema_of("ASO")
    assert "CBCT" in aso["arguments"]["modality"]["choices"]
    assert "Fully-Automated" in aso["arguments"]["automation"]["choices"]

    areg = schema_of("AREG_CBCT")
    assert "Semi-Automated" in areg["arguments"]["automation"]["choices"]

    automatrix = schema_of("AutoMatrix")
    assert "Segmentation" in automatrix["arguments"]["content"]["choices"]
