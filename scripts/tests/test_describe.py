"""describe.py must refuse what it cannot represent, not guess.

Every case here is a schema the client would render wrongly: a dropped argument,
a check box where a number belongs, a required flag that is silently optional.
The script is exercised through its CLI because the exit code is what CI reads.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

DESCRIBE = Path(__file__).resolve().parent.parent / "describe.py"


def make_tool(root, body, package="sadt_fixture"):
    """A minimal tool package on disk: src/<package>/__init__.py holding run()."""
    source = root / "src" / package
    source.mkdir(parents=True)
    (source / "__init__.py").write_text(
        "from pathlib import Path\n\n" + textwrap.dedent(body), encoding="utf-8"
    )
    return root


def describe(tool_dir):
    return subprocess.run(
        [sys.executable, str(DESCRIBE), str(tool_dir)],
        capture_output=True,
        text=True,
    )


GOOD = """
    def run(
        scan: Path,
        output_dir: Path,
        structures: list[str] = ["Mandible", "Maxilla"],
        crop: bool = False,
        margin: float = 1,
    ) -> Path:
        \"\"\"Segment craniofacial structures on a CBCT scan.

        A second paragraph never reaches the client.

        Args:
            scan: The CBCT to segment.
            output_dir: Where the masks are written.
            structures: Which structures to segment.
            crop: Crop each mask to the structure it holds, rather
                than to the whole scan.
            margin: How much the crop leaves around it, in millimetres.
        \"\"\"
        return output_dir
"""


def test_schema_matches_the_signature(tmp_path):
    result = describe(make_tool(tmp_path, GOOD))
    assert result.returncode == 0, result.stderr
    schema = json.loads(result.stdout)

    assert schema["name"] == tmp_path.name
    assert schema["description"] == "Segment craniofacial structures on a CBCT scan."
    assert schema["returns"] == "path"
    assert list(schema["arguments"]) == [
        "scan",
        "output_dir",
        "structures",
        "crop",
        "margin",
    ], "argument order follows the signature, so the client's form does too"
    assert schema["arguments"]["scan"] == {
        "type": "path",
        "required": True,
        "description": "The CBCT to segment.",
    }
    assert schema["arguments"]["structures"] == {
        "type": "list[str]",
        "required": False,
        "default": ["Mandible", "Maxilla"],
        "description": "Which structures to segment.",
    }
    # Wrapped over two source lines, one sentence by the time a panel shows it.
    assert schema["arguments"]["crop"]["description"] == (
        "Crop each mask to the structure it holds, rather than to the whole scan."
    )
    # Declared float, defaulted with an int literal: the client must still get
    # a float widget with a float default.
    assert schema["arguments"]["margin"]["default"] == 1.0
    assert isinstance(schema["arguments"]["margin"]["default"], float)
    assert len(schema["source_hash"]) == 64


def test_source_hash_tracks_the_sources(tmp_path):
    tool = make_tool(tmp_path, GOOD)
    before = json.loads(describe(tool).stdout)["source_hash"]

    helper = tool / "src" / "sadt_fixture" / "helper.py"
    helper.write_text("VALUE = 1\n", encoding="utf-8")
    after_add = json.loads(describe(tool).stdout)["source_hash"]
    assert after_add != before

    helper.rename(helper.with_name("renamed.py"))
    assert json.loads(describe(tool).stdout)["source_hash"] != after_add, (
        "a rename changes what gets imported, so it must change the hash"
    )


def test_bytecode_does_not_change_the_hash(tmp_path):
    tool = make_tool(tmp_path, GOOD)
    before = json.loads(describe(tool).stdout)["source_hash"]

    # Already there: the first describe() imported the package and Python wrote
    # its bytecode next to the sources.
    cache = tool / "src" / "sadt_fixture" / "__pycache__"
    cache.mkdir(exist_ok=True)
    (cache / "__init__.cpython-312.pyc").write_bytes(b"\x00\x01")
    assert json.loads(describe(tool).stdout)["source_hash"] == before


@pytest.mark.parametrize(
    "body, expected",
    [
        pytest.param(
            "from typing import Optional\n"
            "def run(scan: Path, n: Optional[int] = None) -> Path:\n"
            '    """Do a thing."""\n'
            "    return scan\n",
            "unsupported annotation",
            id="Optional is not a type here, a missing default is",
        ),
        pytest.param(
            "def run(scan: Path, n: int = None) -> Path:\n"
            '    """Do a thing."""\n'
            "    return scan\n",
            "defaults to None",
            id="None default would publish a required argument as optional",
        ),
        pytest.param(
            "def run(scan: Path, n: int = True) -> Path:\n"
            '    """Do a thing."""\n'
            "    return scan\n",
            "but defaults to True",
            id="bool passes isinstance(int) and would render a check box",
        ),
        pytest.param(
            "def run(scan: Path, opts: dict = {}) -> Path:\n"
            '    """Do a thing."""\n'
            "    return scan\n",
            "unsupported annotation",
            id="dict argument",
        ),
        pytest.param(
            "def run(scan: Path, tags: list = []) -> Path:\n"
            '    """Do a thing."""\n'
            "    return scan\n",
            "bare list is not supported",
            id="bare list has no element type to build a widget from",
        ),
        pytest.param(
            "def run(scan: Path, **kwargs) -> Path:\n"
            '    """Do a thing."""\n'
            "    return scan\n",
            "must be named",
            id="**kwargs cannot come out of a JSON object",
        ),
        pytest.param(
            "def run(scan: Path, n) -> Path:\n"
            '    """Do a thing."""\n'
            "    return scan\n",
            "no annotation",
            id="unannotated argument",
        ),
        pytest.param(
            "def run(scan: Path) -> str:\n"
            '    """Do a thing."""\n'
            "    return str(scan)\n",
            "must return a Path",
            id="a tool returns files, not text",
        ),
        pytest.param(
            "def run(scan: Path):\n"
            '    """Do a thing."""\n'
            "    return scan\n",
            "no return annotation",
            id="missing return annotation",
        ),
        pytest.param(
            "def run(scan: Path) -> Path:\n    return scan\n",
            "no docstring",
            id="the first line is the description shown to a clinician",
        ),
        pytest.param(
            "run = 3\n",
            "not a function",
            id="run must be callable",
        ),
        pytest.param(
            "def other(scan: Path) -> Path:\n"
            '    """Do a thing."""\n'
            "    return scan\n",
            "defines no run()",
            id="no run at all",
        ),
    ],
)
def test_unsupported_signatures_fail_loudly(tmp_path, body, expected):
    result = describe(make_tool(tmp_path, body))

    assert result.returncode == 2
    assert expected in result.stderr
    assert result.stdout == "", "a refused tool must not emit a partial schema"


# --- Literal: a fixed set of options ---------------------------------------


def test_literal_publishes_choices(tmp_path):
    """`list[Literal[...]]` is several-of, bare `Literal[...]` is exactly-one."""
    tool = make_tool(
        tmp_path,
        "from typing import Literal\n\n"
        "def run(\n"
        "    scan: Path,\n"
        '    structures: list[Literal["MAND", "MAX", "CB"]] = ["MAND", "MAX"],\n'
        '    merge: Literal["MERGED", "SEPARATE"] = "MERGED",\n'
        "    folds: list[Literal[0, 1, 2]] = [0],\n"
        ") -> Path:\n"
        '    """Segment a scan.\n'
        "\n"
        "    Args:\n"
        "        scan: The scan.\n"
        "        structures: Which structures.\n"
        "        merge: One file or several.\n"
        "        folds: Which folds.\n"
        '    """\n'
        "    return scan\n",
    )
    schema = json.loads(describe(tool).stdout)

    assert schema["arguments"]["structures"] == {
        "type": "list[str]",
        "required": False,
        "default": ["MAND", "MAX"],
        "choices": ["MAND", "MAX", "CB"],
        "description": "Which structures.",
    }
    assert schema["arguments"]["merge"] == {
        "type": "str",
        "required": False,
        "default": "MERGED",
        "choices": ["MERGED", "SEPARATE"],
        "description": "One file or several.",
    }
    assert schema["arguments"]["folds"]["type"] == "list[int]"
    assert schema["arguments"]["folds"]["choices"] == [0, 1, 2]


def test_choices_come_last_so_keys_stay_stable(tmp_path):
    """Adding options to an argument must not reorder what a reader expects."""
    tool = make_tool(
        tmp_path,
        "from typing import Literal\n\n"
        'def run(scan: Path, mode: Literal["a", "b"] = "a") -> Path:\n'
        '    """Do a thing.\n'
        "\n"
        "    Args:\n"
        "        scan: The scan.\n"
        "        mode: Which mode.\n"
        '    """\n'
        "    return scan\n",
    )
    schema = json.loads(describe(tool).stdout)

    assert list(schema["arguments"]["mode"]) == [
        "type", "required", "default", "choices", "description",
    ]


def test_a_required_argument_may_still_carry_choices(tmp_path):
    tool = make_tool(
        tmp_path,
        "from typing import Literal\n\n"
        'def run(scan: Path, mode: Literal["a", "b"]) -> Path:\n'
        '    """Do a thing.\n'
        "\n"
        "    Args:\n"
        "        scan: The scan.\n"
        "        mode: Which mode.\n"
        '    """\n'
        "    return scan\n",
    )
    assert json.loads(describe(tool).stdout)["arguments"]["mode"] == {
        "type": "str",
        "required": True,
        "choices": ["a", "b"],
        "description": "Which mode.",
    }


@pytest.mark.parametrize(
    "body, expected",
    [
        pytest.param(
            "from typing import Literal\n\n"
            'def run(scan: Path, mode: Literal["a", "b"] = "c") -> Path:\n'
            '    """Do a thing."""\n'
            "    return scan\n",
            "not one of its options",
            id="a default the published picker cannot produce",
        ),
        pytest.param(
            "from typing import Literal\n\n"
            'def run(scan: Path, m: list[Literal["a", "b"]] = ["a", "z"]) -> Path:\n'
            '    """Do a thing."""\n'
            "    return scan\n",
            "not one of its options",
            id="one bad element in a multi-select default",
        ),
        pytest.param(
            "from typing import Literal\n\n"
            'def run(scan: Path, mode: Literal["a", 1] = "a") -> Path:\n'
            '    """Do a thing."""\n'
            "    return scan\n",
            "must all be str, or all be int",
            id="mixed option types have no single widget",
        ),
        pytest.param(
            "from typing import Literal\n\n"
            "def run(scan: Path, mode: Literal[True, False] = True) -> Path:\n"
            '    """Do a thing."""\n'
            "    return scan\n",
            "must all be str, or all be int",
            id="bool passes isinstance(int) and would render as a number",
        ),
        pytest.param(
            "from typing import Literal, Optional\n\n"
            'def run(scan: Path, mode: Literal[None] = None) -> Path:\n'
            '    """Do a thing."""\n'
            "    return scan\n",
            "Options must be str or int",
            id="None is not an option a client can render",
        ),
    ],
)
def test_bad_choices_fail_loudly(tmp_path, body, expected):
    result = describe(make_tool(tmp_path, body))

    assert result.returncode == 2
    assert expected in result.stderr
    assert result.stdout == ""


def test_dict_of_paths_is_accepted_for_several_outputs(tmp_path):
    tool = make_tool(
        tmp_path,
        "def run(scan: Path) -> dict[str, Path]:\n"
        '    """Segment and align a scan.\n'
        "\n"
        "    Args:\n"
        "        scan: The scan.\n"
        '    """\n'
        "    return {}\n",
    )
    assert json.loads(describe(tool).stdout)["returns"] == "dict[str, path]"


def test_module_level_heavy_import_is_reported_as_such(tmp_path):
    tool = make_tool(
        tmp_path,
        "import torch\n\n"
        "def run(scan: Path) -> Path:\n"
        '    """Do a thing."""\n'
        "    return scan\n",
    )
    result = describe(tool)

    assert result.returncode == 2
    assert "Move it inside run()" in result.stderr


def test_the_template_describes_itself(tmp_path):
    """The reference package must pass the checker it is the reference for."""
    template = Path(__file__).resolve().parents[2] / "tools" / "_template"
    venv_python = template / ".venv" / "bin" / "python"
    if not venv_python.is_file():
        pytest.skip("run `uv sync` in tools/_template first")

    result = subprocess.run(
        [str(venv_python), str(DESCRIBE), str(template)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    schema = json.loads(result.stdout)
    assert schema["arguments"]["scans"]["required"] is True
    assert schema["arguments"]["metrics"]["default"] == ["mean", "max"]
    assert schema["arguments"]["metrics"]["choices"] == ["mean", "max", "min", "std"]
    assert schema["arguments"]["reduction"]["choices"] == ["per_scan", "pooled"]


# ---------------------------------------------------------------------------
# The per-argument help, read from the docstring
# ---------------------------------------------------------------------------


def test_an_undocumented_argument_fails(tmp_path):
    """The whole reason this check exists: descriptions were published nowhere
    for months and nothing said so, because an empty one reads as a tool that
    simply has nothing to add."""
    result = describe(make_tool(
        tmp_path,
        "def run(scan: Path, margin: float = 1.0) -> Path:\n"
        '    """Do a thing.\n'
        "\n"
        "    Args:\n"
        "        scan: The scan.\n"
        '    """\n'
        "    return scan\n",
    ))

    assert result.returncode == 2
    assert "no description for margin" in result.stderr
    assert result.stdout == ""


def test_documenting_an_argument_run_does_not_take_fails(tmp_path):
    """The drift case: an argument renamed in the signature and not in the
    docstring leaves help text that describes something that is gone."""
    result = describe(make_tool(
        tmp_path,
        "def run(scan: Path) -> Path:\n"
        '    """Do a thing.\n'
        "\n"
        "    Args:\n"
        "        scan: The scan.\n"
        "        margin: Removed two releases ago.\n"
        '    """\n'
        "    return scan\n",
    ))

    assert result.returncode == 2
    assert "margin" in result.stderr and "does not take" in result.stderr


def test_a_docstring_with_no_args_section_fails(tmp_path):
    result = describe(make_tool(
        tmp_path,
        'def run(scan: Path) -> Path:\n    """Do a thing."""\n    return scan\n',
    ))

    assert result.returncode == 2
    assert "no 'Args:' section" in result.stderr


def test_the_type_may_be_repeated_in_the_docstring_and_is_dropped(tmp_path):
    """`scan (Path):` is the other common Google style. The annotation is the
    truth about the type, so the parenthesis is read and thrown away rather
    than shipped to a client as part of the sentence."""
    schema = json.loads(describe(make_tool(
        tmp_path,
        "def run(scan: Path) -> Path:\n"
        '    """Do a thing.\n'
        "\n"
        "    Args:\n"
        "        scan (Path): The scan.\n"
        '    """\n'
        "    return scan\n",
    )).stdout)

    assert schema["arguments"]["scan"]["description"] == "The scan."


def test_a_later_section_does_not_leak_into_the_last_description(tmp_path):
    """Returns: follows Args: in every docstring in this repository."""
    schema = json.loads(describe(make_tool(
        tmp_path,
        "def run(scan: Path) -> Path:\n"
        '    """Do a thing.\n'
        "\n"
        "    Args:\n"
        "        scan: The scan.\n"
        "\n"
        "    Returns:\n"
        "        The scan, done to.\n"
        '    """\n'
        "    return scan\n",
    )).stdout)

    assert schema["arguments"]["scan"]["description"] == "The scan."


# ---------------------------------------------------------------------------
# The supervisor -- how a tool calls another tool
# ---------------------------------------------------------------------------

SUPERVISED = """
    def run(scan: Path, output_dir: Path, *, sup=None) -> Path:
        \"\"\"Orient a scan onto a reference.

        Args:
            scan: The scan to orient.
            output_dir: Where the oriented scan is written.
        \"\"\"
        return output_dir
"""


def test_the_supervisor_is_not_published_as_an_argument(tmp_path):
    """It is not data, so a client must never be asked to send it."""
    result = describe(make_tool(tmp_path, SUPERVISED))

    assert result.returncode == 0, result.stderr
    schema = json.loads(result.stdout)
    assert "sup" not in schema["arguments"]
    assert list(schema["arguments"]) == ["scan", "output_dir"]


def test_a_tool_that_takes_one_says_so(tmp_path):
    """The server has to know before it accepts a job: a runner that cannot
    inject a supervisor must refuse the tool, not call it and fail halfway."""
    schema = json.loads(describe(make_tool(tmp_path, SUPERVISED)).stdout)
    assert schema["supervisor"] is True


def test_a_tool_without_one_does_not_carry_the_key(tmp_path):
    """Absent, not `false`: every tool written before supervisors existed keeps
    exactly the schema it had."""
    schema = json.loads(describe(make_tool(tmp_path, GOOD)).stdout)
    assert "supervisor" not in schema


@pytest.mark.parametrize(
    "body, expected",
    [
        (
            # Positional: the runner's first argument would land in it.
            """
            def run(scan: Path, sup=None) -> Path:
                \"\"\"Orient a scan.\"\"\"
                return scan
            """,
            "keyword-only",
        ),
        (
            # Annotated: published as a path, so the form grows a file picker
            # for something no client can produce.
            """
            def run(scan: Path, *, sup: Path = None) -> Path:
                \"\"\"Orient a scan.\"\"\"
                return scan
            """,
            "must not be annotated",
        ),
    ],
)
def test_a_near_miss_supervisor_fails_loudly(tmp_path, body, expected):
    """Both of these would otherwise fail a long way from the signature."""
    result = describe(make_tool(tmp_path, body))

    assert result.returncode == 2
    assert expected in result.stderr
    assert result.stdout == ""


# ---------------------------------------------------------------------------
# The optional panel layout
# ---------------------------------------------------------------------------

LAYOUT_TOOL = """
    from typing import Literal

    def run(scans: Path, output_dir: Path,
            mode: Literal["CBCT", "IOS"] = "CBCT",
            parts: list[Literal["a", "b", "c"]] = ["a"]) -> Path:
        \"\"\"Do a thing.

        Args:
            scans: The scans to do it to.
            output_dir: Where the results are written.
            mode: Which kind of data this is.
            parts: Which parts to do it to.
        \"\"\"
        return output_dir
"""


def make_layout(root, body, package="sadt_fixture"):
    (root / "src" / package / "layout.py").write_text(
        textwrap.dedent(body), encoding="utf-8"
    )
    return root


def test_a_tool_without_a_layout_is_unchanged(tmp_path):
    """Absent is the ordinary case, and must publish exactly what it did before
    layouts existed."""
    schema = json.loads(describe(make_tool(tmp_path, LAYOUT_TOOL)).stdout)

    for spec in schema["arguments"].values():
        assert "section" not in spec and "ui" not in spec


def test_layout_hints_are_merged_into_the_arguments(tmp_path):
    """Into the argument, not beside it: a client reads one spec per argument,
    and a second place to look is a second place to forget."""
    root = make_tool(tmp_path, LAYOUT_TOOL)
    make_layout(root, '''
        LAYOUT = {
            "scans": {"section": "Inputs", "label": "Scans"},
            "parts": {"ui": "tabs", "groups": {"First": ["a"], "Rest": ["b", "c"]}},
            "mode": {"section": "Inputs"},
        }
    ''')

    schema = json.loads(describe(root).stdout)

    assert schema["arguments"]["scans"]["section"] == "Inputs"
    assert schema["arguments"]["scans"]["label"] == "Scans"
    assert schema["arguments"]["parts"]["groups"] == {"First": ["a"], "Rest": ["b", "c"]}
    # And it changed nothing about what the tool accepts.
    assert schema["arguments"]["parts"]["choices"] == ["a", "b", "c"]


def test_a_layout_naming_an_argument_that_does_not_exist_fails(tmp_path):
    root = make_tool(tmp_path, LAYOUT_TOOL)
    make_layout(root, 'LAYOUT = {"nope": {"section": "Inputs"}}')

    result = describe(root)

    assert result.returncode == 2
    assert "no argument 'nope'" in result.stderr


def test_a_tab_listing_an_option_that_is_not_offered_fails(tmp_path):
    """THE check that replaces the drift. The `ArgSpec` tables this succeeds
    listed options by hand, and a landmark added to the catalog was reachable
    through no tab at all -- published by the schema, invisible in the UI."""
    root = make_tool(tmp_path, LAYOUT_TOOL)
    make_layout(root, 'LAYOUT = {"parts": {"groups": {"Tab": ["a", "zzz"]}}}')

    result = describe(root)

    assert result.returncode == 2
    assert "does not offer" in result.stderr and "zzz" in result.stderr


def test_a_condition_on_a_value_the_argument_never_takes_fails(tmp_path):
    """A panel whose field never appears is worse than one that always does."""
    root = make_tool(tmp_path, LAYOUT_TOOL)
    make_layout(root, 'LAYOUT = {"parts": {"visible_when": {"mode": "MRI"}}}')

    result = describe(root)

    assert result.returncode == 2
    assert "which it never is" in result.stderr


def test_a_condition_on_an_argument_that_does_not_exist_fails(tmp_path):
    root = make_tool(tmp_path, LAYOUT_TOOL)
    make_layout(root, 'LAYOUT = {"parts": {"visible_when": {"nope": "CBCT"}}}')

    assert "does not take" in describe(root).stderr


def test_an_unknown_layout_key_fails(tmp_path):
    """A hint the client would silently drop is a hint that reads as working."""
    root = make_tool(tmp_path, LAYOUT_TOOL)
    make_layout(root, 'LAYOUT = {"scans": {"colour": "red"}}')

    result = describe(root)

    assert result.returncode == 2
    assert "unknown key" in result.stderr and "colour" in result.stderr


def test_groups_on_an_argument_with_no_choices_fails(tmp_path):
    root = make_tool(tmp_path, LAYOUT_TOOL)
    make_layout(root, 'LAYOUT = {"scans": {"groups": {"Tab": ["a"]}}}')

    assert "has none" in describe(root).stderr


# ---------------------------------------------------------------------------
# require() states a dependency without making the call

REQUIRES_LITERAL = """
    def _preflight(sup):
        tools.require(sup, "AMASSS", "Fully-Automated registration")

    def run(scan: Path, output_dir: Path, *, sup=None) -> Path:
        \"\"\"Register two timepoints.

        Args:
            scan: The scan to register.
            output_dir: Where the result is written.
        \"\"\"
        _preflight(sup)
        return output_dir
"""

REQUIRES_LOOP = """
    def _preflight(sup):
        for name in ("Crown_Seg", "ALI_IOS"):
            tools.require(sup, name, "Fully-Automated registration")

    def run(scan: Path, output_dir: Path, *, sup=None) -> Path:
        \"\"\"Register two timepoints.

        Args:
            scan: The scan to register.
            output_dir: Where the result is written.
        \"\"\"
        _preflight(sup)
        return output_dir
"""


def test_a_required_tool_is_published_like_a_called_one(tmp_path):
    """`require` names a tool without running it, and the server checks names.

    Collected because it was not: AREG_IOS asked for 'ALI' through `require`
    long after the split renamed it to 'ALI_IOS'. `sup.run` was scanned and
    `require` was not, so neither the schema nor the server's startup check saw
    a name that could never resolve, and the failure waited for a mucogingival
    run to reach it.
    """
    schema = json.loads(describe(make_tool(tmp_path, REQUIRES_LITERAL)).stdout)
    assert schema["calls"] == ["AMASSS"]


def test_a_required_tool_named_in_a_loop_is_published(tmp_path):
    """Several tools one mode needs, written once instead of three times.

    Refusing this form would push a tool towards the repetitive spelling purely
    to satisfy the reader, so the loop is resolved when every element is a
    literal.
    """
    schema = json.loads(describe(make_tool(tmp_path, REQUIRES_LOOP)).stdout)
    assert schema["calls"] == ["ALI_IOS", "Crown_Seg"]


# ---------------------------------------------------------------------------
# Quality control: where a reader may be stopped

DECLARES_ONE = """
    def run(scan: Path, *, sup=None) -> Path:
        \"\"\"Segment a scan.

        Args:
            scan: The scan to segment.
        \"\"\"
        sup.declareQualityControl("after the crop")
        return scan
"""

DECLARES_SEVERAL = """
    def run(scan: Path, *, sup=None) -> Path:
        \"\"\"Orient a scan.

        Args:
            scan: The scan to orient.
        \"\"\"
        sup.declareQualityControl("landmarks")
        if scan.name:
            sup.run("ASO", scan=scan)
        sup.declareQualityControl("orientation")
        sup.declareQualityControl("landmarks")
        return scan
"""

DECLARES_A_VARIABLE = """
    def run(scan: Path, name: str = "x", *, sup=None) -> Path:
        \"\"\"Do something.

        Args:
            scan: The scan.
            name: Where to stop.
        \"\"\"
        sup.declareQualityControl(name)
        return scan
"""


def test_a_declared_checkpoint_is_published(tmp_path):
    """A tool offering a stop in the MIDDLE of its own work, where no
    `sup.run()` boundary exists. AMASSS segmenting five structures calls
    nobody and still has a moment worth looking at."""
    schema = json.loads(describe(make_tool(tmp_path, DECLARES_ONE)).stdout)
    assert schema["quality_controls"] == [
        {"name": "after the crop", "kind": "view"}
    ], "a checkpoint that says nothing is somewhere to LOOK, not to come back to"


def test_they_keep_declaration_order_and_do_not_repeat(tmp_path):
    """Declaration order is the order they happen in; a reader choosing where
    to stop is reading a sequence, not an index."""
    schema = json.loads(describe(make_tool(tmp_path, DECLARES_SEVERAL)).stdout)
    assert [entry["name"] for entry in schema["quality_controls"]] == [
        "landmarks", "orientation"]
    assert schema["calls"] == ["ASO"], "the two are published side by side"


DECLARES_A_KIND = """
    def run(scan: Path, *, sup=None) -> Path:
        \"\"\"Segment a scan.

        Args:
            scan: The scan to segment.
        \"\"\"
        sup.declareQualityControl("the landmarks", kind="landmarks")
        return scan
"""

DECLARES_A_BAD_KIND = """
    def run(scan: Path, *, sup=None) -> Path:
        \"\"\"Segment a scan.

        Args:
            scan: The scan to segment.
        \"\"\"
        sup.declareQualityControl("the landmarks", kind="editable")
        return scan
"""


def test_a_checkpoint_says_what_may_be_edited_there(tmp_path):
    """Which is what decides whether a reader can come BACK to it from a
    later stop. Looking at a bad orientation is useless without a way back to
    the landmarks that caused it."""
    schema = json.loads(describe(make_tool(tmp_path, DECLARES_A_KIND)).stdout)
    assert schema["quality_controls"] == [
        {"name": "the landmarks", "kind": "landmarks"}
    ]


def test_a_kind_no_reviewer_implements_is_refused(tmp_path):
    """The three kinds are what the reviewer knows how to put on screen. A
    fourth would publish a checkpoint promising an edit nothing can make."""
    completed = describe(make_tool(tmp_path, DECLARES_A_BAD_KIND))
    assert completed.returncode != 0
    assert "kind must be one of" in completed.stderr


NAMES_ITS_OUTPUTS = """
    OUTPUT_SUFFIXES = ("_Or", "_lm_Or", "_Or_transform")

    def run(scan: Path) -> Path:
        \"\"\"Orient a scan.

        Args:
            scan: The scan to orient.
        \"\"\"
        return scan
"""

NAMES_THEM_WITHOUT_A_SEPARATOR = """
    OUTPUT_SUFFIXES = ("Or",)

    def run(scan: Path) -> Path:
        \"\"\"Orient a scan.

        Args:
            scan: The scan to orient.
        \"\"\"
        return scan
"""


def test_a_tool_says_what_it_appends_to_a_name(tmp_path):
    """The server has to know which results are about one patient, to replay
    a chain for the cases a clinician marked. It must not learn that from a
    table of its own: it knows no dental tool. So the tool says."""
    schema = json.loads(describe(make_tool(tmp_path, NAMES_ITS_OUTPUTS)).stdout)
    assert schema["output_suffixes"] == ["_Or_transform", "_lm_Or", "_Or"], (
        "longest first, or a caller strips `_Or` from inside `_Or_transform`"
    )


def test_a_marker_that_does_not_separate_is_refused(tmp_path):
    """`Or` without its underscore cuts inside any patient whose name holds
    those two letters."""
    completed = describe(make_tool(tmp_path, NAMES_THEM_WITHOUT_A_SEPARATOR))
    assert completed.returncode != 0
    assert "must begin with" in completed.stderr


def test_a_tool_that_names_nothing_publishes_no_key(tmp_path):
    schema = json.loads(describe(make_tool(tmp_path, GOOD)).stdout)
    assert "output_suffixes" not in schema


def test_a_tool_that_declares_none_publishes_no_key(tmp_path):
    schema = json.loads(describe(make_tool(tmp_path, SUPERVISED)).stdout)
    assert "quality_controls" not in schema


def test_a_name_the_generator_cannot_read_is_refused(tmp_path):
    """The same rule as a call name: a name this cannot see is a name the
    server cannot publish, and a list with a hole in it reads as coverage."""
    completed = describe(make_tool(tmp_path, DECLARES_A_VARIABLE))
    assert completed.returncode != 0
    assert "literal" in completed.stderr


# ---------------------------------------------------------------------------
# vec2: two numbers set together

VEC2 = """
    def run(scan: Path, output_dir: Path,
            corner: tuple[float, float] = (0.5, 0.0)) -> Path:
        \"\"\"Place a corner.

        Args:
            scan: The scan.
            output_dir: Where results go.
            corner: Ratio and adjust of one corner.
        \"\"\"
        return output_dir
"""

VEC3 = """
    def run(scan: Path, output_dir: Path,
            point: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Path:
        \"\"\"Place a point.

        Args:
            scan: The scan.
            output_dir: Where results go.
            point: A point.
        \"\"\"
        return output_dir
"""


def test_a_pair_of_floats_is_published_as_a_vec2(tmp_path):
    """Two numbers that are one position rather than two settings.

    FlexReg's butterfly corners are the case: each has a medio-lateral ratio and
    an antero-posterior adjust, and the client's pad reads both axes spatially -
    the knob sits where the point sits on the arch. Two number fields state two
    numbers and cannot state that.
    """
    schema = json.loads(describe(make_tool(tmp_path, VEC2)).stdout)

    assert schema["arguments"]["corner"]["type"] == "vec2"
    assert schema["arguments"]["corner"]["default"] == [0.5, 0.0]


def test_a_vec2_default_must_be_two_numbers(tmp_path):
    body = VEC2.replace("(0.5, 0.0)", '"middle"')
    result = describe(make_tool(tmp_path, body))

    assert result.returncode != 0
    assert "two numbers" in result.stderr


def test_a_three_number_tuple_is_refused_rather_than_published(tmp_path):
    """A point wants a different widget. Refusing it costs less than publishing
    a type nothing renders."""
    result = describe(make_tool(tmp_path, VEC3))

    assert result.returncode != 0
    assert "tuple[float, float]" in result.stderr


# ---------------------------------------------------------------------------
# ACCEPTS: what a path argument can read
# ---------------------------------------------------------------------------
ACCEPTS_BODY = """
    ACCEPTS = {"ios": (".VTK", ".stl", ".vtk")}


    def run(ios: Path, output_dir: Path) -> Path:
        \"\"\"One line.

        Args:
            ios: The surfaces.
            output_dir: Where to write.

        Returns:
            The output directory.
        \"\"\"
        return output_dir
"""


def test_a_tool_says_what_each_path_argument_accepts(tmp_path):
    """Until this existed, the intraoral picker offered a folder of CBCT
    volumes: the tool knew what it could read and had no way to publish it."""
    tool = make_tool(tmp_path, ACCEPTS_BODY)

    result = describe(tool)

    assert result.returncode == 0, result.stderr
    # Normalised: lowercased, de-duplicated, sorted -- a filter is compared
    # against a file name, not read aloud.
    assert json.loads(result.stdout)["arguments"]["ios"]["extensions"] == [".stl", ".vtk"]


def test_an_accepts_naming_an_unknown_argument_is_refused(tmp_path):
    """A filter on an argument that does not exist narrows nothing, and nothing
    would ever say so."""
    tool = make_tool(tmp_path, ACCEPTS_BODY.replace('"ios":', '"nope":'))

    result = describe(tool)

    assert result.returncode != 0
    assert "no argument 'nope'" in result.stderr, result.stderr


def test_only_a_path_argument_accepts_extensions(tmp_path):
    """The server carries this to `ArgSpec.accepts` and refuses an upload that
    contradicts it, so it has to describe something uploadable."""
    tool = make_tool(tmp_path, ACCEPTS_BODY.replace("run(ios: Path,", "run(ios: str,"))

    result = describe(tool)

    assert result.returncode != 0
    assert "only a path argument" in result.stderr, result.stderr


def test_something_that_is_not_an_extension_is_refused(tmp_path):
    tool = make_tool(tmp_path, ACCEPTS_BODY.replace('".vtk"', '"vtk"'))

    result = describe(tool)

    assert result.returncode != 0
    assert "not an extension" in result.stderr, result.stderr


PAIRED_BODY = """
from pathlib import Path

PAIRED = {"axes": ["t1", "t2"]}


def pairs(t1, t2):
    return {"groups": [], "shared": {}, "unpaired": {}}


def run(t1: Path, t2: Path, output_dir: Path) -> Path:
    \"\"\"Register.

    Args:
        t1: Baseline.
        t2: Follow-up.
        output_dir: Where to write.

    Returns:
        The output directory.
    \"\"\"
    return output_dir
"""


def test_a_tool_says_which_inputs_hold_the_same_patients(tmp_path):
    """Published so a client splits both folders by the same patients."""
    result = describe(make_tool(tmp_path, PAIRED_BODY))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["paired"] == {"axes": ["t1", "t2"]}


@pytest.mark.parametrize("change, message", [
    (('"axes": ["t1", "t2"]', '"axes": ["t1"]'), "two or more"),
    (('"axes": ["t1", "t2"]', '"axes": ["t1", "nope"]'), "no argument 'nope'"),
    (("def pairs(t1, t2):", "def answer(t1, t2):"), "defines no pairs()"),
    (("def pairs(t1, t2):", "def pairs(t1):"), "lacks t2"),
])
def test_a_paired_declaration_that_cannot_be_honoured_is_refused(tmp_path, change, message):
    """A split the tool cannot answer for would send patients unpaired."""
    result = describe(make_tool(tmp_path, PAIRED_BODY.replace(*change)))
    assert result.returncode != 0
    assert message in result.stderr, result.stderr


def test_a_tool_without_paired_inputs_publishes_nothing_about_them(tmp_path):
    result = describe(make_tool(tmp_path, ACCEPTS_BODY))
    assert "paired" not in json.loads(result.stdout)
