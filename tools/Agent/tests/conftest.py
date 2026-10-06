"""Fixtures shared by the suite.

Two guarantees the whole suite rests on:

- **No network, no Ollama, no GPU.** `no_network` is autouse and makes every
  `urllib.request.urlopen` an immediate failure, so a test that reaches for a
  live endpoint fails loudly instead of passing on whatever happens to be
  running on this machine. A test that wants to exercise the HTTP layer stubs
  `urlopen` itself, which overrides the guard for that test only.
- **The catalogue is fabricated.** Nothing here reads the real registry or the
  sibling tools. The point of the port is that the catalogue is an input, so
  the tests supply one.
"""

import json

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any unstubbed HTTP call is a test failure, not a slow test."""

    def refuse(*args, **kwargs):
        raise AssertionError(
            "This test reached the network. Stub it: the suite must pass with "
            "no Ollama, no server and no internet."
        )

    monkeypatch.setattr("urllib.request.urlopen", refuse)


# ---------------------------------------------------------------------------
# A fabricated catalogue, shaped exactly like GET /tools
# ---------------------------------------------------------------------------

def tool(name, description, arguments, **extra):
    spec = {"name": name, "description": description, "arguments": arguments,
            "returns": "path"}
    spec.update(extra)
    return spec


def argument(kind, required=False, **extra):
    spec = {"type": kind, "required": required}
    spec.update(extra)
    return spec


#: Five tools covering every published type, plus a hidden argument and the
#: server-filled `output_dir` every real tool declares.
CATALOG = [
    tool(
        "Bone_Seg",
        "Segment craniofacial bone structures on a cone beam CT scan.",
        {
            "scans": argument("path", True, description="The CBCT volumes to segment."),
            "model": argument("path", True, description="The weight bundle."),
            "output_dir": argument("path", True, description="Filled by the server."),
            "structures": argument(
                "list[str]", False, default=["MAND"],
                choices=["MAND", "MAX", "CB", "UAW"],
                description="Which structures to segment.",
            ),
            "smoothing": argument("int", False, default=5,
                                  description="Surface smoothing iterations."),
            "make_surfaces": argument("bool", False, default=False,
                                      description="Also export meshes."),
            "device": argument("str", False, default="cuda", choices=["cuda", "cpu"],
                               description="Where to run.", hidden=True),
        },
    ),
    tool(
        "Timepoint_Reg",
        "Register a follow-up cone beam CT onto its baseline so the two compare.",
        {
            "baseline": argument("path", True, description="The first timepoint."),
            "followup": argument("path", True, description="The second timepoint."),
            "output_dir": argument("path", True, description="Filled by the server."),
            "tolerance": argument("float", False, default=0.01,
                                  description="Convergence tolerance."),
        },
    ),
    tool(
        "Mesh_Label",
        "Label every tooth of an intraoral surface scan with its dental number.",
        {
            "meshes": argument("path", True, description="The .vtk or .stl scans."),
            "output_dir": argument("path", True, description="Filled by the server."),
            "jaw": argument("str", False, default="Upper", choices=["Upper", "Lower"],
                            description="Which arch."),
        },
    ),
    tool(
        "Note_Reader",
        "Extract structured findings from free-text clinical notes.",
        {
            "notes": argument("path", True, description="The notes folder."),
            "output_dir": argument("path", True, description="Filled by the server."),
        },
    ),
    tool(
        "No_Argument_Tool",
        "A tool that takes nothing but its output directory.",
        {"output_dir": argument("path", True, description="Filled by the server.")},
    ),
]


@pytest.fixture
def catalog_file(tmp_path):
    """The fabricated catalogue, written where `run()` can be pointed at it."""
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(CATALOG), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The language model, stubbed
# ---------------------------------------------------------------------------

class StubModel:
    """Scripted answers, and a record of every message it was sent.

    `llm.chat` is one function with one return value, so a stub of it is one
    list of strings. What the tests actually assert is usually the RECORD: what
    the router was shown decides what it can possibly answer, and that is where
    upstream's truncation and its silent narrowing lived.
    """

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, endpoint, model, messages, **kwargs):
        self.calls.append({
            "endpoint": endpoint, "model": model, "messages": messages,
            "kwargs": kwargs,
        })
        if not self.answers:
            raise AssertionError(
                "The tool asked the model {} time(s); the stub was scripted for "
                "{}.".format(len(self.calls), len(self.calls) - 1)
            )
        return self.answers.pop(0)

    # -- what the tests read ------------------------------------------------

    @property
    def prompts(self):
        """Every message content sent, flattened, in order."""
        return [
            message["content"]
            for call in self.calls
            for message in call["messages"]
        ]

    def user_message(self, index=0):
        return self.calls[index]["messages"][-1]["content"]

    def system_message(self, index=0):
        return self.calls[index]["messages"][0]["content"]


@pytest.fixture
def stub_model(monkeypatch):
    """Install a `StubModel` for `llm.chat` and hand it back."""

    def install(*answers):
        stub = StubModel(answers)
        monkeypatch.setattr("sadt_agent.llm.chat", stub)
        return stub

    return install


def routed(name, confidence=0.9, reason="because"):
    return json.dumps({"tool": name, "confidence": confidence, "reason": reason})


def extracted(values, confidence=0.9, notes=""):
    return json.dumps({"extracted": values, "confidence": confidence, "notes": notes})


# ---------------------------------------------------------------------------
# The supervisor, as a tool sees one
# ---------------------------------------------------------------------------

class FakeSup:
    """Duck-typed, five members, nothing imported. What the server injects.

    Writes a file into the output directory it is handed, so the seam is
    exercised rather than stubbed over: the tool has to give the callee a real
    directory inside its own output, and that is what is checked.
    """

    def __init__(self, tmp_path=None, result=None, fail=None):
        from pathlib import Path

        self.out = Path(tmp_path) if tmp_path else None
        self.tmp = Path(tmp_path) if tmp_path else None
        self.calls = []
        self.spans = []
        self.messages = []
        self.logs = []
        self._result = result
        self._fail = fail

    def run(self, tool_name, **params):
        from pathlib import Path

        # The caller's span of its own bar. The server's supervisor removes it
        # before the callee sees it, so it is recorded apart, never passed on.
        self.spans.append((tool_name, params.pop("_progress", None)))
        self.calls.append((tool_name, params))
        if self._fail is not None:
            raise self._fail
        destination = Path(params["output_dir"])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "produced.txt").write_text(tool_name, encoding="utf-8")
        return self._result if self._result is not None else destination

    def progress(self, fraction, message):
        self.messages.append((fraction, message))

    def log(self, message, level="info", user=False):
        self.messages.append((None, message))
        self.logs.append((level, user, message))
