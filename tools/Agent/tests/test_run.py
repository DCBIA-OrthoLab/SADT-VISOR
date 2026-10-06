"""`run()` end to end, with the language model stubbed.

No Ollama, no network, no GPU. What is exercised is everything around the two
model calls: what is written, what is refused, and -- for `execute` -- what
reaches the supervisor.
"""

import json

import pytest

from sadt_agent import ADVICE_NAME, DECISION_NAME, MODE_ASK, REPORT_NAME, run
from sadt_agent.errors import (
    CatalogError,
    SupervisorRequired,
    ToolInputError,
    ToolUnavailableError,
)

from conftest import FakeSup, extracted, routed


def decision(output_dir):
    return json.loads((output_dir / DECISION_NAME).read_text(encoding="utf-8"))


def report(output_dir):
    return json.loads((output_dir / REPORT_NAME).read_text(encoding="utf-8"))


def invoke(tmp_path, catalog_file, **overrides):
    arguments = {
        "prompt": "segment the bone structures on my CBCTs",
        "output_dir": tmp_path / "out",
        "catalog_file": catalog_file,
    }
    arguments.update(overrides)
    return run(**arguments)


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------

def test_a_request_is_routed_and_its_parameters_extracted(
    tmp_path, catalog_file, stub_model
):
    stub_model(
        routed("Bone_Seg", 0.92, "the request asks for bone segmentation"),
        extracted({"scans": "/data/cohort", "model": "/models/v1"}, 0.8),
    )
    output_dir = invoke(tmp_path, catalog_file, folders=["/data/cohort"])

    made = decision(output_dir)
    assert made["tool"] == "Bone_Seg"
    assert made["arguments"] == {"scans": "/data/cohort", "model": "/models/v1"}
    assert made["confidence"] == 0.92
    assert made["reasoning"] == "the request asks for bone segmentation"
    assert made["missing_required"] == []


def test_the_decision_carries_the_five_fields_a_caller_reads(
    tmp_path, catalog_file, stub_model
):
    stub_model(routed("Bone_Seg"), extracted({"scans": "/d", "model": "/m"}))
    made = decision(invoke(tmp_path, catalog_file))
    for field in ("tool", "arguments", "confidence", "reasoning", "alternatives"):
        assert field in made


def test_the_alternatives_are_the_other_candidates_with_their_scores(
    tmp_path, catalog_file, stub_model
):
    stub_model(routed("Bone_Seg"), extracted({}))
    made = decision(invoke(tmp_path, catalog_file, candidates=3))
    assert len(made["alternatives"]) == 2
    assert "Bone_Seg" not in [entry["tool"] for entry in made["alternatives"]]
    assert all("score" in entry for entry in made["alternatives"])


def test_run_returns_its_output_directory(tmp_path, catalog_file, stub_model):
    stub_model(routed("Bone_Seg"), extracted({}))
    assert invoke(tmp_path, catalog_file) == tmp_path / "out"


def test_a_report_is_written_beside_the_decision(tmp_path, catalog_file, stub_model):
    stub_model(routed("Bone_Seg"), extracted({}))
    written = report(invoke(tmp_path, catalog_file, candidates=2))
    assert written["tool"] == "Agent"
    assert written["catalog_size"] == 5
    assert written["candidates_narrowed"] is True
    assert len(written["candidates"]) == 2
    assert written["model_tag"] == "qwen3:8b"
    assert "duration_seconds" in written


def test_nothing_is_written_outside_the_output_directory(
    tmp_path, catalog_file, stub_model
):
    stub_model(routed("Bone_Seg"), extracted({"scans": "/d", "model": "/m"}))
    output_dir = invoke(tmp_path, catalog_file)
    assert sorted(path.name for path in output_dir.iterdir()) == sorted(
        [REPORT_NAME, DECISION_NAME]
    )
    assert sorted(path.name for path in tmp_path.iterdir()) == ["catalog.json", "out"]


def test_nothing_is_printed_to_stdout(tmp_path, catalog_file, stub_model, capsys):
    """The result travels in a file the runner writes. Upstream printed both the
    result AND its own diagnostics to stdout, so a fallback message broke the
    JSON the caller parsed."""
    stub_model(routed("Bone_Seg"), extracted({"scans": "/d", "model": "/m"}))
    invoke(tmp_path, catalog_file)
    assert capsys.readouterr().out == ""


def test_nothing_is_written_to_the_home_directory(
    tmp_path, catalog_file, stub_model, monkeypatch
):
    """`Agent/Agent.py:1078` wrote chat transcripts -- which carry the patient
    folder paths the user typed -- straight into `$HOME`, outside any job."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    stub_model(routed("Bone_Seg"), extracted({"scans": "/d", "model": "/m"}))
    invoke(tmp_path, catalog_file)
    assert list(home.iterdir()) == []


# ---------------------------------------------------------------------------
# Routing refusals
# ---------------------------------------------------------------------------

def test_a_null_tool_is_a_refusal_a_person_can_read(tmp_path, catalog_file, stub_model):
    stub_model(routed(None, 0.1, "nothing here does that"))
    made = decision(invoke(tmp_path, catalog_file))
    assert made["tool"] is None
    assert made["arguments"] == {}
    assert made["errors"]


def test_the_string_null_is_read_as_no_tool(tmp_path, catalog_file, stub_model):
    """qwen3 emits both `null` and the string `"null"`, and upstream had the
    same two special cases."""
    stub_model(routed("null", 0.1))
    assert decision(invoke(tmp_path, catalog_file))["tool"] is None


def test_a_hallucinated_tool_name_is_refused_and_recorded(
    tmp_path, catalog_file, stub_model
):
    """"Choose ONLY from the candidates" is an instruction, not a type.
    Upstream passed the name straight into `build_cli_args`, where it raised a
    `KeyError` about a manifest the user has never seen."""
    stub_model(routed("Segmentation3000", 0.95))
    output_dir = invoke(tmp_path, catalog_file)
    made = decision(output_dir)
    assert made["tool"] is None
    assert made["rejected_tool_name"] == "Segmentation3000"
    assert made["confidence"] <= 0.2
    assert any("Segmentation3000" in line for line in report(output_dir)["warnings"])


def test_a_tool_named_with_different_separators_still_resolves(
    tmp_path, catalog_file, stub_model
):
    stub_model(routed("boneseg", 0.9), extracted({}))
    assert decision(invoke(tmp_path, catalog_file))["tool"] == "Bone_Seg"


def test_a_router_answer_that_is_not_json_is_an_error(
    tmp_path, catalog_file, stub_model
):
    stub_model("I would rather not.")
    with pytest.raises(ToolInputError):
        invoke(tmp_path, catalog_file)


def test_a_confidence_outside_zero_to_one_is_clamped(
    tmp_path, catalog_file, stub_model
):
    stub_model(routed("Bone_Seg", 42.0), extracted({}))
    assert decision(invoke(tmp_path, catalog_file))["confidence"] == 1.0


def test_a_confidence_that_is_not_a_number_becomes_zero(
    tmp_path, catalog_file, stub_model
):
    stub_model(
        json.dumps({"tool": "Bone_Seg", "confidence": "very sure", "reason": ""}),
        extracted({}),
    )
    assert decision(invoke(tmp_path, catalog_file))["confidence"] == 0.0


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def test_a_missing_required_argument_is_named_rather_than_invented(
    tmp_path, catalog_file, stub_model
):
    stub_model(routed("Bone_Seg"), extracted({"scans": "/data"}))
    made = decision(invoke(tmp_path, catalog_file))
    assert made["missing_required"] == ["model"]
    assert made["arguments"] == {"scans": "/data"}


def test_an_extraction_error_lowers_the_parameter_confidence(
    tmp_path, catalog_file, stub_model
):
    stub_model(routed("Bone_Seg"), extracted({"structures": ["NOPE"]}, 1.0))
    made = decision(invoke(tmp_path, catalog_file))
    assert made["parameter_confidence"] == 0.6
    assert made["errors"]


def test_a_dropped_argument_is_reported_by_name(tmp_path, catalog_file, stub_model):
    stub_model(routed("Bone_Seg"), extracted({"sharpen": True}))
    assert decision(invoke(tmp_path, catalog_file))["dropped_arguments"] == ["sharpen"]


def test_a_tool_with_no_fillable_argument_needs_no_extraction_call(
    tmp_path, catalog_file, stub_model
):
    """The stub is scripted for ONE answer, so a second call would fail it."""
    stub = stub_model(routed("No_Argument_Tool"))
    made = decision(invoke(tmp_path, catalog_file, prompt="do the argumentless thing"))
    assert made["tool"] == "No_Argument_Tool"
    assert len(stub.calls) == 1


# ---------------------------------------------------------------------------
# What the model is actually sent
# ---------------------------------------------------------------------------

def test_the_history_reaches_both_calls_as_turns(tmp_path, catalog_file, stub_model):
    stub = stub_model(routed("Bone_Seg"), extracted({}))
    invoke(
        tmp_path, catalog_file,
        history=json.dumps([{"role": "user", "content": "earlier question"}]),
    )
    for call in stub.calls:
        roles = [message["role"] for message in call["messages"]]
        assert roles == ["system", "user", "user"]
        assert call["messages"][1]["content"] == "earlier question"


def test_a_folder_containing_a_comma_stays_one_folder(
    tmp_path, catalog_file, stub_model
):
    """`Agent_CLI.py:88` split a single string on commas with no escaping, so
    `/data/Smith, John/T1` became two paths, neither of which exists."""
    stub = stub_model(routed("Bone_Seg"), extracted({}))
    invoke(tmp_path, catalog_file, folders=["/data/Smith, John/T1", "/data/out"])
    message = stub.user_message(0)
    assert "- /data/Smith, John/T1" in message
    assert "- /data/out" in message
    assert message.count("- /data/") == 2


def test_the_candidate_count_decides_how_many_tools_the_model_sees(
    tmp_path, catalog_file, stub_model
):
    for limit, expected in ((1, 1), (2, 2), (0, 5)):
        stub = stub_model(routed("Bone_Seg"), extracted({}))
        invoke(tmp_path / str(limit), catalog_file, candidates=limit)
        listed = [
            name
            for name in ("Bone_Seg", "Timepoint_Reg", "Mesh_Label", "Note_Reader",
                         "No_Argument_Tool")
            if name in stub.user_message(0)
        ]
        assert len(listed) == expected


def test_the_temperature_and_seed_reach_the_model(tmp_path, catalog_file, stub_model):
    stub = stub_model(routed("Bone_Seg"), extracted({}))
    invoke(tmp_path, catalog_file, temperature=0.3, seed=11)
    assert stub.calls[0]["kwargs"]["temperature"] == 0.3
    assert stub.calls[0]["kwargs"]["seed"] == 11


def test_the_endpoint_is_resolved_once_and_used_for_both_calls(
    tmp_path, catalog_file, stub_model
):
    stub = stub_model(routed("Bone_Seg"), extracted({}))
    invoke(tmp_path, catalog_file, endpoint="http://gpu-box:11434")
    assert {call["endpoint"] for call in stub.calls} == {"http://gpu-box:11434"}


# ---------------------------------------------------------------------------
# Ask mode
# ---------------------------------------------------------------------------

def test_ask_mode_writes_advice_and_proposes_no_tool(
    tmp_path, catalog_file, stub_model
):
    stub_model("## Suggested workflow\n\n1. Run **Bone_Seg** first.")
    output_dir = invoke(tmp_path, catalog_file, mode=MODE_ASK)
    made = decision(output_dir)
    assert made["tool"] is None
    assert made["arguments"] == {}
    assert "Bone_Seg" in made["reasoning"]
    assert (output_dir / ADVICE_NAME).read_text(encoding="utf-8").startswith("## ")


def test_ask_mode_keeps_the_markdown(tmp_path, catalog_file, stub_model):
    """Upstream stripped every `*` and `#` from the answer
    (`Agent_CLI.py:215-216`) so it would look like plain text in a Qt label.
    This writes a `.md` file."""
    stub_model("# Heading\n\n**bold** and *emphasis*")
    output_dir = invoke(tmp_path, catalog_file, mode=MODE_ASK)
    written = (output_dir / ADVICE_NAME).read_text(encoding="utf-8")
    assert "#" in written and "**" in written


def test_ask_mode_never_runs_anything_even_when_execute_is_set(
    tmp_path, catalog_file, stub_model
):
    stub_model("advice, not an action")
    supervisor = FakeSup(tmp_path)
    output_dir = invoke(
        tmp_path, catalog_file, mode=MODE_ASK, execute=True, sup=supervisor
    )
    assert supervisor.calls == []
    assert decision(output_dir)["tool"] is None


def test_ask_mode_asks_for_prose_not_json(tmp_path, catalog_file, stub_model):
    stub = stub_model("prose")
    invoke(tmp_path, catalog_file, mode=MODE_ASK)
    assert stub.calls[0]["kwargs"]["json_format"] is False


# ---------------------------------------------------------------------------
# execute -- off by default, and refused when the proposal is incomplete
# ---------------------------------------------------------------------------

def test_execute_is_off_by_default(tmp_path, catalog_file, stub_model):
    """A model choosing what to run on patient data unattended is a decision a
    person should make. Upstream asked with a Yes/No dialog; that approval had
    to become a field rather than vanish with the Qt."""
    stub_model(routed("Bone_Seg"), extracted({"scans": "/d", "model": "/m"}))
    supervisor = FakeSup(tmp_path)
    output_dir = invoke(tmp_path, catalog_file, sup=supervisor)
    assert supervisor.calls == []
    assert decision(output_dir)["executed"] is False


def test_execute_runs_the_chosen_tool_through_the_supervisor(
    tmp_path, catalog_file, stub_model
):
    stub_model(routed("Bone_Seg"), extracted({"scans": "/d", "model": "/m"}))
    supervisor = FakeSup(tmp_path)
    output_dir = invoke(tmp_path, catalog_file, execute=True, sup=supervisor)

    assert len(supervisor.calls) == 1
    name, parameters = supervisor.calls[0]
    assert name == "Bone_Seg"
    assert parameters["scans"] == "/d" and parameters["model"] == "/m"
    assert decision(output_dir)["executed"] is True


def test_the_chosen_tool_fills_the_bar_from_the_waypoint_to_the_end(
    tmp_path, catalog_file, stub_model
):
    """The callee's own 0..1 is folded into the span it is handed, which starts
    at the waypoint this tool writes before the call and ends the bar."""
    stub_model(routed("Bone_Seg"), extracted({"scans": "/d", "model": "/m"}))
    supervisor = FakeSup(tmp_path)
    invoke(tmp_path, catalog_file, execute=True, sup=supervisor)

    assert supervisor.spans == [("Bone_Seg", (0.1, 1.0))]
    assert "_progress" not in supervisor.calls[0][1]
    assert supervisor.messages[-1] == (0.1, "running Bone_Seg")


def test_the_supervised_run_writes_inside_this_run_s_own_output(
    tmp_path, catalog_file, stub_model
):
    """`output_dir` is ours to give and never the model's to propose."""
    stub_model(routed("Bone_Seg"), extracted({"scans": "/d", "model": "/m"}))
    supervisor = FakeSup(tmp_path)
    output_dir = invoke(tmp_path, catalog_file, execute=True, sup=supervisor)

    _name, parameters = supervisor.calls[0]
    assert parameters["output_dir"] == str(output_dir / "run")
    assert (output_dir / "run" / "produced.txt").exists()
    assert decision(output_dir)["execution"]["output"] == "run"


def test_an_output_dir_the_model_proposed_never_reaches_the_supervisor(
    tmp_path, catalog_file, stub_model
):
    stub_model(
        routed("Bone_Seg"),
        extracted({"scans": "/d", "model": "/m", "output_dir": "/etc"}),
    )
    supervisor = FakeSup(tmp_path)
    output_dir = invoke(tmp_path, catalog_file, execute=True, sup=supervisor)
    _name, parameters = supervisor.calls[0]
    assert parameters["output_dir"] == str(output_dir / "run")


def test_execute_without_a_supervisor_says_what_to_do_instead(
    tmp_path, catalog_file, stub_model
):
    stub_model(routed("Bone_Seg"), extracted({"scans": "/d", "model": "/m"}))
    with pytest.raises(SupervisorRequired) as raised:
        invoke(tmp_path, catalog_file, execute=True)
    assert "run it yourself" in str(raised.value)


def test_execute_is_refused_when_a_required_argument_is_missing(
    tmp_path, catalog_file, stub_model
):
    stub_model(routed("Bone_Seg"), extracted({"scans": "/d"}))
    supervisor = FakeSup(tmp_path)
    output_dir = invoke(tmp_path, catalog_file, execute=True, sup=supervisor)
    made = decision(output_dir)
    assert supervisor.calls == []
    assert made["executed"] is False
    assert "model" in made["execution_refused"]


def test_execute_is_refused_when_no_tool_was_chosen(
    tmp_path, catalog_file, stub_model
):
    stub_model(routed(None, 0.1))
    supervisor = FakeSup(tmp_path)
    output_dir = invoke(tmp_path, catalog_file, execute=True, sup=supervisor)
    assert supervisor.calls == []
    assert decision(output_dir)["executed"] is False


def test_execute_is_refused_when_the_router_says_it_is_guessing(
    tmp_path, catalog_file, stub_model
):
    """A floor, not a quality bar: the model has said in its own answer that it
    does not know, and nothing should run unattended on that."""
    stub_model(routed("Bone_Seg", 0.3), extracted({"scans": "/d", "model": "/m"}))
    supervisor = FakeSup(tmp_path)
    output_dir = invoke(tmp_path, catalog_file, execute=True, sup=supervisor)
    made = decision(output_dir)
    assert supervisor.calls == []
    assert "confidence" in made["execution_refused"]


def test_a_refusal_to_execute_is_a_warning_in_the_report(
    tmp_path, catalog_file, stub_model
):
    stub_model(routed("Bone_Seg", 0.3), extracted({"scans": "/d", "model": "/m"}))
    output_dir = invoke(
        tmp_path, catalog_file, execute=True, sup=FakeSup(tmp_path)
    )
    assert any("Not executed" in line for line in report(output_dir)["warnings"])


def test_a_failing_supervised_tool_carries_its_own_failure_up(
    tmp_path, catalog_file, stub_model
):
    stub_model(routed("Bone_Seg"), extracted({"scans": "/d", "model": "/m"}))
    supervisor = FakeSup(tmp_path, fail=FileNotFoundError("No scan found in '/d'."))
    with pytest.raises(FileNotFoundError) as raised:
        invoke(tmp_path, catalog_file, execute=True, sup=supervisor)
    assert "No scan found" in str(raised.value)


def test_the_supervisor_is_told_what_is_running(tmp_path, catalog_file, stub_model):
    stub_model(routed("Bone_Seg"), extracted({"scans": "/d", "model": "/m"}))
    supervisor = FakeSup(tmp_path)
    invoke(tmp_path, catalog_file, execute=True, sup=supervisor)
    assert any("Bone_Seg" in str(message) for _f, message in supervisor.messages)


def test_a_dict_of_named_outputs_is_recorded(tmp_path, catalog_file, stub_model):
    """A tool returns a `Path` or a `dict[str, Path]`; both have to survive into
    the decision, because the names are what a later chain would wire on."""
    stub_model(routed("Bone_Seg"), extracted({"scans": "/d", "model": "/m"}))
    supervisor = FakeSup(tmp_path, result={"mandible": tmp_path / "m.nii.gz"})
    output_dir = invoke(tmp_path, catalog_file, execute=True, sup=supervisor)
    assert decision(output_dir)["execution"]["returned"] == {
        "mandible": str(tmp_path / "m.nii.gz")
    }


# ---------------------------------------------------------------------------
# Argument refusals
# ---------------------------------------------------------------------------

def test_an_empty_prompt_is_refused_before_any_model_call(
    tmp_path, catalog_file, stub_model
):
    stub = stub_model()
    with pytest.raises(ToolInputError) as raised:
        invoke(tmp_path, catalog_file, prompt="   ")
    assert "empty" in str(raised.value)
    assert stub.calls == []


def test_a_negative_candidate_count_is_refused_and_names_zero(
    tmp_path, catalog_file, stub_model
):
    stub_model()
    with pytest.raises(ToolInputError) as raised:
        invoke(tmp_path, catalog_file, candidates=-1)
    assert "0 to show the model every tool" in str(raised.value)


def test_a_bad_history_is_refused_before_any_model_call(
    tmp_path, catalog_file, stub_model
):
    stub = stub_model()
    with pytest.raises(ToolInputError):
        invoke(tmp_path, catalog_file, history="{not json")
    assert stub.calls == []


def test_a_missing_catalogue_is_refused_before_any_model_call(
    tmp_path, stub_model, monkeypatch
):
    monkeypatch.delenv("SADT_API", raising=False)
    stub = stub_model()
    with pytest.raises(CatalogError):
        run(prompt="anything", output_dir=tmp_path / "out")
    assert stub.calls == []


def test_an_unreachable_model_is_a_deployment_failure(
    tmp_path, catalog_file, monkeypatch
):
    def unreachable(*args, **kwargs):
        raise ToolUnavailableError("No Ollama server answered at 'http://h:1'.")

    monkeypatch.setattr("sadt_agent.llm.chat", unreachable)
    with pytest.raises(ToolUnavailableError):
        invoke(tmp_path, catalog_file)


def test_the_output_directory_is_created(tmp_path, catalog_file, stub_model):
    stub_model(routed("Bone_Seg"), extracted({}))
    output_dir = invoke(tmp_path, catalog_file, output_dir=tmp_path / "a" / "b" / "c")
    assert output_dir.is_dir()
