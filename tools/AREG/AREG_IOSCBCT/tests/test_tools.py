"""The four tools this one drives, and what it does when it cannot reach them.

AREG_IOSCBCT predicts nothing itself: the landmarks come from ALI_CBCT and
ALI_IOS, the tooth labels from Crown_Seg and the orientation from ASO -- each
in its own virtualenv, started as a subprocess by the supervisor. That is what
lets this tool drive an engine pinned to torch 2.8 and another pinned to 2.11
while depending on neither.

So the seam is a set of STRINGS and a set of parameter names, and it is the
only thing a test can hold: the arguments are the callee's published schema,
not this tool's vocabulary, and when a tool renames one this file is what
breaks. `test_schema_seam.py` checks those names against the real schemas.
"""

import logging
import os

import pytest

from conftest import FakeSup, cohort, read_report, write_markups, write_mesh, UPPER_LABELS, moved
from sadt_areg_ioscbct import dispatch, tools
from sadt_areg_common import catalogs
from sadt_areg_common.errors import SupervisorRequired, ToolInputError

FOUR = ("Crown_Seg", "ALI_CBCT", "ALI_IOS", "ASO")


def planted(tmp_path, name):
    """A directory a fake callee 'produced'."""
    destination = tmp_path / "planted" / name
    destination.mkdir(parents=True, exist_ok=True)
    return destination


# ---------------------------------------------------------------------------
# require: the mode that cannot run
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool", FOUR)
def test_a_mode_needing_a_tool_says_so_when_there_is_no_supervisor(tool):
    """Nothing about the request is wrong -- there is simply no way to reach
    the other tool. So the message names the mode that DOES work rather than an
    argument to change, because "deploy a tool" is not something the person who
    sent the request can act on."""
    with pytest.raises(SupervisorRequired) as raised:
        tools.require(None, tool, "Semi-Automated IOSCBCT registration")
    message = str(raised.value)
    assert tool in message
    assert "Semi-Automated IOSCBCT registration" in message
    assert "no supervisor" in message


@pytest.mark.parametrize("tool", FOUR)
def test_every_refusal_carries_an_alternative_the_caller_can_act_on(tool):
    """The advice is the useful half. Somebody who cannot reach ALI_CBCT can
    still send their own CBCT landmarks."""
    with pytest.raises(SupervisorRequired) as raised:
        tools.require(None, tool, "a mode")
    assert len(str(raised.value).split("no supervisor was supplied.")[1].strip()) > 20


def test_the_advice_names_the_argument_or_the_mode_that_replaces_the_tool():
    expectations = {
        "Crown_Seg": "Registration",
        "ALI_CBCT": "cbct_landmarks",
        "ALI_IOS": "ios_landmarks",
        "ASO": "already oriented",
    }
    for tool, expected in expectations.items():
        with pytest.raises(SupervisorRequired) as raised:
            tools.require(None, tool, "a mode")
        assert expected in str(raised.value), tool


def test_a_supervisor_makes_the_same_mode_acceptable(tmp_path):
    assert tools.require(FakeSup(tmp_path), "ALI_IOS", "anything") is None


def test_an_unknown_tool_still_refuses_rather_than_passing_through():
    """`_ADVICE.get(...)` returns an empty string for a name nobody wrote
    advice for; the refusal itself must not depend on the table."""
    with pytest.raises(SupervisorRequired, match="Nonexistent"):
        tools.require(None, "Nonexistent", "a mode")


# ---------------------------------------------------------------------------
# What each call asks for
# ---------------------------------------------------------------------------

def test_the_crown_request_names_crown_segs_arguments(tmp_path):
    """`skip_segmented` is left at its default: a mesh already carrying a
    tooth-label array passes through untouched, so a batch mixing segmented and
    raw meshes costs network time only for the ones that need it."""
    sup = FakeSup(tmp_path, {"Crown_Seg": lambda params: planted(tmp_path, "seg")})
    returned = tools.label_crowns(sup, str(tmp_path / "ios"), "/models/crown")

    params = sup.asked("Crown_Seg")
    assert params["meshes"] == str(tmp_path / "ios")
    assert params["suffix"] == "Seg"
    assert params["model"] == "/models/crown"
    assert "skip_segmented" not in params
    assert returned == str(planted(tmp_path, "seg"))


def test_the_crown_request_omits_the_model_when_none_was_named(tmp_path):
    """Sending `model=""` is not the same as not sending it: the callee's own
    default is what a caller who named nothing is asking for."""
    sup = FakeSup(tmp_path, {"Crown_Seg": lambda params: planted(tmp_path, "seg")})
    tools.label_crowns(sup, str(tmp_path / "ios"), "")
    assert "model" not in sup.asked("Crown_Seg")


def test_the_cbct_landmarks_are_asked_for_by_name_not_by_region(tmp_path):
    """The registration uses a handful of points, and asking by region would
    run every agent of every region containing one of them. One agent is a full
    two-scale walk of the volume."""
    sup = FakeSup(tmp_path, {"ALI_CBCT": lambda params: planted(tmp_path, "lm")})
    tools.predict_cbct_landmarks(sup, str(tmp_path / "cbct"), "/models/ali")

    params = sup.asked("ALI_CBCT")
    assert params["landmarks"] == list(tools.CBCT_LANDMARKS)
    assert "regions" not in params
    assert params["model"] == "/models/ali"
    # NOT sent, and asserted rather than simply dropped: ALI fixed its marker
    # at `Pred` precisely because a caller that moved it made the output file
    # name unpredictable. Sending it again is an unexpected keyword now.
    assert "prediction_ID" not in params


def test_the_twelve_cbct_landmarks_are_three_per_quadrant():
    """Verbatim from upstream's IOSCBCT parameter dict: the central incisor,
    the canine and the first molar of each quadrant -- the points visible in
    BOTH modalities. A crown tip is a crown tip either way."""
    assert len(tools.CBCT_LANDMARKS) == 12
    for quadrant in ("UR", "UL", "LR", "LL"):
        assert [name for name in tools.CBCT_LANDMARKS if name.startswith(quadrant)] == [
            f"{quadrant}1O", f"{quadrant}3O", f"{quadrant}6O"
        ]
    # Occlusal points only, which is why the intraoral side asks for that
    # network alone.
    assert all(name.endswith("O") for name in tools.CBCT_LANDMARKS)


def test_the_intraoral_request_asks_for_the_occlusal_family_alone(tmp_path):
    """The cross-modality alignment matches crown points against their CBCT
    counterparts; the cervical and mucogingival passes would cost a run over
    every mesh for points nothing here reads."""
    sup = FakeSup(tmp_path, {"ALI_IOS": lambda params: planted(tmp_path, "lm")})
    tools.predict_ios_landmarks(sup, str(tmp_path / "seg"), "")

    params = sup.asked("ALI_IOS")
    assert params["networks"] == ["Occlusal"]
    assert "model" not in params
    assert "prediction_ID" not in params


def test_the_orientation_request_is_asos_fully_automated_cbct_mode(tmp_path):
    """Upstream reaches this through three separate CLI modules
    (PRE_ASO_CBCT, SEMI_ASO_CBCT, PRE_ASO_IOS); ours is one tool taking the
    mode as data."""
    sup = FakeSup(tmp_path, {"ASO": lambda params: planted(tmp_path, "or")})
    tools.orient_cbct(sup, str(tmp_path / "cbct"), "/models/gold", "/models/ali")

    params = sup.asked("ASO")
    assert params["modality"] == "CBCT"
    assert params["automation"] == "Fully-Automated"
    assert params["reference"] == "/models/gold"
    assert params["landmark_model"] == "/models/ali"


def test_with_no_reference_named_aso_is_asked_for_the_frame(tmp_path):
    """The reference bundles are ASO's own: this engine names the frame and
    hands over none of them."""
    sup = FakeSup(tmp_path, {"ASO": lambda params: planted(tmp_path, "or")})
    tools.orient_cbct(sup, str(tmp_path / "cbct"), "", frame="Frankfurt horizontal")

    params = sup.asked("ASO")
    assert params["frame"] == "Frankfurt horizontal"
    assert "reference" not in params


def test_no_callee_is_told_where_to_write(tmp_path):
    """Where a supervised tool writes is the SUPERVISOR's business.

    This used to name a scratch directory per callee, so that two of them could
    not land in one folder. The supervisor already solves that -- it gives each
    nested call its own slot -- and naming one instead pointed the callee AWAY
    from that slot, which is the only place `keep_intermediate` collects from.
    Every step then came back to the reader as "a call that wrote nowhere".

    The directory each call RETURNS is what the caller reads from, so nothing
    downstream needs the path to have been chosen here.
    """
    sup = FakeSup(
        tmp_path,
        {
            "Crown_Seg": lambda params: planted(tmp_path, "a"),
            "ALI_IOS": lambda params: planted(tmp_path, "b"),
        },
    )
    crown = tools.label_crowns(sup, str(tmp_path / "ios"), "")
    ios = tools.predict_ios_landmarks(sup, str(tmp_path / "seg"), "")

    assert "output_dir" not in sup.asked("Crown_Seg")
    assert "output_dir" not in sup.asked("ALI_IOS")
    # And the caller still knows where each one put its files.
    assert crown != ios
    for directory in (crown, ios):
        assert os.path.isdir(directory)


def test_a_tool_returning_a_dict_of_paths_is_reduced_to_a_directory(tmp_path):
    """A tool returns a `Path`, or a dict of named ones. AREG wants a folder to
    walk either way."""
    assert tools._returned({"landmarks": "/x/y"}) == "/x/y"
    assert tools._returned("/x/y") == "/x/y"


def test_the_long_running_calls_report_progress(tmp_path):
    """A supervised chain is three tools deep and minutes long; a run that
    prints nothing reads as a hang."""
    sup = FakeSup(tmp_path, {"ALI_CBCT": lambda params: planted(tmp_path, "lm")})
    tools.predict_cbct_landmarks(sup, str(tmp_path / "cbct"), "")
    assert sup.messages
    assert "ALI_CBCT" in sup.messages[0][1]


def test_every_tool_is_reachable_without_a_progress_capable_supervisor(tmp_path):
    """`sup` is duck-typed. A supervisor with no `progress` is a supervisor."""
    class Minimal:
        tmp = tmp_path / "tmp"

        def run(self, tool, **params):
            return planted(tmp_path, "x")

    (tmp_path / "tmp").mkdir()
    assert tools.predict_cbct_landmarks(Minimal(), str(tmp_path / "cbct"), "")


# ---------------------------------------------------------------------------
# The modes, end to end, with the four tools stood in for
# ---------------------------------------------------------------------------

def semi_automated(tmp_path, **overrides):
    """A Semi-Automated request whose callees plant real, usable output."""
    cohort(tmp_path)
    outputs = {
        "Crown_Seg": lambda params: tmp_path / "ios",
        "ALI_IOS": lambda params: tmp_path / "ios_lm",
        "ALI_CBCT": lambda params: tmp_path / "cbct_lm",
        "ASO": lambda params: tmp_path / "cbct",
    }
    outputs.update(overrides.pop("outputs", {}))
    sup = FakeSup(tmp_path, outputs)

    arguments = {
        "ios": str(tmp_path / "ios"),
        "cbct": str(tmp_path / "cbct"),
        "output_dir": str(tmp_path / "out"),
        "automation": catalogs.AUTOMATION_SEMI,
        "sup": sup,
    }
    arguments.update(overrides)
    return sup, arguments


def test_semi_automated_labels_the_crowns_and_predicts_both_landmark_sets(tmp_path):
    sup, arguments = semi_automated(tmp_path)
    dispatch.main(**arguments)

    assert [name for name, _params in sup.calls] == ["Crown_Seg", "ALI_IOS", "ALI_CBCT"]
    assert read_report(tmp_path / "out")["patients"]["1"]["status"] == "ok"


def test_semi_automated_does_not_orient_anything(tmp_path):
    """Orienting the CBCT is what Fully-Automated adds, and it needs a
    reference the Semi-Automated caller was never asked for."""
    sup, arguments = semi_automated(tmp_path)
    dispatch.main(**arguments)
    assert "ASO" not in [name for name, _params in sup.calls]


def test_the_landmark_predictions_run_on_the_labelled_meshes(tmp_path):
    """Crown_Seg's output is what ALI_IOS is pointed at, and what is finally
    registered -- not the raw meshes the caller sent."""
    sup, arguments = semi_automated(tmp_path)
    sup.outputs["Crown_Seg"] = lambda params: tmp_path / "labelled"
    (tmp_path / "labelled").mkdir()
    for name in os.listdir(str(tmp_path / "ios")):
        os.link(str(tmp_path / "ios" / name), str(tmp_path / "labelled" / name))

    dispatch.main(**arguments)
    assert sup.asked("ALI_IOS")["input"] == str(tmp_path / "labelled")


def test_landmarks_the_caller_supplied_are_used_instead_of_predicted(tmp_path):
    """A caller who already has one side's points should not pay for them
    again -- and the tool that would have produced them is not called."""
    sup, arguments = semi_automated(tmp_path, ios_landmarks=str(tmp_path / "ios_lm"))
    dispatch.main(**arguments)
    assert [name for name, _params in sup.calls] == ["Crown_Seg", "ALI_CBCT"]


def test_fully_automated_orients_the_cbct_first(tmp_path):
    """The CBCT is put in the reference frame BEFORE anything is matched onto
    it, so the landmarks ALI_CBCT predicts are already in that frame."""
    sup, arguments = semi_automated(
        tmp_path,
        automation=catalogs.AUTOMATION_FULLY,
        cbct_reference="/models/gold",
    )
    dispatch.main(**arguments)

    assert [name for name, _params in sup.calls][0] == "ASO"
    assert sup.asked("ALI_CBCT")["input"] == str(tmp_path / "cbct")


def test_fully_automated_asks_aso_for_its_frame_and_no_bundle(tmp_path):
    """Every tool owns its model. With no reference named, ASO is asked for
    the Frankfurt horizontal frame by name and resolves its own bundle; this
    engine no longer refuses for a bundle it was never meant to hold."""
    sup, arguments = semi_automated(tmp_path, automation=catalogs.AUTOMATION_FULLY)
    dispatch.main(**arguments)

    params = sup.asked("ASO")
    assert params["frame"] == "Frankfurt horizontal"
    assert "reference" not in params


def test_a_sibling_tool_failing_takes_the_run_with_it(tmp_path):
    """Nothing downstream can be salvaged: there are no landmarks to register
    on, and inventing some would be a confident wrong answer."""
    sup, arguments = semi_automated(
        tmp_path,
        outputs={"ALI_IOS": lambda params: RuntimeError("ALI_IOS exited 1")},
    )
    with pytest.raises(RuntimeError, match="ALI_IOS exited 1"):
        dispatch.main(**arguments)


def test_a_sibling_producing_nothing_usable_is_a_server_fault_naming_it(tmp_path, caplog):
    """ALI_IOS returning an empty folder is a real outcome -- but not the
    caller's fault, so not a 422, and the message has to say which tool's
    output came back empty. The shortfall is logged as soon as it returns."""
    empty = tmp_path / "empty"
    empty.mkdir()
    sup, arguments = semi_automated(
        tmp_path, outputs={"ALI_IOS": lambda params: empty}
    )
    with caplog.at_level(logging.INFO, logger="sadt_areg_ioscbct"):
        with pytest.raises(RuntimeError) as raised:
            dispatch.main(**arguments)
    assert not isinstance(raised.value, ValueError)
    assert "No intraoral landmark could be read from ALI_IOS's output" in str(raised.value)
    assert any("ALI_IOS returned 0 landmark file(s) for 1 sent" in r.getMessage()
               and r.levelno == logging.WARNING for r in caplog.records)


def test_a_supervised_run_leaves_no_working_directory_behind(tmp_path):
    sup, arguments = semi_automated(tmp_path)
    dispatch.main(**arguments)
    assert not os.path.exists(str(tmp_path / "out" / dispatch.WORK_DIRNAME))


def test_the_predicted_landmarks_really_reach_the_registration(tmp_path):
    """The whole point of the supervised modes: what comes back from ALI is
    what the fit is computed from."""
    sup, arguments = semi_automated(tmp_path)
    write_markups(
        tmp_path / "cbct_lm" / "P_0001_T2_lm_Pred.mrk.json",
        moved(UPPER_LABELS, shift=(7.0, 0.0, 0.0)),
    )
    dispatch.main(**arguments)

    import numpy as np

    matrix = np.load(str(tmp_path / "out" / "1" / "P001_T2_U_Reg_matrix.npy"))
    assert matrix[0, 3] == pytest.approx(7.0, abs=1e-5)


def test_a_sibling_returning_fewer_files_than_it_was_sent_is_warned_about(tmp_path, caplog):
    """The tool ran to the end, so nothing raised: the shortfall would only show
    later, as a patient with nothing to register."""
    sup, arguments = semi_automated(tmp_path)
    # Two meshes sent, Crown_Seg hands back a folder holding one.
    write_mesh(tmp_path / "ios" / "P001_T2_L.vtk")
    labelled = tmp_path / "labelled"
    write_mesh(labelled / "P001_T2_U.vtk")
    sup.outputs["Crown_Seg"] = lambda params: labelled

    with caplog.at_level(logging.INFO, logger="sadt_areg_ioscbct"):
        dispatch.main(**arguments)
    shortfall = [r for r in caplog.records
                 if r.getMessage() == "Crown_Seg returned 1 labelled mesh(es) for 2 sent"]
    assert shortfall and shortfall[0].levelno == logging.WARNING
