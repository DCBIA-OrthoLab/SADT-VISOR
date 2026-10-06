"""The six calls VFACE makes, and the parameters each one sends.

Asserted on a fake supervisor, which is all a tool can ever see of a real one.
Nothing is imported across virtualenvs here, any more than it is in the server:
`sup.run("ASO", ...)` starts that tool in its own interpreter and blocks.

What these tests pin is the SEAM -- that each call names the tool it means and
sends the arguments that tool publishes. `test_schema_seam.py` is the other
half: it reads the six real schemas out of process and checks every argument
sent exists.
"""

import os

import pytest

from conftest import FakeSup
from sadt_vface import catalogs, tools
from sadt_vface.errors import SupervisorRequired


@pytest.fixture
def sup(tmp_path):
    def slot(params):
        return params["output_dir"]

    def registration(params):
        # AREG_CBCT groups its results by region and writes one matrix per
        # patient inside. `register` reads that shape back, so a slot with
        # nothing in it would not exercise what the caller actually receives.
        import os
        produced = params["output_dir"]
        region = os.path.join(produced, "CB")
        os.makedirs(region, exist_ok=True)
        open(os.path.join(region, "C_0001_Reg_transform.tfm"), "w").close()
        return produced

    return FakeSup(tmp_path, dict(
        {name: slot for name in ("ASO", "AMASSS", "AutoMatrix", "ALI_CBCT",
                                 "Batch_Dental_Seg")},
        AREG_CBCT=registration,
    ))


# ---------------------------------------------------------------------------
# Refusing at the door
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool", [
    "ASO", "AMASSS", "AutoMatrix", "AREG_CBCT", "ALI_CBCT", "Batch_Dental_Seg",
])
def test_a_mode_that_needs_a_tool_refuses_without_a_supervisor(tool):
    with pytest.raises(SupervisorRequired) as raised:
        tools.require(None, tool, "This mode")
    assert tool in str(raised.value)


@pytest.mark.parametrize("tool", list(tools._ADVICE))
def test_the_refusal_names_a_mode_that_works_instead(tool):
    """"Deploy a tool" is not an answer a clinician can act on. Every one of
    these has a mode that does not need it, and the refusal says which."""
    with pytest.raises(SupervisorRequired) as raised:
        tools.require(None, tool, "This mode")
    assert tools._ADVICE[tool] in str(raised.value)


def test_every_tool_called_carries_advice():
    """Derived from the source, not listed: a call added without advice would
    refuse by naming a deployment problem and nothing else."""
    import re

    called = set(re.findall(r'sup\.run\(\s*"([A-Za-z_]+)"', open(
        os.path.join(os.path.dirname(tools.__file__), "tools.py")
    ).read()))
    assert called, "no sup.run call found -- has the call graph moved?"
    assert called <= set(tools._ADVICE)


def test_a_supervisor_that_is_there_is_not_refused(sup):
    tools.require(sup, "ASO", "This mode")


# ---------------------------------------------------------------------------
# ASO -- the orientation, once per frame
# ---------------------------------------------------------------------------

def test_the_orientation_is_fully_automated_on_cbct(sup, tmp_path):
    frame = catalogs.FRAMES[catalogs.FRAME_CRANIAL_BASE]
    tools.orient_scans(
        sup, str(tmp_path / "scans"), str(tmp_path / "ref"),
        frame["landmarks"], frame["suffix"], str(tmp_path / "bundle"),
        label=catalogs.FRAME_CRANIAL_BASE,
    )
    asked = sup.asked("ASO")
    assert asked["modality"] == "CBCT"
    assert asked["automation"] == "Fully-Automated"
    assert asked["cbct_landmarks"] == ["Ba", "LPo", "N", "RPo", "S", "LOr", "ROr"]
    assert asked["output_suffix"] == "CB_Or"
    assert asked["landmark_model"] == str(tmp_path / "bundle")


def test_each_frame_is_fitted_on_its_own_landmarks(sup, tmp_path):
    """The cranial base frame is the Frankfort horizontal and the mid-sagittal
    plane; the maxillary frame is the occlusal and mid-sagittal plane. Each
    reference is built on its own set of points, and sending one frame's
    landmarks against the other's reference orients the scan onto nothing."""
    for frame_name, frame in catalogs.FRAMES.items():
        tools.orient_scans(
            sup, str(tmp_path / "scans"), str(tmp_path / frame["reference"]),
            frame["landmarks"], frame["suffix"], label=frame_name,
        )
    sent = [params for name, params in sup.calls if name == "ASO"]
    assert [params["cbct_landmarks"] for params in sent] == [
        catalogs.FRAMES[name]["landmarks"] for name in catalogs.FRAMES
    ]
    assert len({params["reference"] for params in sent}) == len(catalogs.FRAMES)


def test_the_two_orientations_land_apart_without_naming_a_folder(sup, tmp_path):
    """Two calls to the same tool, which must not read each other's scans.

    VFACE used to keep them apart by naming a directory per call. That is the
    supervisor's job -- it gives every call its own slot -- and naming one here
    pointed the callee away from the slot `keep_intermediate` collects from. So
    the property is still required, and it is now asserted where it lives: on
    what the calls RETURN, no `output_dir` being sent at all.
    """
    produced = []
    for frame_name, frame in catalogs.FRAMES.items():
        produced.append(tools.orient_scans(
            sup, str(tmp_path / "scans"), str(tmp_path / "ref"),
            frame["landmarks"], frame["suffix"], label=frame_name))
    assert len(set(produced)) == len(produced)
    assert not [params for name, params in sup.calls
                if name == "ASO" and "output_dir" in params]


def test_an_orientation_with_no_bundle_named_does_not_send_an_empty_one(sup, tmp_path):
    """ASO resolves nothing itself. An empty path is not "use the default", it
    is a path that does not exist."""
    frame = catalogs.FRAMES[catalogs.FRAME_MAXILLA]
    tools.orient_scans(sup, str(tmp_path / "scans"), str(tmp_path / "ref"),
                       frame["landmarks"], frame["suffix"])
    assert "landmark_model" not in sup.asked("ASO")


# ---------------------------------------------------------------------------
# AMASSS -- the masks
# ---------------------------------------------------------------------------

def test_the_masks_are_asked_for_one_file_per_structure(sup, tmp_path):
    """A region's mask is looked up by name. A merged multi-label volume makes
    every region resolve to the same file."""
    tools.segment_masks(sup, str(tmp_path / "oriented"), str(tmp_path / "bundle"),
                        ["CBMASK", "MANDMASK"], label="CB")
    asked = sup.asked("AMASSS")
    assert asked["merge"] == ["SEPARATE"]
    assert asked["structures"] == ["CBMASK", "MANDMASK"]
    assert asked["generate_surface"] is False


def test_the_structures_of_a_frame_are_asked_for_in_one_call(sup, tmp_path):
    """AMASSS loads a network per structure and a cohort per call, so both of
    the cranial base frame's structures in one call is one pass over the scans
    rather than two."""
    structures = catalogs.structures_for(catalogs.FRAME_CRANIAL_BASE, catalogs.REGIONS)
    tools.segment_masks(sup, str(tmp_path / "oriented"), str(tmp_path / "bundle"),
                        structures, label="CB")
    assert sup.asked("AMASSS")["structures"] == ["CBMASK", "MANDMASK"]


def test_the_structures_asked_for_are_masks_not_segmentations():
    """AMASSS publishes both, and they are different volumes. `CB` is the
    anatomical segmentation, which follows the bone; `CBMASK` is the region a
    registration is confined to. Sending the first would hand AREG a
    segmentation where it expects a mask -- it would run, produce a transform,
    and report success on the wrong anatomy."""
    asked = {entry["structure"] for entry in catalogs.REGION_TABLE.values()}
    assert asked == {"CBMASK", "MANDMASK", "MAXMASK"}
    assert not asked & {"CB", "MAND", "MAX"}


# ---------------------------------------------------------------------------
# AutoMatrix -- the mirror
# ---------------------------------------------------------------------------

def test_the_mirror_uses_one_transform_for_every_patient(sup, tmp_path):
    """It is a reflection of the frame, not something fitted per patient."""
    tools.mirror(sup, str(tmp_path / "oriented"), str(tmp_path / "mirror.tfm"))
    asked = sup.asked("AutoMatrix")
    assert asked["same_transform_for_every_patient"] is True
    assert asked["output_suffix"] == "mir"
    assert asked["transforms"] == str(tmp_path / "mirror.tfm")


def test_a_mask_is_mirrored_as_a_segmentation_not_as_a_scan(sup, tmp_path):
    """Interpolating a label map linearly invents labels that are neither of
    the two it sits between."""
    tools.mirror(sup, str(tmp_path / "masks"), str(tmp_path / "mirror.tfm"),
                 content="Segmentation", label="masks")
    assert sup.asked("AutoMatrix")["content"] == "Segmentation"


# ---------------------------------------------------------------------------
# AREG_CBCT -- the registration
# ---------------------------------------------------------------------------

def test_the_registration_is_semi_automated_on_masks_vface_already_has(sup, tmp_path):
    """Asking AREG to segment and orient again would redo two steps and --
    worse -- orient the mirror independently of the original. A mirror is only
    a mirror while it shares the original's frame."""
    tools.register(sup, str(tmp_path / "t1"), str(tmp_path / "t2"),
                   catalogs.REGION_MANDIBLE, str(tmp_path / "masks"))
    asked = sup.asked("AREG_CBCT")
    assert asked["automation"] == "Semi-Automated"
    assert asked["regions"] == ["Mandible"]
    assert asked["t1_masks"] == str(tmp_path / "masks")


def test_each_region_is_registered_into_its_own_slot(sup, tmp_path):
    """Same property as the orientations, for the three registrations: each
    comes back with its own directory, and none of them asked for one."""
    produced = []
    for region in catalogs.REGIONS:
        produced.append(tools.register(
            sup, str(tmp_path / "t1"), str(tmp_path / "t2"),
            catalogs.REGION_TABLE[region]["areg"], str(tmp_path / "masks")))
    assert len(set(produced)) == len(catalogs.REGIONS)
    assert not [params for name, params in sup.calls
                if name == "AREG_CBCT" and "output_dir" in params]


def test_the_region_names_sent_are_the_ones_areg_publishes():
    """`REGION_TABLE` holds VFACE's names on one side and AREG's on the other,
    and they are not the same vocabulary. The `areg` column is what travels."""
    assert sorted(entry["areg"] for entry in catalogs.REGION_TABLE.values()) == [
        "Cranial base", "Mandible", "Maxilla"
    ]


# ---------------------------------------------------------------------------
# ALI_CBCT -- the landmarks
# ---------------------------------------------------------------------------

def test_the_landmarks_are_asked_for_by_name_not_by_region(sup, tmp_path):
    """ALI spawns one search agent per landmark, at about a minute each.
    Asking for a region's whole catalogue searches dozens of points no
    measurement reads."""
    tools.predict_landmarks(sup, str(tmp_path / "registered"), ["Ba", "S", "N"],
                            str(tmp_path / "bundle"))
    asked = sup.asked("ALI_CBCT")
    assert asked["landmarks"] == ["Ba", "S", "N"]
    assert asked["model"] == str(tmp_path / "bundle")


# ---------------------------------------------------------------------------
# Batch_Dental_Seg -- the surfaces
# ---------------------------------------------------------------------------

def test_the_surfaces_are_asked_for_with_the_bundle_named(sup, tmp_path):
    """`Batch_Dental_Seg.run` takes `model` with no default, so an omitted one
    is a TypeError one tool down rather than a refusal here."""
    tools.segment_surfaces(sup, str(tmp_path / "registered"), str(tmp_path / "bundle"))
    assert sup.asked("Batch_Dental_Seg")["model"] == str(tmp_path / "bundle")


# ---------------------------------------------------------------------------
# The bundles this deployment fills in
# ---------------------------------------------------------------------------

class TestTheBundlesVfaceResolvesItself:
    """`models/` holds six bundles at once, so an unnamed argument arrived as
    that FOLDER and the run died after the segmentation and the orientation had
    already been paid for. None of them is a clinical choice, so the panel asks
    for none and the deployment names them instead."""

    def _root(self, tmp_path, *names):
        for name in names:
            # exist_ok: two arguments share `DefaultList`, the measurement
            # lists and the feature template living in one bundle.
            (tmp_path / "VFACE" / "models" / name).mkdir(parents=True, exist_ok=True)
        return str(tmp_path)

    def test_each_bundle_is_found_under_the_data_root(self, tmp_path):
        from sadt_vface import dispatch

        root = self._root(tmp_path, *dispatch._BUNDLES.values())
        for argument, name in dispatch._BUNDLES.items():
            assert dispatch._own_bundle(root, argument) == str(
                tmp_path / "VFACE" / "models" / name), argument

    def test_a_deployment_publishing_none_gets_an_empty_string(self, tmp_path):
        """Empty, not a path that does not exist: the checks downstream say
        what is missing, and they can only do that if handed nothing."""
        from sadt_vface import dispatch

        root = self._root(tmp_path)
        assert dispatch._own_bundle(root, "segmentation_model") == ""
        assert dispatch._own_bundle(None, "segmentation_model") == ""

    def test_the_landmark_bundle_is_not_resolved_here(self, tmp_path):
        """ALI_CBCT resolves its own weights from the same data root, so the
        empty string it gets is the right answer rather than an omission."""
        from sadt_vface import dispatch

        assert "landmark_model" not in dispatch._BUNDLES
        assert dispatch._own_bundle(self._root(tmp_path), "landmark_model") == ""

    def test_the_surface_bundle_is_not_resolved_here(self, tmp_path):
        """Nothing publishes it yet, and heat maps are the only path that asks
        for one. Pinned so its absence stays a known gap."""
        from sadt_vface import dispatch

        assert "surface_model" not in dispatch._BUNDLES


class TestWhereTheRegistrationPutItsMatrices:
    """AREG_CBCT groups its results by region, so the directory it returns holds
    no transform at all -- the matrices are one level down, beside its report.

    Nothing caught this until the chain was run for the first time: AutoMatrix,
    handed that directory three steps later, could only say it held no
    transform, with no way to name the tool that produced it. The segmentation
    and the three registrations had been paid for by then.
    """

    def _with_transform(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "C_0001_Reg_transform.tfm").write_text("")
        return folder

    def test_one_subfolder_holding_the_matrices_is_descended_into(self, tmp_path):
        from sadt_vface import tools

        root = tmp_path / "output"
        self._with_transform(root / "CB")
        (root / "AREG_report.json").write_text("{}")
        assert tools._region_folder(str(root)) == str(root / "CB")

    def test_a_flat_output_is_left_alone(self, tmp_path):
        """A tool that writes its matrices at the top level needs no descent."""
        from sadt_vface import tools

        root = self._with_transform(tmp_path / "output")
        assert tools._region_folder(str(root)) == str(root)

    def test_no_transform_anywhere_is_refused_with_what_was_found(self, tmp_path):
        """The refusal names the directory and its contents, so a reader can
        see whether the registration wrote nothing or wrote it elsewhere."""
        from sadt_vface import tools
        from sadt_vface.errors import ToolInputError

        root = tmp_path / "output"
        (root / "CB").mkdir(parents=True)
        (root / "AREG_report.json").write_text("{}")
        with pytest.raises(ToolInputError, match="AREG_report.json"):
            tools._region_folder(str(root))

    def test_the_refusal_carries_areg_unmatched_counts_never_the_names(self, tmp_path):
        """AREG_CBCT writes "unmatched", not "unpaired". Read under the old key
        only, the one explanation that survives the job's deletion was empty."""
        import json

        from sadt_vface import tools
        from sadt_vface.errors import ToolInputError

        root = tmp_path / "output"
        (root / "CB").mkdir(parents=True)
        (root / "AREG_report.json").write_text(json.dumps({
            "unmatched": {"t1_without_t2": ["P1", "P2"], "t2_without_t1": ["P3"]},
            "patients": {},
        }))
        with pytest.raises(ToolInputError, match="unmatched: 2 T1-only, 1 T2-only") as raised:
            tools._region_folder(str(root))
        assert "P1" not in str(raised.value) and "P3" not in str(raised.value)

    def test_two_region_folders_are_refused_rather_than_guessed(self, tmp_path):
        """Only a single-region call can be descended into unambiguously."""
        from sadt_vface import tools
        from sadt_vface.errors import ToolInputError

        root = tmp_path / "output"
        for name in ("CB", "MAND"):
            self._with_transform(root / name)
        with pytest.raises(ToolInputError, match="no transform"):
            tools._region_folder(str(root))
