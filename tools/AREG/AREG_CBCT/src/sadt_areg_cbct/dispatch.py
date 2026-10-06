"""AREG -- Automated REGistration of two timepoints.

Ported from the Slicer extension's `AREG/` module and its CLI modules
(`AREG_CBCT`, `AREG_IOS`). One tool, two engines, five modes:

|          | Semi-Automated                | Fully-Automated              | Oriented + Fully-Automated |
|----------|-------------------------------|------------------------------|----------------------------|
| **CBCT** | your T1 masks, masked Elastix | AMASSS segments the T1 masks | ASO orients the T1 first   |
| **IOS**  | your segmented meshes         | CrownSeg labels + ASO orients| --                         |

The Slicer envelope is gone entirely: no `<filter-progress>` prints, no
`time.sleep(0.2)` progress theatre, no `sys.exit`, no log file the client
polls, and nothing written into the caller's input tree.

Two entry points, for the same reason AMASSS and ASO have two:

* `register(...)` -> `RegistrationRun`, the real API: the output directory plus
  a structured report. This is what another server-side tool calls.
* `main(...)` -> the output directory's path, the schema adapter `AREG.py` uses.

The Slicer widget built a list of CLI invocations per mode and ran them in
order, passing folders between them. That structure survives, but the steps are
the other packaged tools -- see `tools.py` for how they are called
in-process and through the registry.
"""

import json
import logging
import os
import shutil

from sadt_areg_common.errors import ToolInputError

from sadt_areg_common import catalogs, pairing
from . import dicom, progress, tools

logger = logging.getLogger(__name__)

# This tool IS the modality: it is no longer an argument, so the value the
# report carries and the automation table is keyed by is fixed here.
MODALITY = catalogs.MODALITY_CBCT

REPORT_NAME = "AREG_report.json"

# Intermediates live here, under the output directory the caller owns, and are
# removed before `register` returns. A surviving `.areg_work/` means a run
# crashed.
WORK_DIRNAME = ".areg_work"


class RegistrationRun:
    """Result of `register()`: where the files are, and what actually happened.

    Reported per patient AND per region, because a CBCT run registering on the
    cranial base and the mandible is two registrations of every patient and one
    of them can fail on its own. The original caught each per-patient exception
    into a log line and finished by printing a count; the archive said nothing.
    """

    def __init__(self, output_dir: str, report: dict):
        self.output_dir = output_dir
        self.report = report

    @property
    def patients(self) -> dict:
        return self.report["patients"]

    @property
    def succeeded(self) -> list:
        return [key for key, entry in self.patients.items() if entry.get("status") == "ok"]



# The DATA folder this tool's bundles live in, and the segmentation bundle
# inside it. Written rather than derived, for the reason ALI_CBCT gives about
# its own: which folder serves which engine is a DEPLOYMENT fact (AREG_CBCT,
# AREG_IOS and AREG_IOSCBCT share `DATA/AREG/`), and a wrong guess is a folder
# that is simply not there.
#
# The bundle name is the one `scripts/data-manifest.yml` unpacks AMASSS's
# archive to for AREG, so an installation set up by `setup-models.sh` has it
# without anybody choosing anything.
_DATA_NAME = "AREG"
_SEGMENTATION_BUNDLE = "AMASSS_Models"
# Which bundle each frame is defined by lives in the shared catalog
# (`catalogs.ORIENTATION_BUNDLES`), because the panel offers the frame by name
# and this resolves the name -- one table, read from both ends.


def _own_segmentation(data_root):
    """The AMASSS bundle this deployment publishes for AREG, or "".

    There is nothing for a clinician to decide here: the modes that segment
    segment with AMASSS, and the only bundle that answers is the one the
    deployment already holds. Asking which folder to use was asking a question
    with one possible answer -- and one that a caller could get wrong in ways
    that only surface fifteen seconds into a child process.

    Returns "" rather than raising, so the single refusal below keeps saying
    what is missing, now including where this looked.
    """
    if not data_root:
        return ""
    candidate = os.path.join(str(data_root), _DATA_NAME, "models", _SEGMENTATION_BUNDLE)
    return candidate if os.path.isdir(candidate) else ""


def _own_reference(data_root, orientation: str) -> str:
    """The bundle this deployment publishes for the frame `orientation` names.

    A reference defines its frame through what it CARRIES -- the two published
    bundles hold disjoint landmark sets -- so naming the frame is naming the
    bundle, and this is the whole of the translation.

    Two reasons the path is resolved here rather than asked for. `reference`
    carries `server_selectable = "model"`, so a request that names nothing
    arrives holding `DATA/AREG/models/`: the FOLDER, eight bundles deep, which
    ASO cannot orient onto. And a dropdown of bundle FOLDERS asks a clinician to
    recognise `CBCT_Gold_Frankfurt_Horizontal_Midsagittal_Plane` where the
    question is "Frankfurt horizontal or occlusal plane".

    Returns "" for the frame that is no frame, and for a deployment that does
    not publish the bundle -- the refusal in `_check_cbct` then keeps saying what
    is missing.
    """
    bundle = catalogs.ORIENTATION_BUNDLES.get(str(orientation or ""))
    if not data_root or not bundle:
        return ""
    candidate = os.path.join(str(data_root), _DATA_NAME, "models", bundle)
    return candidate if os.path.isdir(candidate) else ""


def derive_automation(automation: str, t1_masks,
                     orientation: str = catalogs.ORIENTATION_NONE) -> tuple:
    """The mode this request really is. Returns `(mode, source)`.

    `source` is "from the data" or "requested".

    One of the three modes is written in the request and the other two differ by
    a preference:

    * `t1_masks` -- masks the caller made -- is Semi-Automated. There is nothing
      else to do with them, and asking anyway meant a Fully-Automated run could
      segment over a folder of masks somebody had prepared;
    * with none, AMASSS makes them, and what is left to decide is WHICH FRAME the
      scans come back in. No folder answers that -- there are two published
      frames and they mean different things -- so it is asked as `orientation`,
      by name, rather than hidden inside a three-valued mode or behind a boolean
      that could only say "the default one".

    **The orientation `reference` is deliberately NOT read here**, though it is
    what the oriented mode needs. It carries `server_selectable = "model"`, so
    the server fills it from `DATA/AREG/models/` whenever the client leaves it
    empty -- it is never absent, and a run sending masks was refused for
    "sending both" on a reference nobody had named. Measured through the server
    on 2026-09-29, after 113 unit tests had passed on the assumption.

    A named mode overrides all of it, and that is the one thing naming a mode
    still does: "Fully-Automated" beside a folder of masks means "segment anyway,
    I know they are there".
    """
    if automation and automation != catalogs.AUTOMATION_AUTO:
        return automation, "requested"
    if t1_masks:
        return catalogs.AUTOMATION_SEMI, "from the data"
    chose_a_frame = (str(orientation or catalogs.ORIENTATION_NONE)
                     != catalogs.ORIENTATION_NONE)
    return (catalogs.AUTOMATION_ORIENTED if chose_a_frame
            else catalogs.AUTOMATION_FULLY), "from the data"


def _check_cbct(automation: str, regions: list, t1_masks, reference,
                segmentation_model=None, sup=None, landmark_model=None) -> None:
    if not regions:
        raise ToolInputError(
            "Select at least one anatomical region to register on in 'cbct_regions' "
            f"({', '.join(catalogs.REGION_CHOICES)}). Each one is a separate "
            "registration with its own output folder."
        )

    if automation == catalogs.AUTOMATION_SEMI:
        if not t1_masks:
            raise ToolInputError(
                "Semi-Automated CBCT registers inside masks you provide: send the T1 "
                "segmentations in 't1_masks', or use Fully-Automated mode to have "
                "them produced server-side."
            )
        return

    # Both automated modes need the segmentation; the oriented one also needs
    # the orientation. Checked before the input is extracted -- with the tool
    # absent, the answer is the same whatever the rest of the request says.
    tools.require(sup, "AMASSS", f"{automation} CBCT registration")
    if automation == catalogs.AUTOMATION_ORIENTED:
        tools.require(sup, "ASO", "Oriented + Fully-Automated CBCT registration")
        if not reference:
            raise ToolInputError(
                "Oriented + Fully-Automated CBCT orients the T1 scans before "
                "registering onto them, which needs an orientation reference: name "
                "one in 'cbct_reference' (see GET /tools/AREG_CBCT/data)."
            )
        # `landmark_model` is NOT required here any more. Which weights the
        # landmark tool predicts with is that tool's business -- ALI_CBCT
        # resolves its own from the deployment's data folder, the way ASO's
        # comment says it should -- and demanding a name here made AREG hold a
        # name for its neighbour's storage. An explicit one is still obeyed.

    if not segmentation_model:
        # Named here rather than left to AMASSS, which receives None and fails on
        # `TypeError: expected str, bytes or os.PathLike object, not NoneType` --
        # fifteen seconds in, from inside a child process, and opaque to whoever
        # sent the request. The tool is reachable; what is missing is which
        # weights it should load.
        raise ToolInputError(
            f"{automation} CBCT segments the T1 scans before registering, which "
            f"needs the segmentation weights. This deployment publishes none: "
            f"no '{_SEGMENTATION_BUNDLE}' under DATA/{_DATA_NAME}/models/. Add it "
            f"(scripts/setup-models.sh --tool AREG), or name another bundle in "
            f"'segmentation_model' (see GET /tools/AREG_CBCT/data)."
        )



# Where the anatomical segmentations land in the caller's output. A folder of
# their own: the registration writes one tree per region, and a mandible
# segmentation belongs to neither of them.
SEGMENTATION_DIRNAME = "Segmentations"


def _collect_segmentations(amasss_dir, codes, output_dir, report) -> None:
    """Copy the requested anatomy out of AMASSS's folder into the caller's.

    AMASSS runs here as a STEP of the chain, so its output sits in the
    supervisor's scratch and comes back only to someone who ticked
    `keep_intermediate`. The original module writes these segmentations into
    the user's own output folder, and a check box that produces files nobody
    receives would be worse than no check box -- so what was asked for is
    copied out, and only that.

    The names are AMASSS's own and deterministic:
    `<base>_<ID>_SegOut/<base>_<ID>_<CODE><extension>` (see its
    `_assemble_scan_outputs`), which is what makes picking the requested
    structures out of a folder holding the masks too a match rather than a
    guess.
    """
    if not codes or not amasss_dir or not os.path.isdir(amasss_dir):
        return
    wanted = set(codes)
    destination = os.path.join(output_dir, SEGMENTATION_DIRNAME)
    collected = []
    for root, _dirs, files in os.walk(amasss_dir):
        for name in sorted(files):
            stem = name.split(".")[0]
            code = stem.rsplit("_", 1)[-1] if "_" in stem else ""
            if code not in wanted:
                continue
            target_dir = os.path.join(destination, os.path.basename(root))
            os.makedirs(target_dir, exist_ok=True)
            shutil.copy2(os.path.join(root, name), os.path.join(target_dir, name))
            collected.append(os.path.join(SEGMENTATION_DIRNAME,
                                          os.path.basename(root), name))
    report["segmentations"] = sorted(collected)


# Where the oriented T1 lands in the caller's output, with ASO's landmarks and
# transform beside it.
ORIENTED_DIRNAME = "T1_Oriented"


def _collect_oriented(oriented_dir, output_dir, report) -> None:
    """Copy ASO's oriented T1 tree into the caller's output.

    The registered T2 is written in the ORIENTED T1's frame, so it is only
    readable next to that T1 -- laid over the T1 the caller sent, it sits as far
    off as the orientation moved it. The oriented scans live in the
    supervisor's scratch and were thrown away with it, so the one volume the
    result has to be read against never reached anyone. Copied whole: the
    scan, `<name>_lm_Or.mrk.json` and `<name>_Or_transform.tfm` each say
    something the other two do not, and ASO's own report says how it went.
    """
    if not oriented_dir or not os.path.isdir(oriented_dir):
        return
    destination = os.path.join(output_dir, ORIENTED_DIRNAME)
    shutil.copytree(oriented_dir, destination, dirs_exist_ok=True)
    report["oriented"] = sorted(
        os.path.relpath(os.path.join(root, name), output_dir)
        for root, _dirs, files in os.walk(destination) for name in files
    )


# How a CBCT run shares its bar, as fractions of the whole. The supervised
# steps come first because they run first, and the registration loop takes
# what is left. Orientation is a full ASO run, landmark prediction through ALI
# included; segmentation is AMASSS over the T1 cohort; registration is elastix
# per region per subject, the longest of the three on any real cohort, so it
# keeps the larger share. A weighting, not a measurement: what is exact is the
# counter in each message.
ORIENT_SHARE = 0.25
SEGMENT_SHARE = 0.25


def _cbct_spans(orient: bool, segment: bool) -> tuple:
    """(orientation, segmentation, registration) spans, None for a step not run.

    Each step starts where the one before it ended, so the bar only moves
    forward, and a step a mode skips takes no slice at all rather than leaving
    a jump where it would have been.
    """
    position = 0.0
    orientation = segmentation = None
    if orient:
        orientation = (position, position + ORIENT_SHARE)
        position += ORIENT_SHARE
    if segment:
        segmentation = (position, position + SEGMENT_SHARE)
        position += SEGMENT_SHARE
    return orientation, segmentation, (position, 1.0)


def _run_cbct(
    t1_root, t2_root, t1_masks_path, automation, regions, segmentation_model,
    segmentation_label, orientation_reference, dicom_input, output_dir, work_dir,
    suffix, report, sup=None, landmark_model=None, segmentations=None, num_workers=0,
) -> None:
    # Imported here rather than at module level: the CBCT engine pulls in
    # SimpleITK and itk-elastix, and AREG must load on a server without them so
    # its schema is still published and its IOS mode still runs.
    from . import elastix
    from . import pipeline as cbct_pipeline

    elastix.check_dependencies()

    # Asked of the DATA, not of the caller, the way ASO's CBCT engine asks it.
    # DICOM slices routinely carry no extension, so a clinician could not tell
    # from a file name either -- and answering wrong produced a run that failed
    # for a reason nobody could see. `dicom_input` remains an OVERRIDE for a
    # caller who knows better than the detector, which is why it stays in the
    # signature and leaves the panel.
    #
    # The two timepoints are asked separately: a cohort half exported as DICOM
    # and half already converted is somebody's real Tuesday, and one flag for
    # both would have made them choose which half to break.
    if dicom_input or dicom.holds_a_series(t1_root):
        t1_root = dicom.convert_tree(t1_root, os.path.join(work_dir, "dicom_t1"))
    if dicom_input or dicom.holds_a_series(t2_root):
        t2_root = dicom.convert_tree(t2_root, os.path.join(work_dir, "dicom_t2"))

    # Descended ONCE, here, before anything reads either folder: a hosted test
    # entry is a whole cohort (`<name>/{T1,T2}/`) because that is all a picker
    # can offer, and every step below -- the segmentation, the pairing, the
    # output names -- has to be looking at the same directory.
    t1_root = pairing.timepoint_root(t1_root, "T1")
    t2_root = pairing.timepoint_root(t2_root, "T2")

    codes = [catalogs.region_code(name) for name in regions]
    report["regions"] = list(regions)
    report["segmentation_label"] = segmentation_label or None

    orient_span, segment_span, register_span = _cbct_spans(
        orient=automation == catalogs.AUTOMATION_ORIENTED,
        segment=automation != catalogs.AUTOMATION_SEMI,
    )

    # Step 1 -- orient the T1 scans, when the mode asks for it. The T2 is NOT
    # oriented: it is about to be resampled into the T1's frame anyway, and
    # orienting it first would be one more interpolation of the same data.
    if automation == catalogs.AUTOMATION_ORIENTED:
        progress.emit(orient_span[0], "orienting the T1 scans with ASO")
        oriented = tools.orient_scans(
            sup,
            t1_root, orientation_reference, catalogs.MODALITY_CBCT,
            landmark_model=landmark_model or "",
            span=orient_span,
        )
        report["oriented_t1"] = True
        _collect_oriented(oriented, output_dir, report)
        t1_root = oriented

    # Step 2 -- the masks the registration is confined to.
    mask_roots = []
    if t1_masks_path:
        mask_roots.append(_as_directory(t1_masks_path, os.path.join(work_dir, "masks_input")))
    if automation == catalogs.AUTOMATION_SEMI:
        # Where the original looked when no mask folder was given.
        mask_roots.append(t1_root)
    else:
        masks = [catalogs.REGION_MASK_STRUCTURES[code] for code in codes]
        # One AMASSS call for both: the masks the registration consumes and the
        # anatomy the caller ticked. Asking twice would segment the same scan
        # twice, and the card is serialised.
        wanted = [catalogs.SEGMENTATION_CODES[name]
                  for name in _selected(segmentations, catalogs.SEGMENTATION_CHOICES)]
        structures = masks + [code for code in wanted if code not in masks]
        progress.emit(segment_span[0], "segmenting the T1 scans with AMASSS")
        amasss_dir = tools.segment_masks(
            sup, t1_root, segmentation_model, structures, span=segment_span
        )
        mask_roots.append(amasss_dir)
        report["segmented_t1"] = sorted(structures)
        _collect_segmentations(amasss_dir, wanted, output_dir, report)

    # Step 3 -- pair the timepoints, then register once per region.
    matched = pairing.pair(t1_root, t2_root, suffix)
    report["unmatched"] = matched.unmatched_report()
    if not matched:
        raise ToolInputError(
            "No subject appears in both the T1 and the T2 folder. They are paired by "
            "name, up to the timepoint token and a trailing "
            f"{', '.join(catalogs.PATIENT_SUFFIXES[:4])}... -- so 'P1_T1_scan.nii.gz' "
            f"in one folder pairs with 'P1_T2.nii.gz' in the other. Found "
            f"{len(matched.t1_only)} T1-only and {len(matched.t2_only)} T2-only subject(s)."
        )

    # Every registration is independent -- one region of one subject, on that
    # subject's two scans -- so they are listed first and run as wide as the
    # machine pays for. Each is still one elastix thread and the same
    # deterministic computation, so running them side by side changes the
    # order they finish in and nothing they produce.
    jobs = []
    for code in codes:
        masks = cbct_pipeline.find_masks(mask_roots, code, scan_keys=matched.matched)
        if not masks:
            # Every subject of this region is about to fail on the same missing
            # mask. Said once, to the clinician who asked for the region: their
            # result will hold none of it, and the report says so only per
            # subject.
            _log(sup, f"no {catalogs.region_name(code)} mask for any subject; "
                      "that region is not registered", level="warning", user=True)
        for key, entry in sorted(matched.matched.items()):
            record = report["patients"].setdefault(key, {"status": "ok", "regions": {}})
            mask_path = masks.get(key)
            if not mask_path:
                record["regions"][code] = {
                    "status": "failed",
                    "reason": _no_mask_reason(automation, code),
                }
                continue
            jobs.append((code, key, {
                "t1_path": entry["t1"],
                "t2_path": entry["t2"],
                "mask_path": mask_path,
                "region": code,
                "output_dir": output_dir,
                "relative_key": key,
                "suffix": suffix,
                "segmentation_label": segmentation_label or None,
            }))

    register_start, register_end = register_span
    width = _registration_width(sup, len(jobs), num_workers)
    logger.info("AREG CBCT: %d registration(s), %d at a time", len(jobs), width)
    # Declared around the registrations and nowhere else: the peak is there,
    # one pair of volumes per channel, and what runs before it is the chain's.
    progress.set_width(width)
    # Said BEFORE the registrations start, not only as each one ends: they are
    # the long step, and a failure inside them is diagnosed by the last thing
    # this tool said it was doing.
    progress.emit(register_start, "registering {} region(s) of {} subject(s)".format(
        len(codes), len(matched.matched)))
    try:
        finished = cbct_pipeline.register_all(
            [kwargs for _code, _key, kwargs in jobs], width,
            # Inside the registration's span of the bar, which starts where the
            # supervised steps above ended -- not 0..1 again.
            on_done=lambda done, total: progress.emit(
                register_start + (register_end - register_start) * done / total,
                "registration {} of {}".format(done, total)),
        )
    finally:
        progress.set_width(None)
    # In the order they were listed, whatever order they finished in.
    for (code, key, _kwargs), entry in zip(jobs, finished):
        report["patients"][key]["regions"][code] = entry

    _roll_up_regions(report["patients"])


def _registration_width(sup, wanted: int, declared: int = 0) -> int:
    """How many registrations to run at once: what the machine will pay for.

    The supervisor answers from what one registration was measured to cost and
    what this run's reservation holds. A number the caller named is a ceiling on
    the ask, never a floor. One without a supervisor unless the caller named a
    number -- how this tool runs from a CLI and in its own tests.
    """
    wanted = max(1, int(wanted or 1))
    try:
        declared = int(declared or 0)
    except (TypeError, ValueError):
        declared = 0
    if declared > 0:
        wanted = min(wanted, declared)
    ask = getattr(sup, "channels", None)
    if ask is None:
        return wanted if declared > 0 else 1
    try:
        return max(1, min(wanted, int(ask(wanted))))
    except Exception:  # noqa: BLE001 - a grant must never fail a run
        logger.warning("Could not ask for channels; registering one at a time",
                       exc_info=True)
        return 1


def _log(sup, message: str, level: str = "info", user: bool = False) -> None:
    """`sup.log` when there is a supervisor, the progress file's log otherwise.

    Never a file name: the line reaches the operator page, and with `user`
    the clinician's panel.
    """
    if sup is not None and hasattr(sup, "log"):
        sup.log(message, level=level, user=user)
    else:
        progress.log(message, level=level, user=user)


def _no_mask_reason(automation: str, code: str) -> str:
    region = catalogs.region_name(code)
    if automation == catalogs.AUTOMATION_SEMI:
        return (
            f"no {region} mask for this subject. A mask is matched to its scan by name "
            f"and has to say both that it is a segmentation (mask/seg/pred) and which "
            f"structure it covers ({'/'.join(catalogs.REGION_TOKENS[code][:2])}) -- "
            f"e.g. 'P1_T1_{code}_seg.nii.gz' next to 'P1_T1_scan.nii.gz'"
        )
    return (
        f"the segmentation step produced no {region} mask for this subject -- see the "
        f"AMASSS report if one was included in this archive"
    )


def _roll_up_regions(patients: dict) -> None:
    """A patient is 'ok' when at least one of its regions registered."""
    for entry in patients.values():
        statuses = [region.get("status") for region in entry["regions"].values()]
        entry["status"] = "ok" if "ok" in statuses else "failed"


def _selected(value, choices: dict) -> list:
    """The enabled options of a multichoice argument, in declaration order.

    Accepts the `Selection` validate() produces, a plain dict, or a sequence --
    so `register()` stays directly callable with `["Mandible"]`.

    An option nobody offers is REFUSED, not dropped. `Literal` is published,
    not enforced -- the runner calls `run(**params)` from a JSON object -- so a
    stale client naming a region that no longer exists used to be handed a
    narrower registration than it asked for, with nothing in the report saying
    which of its regions had gone missing. In the dict form only an ENABLED
    unknown counts, because a client sending back the whole `{option: checked}`
    dict it was given is not asking for the boxes it left unticked.
    """
    if value is None:
        return [name for name, on in choices.items() if on]
    if isinstance(value, dict):
        wanted = {name for name, on in value.items() if on}
    else:
        wanted = set(value)
    unknown = sorted(wanted - set(choices))
    if unknown:
        raise ToolInputError(
            f"{', '.join(repr(name) for name in unknown)} is not something this tool "
            f"can register on. It offers: {', '.join(choices)}."
        )
    return [name for name in choices if name in wanted]


def _as_directory(path: str, destination: str) -> str:
    """A directory holding the input, whatever shape it arrived in.

    A single uploaded file is linked into a directory of its own rather than
    used from where it landed: main.py streams every upload of a request into
    ONE work directory, so treating a file's parent as an input root would make
    the T2 folder part of the T1 one.
    """
    path = str(path)
    if os.path.isdir(path):
        return path

    os.makedirs(destination, exist_ok=True)
    linked = os.path.join(destination, os.path.basename(path))
    try:
        os.link(path, linked)
    except OSError:
        shutil.copy2(path, linked)
    return destination


def _summarize(report: dict) -> None:
    statuses = [entry.get("status") for entry in report["patients"].values()]
    report["summary"] = {
        "patients": len(statuses),
        "registered": statuses.count("ok"),
        "failed": statuses.count("failed"),
    }
    logger.info(
        "AREG %s %s: %d/%d registered",
        report["modality"],
        report["automation"],
        report["summary"]["registered"],
        report["summary"]["patients"],
    )


def register(
    t1_path: str,
    t2_path: str,
    automation: str,
    regions=None,
    t1_masks_path: str = None,
    segmentation_model: str = None,
    segmentation_label: int = 0,
    orientation_reference: str = None,
    landmark_model: str = None,
    dicom_input: bool = False,
    orientation: str = catalogs.ORIENTATION_NONE,
    output_suffix: str = "Reg",
    output_dir: str = None,
    sup=None,
    segmentations=None,
    num_workers=0,
) -> RegistrationRun:
    """Register every T2 under `t2_path` onto its T1 under `t1_path`.

    Each path is a directory or a `.zip`. `regions` are the display names
    declared in `catalogs.REGION_CHOICES` (CBCT only).
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    work_dir = os.path.join(output_dir, WORK_DIRNAME)
    os.makedirs(work_dir, exist_ok=True)

    t1_root = _as_directory(t1_path, os.path.join(work_dir, "t1_input"))
    t2_root = _as_directory(t2_path, os.path.join(work_dir, "t2_input"))

    # Resolved here, in the real API, so a direct caller gets the same rule the
    # schema adapter gets. Idempotent: a named mode passes through unchanged.
    automation, automation_source = derive_automation(
        automation, t1_masks_path, orientation
    )

    report = {
        "modality": MODALITY,
        "automation": automation,
        # Which of the two it was, because "Semi-Automated" in a report does not
        # say whether anybody chose it.
        "automation_source": automation_source,
        "output_suffix": output_suffix,
        "patients": {},
    }

    _run_cbct(
        t1_root=t1_root,
        t2_root=t2_root,
        t1_masks_path=t1_masks_path,
        automation=automation,
        regions=list(regions or ()),
        segmentation_model=segmentation_model,
        segmentation_label=int(segmentation_label or 0),
        orientation_reference=orientation_reference,
        dicom_input=dicom_input,
        output_dir=output_dir,
        work_dir=work_dir,
        suffix=output_suffix,
        report=report,
        sup=sup,
        landmark_model=landmark_model,
        segmentations=segmentations,
        num_workers=num_workers,
    )

    # Extracted inputs, converted DICOM, the oriented copies and whatever the
    # tools it drove wrote. Removed whether or not the run succeeded, so what is
    # left under output_dir is results and nothing else.
    shutil.rmtree(work_dir, ignore_errors=True)

    _summarize(report)
    with open(os.path.join(output_dir, REPORT_NAME), "w") as handle:
        json.dump(report, handle, indent=2)
    return RegistrationRun(output_dir, report)


def main(
    automation,
    t1,
    t2,
    t1_masks=None,
    cbct_regions=None,
    segmentations=None,
    segmentation_label=0,
    segmentation_model=None,
    cbct_reference=None,
    landmark_model=None,
    dicom_input=False,
    orientation=None,
    output_suffix="Reg",
    output_dir=None,
    sup=None,
    data_root=None,
    num_workers=0,
) -> str:
    """Translate the schema's arguments into `register()` and return its output
    directory, which main.py zips and streams.

    Every cross-argument rule is checked HERE, before any file is read: a
    request that cannot work must come back in a second, not after an hour of
    registration. `require` in tools.py is part of that: a mode that needs
    another tool fails at the door when there is no supervisor to reach it.
    """
    automation = str(automation)
    suffix = (output_suffix or "Reg").strip() or "Reg"
    if os.sep in suffix or (os.altsep and os.altsep in suffix):
        raise ToolInputError("'output_suffix' is a name fragment, not a path.")

    allowed = catalogs.AUTOMATION_BY_MODALITY.get(MODALITY, ())
    if automation not in allowed:
        raise ToolInputError(
            f"'{automation}' is not a mode {MODALITY} has. {MODALITY} offers: "
            f"{', '.join(allowed)}."
        )

    regions = _selected(cbct_regions, catalogs.REGION_CHOICES)
    reference = cbct_reference
    # Resolved BEFORE the checks, so the refusal below judges what will
    # actually be loaded rather than what the caller happened to name.
    if not segmentation_model:
        segmentation_model = _own_segmentation(data_root)
    # Same treatment, and the hidden field makes it the only one: a `reference`
    # nobody named arrives as the whole models FOLDER (see `_own_reference`), so
    # "did the caller name one" is asked of the bundle, not of the string.
    # A `reference` nobody named arrives as the whole models FOLDER (see
    # `_own_reference`), so "did the caller name a bundle" is asked of the
    # bundle, not of the string -- and the frame the panel offered answers it.
    if not reference or os.path.basename(str(reference).rstrip(os.sep)) == "models":
        reference = _own_reference(data_root, orientation) or reference
    # The checks judge the mode that will RUN, not the word the request carried:
    # with `automation` left on its default, that word names no mode. `register`
    # derives it again from the same inputs, so the rule lives in one place and
    # the report says which of the two it was.
    resolved, _source = derive_automation(automation, t1_masks, orientation)
    _check_cbct(
        resolved, regions, t1_masks, reference, segmentation_model, sup,
        landmark_model,
    )

    run = register(
        t1_path=str(t1),
        t2_path=str(t2),
        automation=automation,
        regions=regions,
        t1_masks_path=str(t1_masks) if t1_masks else None,
        segmentation_model=str(segmentation_model) if segmentation_model else None,
        segmentation_label=int(segmentation_label or 0),
        orientation_reference=str(reference) if reference else None,
        landmark_model=landmark_model,
        dicom_input=bool(dicom_input),
        orientation=orientation,
        output_suffix=suffix,
        output_dir=output_dir,
        sup=sup,
        segmentations=segmentations,
        num_workers=num_workers,
    )

    return run.output_dir
