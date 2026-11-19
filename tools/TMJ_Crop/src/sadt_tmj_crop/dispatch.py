"""The chain: ASO orients, AMASSS segments the mandible, then the crop.

Both supervised tools take a whole folder, so each is called ONCE for the batch
(the server pays a process start-up cost per call). The crop then runs per scan
on what they returned.

A scan that fails at any step gets a short JSON saying so and the rest of the
batch goes on; a batch that produces nothing raises, so a caller never receives
an empty folder and a success.

Messages carry the position in the batch ("scan 3 of 40"), never a file name:
they are stored on the server and shown, and a name is patient metadata.
"""

import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path

from . import catalogs, crop, progress, tools
from .errors import ToolInputError

logger = logging.getLogger(__name__)

# The slices of the whole run each stage occupies, for the caller's bar.
ORIENT_SPAN = (0.0, 0.45)
SEGMENT_SPAN = (0.45, 0.85)
CROP_SPAN = (0.85, 1.0)


_SCAN_EXTENSIONS = (".nii.gz", ".nii", ".nrrd", ".gipl.gz", ".gipl")
_PATH_LIKE = re.compile(r"\S*[/\\]\S*|\S+\.(?:nii|nrrd|gz|json|vtk|stl|gipl)\S*", re.IGNORECASE)


def _safe_reason(reason, limit: int = 120) -> str:
    """A reason from another tool, with anything that looks like a path replaced.

    ASO's reasons come from libraries that quote the file they were reading, and a
    file name is patient metadata: the report is stored and shown.
    """
    text = str(reason or "").strip()
    # Everything from the first path-like word on is dropped, not just that word: a file name
    # with spaces in it would otherwise leave its middle words behind.
    found = _PATH_LIKE.search(text)
    if found:
        text = text[: found.start()] + "<file>"
    return " ".join(text.split())[:limit]


def _tell(sup, message: str) -> None:
    """A warning for the clinician's panel, through the supervisor this tool has."""
    if hasattr(sup, "log"):
        sup.log(message, level="warning", user=True)
    else:
        progress.log(message, level="warning", user=True)


def _strip_extension(name: str) -> str:
    for extension in _SCAN_EXTENSIONS:
        if name.lower().endswith(extension):
            return name[: -len(extension)]
    return name


def _scan_end(name: str, suffix: str) -> str:
    """The `_<suffix><extension>` a file name ends in, or "" when it is not an oriented scan."""
    for extension in _SCAN_EXTENSIONS:
        if name.endswith(f"_{suffix}{extension}"):
            return f"_{suffix}{extension}"
    return ""


def _as_nifti(cases, suffix: str, sup, output_dir: Path) -> list:
    """The same cases with every oriented scan a NIfTI file, converted by Convert.

    ASO writes the oriented scan in the input's family, so a `.nrrd` gives a `.nrrd`.
    Convert is called once, and only when something needs it. A scan Convert could
    not read turns its case into a failed one rather than stopping the batch.
    """
    wanted = [(i, scan) for i, (_, _, _, scan, _, _) in enumerate(cases)
              if scan is not None and not scan.name.endswith(".nii.gz")]
    if not wanted:
        return cases
    stage = Path(tempfile.mkdtemp(prefix=".tmj_convert_", dir=output_dir))
    try:
        for i, scan in wanted:
            extension = _scan_end(scan.name, suffix)[len(f"_{suffix}"):]
            os.symlink(scan, stage / f"{i}_{suffix}{extension}")
        try:
            converted = Path(tools.convert(sup, str(stage)))
        except Exception as error:  # noqa: BLE001 - the NIfTI scans are still good
            logger.warning("Convert failed (%s)", type(error).__name__)
            converted = stage / "_none"
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    result = list(cases)
    for i, _ in wanted:
        path = converted / f"{i}_{suffix}.nii.gz"
        stem, folder, case, _, landmarks, key = cases[i]
        if path.is_file():
            result[i] = (stem, folder, case, path, landmarks, key)
        else:
            case = {**case, "status": "failed", "reason": "the oriented scan could not be converted to NIfTI"}
            result[i] = (stem, folder, case, None, landmarks, key)
    return result


def _oriented_cases(oriented_dir: Path):
    """[(stem, folder, case, oriented scan, landmarks file, key)], the report and the suffix.

    Every file is taken from the paths ASO's report lists (relative to the folder the
    report sits in), never searched for by name: two cases can share a file name in
    different subfolders, and a name is not a path.
    """
    report_path = next(iter(sorted(oriented_dir.rglob("ASO_report.json"))), None)
    if report_path is None:
        raise ToolInputError("ASO wrote no report, so there is nothing to crop.")
    root = report_path.parent
    report = json.loads(report_path.read_text())
    suffix = report.get("output_suffix", "Or")
    landmarks_end = f"_lm_{suffix}.mrk.json"
    cases = []
    for key, case in sorted(report.get("cases", {}).items()):
        scan = landmarks = None
        for produced in case.get("produced", []):
            path = root / produced
            if _scan_end(produced, suffix) and path.is_file():
                scan = path
            elif produced.endswith(landmarks_end) and path.is_file():
                landmarks = path
        folder = os.path.dirname(key)
        stem = (scan.name[: -len(_scan_end(scan.name, suffix))] if scan is not None
                else _strip_extension(os.path.basename(key)))
        cases.append((stem, folder, case, scan, landmarks, key))
    return cases, report, suffix


def _unique_names(cases) -> list:
    """One output name per case, different even on a disk that ignores case. The file stem
    when those are all different; when two cases share one (a T1 and a T2 folder holding
    the same name) the subfolder is part of every name; a clash that remains gets its
    position, so nothing is overwritten and nothing is matched to the wrong scan."""
    stems = [stem for stem, *_ in cases]
    if len({stem.casefold() for stem in stems}) == len(stems):
        return stems
    names = [
        "_".join([part for part in re.split(r"[\\/]+", folder) if part] + [stem])
        for stem, folder, *_ in cases
    ]
    taken, clashing = set(), []
    for position, name in enumerate(names):
        if name.casefold() in taken:
            clashing.append(position)
        else:
            taken.add(name.casefold())
    for position in clashing:
        # The first of a clash keeps its name; the others get a number, until it is free.
        number = position + 1
        while f"{names[position]}_{number}".casefold() in taken:
            number += 1
        names[position] = f"{names[position]}_{number}"
        taken.add(names[position].casefold())
    return names


def _reference(report: dict) -> dict:
    """The reference landmarks (RAS) ASO oriented onto, from its own report."""
    positions = report.get("reference_positions")
    if not positions:
        raise ToolInputError(
            "ASO's report carries no reference positions, so the crop cannot be "
            "placed. This tool needs a version of ASO that publishes them."
        )
    return {name: crop.lps_to_ras(point) for name, point in positions.items()}


def _discard(output_dir: Path, name: str) -> None:
    """Remove what a case wrote before it failed, so no file is left that the report disowns."""
    for file_name in (f"{name}_R.nii.gz", f"{name}_L.nii.gz", f"{name}_qc.png",
                      f"{name}_oriented.nii.gz", f"{name}_mandible.nii.gz"):
        (output_dir / file_name).unlink(missing_ok=True)


def _write_failure(output_dir: Path, name: str, reason: str) -> None:
    (output_dir / f"{name}.json").write_text(
        json.dumps({"scan": name, "status": reason}, indent=1)
    )


SIDES = {"Left and Right": ("R", "L"), "Left": ("L",), "Right": ("R",)}

# The run report every tool of the catalogue writes beside its results: the server
# reads `cases` (what each case was given and produced) to relate files to cases.
REPORT_NAME = "run_report.json"


def _inputs_by_key(scans: Path) -> dict:
    """{ASO's case key: path of that scan relative to what the caller sent}.

    ASO keys a case by its relative path without the extension; the report states
    the path as the caller sent it, which is what `input` means.
    """
    if scans.is_file():
        return {_strip_extension(scans.name): scans.name}
    found = {}
    for folder, _, file_names in os.walk(scans):
        for file_name in sorted(file_names):
            relative = os.path.relpath(os.path.join(folder, file_name), scans)
            found.setdefault(_strip_extension(relative), relative)
            found.setdefault(relative, relative)
    return found


def crop_scans(scans, output_dir, scale_by, scale, sup, sides="Left and Right",
               outputs=()) -> Path:
    """Crop the requested joints of every CBCT under `scans`; returns `output_dir`.

    The oriented scan and the mandible segmentation are copied next to the
    crops only when asked for: they are the size of the scan, and most callers
    want the crops alone.
    """
    if sides not in SIDES:
        raise ToolInputError(f"sides must be one of {', '.join(SIDES)}.")
    unknown = sorted(set(outputs) - set(catalogs.OUTPUT_HELP))
    if unknown:
        raise ToolInputError(f"Unknown extra output(s): {', '.join(unknown)}.")
    tools.require(sup)
    scans = Path(scans)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        settings = crop.CropSettings(scale_by=scale_by, scale=scale)
    except ValueError as error:
        raise ToolInputError(str(error)) from None

    oriented_dir = Path(tools.orient(sup, str(scans), span=ORIENT_SPAN))
    cases, report, suffix = _oriented_cases(oriented_dir)
    cases = _as_nifti(cases, suffix, sup, output_dir)
    reference = _reference(report)
    names = _unique_names(cases)

    given = _inputs_by_key(scans)
    entries = {}

    def note(name, key, status, produced):
        entry = {"status": status, "produced": produced}
        if given.get(key):
            entry["input"] = given[key]
        entries[name] = entry

    ready, reasons = [], []
    for position, (name, (_, _, case, scan, landmarks, key)) in enumerate(zip(names, cases), start=1):
        if case.get("status") != "ok" or scan is None or landmarks is None:
            reason = _safe_reason(case.get("reason")) or str(case.get("status") or "no result")
            reasons.append(reason)
            step = "conversion failed" if "converted to NIfTI" in reason else "orientation failed"
            _write_failure(output_dir, name, f"{step}: {reason}")
            note(name, key, f"{step}: {reason}", [f"{name}.json"])
            logger.warning("scan %d of %d: %s", position, len(cases), step)
            _tell(sup, f"scan {position} of {len(cases)}: {step} ({reason})")
            continue
        ready.append((name, case, scan, landmarks, key))
    logger.info("TMJ_Crop: %d of %d scan(s) oriented", len(ready), len(cases))
    if not ready:
        raise ToolInputError(
            f"ASO oriented none of the {len(cases)} scan(s), so nothing was cropped."
            + (f" First reason: {reasons[0]}." if reasons and reasons[0] else "")
        )

    stage = Path(tempfile.mkdtemp(prefix=".tmj_", dir=output_dir))
    try:
        for name, _, scan, _, _ in ready:
            os.symlink(scan, stage / f"{name}_{suffix}.nii.gz")
        mask_dir = Path(tools.segment_mandible(sup, str(stage), span=SEGMENT_SPAN))
        # Looked up by exact name, never as a glob pattern: a scan called `scan[1]`
        # must not be read as a character class.
        masks = {
            file_name: Path(folder) / file_name
            for folder, _, file_names in os.walk(mask_dir) for file_name in file_names
        }

        done = partial = 0
        for index, (name, case, scan, landmarks, key) in enumerate(ready, start=1):
            progress.report(index, len(ready), "scan", start=CROP_SPAN[0], end=CROP_SPAN[1])
            mask = masks.get(f"{name}_{suffix}_Pred_MAND.nii.gz")
            if mask is None:
                logger.warning("scan %d of %d: AMASSS wrote no mandible mask", index, len(ready))
                _write_failure(output_dir, name, "no mandible mask")
                note(name, key, "no mandible mask", [f"{name}.json"])
                _tell(sup, f"scan {index} of {len(ready)}: no mandible mask")
                continue
            try:
                result = crop.crop_scan(
                    scan, landmarks, mask, reference,
                    used=case.get("landmarks_used", []),
                    dropped=case.get("landmarks_dropped", {}),
                    settings=settings, name=name, output_dir=str(output_dir),
                    sides=SIDES[sides],
                )
                crop.write_quality_image(result, reference, name, str(output_dir))
            except ValueError as error:
                reason = _safe_reason(error)
                logger.warning("scan %d of %d: crop refused: %s", index, len(ready), reason)
                _discard(output_dir, name)
                _write_failure(output_dir, name, f"crop refused: {reason}")
                note(name, key, f"crop refused: {reason}", [f"{name}.json"])
                _tell(sup, f"scan {index} of {len(ready)}: crop refused ({reason})")
                continue
            except Exception as error:  # noqa: BLE001 - one damaged scan must not cost the batch
                # Only the class: the message of an I/O or library error can quote a path.
                logger.warning("scan %d of %d: crop failed (%s)", index, len(ready), type(error).__name__)
                _discard(output_dir, name)
                _write_failure(output_dir, name, f"crop failed: {type(error).__name__}")
                note(name, key, f"crop failed: {type(error).__name__}", [f"{name}.json"])
                _tell(sup, f"scan {index} of {len(ready)}: crop failed ({type(error).__name__})")
                continue
            produced = [f"{name}_{side}.nii.gz" for side in ("R", "L") if side in result.crops]
            produced += [f"{name}.json", f"{name}_qc.png"]
            try:
                if catalogs.OUTPUT_ORIENTED_SCAN in outputs:
                    shutil.copyfile(scan, output_dir / f"{name}_oriented.nii.gz")
                    produced.append(f"{name}_oriented.nii.gz")
                if catalogs.OUTPUT_MANDIBLE in outputs:
                    shutil.copyfile(mask, output_dir / f"{name}_mandible.nii.gz")
                    produced.append(f"{name}_mandible.nii.gz")
            except OSError as error:
                logger.warning("scan %d of %d: extra output failed (%s)", index, len(ready), type(error).__name__)
                _discard(output_dir, name)
                _write_failure(output_dir, name, f"crop failed: {type(error).__name__}")
                note(name, key, f"crop failed: {type(error).__name__}", [f"{name}.json"])
                _tell(sup, f"scan {index} of {len(ready)}: crop failed ({type(error).__name__})")
                continue
            problems = [
                f"{side}: {result.report['sides'][side]['status']}"
                for side in SIDES[sides]
                if result.report["sides"].get(side, {}).get("status") != "ok"
            ]
            if not result.crops:
                note(name, key, "no requested joint found", produced)
                _tell(sup, f"scan {index} of {len(ready)}: no requested joint found")
            elif problems:
                done += 1
                partial += 1
                note(name, key, "partial: " + "; ".join(problems), produced)
                _tell(sup, f"scan {index} of {len(ready)}: partial ({'; '.join(problems)})")
            else:
                done += 1
                note(name, key, "ok", produced)
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    logger.info("TMJ_Crop: %d of %d scan(s) cropped", done, len(cases))
    if done == 0:
        raise ToolInputError(
            f"None of the {len(ready)} oriented scan(s) could be cropped. "
            "The per-scan reports in the output folder say why."
        )
    (output_dir / REPORT_NAME).write_text(json.dumps({
        "tool": "TMJ_Crop",
        "sides": sides,
        "outputs": sorted(outputs),
        "summary": {"cases": len(cases), "cropped": done, "partial": partial,
                    "failed": len(cases) - done},
        "cases": entries,
    }, indent=1))
    return output_dir
