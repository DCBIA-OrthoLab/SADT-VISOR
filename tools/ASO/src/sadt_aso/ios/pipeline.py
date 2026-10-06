"""The IOS half of ASO: pair meshes with their landmarks per jaw, register each
jaw onto the reference, write the oriented meshes and transforms.

Ported from `ASO_IOS/PRE_ASO_IOS/PRE_ASO_IOS.py` (fully-automated, registering
on tooth centroids) and `ASO_IOS/SEMI_ASO_IOS/SEMI_ASO_IOS.py` (semi-automated,
registering on landmarks). Both drove the same `ICP` class with a different
`option` callable, so they are one function here with that callable as the
difference.

Crown segmentation is NOT part of this. A fully-automated run needs meshes that
already carry a per-point tooth-label array; `segment_unlabelled` marks the seam
where a future `tools/CrownSeg` plugs in (see ALI_PORT_CONTEXT.md section 3.2).
"""

import logging
import os
import re

import numpy as np
import SimpleITK as sitk
import sadt_naming

from .. import catalogs, markups
from . import icp as ios_icp
from . import pre_icp, surfaces

logger = logging.getLogger(__name__)

# Tokens naming a jaw inside a file name. The original tested only for "_U_" and
# the substring "upper", and DEFAULTED TO LOWER when neither was found -- so a
# maxillary scan named `patient1.vtk` was quietly registered against the
# mandibular reference and returned as a success.
#
# Shared, not local, and that is a widening: this tool knew ten spellings and
# its neighbours knew thirteen and four, so `P1_MX.vtk` was read by AREG and
# refused HERE. A clinician cannot be expected to know which tool learnt which
# word. Nothing that worked stops working -- the table is the union.
_JAW_TOKENS = sadt_naming.JAW_TOKENS

_SPLIT = re.compile(r"[_\-.]+")


class JawError(Exception):
    """A file's jaw could not be determined from its name."""


def patient_and_jaw(filename: str, output_suffix: str = "") -> tuple:
    """('P1_U_Seg.vtk') -> ('P1', 'Upper'), ('Upper_gold.vtk') -> ('gold', 'Upper').

    Everything from the jaw token onwards is normally decoration a previous
    step added (`_Seg`, `_Or`, `_lm`, ...), so the patient is what comes before
    it.

    **The jaw token may also come first**, and that is not a corner case: the
    published IOS reference bundle is `Upper_gold.vtk` / `Lower_gold.vtk`.
    Requiring something before the token rejected the whole bundle with "no mesh
    whose name says which jaw it is" -- verified against
    HUTIN1/ASO v1.0.0 Gold_file.zip. When nothing precedes the token, what
    follows it is the identifier rather than decoration.

    **The identifier ends at the first decoration token**, which is a second jaw
    token or `output_suffix`. Both cases are real and both were broken:

    * a predictor names the jaw again -- ALI and Crown_Seg write
      `<mesh stem>_<Jaw>_<type>_Pred.json`, so the landmark file of
      `Upper_new_9.vtk` is `Upper_new_9_Upper_O_Pred.json`. Without this the
      identifier ran to the end of the name, the pair landed under two
      patients, and Semi-Automated IOS reported "no landmark file for this
      jaw" on the dataset this repository ships;
    * this tool's OWN output keeps the suffix -- `Upper_new_9_Or.vtk` gave
      `new_9_Or`. Feeding a previous run back in, which the caller is
      explicitly allowed to do (see the fallback below), split the same
      patient in two all over again.

    When the jaw token is NOT first, both cases are already covered: everything
    from the jaw token onwards is dropped, decoration included. This branch is
    the one that had no end.

    Upstream paired the first case only because it compared names with `in`,
    which is what made patient `1` match patient `10`; the exact-stem rule that
    replaced it is right, it just has to be given the right stem.

    Raises JawError when no token names a jaw, rather than guessing: the
    original defaulted to Lower, so a maxillary mesh named `patient1.vtk` was
    registered against the mandibular reference and returned as a success.
    """
    stem = _strip_extension(filename)
    tokens = [token for token in _SPLIT.split(stem) if token]
    for index, token in enumerate(tokens):
        # `jaw_in_token`, not a lookup in the table: it also reads a jaw word
        # with a timepoint glued to it, which is how upstream's published IOS
        # test set is named (`A2_UpperT1.vtk`). A plain lookup found no jaw in
        # that name and the mesh was refused.
        jaw = sadt_naming.jaw_in_token(token)
        if jaw is None:
            continue
        before = tokens[:index]
        if before:
            return "_".join(before), jaw
        after = tokens[index + 1:]
        decoration = {output_suffix.lower()} if output_suffix else set()
        for offset, later in enumerate(after):
            if sadt_naming.jaw_in_token(later) or later.lower() in decoration:
                after = after[:offset]
                break
        return "_".join(after) or stem, jaw
    raise JawError(
        f"'{filename}': cannot tell which jaw this is. Name the files so a "
        f"token says it, e.g. 'P1_U_Seg.vtk' / 'P1_Lower.vtk' / 'Upper_gold.vtk'."
    )


def is_previous_output(filename: str, suffix: str) -> bool:
    """True if this file looks like something a previous ASO run wrote.

    Same reason as the CBCT engine's: a second run on the same folder must
    orient the original mesh, not the first run's result.
    """
    return _strip_extension(filename).endswith(f"_{suffix}")


def discover(input_root: str, output_suffix: str = "Or") -> dict:
    """{patient key: {jaw: {"surface": path, "markups": path}}}.

    The key is the patient's path relative to the input root, so two patients
    with the same name in different folders stay apart -- and files are paired
    only within one directory, by exact stem, never by the `vtk_name in
    json_name` substring test that made patient `1` match patient `10`.

    A previous run's outputs are used only when a patient has nothing else.
    """
    patients: dict = {}
    unnamed = 0

    for directory, _, file_names in os.walk(input_root):
        relative = os.path.relpath(directory, input_root)
        prefix = "" if relative == "." else relative
        for file_name in sorted(file_names):
            if file_name.startswith("."):
                continue
            is_surface = surfaces.is_surface_file(file_name)
            is_markups = markups.is_markups_file(file_name)
            if not (is_surface or is_markups):
                continue
            try:
                stem, jaw = patient_and_jaw(file_name, output_suffix)
            except JawError:
                # Counted, not collected: every JawError quotes the file it
                # came from, and that name is patient metadata the moment it
                # reaches a log.
                unnamed += 1
                continue
            entry = patients.setdefault(os.path.join(prefix, stem), {})
            jaw_entry = entry.setdefault(
                jaw, {"surface": [], "old_surface": [], "markups": [], "old_markups": []}
            )
            kind = "surface" if is_surface else "markups"
            if is_previous_output(file_name, output_suffix):
                kind = f"old_{kind}"
            jaw_entry[kind].append(os.path.join(directory, file_name))

    if unnamed:
        # How many were skipped and what to do about it, which is the whole of
        # the diagnosis here: the reason is the same for every one of them.
        logger.warning(
            "%d file(s) skipped: nothing in the name says which jaw it is. Name "
            "them so a token does, e.g. 'P1_U_Seg.vtk' / 'P1_Lower.vtk'.",
            unnamed,
        )

    return {
        key: {
            jaw: {
                "surface": _first(entry["surface"], entry["old_surface"]),
                "markups": _first(entry["markups"], entry["old_markups"]),
            }
            for jaw, entry in jaws.items()
        }
        for key, jaws in patients.items()
    }


def _first(preferred: list, fallback: list):
    candidates = preferred or fallback
    return candidates[0] if candidates else None


def load_reference(reference_dir: str, need_surfaces=None) -> dict:
    """{jaw: {"surface": path, "markups": path}} for the reference bundle.

    Registering tooth centroids needs the reference's MESHES; registering
    landmarks needs its MARKUPS. Which of the two a run uses is now decided per
    jaw, from what each patient came with, so a cohort can legitimately need
    both -- and this cannot know in advance which.

    So it asks only that the bundle hold SOMETHING for some jaw, and leaves
    "which form is missing" to the point where it is actually wanted: both
    `_fully_automated_matrix` and `_semi_automated_matrix` already say it
    precisely, for that jaw. `need_surfaces` is kept for a caller that does know
    (True or False), and None -- the normal case now -- means either will do.
    """
    reference = discover(reference_dir)
    merged: dict = {}
    for jaws in reference.values():
        for jaw, entry in jaws.items():
            target = merged.setdefault(jaw, {"surface": None, "markups": None})
            target["surface"] = target["surface"] or entry["surface"]
            target["markups"] = target["markups"] or entry["markups"]

    if need_surfaces is None:
        wanted, what = ("surface", "markups"), "mesh or landmark file"
    elif need_surfaces:
        wanted, what = ("surface",), "mesh"
    else:
        wanted, what = ("markups",), "landmark file"
    if not any(entry[key] for entry in merged.values() for key in wanted):
        # A RuntimeError, not a ValueError: the reference is a bundle this
        # server hosts, and a 422 would send the caller to fix a request that
        # is fine.
        raise RuntimeError(
            f"reference bundle unreadable: it holds no {what} whose name says "
            f"which jaw it is (e.g. 'Gold_Upper.vtk')."
        )
    return merged


class FileCache:
    """The meshes and landmark files a run reads more than once, read once.

    Nothing about the pairing or the registration changed; what changed is how
    often the same bytes are parsed. A fully-automated jaw read THREE meshes to
    orient one: the patient's, the reference's, and then the patient's again in
    `_write_jaw`, which needed the untransformed mesh to apply the matrix to.
    A semi-automated jaw parsed its landmark file twice for the same reason.
    And the reference -- one file for the whole cohort -- was re-read once per
    jaw per patient, so a forty-patient batch parsed the gold meshes eighty
    times.

    **This can only be a speedup, never a change**, because every reader here
    treats a mesh as immutable: `surfaces.transform_surface` deep-copies before
    it transforms, `labels_of`, `points_of`, `label_array_name` and
    `write_surface` only read, and the landmark dicts are handed out as fresh
    dicts over shared coordinate arrays that nothing writes into.
    `test_the_cache_hands_back_an_unmodified_mesh` holds that.

    Two lifetimes, because holding everything would trade a second for a
    gigabyte: `keep=True` is for the reference, which is the same file for
    every patient, and everything else is dropped by `release()` when the
    patient it belongs to is done. Forty intra-oral scans at ~10 MB of parsed
    geometry each is not memory a shared server should spend on files nobody
    will look at again.
    """

    def __init__(self):
        self._kept: dict = {}
        self._patient: dict = {}

    def surface(self, path: str, keep: bool = False):
        return self._get(surfaces.read_surface, path, keep)

    def landmarks(self, path: str, keep: bool = False) -> dict:
        # A fresh dict over the same arrays: a caller may filter or extend what
        # it was handed, and must not be able to edit the cache by doing so.
        return dict(self._get(markups.load_landmarks, path, keep))

    def release(self) -> None:
        """Forget everything belonging to the patient just finished."""
        self._patient.clear()

    def _get(self, read, path: str, keep: bool):
        store = self._kept if keep else self._patient
        key = (read.__name__, path)
        value = store.get(key)
        if value is None:
            # The other store may already hold it: a caller can legitimately
            # ask for the same file both ways (a reference bundle used as the
            # input, which re-running on an output folder does).
            value = self._kept.get(key) or self._patient.get(key)
        if value is None:
            value = read(path)
            store[key] = value
        return value


def orient_patient(
    jaws: dict,
    reference: dict,
    automation: str,
    selected_teeth: dict,
    landmark_keys: dict,
    wanted_jaws: list,
    driving_jaw: str,
    output_dir: str,
    relative_key: str,
    suffix: str,
    max_triplets: int,
    seed: int,
    cache: "FileCache" = None,
    patient=None,
) -> dict:
    """Orient one patient's jaws. Returns a report entry.

    `patient`, when given, is `(index, total)`: a jaw that fails is logged
    with it, since the report that also records the failure is deleted with
    the job directory when the whole run fails.

    With `driving_jaw` set, that jaw's transform is applied to the other one as
    well -- occlusion is preserved by moving both halves rigidly together, which
    only makes sense if the two meshes were in occlusion to begin with.

    `cache` is shared across the cohort so the reference bundle is parsed once
    rather than once per patient; without one, each call gets a fresh cache and
    still avoids re-reading a patient's own mesh.
    """
    entry: dict = {"status": "ok", "jaws": {}, "produced": []}
    matrices: dict = {}
    cache = cache if cache is not None else FileCache()

    try:
        order = _ordered_jaws(wanted_jaws, driving_jaw)
        for jaw in order:
            available = jaws.get(jaw)
            if not available or not available["surface"]:
                entry["jaws"][jaw] = {
                    "status": "skipped",
                    "reason": "no mesh for this jaw",
                }
                continue
            try:
                matrices[jaw] = _matrix_for(
                    jaw,
                    available,
                    reference,
                    automation,
                    selected_teeth,
                    landmark_keys,
                    driving_jaw,
                    matrices,
                    max_triplets,
                    seed,
                    cache,
                )
            except (ios_icp.RegistrationError, surfaces.SurfaceError, ValueError) as exc:
                logger.warning(
                    "%s%s jaw: registration failed (%s: %s)",
                    f"patient {patient[0]} of {patient[1]}: " if patient else "",
                    jaw, type(exc).__name__, exc,
                )
                entry["jaws"][jaw] = {"status": "failed", "reason": str(exc)}
                continue

            written = _write_jaw(
                available, matrices[jaw], output_dir, relative_key, jaw, suffix, cache
            )
            entry["jaws"][jaw] = {
                "status": "ok",
                "registered_on": (
                    f"the {driving_jaw} jaw's transform"
                    if driving_jaw and jaw != driving_jaw
                    else ("tooth centroids" if automation == catalogs.AUTOMATION_FULLY
                          else "landmarks")
                ),
            }
            entry["produced"].extend(written)
    finally:
        # This patient's meshes are of no further use, and the next one's are
        # the same size. Released even when a jaw raised something unexpected.
        cache.release()

    if not any(jaw.get("status") == "ok" for jaw in entry["jaws"].values()):
        entry["status"] = "failed"
        entry["reason"] = "; ".join(
            f"{jaw}: {detail.get('reason', detail['status'])}"
            for jaw, detail in entry["jaws"].items()
        ) or "no jaw could be oriented"
    entry["produced"].sort()
    return entry


def segment_unlabelled(surface_path: str) -> None:
    """Seam for server-side crown segmentation, not implemented yet.

    A mesh with no tooth-label array cannot be oriented by the fully-automated
    mode. Producing that array means running `shapeaxi`'s `dental_model_seg`,
    which needs pytorch3d -- absent from the deployment image (see
    ALI_PORT_CONTEXT.md section 4). It belongs in a `tools/CrownSeg` of its own
    because ALI, AREG and FlexReg need it too, so this function is where ASO
    will call it, and nothing else here changes when it lands.
    """
    raise ios_icp.RegistrationError(
        f"'{os.path.basename(surface_path)}' carries no tooth labels. The "
        f"fully-automated mode needs a mesh with a per-point array named one of "
        f"{', '.join(markups.LABEL_ARRAY_NAMES)}. Segment it first, or use the "
        f"semi-automated mode with landmark files."
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _ordered_jaws(wanted_jaws: list, driving_jaw: str) -> list:
    """The driving jaw first: the other one reuses its matrix."""
    if driving_jaw and driving_jaw in wanted_jaws:
        return [driving_jaw] + [jaw for jaw in wanted_jaws if jaw != driving_jaw]
    return list(wanted_jaws)


def _matrix_for(
    jaw: str,
    available: dict,
    reference: dict,
    automation: str,
    selected_teeth: dict,
    landmark_keys: dict,
    driving_jaw: str,
    matrices: dict,
    max_triplets: int,
    seed: int,
    cache: FileCache,
) -> np.ndarray:
    if driving_jaw and jaw != driving_jaw:
        if driving_jaw not in matrices:
            raise ios_icp.RegistrationError(
                f"the {driving_jaw} jaw, which drives this one, was not oriented"
            )
        return matrices[driving_jaw]

    reference_entry = reference.get(jaw)
    if not reference_entry:
        raise ios_icp.RegistrationError(f"the reference has no {jaw} jaw")

    # Decided by what this JAW came with, not by a mode the caller declared.
    # Landmarks beside the mesh are landmarks to register on; a mesh with none
    # is one to register by its tooth centroids -- and `_fully_automated_matrix`
    # segments it first if it carries no tooth labels either. The caller used to
    # say which, and saying it wrong was silent: a fully-automated run ignored
    # landmarks already on disk, and a semi-automated one failed jaw by jaw with
    # "no landmark file".
    #
    # `automation` survives as an OVERRIDE, and only in the direction that adds
    # nothing back: asking for Fully-Automated over a jaw that HAS landmarks is
    # asking for them to be ignored, which is a legitimate thing to want.
    forced = automation == catalogs.AUTOMATION_FULLY
    if forced or not available["markups"]:
        return _fully_automated_matrix(
            available, reference_entry, selected_teeth[jaw], max_triplets, seed, cache
        )
    return _semi_automated_matrix(
        available, reference_entry, landmark_keys[jaw], max_triplets, seed, cache
    )


def _fully_automated_matrix(
    available: dict,
    reference_entry: dict,
    teeth: list,
    max_triplets: int,
    seed: int,
    cache: FileCache,
) -> np.ndarray:
    if not reference_entry["surface"]:
        raise ios_icp.RegistrationError("the reference has no mesh for this jaw")

    source = cache.surface(available["surface"])
    array_name = surfaces.label_array_name(source)
    if array_name is None:
        segment_unlabelled(available["surface"])

    target = cache.surface(reference_entry["surface"], keep=True)
    reference_array = surfaces.label_array_name(target)
    if reference_array is None:
        raise ios_icp.RegistrationError(
            "the reference mesh carries no tooth labels, so there is nothing to "
            "register the tooth centroids against"
        )

    coarse = pre_icp.align(source, target, teeth, array_name)
    aligned = surfaces.transform_surface(source, coarse)

    tooth_ids = catalogs.teeth_to_ids(teeth)
    fine = ios_icp.register(
        ios_icp.mean_teeth(aligned, tooth_ids, array_name),
        ios_icp.mean_teeth(target, tooth_ids, reference_array),
        max_triplets=max_triplets,
        seed=seed,
    )
    return fine @ coarse


def _semi_automated_matrix(
    available: dict,
    reference_entry: dict,
    keys: list,
    max_triplets: int,
    seed: int,
    cache: FileCache,
) -> np.ndarray:
    if not available["markups"]:
        raise ios_icp.RegistrationError(
            "no landmark file for this jaw; the semi-automated mode registers on "
            "landmarks you provide"
        )
    if not reference_entry["markups"]:
        raise ios_icp.RegistrationError("the reference has no landmark file for this jaw")

    source = ios_icp.select_keys(cache.landmarks(available["markups"]), keys)
    target = ios_icp.select_keys(
        cache.landmarks(reference_entry["markups"], keep=True), keys
    )
    return ios_icp.register(source, target, max_triplets=max_triplets, seed=seed)


def _write_jaw(
    available: dict,
    matrix: np.ndarray,
    output_dir: str,
    relative_key: str,
    jaw: str,
    suffix: str,
    cache: FileCache,
) -> list:
    relative_dir, patient = os.path.split(relative_key)
    destination = os.path.join(output_dir, relative_dir)
    os.makedirs(destination, exist_ok=True)
    written = []

    surface = cache.surface(available["surface"])
    oriented = surfaces.transform_surface(surface, matrix)
    stem = _strip_extension(os.path.basename(available["surface"]))
    extension = surfaces.output_extension(available["surface"])
    written.append(
        surfaces.write_surface(oriented, os.path.join(destination, f"{stem}_{suffix}{extension}"))
    )

    if available["markups"]:
        landmarks = cache.landmarks(available["markups"])
        moved = {
            name: (matrix @ np.append(point, 1.0))[:3] for name, point in landmarks.items()
        }
        landmark_stem = _strip_extension(os.path.basename(available["markups"]))
        written.append(
            markups.rewrite_landmarks(
                moved,
                available["markups"],
                os.path.join(destination, f"{landmark_stem}_{suffix}.mrk.json"),
            )
        )

    # Named per jaw. The original wrote `<patient>_SegOr.tfm` for both, so the
    # second jaw silently overwrote the first one's transform.
    written.append(
        _write_transform(
            matrix, os.path.join(destination, f"{patient}_{jaw}_{suffix}.tfm")
        )
    )
    return [os.path.relpath(path, output_dir) for path in written]


def _write_transform(matrix: np.ndarray, path: str) -> str:
    inverted = np.linalg.inv(matrix)
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(inverted[:3, :3].flatten().tolist())
    transform.SetTranslation(inverted[:3, 3].tolist())
    sitk.WriteTransform(transform, path)
    return path


def _strip_extension(filename: str) -> str:
    lower = filename.lower()
    for extension in markups.MARKUPS_EXTENSIONS + surfaces.SURFACE_EXTENSIONS:
        if lower.endswith(extension):
            return filename[: -len(extension)]
    return os.path.splitext(filename)[0]
