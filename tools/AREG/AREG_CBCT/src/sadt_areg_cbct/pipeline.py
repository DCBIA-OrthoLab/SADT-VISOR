"""The CBCT half of AREG: mask the T1 scan to one anatomical region, register
the T2 onto it, write the registered volume and the transform.

Ported from `AREG_CBCT/AREG_CBCT.py` (the driver) and the registration half of
`AREG_CBCT_utils/utils.py`. The Slicer envelope is gone: no `<filter-progress>`
prints, no `time.sleep(0.2)` progress theatre (0.6 s per patient), no
`sys.exit`, and nothing written into the caller's input tree.

Three behaviours are deliberately different from the original:

* **The written `.tfm` is usable.** The original registered the T1 against a
  RECENTRED COPY of the T2 and wrote the transform between those two spaces,
  while that copy lived in a `<t2_folder>_Center` directory next to the user's
  own data and was never returned -- so the one file saying how the scans were
  aligned referred to a volume the caller did not have. There is no recentring
  here (see `elastix._RIGID_PARAMETERS`), so the transform maps the T1 frame to
  the T2 frame the caller sent.
* **The T2 is interpolated once instead of twice.** Recentring resampled every
  moving volume before the registration resampled it again; elastix's
  `AutomaticTransformInitialization` aligns the centres itself, so the first
  pass bought nothing and cost a blur.
* **A registration that cannot be done is reported, not raised.** The original
  caught every per-patient exception into a log line and printed how many had
  failed; the archive gave no clue. Each patient gets a report entry.
"""

import logging
import os

import numpy as np
import SimpleITK as sitk

from sadt_areg_common import catalogs, pairing
from . import elastix, gpu_rigid

logger = logging.getLogger(__name__)


def register_patient(
    t1_path: str,
    t2_path: str,
    mask_path: str,
    region: str,
    output_dir: str,
    relative_key: str,
    suffix: str,
    segmentation_label: int = None,
) -> dict:
    """Register one T2 onto one T1 and write the results. Returns a report entry.

    Raises `elastix.RegistrationError` when this patient cannot be registered;
    the caller records that and moves on to the next.
    """
    fixed = _read(t1_path, "T1 scan")
    mask = _read(mask_path, "mask", cause="mask")
    masked, note = elastix.apply_mask(fixed, mask, label=segmentation_label)

    moving = _read(t2_path, "T2 scan")
    transform, engine = _register(masked, moving)

    # The T2's own size, spacing and direction, in the T1's frame -- placed where
    # the T2 now IS. (Resampling onto the T1 grid instead would crop the T2 to the
    # T1's field of view and re-sample it to the T1's spacing.)
    #
    # Not the T2's grid as it stands: the original could use that only because
    # it had recentred the T2 first, so the T2's box and the oriented T1's both
    # sat around the origin. Without the recentring, a T2 whose box runs from 0
    # to +197 mm against a T1 oriented around the origin overlaps it on one
    # octant, and seven eighths of the registered volume came back empty --
    # measured on a clinical pair, 12.5 % of its voxels nonzero.
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(moving)
    resampler.SetOutputOrigin(_origin_in_fixed_frame(moving, transform))
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    registered = sitk.Cast(resampler.Execute(moving), sitk.sitkInt16)

    relative_dir, patient = os.path.split(relative_key)
    destination = os.path.join(output_dir, region, relative_dir)
    os.makedirs(destination, exist_ok=True)

    _, extension = pairing.split_scan_extension(os.path.basename(t2_path))
    scan_output = os.path.join(
        destination, f"{patient}_{region}_{suffix}{pairing.compressed_extension(extension)}"
    )
    sitk.WriteImage(registered, scan_output, useCompression=True)

    transform_output = os.path.join(destination, f"{patient}_{region}_{suffix}_transform.tfm")
    sitk.WriteTransform(transform, transform_output)

    entry = {
        "status": "ok",
        "region": catalogs.region_name(region),
        "mask": os.path.basename(mask_path),
        # Stated rather than assumed: getting the direction backwards is silent
        # (the file still loads, and still transforms), and it is the only thing
        # a downstream tool needs to know to reuse it.
        "transform_maps": "T1 space -> T2 space (what sitk.ResampleImageFilter consumes)",
        "engine": engine,
        "outputs": sorted(
            os.path.relpath(path, output_dir) for path in (scan_output, transform_output)
        ),
    }
    if note:
        entry["note"] = note
    return entry


def _register(masked: sitk.Image, moving: sitk.Image) -> tuple:
    """`(transform, engine)`: on the card when there is one, elastix otherwise.

    The GPU engine solves the same problem six times faster (see `gpu_rigid`).
    A failure there -- the card full of other runs, a driver error -- falls
    back to elastix rather than failing the patient: the answer is the same to
    within a few hundredths of a millimetre, only slower.
    """
    if gpu_rigid.available():
        try:
            return gpu_rigid.register(masked, moving), "gpu"
        except Exception as exc:  # noqa: BLE001 - elastix is the fallback, not the patient's failure
            logger.warning("GPU registration failed (%s); registering with elastix", elastix.describe(exc))
    return elastix.register(masked, moving), "elastix"


def register_all(jobs: list, width: int, on_done=None) -> list:
    """Run `register_patient(**job)` for every job, `width` at a time.

    Returns one report entry per job, in the order given. A registration that
    cannot be done is a failed entry, never an exception, exactly as one at a
    time. `spawn` workers, so none inherits state it did not build; each holds
    its own pair of volumes, which is what a channel costs.
    """
    total = len(jobs)
    entries = [None] * total
    if width <= 1 or total <= 1:
        # The same thread count a worker would get: elastix's preprocessing
        # sums in an order set by it, so this is what keeps a registration
        # identical to the bit whichever path ran it.
        _worker_setup(_threads_per_worker(1))
        for index, job in enumerate(jobs):
            entries[index] = _register_safely(job)
            if on_done:
                on_done(index + 1, total)
        return entries
    import multiprocessing
    from concurrent import futures

    threads = _threads_per_worker(width)
    with futures.ProcessPoolExecutor(
            max_workers=width, mp_context=multiprocessing.get_context("spawn"),
            initializer=_worker_setup, initargs=(threads,)) as pool:
        pending = {pool.submit(_register_safely, job): index for index, job in enumerate(jobs)}
        for done, future in enumerate(futures.as_completed(pending), start=1):
            entries[pending[future]] = future.result()
            if on_done:
                on_done(done, total)
    return entries


def _register_safely(job: dict) -> dict:
    """One registration, its failure reported rather than raised."""
    try:
        return register_patient(**job)
    except elastix.RegistrationError as exc:
        # `error` and `cause` travel back to the parent, which logs the failure
        # and decides whose fault it was; it takes them out of the report.
        return {"status": "failed", "reason": str(exc), "error": type(exc).__name__,
                "cause": getattr(exc, "cause", None)}
    except RuntimeError as exc:
        # Straight from ITK: its source path and an object address come before
        # the sentence, so only the sentence is kept.
        return {"status": "failed", "reason": elastix.describe(exc),
                "error": type(exc).__name__, "cause": "engine"}


def _threads_per_worker(width: int) -> int:
    """The threads ONE registration may open.

    elastix speeds up with ITK's thread count -- 97 s on ten threads, 43 s on
    fifty-six, for one region of the test pair -- so this is the number that
    decides how long a registration takes. Under a server it is what the
    server set for one CHANNEL: AREG_CBCT's channels each bring their own share
    of cores (`cores_per_channel` in the server's deployment.toml), so the
    variable already holds one registration's threads and is taken as it is.
    With no server, the machine's cores are shared between the workers.
    """
    try:
        granted = int(os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS") or 0)
    except ValueError:
        granted = 0
    if granted > 0:
        return granted
    return max(1, (os.cpu_count() or 1) // max(1, width))


def _worker_setup(threads: int) -> None:
    """Pin this process's thread count, in SimpleITK AND in elastix's itk.

    They are two libraries with two defaults: setting SimpleITK's left elastix
    on its own count, read once when `itk` was first imported, so a serial run
    and a worker could still disagree by the order their sums ran in.
    """
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(threads)
    try:
        elastix._import_elastix().MultiThreaderBase.SetGlobalDefaultNumberOfThreads(threads)
    except Exception:  # noqa: BLE001 - a missing itk is reported where it is used
        pass


def _read(path: str, what: str, cause: str = "input") -> sitk.Image:
    """Read one volume, or say which of the three could not be read.

    A file the caller sent that ITK cannot open is the caller's to fix, so it
    is reported as their input (`cause="input"`) rather than as an engine
    failure -- and by its role, never by its name. A mask is "mask": whose
    fault it is depends on who made it, which only the caller knows.
    """
    try:
        return sitk.ReadImage(path)
    except RuntimeError as exc:
        raise elastix.RegistrationError(
            f"the {what} could not be read: {elastix.describe(exc)}", cause=cause
        ) from exc


def _origin_in_fixed_frame(moving: sitk.Image, transform: sitk.Transform) -> tuple:
    """The origin that puts the T2's grid, unchanged in size, spacing and
    direction, centred on where the T2's centre lands in the T1's frame.

    `transform` maps the T1 frame to the T2 frame, so its inverse is what
    carries the T2's centre over.
    """
    middle = (np.array(moving.GetSize()) - 1) / 2.0
    centre = np.array(transform.GetInverse().TransformPoint(
        moving.TransformContinuousIndexToPhysicalPoint(middle.tolist())))
    direction = np.array(moving.GetDirection()).reshape(3, 3)
    origin = centre - direction @ (middle * np.array(moving.GetSpacing()))
    # Rounded to what a NIfTI header can hold (float32), so the file written
    # describes the very grid that was sampled: resampling the T2 again with
    # the written transform onto the written file's grid reproduces it exactly.
    return tuple(float(value) for value in origin.astype(np.float32))


def find_masks(mask_roots: list, region: str, scan_keys=()) -> dict:
    """{patient key: mask path} for one region, across several folders.

    Several roots because a mask can come from three places -- a folder the
    caller sent, the T1 folder itself (which is where the original looked when
    no mask folder was given), or AMASSS's output in the automated modes. The
    first root that has a patient's mask wins, so an explicit mask folder
    always beats one found next to the scans.

    `scan_keys` are the patients the scans were discovered under, and matching
    falls back to the LEAF of a key when the full relative path does not line
    up. It has to: a mask folder a caller sends is rarely laid out like their
    scan folder, and AMASSS's output is never laid out like it -- it writes one
    `<scan>_<id>_SegOut/` directory per scan, so a mask discovered under it
    keys to `P1_seg_SegOut/P1` while its scan keys to `P1`. The fallback is
    only taken when the leaf is unambiguous across the whole tree, so two
    subjects genuinely called `P1` in different folders never borrow each
    other's mask.
    """
    found: dict = {}
    for root in mask_roots:
        if not root or not os.path.isdir(root):
            continue
        for key, path in pairing.discover_masks(root, region).items():
            found.setdefault(key, path)

    missing = [key for key in scan_keys if key not in found]
    if not missing:
        return found

    by_leaf: dict = {}
    for key, path in found.items():
        by_leaf.setdefault(os.path.basename(key), []).append(path)
    for key in missing:
        candidates = by_leaf.get(os.path.basename(key), ())
        if len(candidates) == 1:
            found[key] = candidates[0]
    return found
