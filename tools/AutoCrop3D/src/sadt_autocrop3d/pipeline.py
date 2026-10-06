"""Crop a volume to a Slicer ROI box, and turn a cropped label map into a surface.

Ported from `AutoCrop3D/Crop_Volumes_CLI/`. The geometry is upstream's and is
kept deliberately intact: the ROI's two physical corners are mapped to
continuous indices, truncated to integers, clamped to the volume, and the
volume is sliced between them. Changing that changes every result, so it did
not change.

What did change is everything around it -- how a scan is matched to its ROI,
where the temporary files go, which labels get a colour, and what happens when
one file in a batch is unreadable. Each of those is a defect with a name in
README.md.
"""

import json
import logging
import os

from sadt_areg_common import pairing

logger = logging.getLogger("AutoCrop3D")

# What discovery accepts, and it is upstream's list rather than the panel's.
# The panel offered only the three `.gz` forms and REFUSED a folder of plain
# `.nii`/`.nrrd`/`.gipl`, which the CLI behind it reads perfectly well.
SCAN_EXTENSIONS = (".nii.gz", ".nrrd.gz", ".gipl.gz", ".nii", ".nrrd", ".gipl")

ROI_EXTENSION = ".mrk.json"

# ITK's NrrdImageIO accepts a name ending in `.nrrd` or `.nhdr` and nothing
# else: NRRD compresses INSIDE the file, so `.nrrd.gz` is a gzipped NRRD that
# no ITK reader or writer can open. Upstream discovered these files anyway,
# read them OUTSIDE its try block (so one aborted the whole batch) and wrote
# them inside a bare `except:` (so it produced nothing while the progress
# counter advanced). They are still discovered here -- reporting a file the
# user pointed at beats ignoring it -- but the failure is per file and named.
UNREADABLE_EXTENSIONS = (".nrrd.gz",)


# ---------------------------------------------------------------------------
# The identifier
# ---------------------------------------------------------------------------
# Tokens that say what a file IS rather than whose it is. Matched as WHOLE
# tokens of the stem, never as substrings: that is the difference between
# dropping `_MAX` from `P1_MAX_seg` and truncating a patient genuinely called
# `MAX_01`.
#
# The list is upstream's own chained-split vocabulary, plus the ROI words the
# ROI side needs. Both sides run through the same function, which is the whole
# point -- see `patient_key`.
IDENTITY_TOKENS_TO_DROP = frozenset({
    # what the CLI split on
    "scan", "scans", "seg", "segmentation", "segmentations", "or", "mand",
    "md", "max", "mx", "cb", "lm", "cl", "merged", "mask", "pred",
    # what an ROI file calls itself
    "roi", "rois", "mrk", "crop", "cropped", "box", "bbox",
})


def patient_key(filename: str) -> str:
    """The subject a scan or an ROI belongs to, from its name alone.

    ONE rule for both sides. Upstream had two that could not agree:

        scans: basename.split('_Scan')[0]...split('_T1')[0].split('.')[0]
        ROIs:  os.path.basename(file).split('_')[0]

    so `PatientA_01_Scan.nii.gz` keyed as `PatientA_01` while its own
    `PatientA_01_ROI.mrk.json` keyed as `PatientA`. The lookup missed, a bare
    `except` logged and continued, and every cohort whose identifiers contain
    an underscore produced an empty output folder and exit code 0.

    Three further properties, each of which upstream got wrong:

    * the timepoint is KEPT. `_T1` and `_T2` are two scans of one subject and
      each has its own ROI; collapsing them made `P01_T1` and `P01_T2` the same
      key, so the T1 scan was cropped with the T2 box and looked plausible;
    * tokens are dropped whole. `SMITH_ORTHO` survives as `SMITH_ORTHO` --
      upstream's `.split('_OR')[0]` truncated it to `SMITH`;
    * the extension is split off properly, compound extensions included, so
      `Patient.01_Scan.nii.gz` keys as `Patient_01` rather than `Patient`.

    Built on `sadt_areg_common.pairing` for the two things that must not be
    re-derived per tool -- what a compound scan extension is, and how a stem
    breaks into tokens -- but NOT on its `patient_stem`, which drops the
    timepoint and may drop the leading token. Both are right for pairing two
    timepoints and wrong for pairing a scan with its box: `P01_T1` and `P01_T2`
    may want two different boxes, and a patient can be called `MAX_01`.
    README.md says so at greater length.
    """
    stem, extension = pairing.split_scan_extension(filename)
    if extension.lower() not in SCAN_EXTENSIONS:
        # `.mrk.json` and anything else: take the outer extension off and let
        # the `mrk` token be dropped like any other below.
        stem = os.path.splitext(filename)[0]

    kept, seen_a_word = [], False
    for part in pairing.split_parts(stem):
        is_word = bool(part.strip(" _-."))
        # The FIRST word is always part of the identity, whatever it says. A
        # patient really can be called `MAX_01`, and dropping its leading token
        # would key them as `01` -- upstream's family of failure, arrived at
        # from the other direction.
        if is_word and seen_a_word and part.lower() in IDENTITY_TOKENS_TO_DROP:
            continue
        seen_a_word = seen_a_word or is_word
        kept.append(part)
    key = "".join(kept)
    # Collapse the separators the dropped tokens left behind.
    for separator in ("_", "-", ".", " "):
        while separator * 2 in key:
            key = key.replace(separator * 2, separator)
    return key.strip("_-. ") or stem


def is_scan_file(filename: str) -> bool:
    return filename.lower().endswith(SCAN_EXTENSIONS)


def is_roi_file(filename: str) -> bool:
    return filename.lower().endswith(ROI_EXTENSION)


def is_segmentation_name(filename: str) -> bool:
    """Whether a file's own NAME says it is a segmentation.

    Upstream asked `if "seg" in ScanOutPath.lower()` -- a substring test on the
    whole OUTPUT PATH, so an output folder called `/data/Segmentations/` sent
    every cropped CBCT through marching cubes while a segmentation named
    `Mandible.nii.gz` got none. The path is not part of the question, and the
    match is on a whole token of the file's stem.
    """
    stem, _ = pairing.split_scan_extension(os.path.basename(filename))
    return pairing.has_token(stem, ("seg", "segmentation", "mask", "pred", "label", "labels"))


# ---------------------------------------------------------------------------
# The ROI
# ---------------------------------------------------------------------------

class Roi:
    """A Slicer ROI markup, reduced to what the crop uses.

    `center` and `size` only, which is upstream's reading and therefore
    upstream's limitation: the box's `orientation` is not applied, so a ROTATED
    ROI is cropped as though it were axis-aligned. That is recorded rather than
    silently accepted -- `orientation_ignored` ends up in the run report.
    """

    def __init__(self, center, size, coordinate_system: str, orientation_ignored: bool, path: str):
        self.center = center
        self.size = size
        self.coordinate_system = coordinate_system
        self.orientation_ignored = orientation_ignored
        self.path = path


def read_roi(path: str, where: str = "") -> Roi:
    """Read a `.mrk.json` ROI box.

    Raises `ValueError` naming the file for anything a caller can fix: a file
    that is not JSON, a markups file holding no ROI, or one whose ROI has no
    box. Upstream ran `json.load(open(ROI_Path))['markups'][0]` unguarded
    inside the per-patient loop, so a single malformed file ended the batch
    with a traceback rather than a message.

    `where` is the scan's position in the batch ("scan 2 of 5"), used to say
    which crop a warning concerns without naming any file.
    """
    name = os.path.basename(path)
    try:
        with open(path) as handle:
            document = json.load(handle)
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"Could not read the ROI file '{name}': {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"'{name}' is not valid JSON: {error}") from error

    markups = document.get("markups")
    if not isinstance(markups, list) or not markups:
        raise ValueError(f"'{name}' holds no markup. Expected a Slicer ROI saved as .mrk.json.")

    markup = markups[0]
    center = markup.get("center")
    size = markup.get("size")
    if not _is_triple(center) or not _is_triple(size):
        raise ValueError(
            f"'{name}' has no ROI box: a markup of type '{markup.get('type', 'unknown')}' "
            f"carries no 'center' and 'size'. Draw a Region of Interest in Slicer and save that."
        )

    # Slicer writes the frame the coordinates are in. Upstream never read it,
    # so an ROI saved in RAS -- which Slicer does write -- was applied to an
    # LPS volume and cropped the mirror image of what the user drew. LPS is the
    # default because it is what Slicer writes for a markups file and what
    # upstream assumed.
    system = str(document.get("coordinateSystem", markup.get("coordinateSystem", "LPS"))).upper()
    center = [float(value) for value in center]
    if system in ("RAS", "0"):
        center = [-center[0], -center[1], center[2]]
    elif system not in ("LPS", "1"):
        raise ValueError(
            f"'{name}' declares an unknown coordinate system '{system}'. Expected LPS or RAS."
        )

    orientation = markup.get("orientation")
    ignored = bool(orientation) and not _is_axis_aligned(orientation)
    if ignored:
        # Not fatal: upstream cropped the axis-aligned bounding behaviour
        # silently and this keeps doing exactly that, but it says so.
        # By the scan's position, not the ROI's name: the name is patient
        # data, and the position is what lets an operator find the crop.
        logger.warning(
            "%sthe ROI is rotated; AutoCrop3D crops an axis-aligned box, so the "
            "rotation is ignored", f"{where}: " if where else "",
        )

    return Roi(
        center=center,
        size=[abs(float(value)) for value in size],
        coordinate_system=system,
        orientation_ignored=ignored,
        path=path,
    )


def _is_triple(value) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 3 and all(
        isinstance(item, (int, float)) for item in value
    )


def _is_axis_aligned(orientation) -> bool:
    """Whether ignoring this orientation changes the box at all.

    Not identity: a signed axis permutation -- a 180-degree flip about z, which
    is what the ROI shipped with upstream's own test data carries -- maps an
    axis-aligned box onto itself, so treating it as axis-aligned is exact. Only
    a genuine rotation makes the crop an approximation, and only that is worth
    telling the caller about.
    """
    try:
        flat = [float(value) for value in orientation]
    except (TypeError, ValueError):
        return True
    if len(flat) != 9:
        return True
    rows = [flat[0:3], flat[3:6], flat[6:9]]
    for vectors in (rows, list(zip(*rows))):
        for vector in vectors:
            magnitudes = sorted(abs(value) for value in vector)
            if magnitudes[2] < 1.0 - 1e-9 or magnitudes[1] > 1e-9:
                return False
    return True


# ---------------------------------------------------------------------------
# The crop
# ---------------------------------------------------------------------------

def crop_bounds(image, roi: Roi):
    """`(lower, upper, clamped)` in index space, upstream's arithmetic exactly.

    The two physical corners of the box go through
    `TransformPhysicalPointToContinuousIndex` and are TRUNCATED to integers --
    not floored and not rounded. Truncation toward zero is what upstream's
    `.astype(int)` does and it is kept, because the alternative moves every
    boundary voxel of every result this tool has ever produced.

    `clamped` says whether the box reached outside the volume, which is the
    difference between "you cropped what you drew" and "you cropped the part of
    what you drew that exists".
    """
    lower_point = [c - s / 2.0 for c, s in zip(roi.center, roi.size)]
    upper_point = [c + s / 2.0 for c, s in zip(roi.center, roi.size)]

    lower = [int(value) for value in image.TransformPhysicalPointToContinuousIndex(lower_point)]
    upper = [int(value) for value in image.TransformPhysicalPointToContinuousIndex(upper_point)]

    for axis in range(3):
        if lower[axis] > upper[axis]:
            lower[axis], upper[axis] = upper[axis], lower[axis]

    size = image.GetSize()
    clamped = any(lower[axis] < 0 or upper[axis] > size[axis] for axis in range(3))
    lower = [max(0, value) for value in lower]
    upper = [min(size[axis], value) for axis, value in enumerate(upper)]
    return lower, upper, clamped


def crop(image, lower, upper):
    """The sub-volume between two index corners, geometry preserved.

    SimpleITK's slicing carries origin, spacing and direction across, so the
    crop still sits exactly where its voxels were. An empty box is refused with
    a message rather than sliced: upstream produced a zero-sized image, wrote
    it inside a bare `except:` and counted the patient as a success.
    """
    extent = [upper[axis] - lower[axis] for axis in range(3)]
    if any(value <= 0 for value in extent):
        raise ValueError(
            "The ROI does not overlap this volume: the box maps to an empty index range "
            f"{tuple(lower)}..{tuple(upper)} inside a volume of size {tuple(image.GetSize())}. "
            "Check that the ROI was drawn on this patient."
        )
    return image[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]


def repad(image, cropped, lower):
    """The crop placed back into a blank volume of the original geometry.

    What `keep_original_size` produces: everything outside the box is zero and
    everything inside it is where it always was, so the result still overlays
    the scan it came from. Upstream's arithmetic, kept.
    """
    import numpy as np
    import SimpleITK as sitk

    blank = sitk.Image(image.GetSize(), image.GetPixelID())
    blank.CopyInformation(image)

    blank_array = sitk.GetArrayFromImage(blank)
    cropped_array = sitk.GetArrayFromImage(cropped)

    # GetArrayFromImage indexes (z, y, x); `lower` is (x, y, z).
    blank_array[
        lower[2]:lower[2] + cropped_array.shape[0],
        lower[1]:lower[1] + cropped_array.shape[1],
        lower[0]:lower[0] + cropped_array.shape[2],
    ] = np.asarray(cropped_array)

    padded = sitk.GetImageFromArray(blank_array)
    padded.CopyInformation(blank)
    return padded


def output_name(filename: str, suffix: str) -> str:
    """`P1_scan.nii.gz` + `cropped` -> `P1_scan_cropped.nii.gz`.

    Upstream built this with `basename.split('.')[0]`, which truncated any name
    holding a dot, and then placed it with
    `os.path.join(out, relative).replace(os.path.basename(relative), filename)`.
    In single-file mode `relative` is `"."`, `os.path.basename(".")` is `"."`,
    and `str.replace` therefore substituted the file name for EVERY DOT IN THE
    WHOLE PATH before `os.makedirs` created the resulting tree.
    """
    stem, extension = pairing.split_scan_extension(filename)
    tail = f"_{suffix}" if suffix else ""
    return f"{stem}{tail}{extension}"


def surface_name(filename: str, suffix: str) -> str:
    """The `.vtk` beside a cropped segmentation. Upstream's `_vtk.vtk` kept."""
    stem, _ = pairing.split_scan_extension(filename)
    tail = f"_{suffix}" if suffix else ""
    return f"{stem}{tail}_vtk.vtk"


# ---------------------------------------------------------------------------
# The surface
# ---------------------------------------------------------------------------
# Upstream's table, kept to the digit so a five-structure segmentation still
# comes out the colours a clinician recognises.
LABEL_COLORS = {
    1: (216, 101, 79),
    2: (128, 174, 128),
    3: (0, 0, 0),
    4: (230, 220, 70),
    5: (111, 184, 210),
    6: (172, 122, 101),
}


def color_for_label(label: int):
    """A colour for any label, not only the six that were written down.

    Upstream indexed `LABEL_COLORS[np.max(img_arr)]`, so a segmentation with
    more than six labels raised `KeyError` -- and an empty crop, whose maximum
    is 0, raised it too. Both were swallowed by a bare `except: pass` in the
    caller, so the run reported success and wrote no surface.

    Beyond six the colour is generated from the label with the golden-ratio
    hue step, which is deterministic (the same label is the same colour in
    every run and in every patient) and keeps adjacent labels far apart.
    """
    if label in LABEL_COLORS:
        return LABEL_COLORS[label]
    hue = ((label - len(LABEL_COLORS)) * 0.618033988749895) % 1.0
    return _hsv_to_rgb(hue, 0.55, 0.85)


def _hsv_to_rgb(hue: float, saturation: float, value: float):
    sector = int(hue * 6.0) % 6
    fraction = hue * 6.0 - int(hue * 6.0)
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * fraction)
    t = value * (1.0 - saturation * (1.0 - fraction))
    channels = {
        0: (value, t, p), 1: (q, value, p), 2: (p, value, t),
        3: (p, q, value), 4: (t, p, value), 5: (value, p, q),
    }[sector]
    return tuple(int(round(channel * 255)) for channel in channels)


def present_labels(image) -> list:
    """The non-zero label values actually in the image, sorted.

    Upstream computed exactly this list at lines 29-32 and then never used it,
    reaching for `np.max(img_arr)` instead. Using it is what makes the surface
    per-label rather than one colour for everything, and what makes an empty
    crop produce no surface instead of a `KeyError`.
    """
    import numpy as np
    import SimpleITK as sitk

    values = np.unique(sitk.GetArrayFromImage(image))
    return [int(value) for value in values if value > 0]


def pad_voxels(image, padding_mm: float) -> list:
    """`padding_mm` converted to whole voxels per axis, at least one.

    Upstream padded a fixed `[50, 50, 10]` VOXELS, so the physical thickness of
    the margin changed with the scan: 25 mm on a 0.5 mm CBCT and 8 mm on a
    0.16 mm one. The margin exists only to give marching cubes a shell of
    background to close the isosurface against, which is a physical quantity,
    so it is declared in millimetres and converted here.
    """
    if padding_mm < 0:
        raise ValueError(f"surface_padding_mm must not be negative, got {padding_mm}.")
    return [max(1, int(round(padding_mm / spacing))) for spacing in image.GetSpacing()]


# Upstream's LAPLACIAN smoothing recipe, kept as constants rather than
# published as arguments. The iteration count IS an argument -- it is the one
# a user turns down when a thin structure loses its detail -- but the feature
# angle and the relaxation factor are properties of the recipe, not per-request
# clinical choices, and three more floats on a panel buy nothing.
SMOOTHING_FEATURE_ANGLE = 120.0
SMOOTHING_RELAXATION_FACTOR = 0.6


def write_surface(image, destination: str, scratch_dir: str, padding_mm: float,
                  smoothing_iterations: int,
                  feature_angle: float = SMOOTHING_FEATURE_ANGLE,
                  relaxation_factor: float = SMOOTHING_RELAXATION_FACTOR) -> list:
    """Marching cubes over a cropped label map. Returns the labels it drew.

    An empty label map produces no file and an empty list -- there is no
    surface of nothing, and writing a zero-cell `.vtk` only moves the failure
    into whatever opens it.

    The padded volume goes to a file under `scratch_dir`. Upstream wrote
    `"image_padded.nii.gz"` as a RELATIVE path into the process working
    directory and deleted it only if nothing raised, so two concurrent requests
    overwrote each other's temporary volume and a failure left it behind. On a
    server running several tools at once that is a correctness bug, not tidiness.
    """
    import SimpleITK as sitk
    import vtk

    labels = present_labels(image)
    if not labels:
        return []

    margin = pad_voxels(image, padding_mm)
    padded = sitk.ConstantPad(image, margin, margin, 0)
    padded_path = os.path.join(scratch_dir, "padded.nii.gz")
    sitk.WriteImage(padded, padded_path)
    try:
        # VTK reports a failure by printing to stderr -- which no operator
        # sees -- and handing on an empty output, so each stage is checked
        # here rather than discovered as a missing or zero-cell file later.
        reader = vtk.vtkNIFTIImageReader()
        reader.SetFileName(padded_path)
        reader.Update()
        if reader.GetErrorCode() or reader.GetOutput().GetNumberOfPoints() == 0:
            raise RuntimeError(
                "VTK could not read back the padded label map "
                f"(error code {reader.GetErrorCode()})"
            )

        marching = vtk.vtkDiscreteMarchingCubes()
        marching.SetInputConnection(reader.GetOutputPort())
        # One contour per label PRESENT, rather than upstream's blanket
        # `GenerateValues(100, 1, 100)`: a label above 100 was never contoured
        # at all, and 100 contour values were computed for a five-label map.
        marching.SetNumberOfContours(len(labels))
        for index, label in enumerate(labels):
            marching.SetValue(index, label)
        marching.Update()
        if marching.GetOutput().GetNumberOfCells() == 0:
            raise RuntimeError(
                f"marching cubes produced no surface for {len(labels)} label(s) "
                f"present in the crop"
            )

        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(marching.GetOutputPort())
        smoother.SetNumberOfIterations(smoothing_iterations)
        smoother.SetFeatureAngle(feature_angle)
        smoother.SetRelaxationFactor(relaxation_factor)
        smoother.Update()

        model = _to_patient_space(smoother.GetOutput(), padded,
                                  reader.GetOutput().GetSpacing())
        _color_by_label(model, labels)

        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(destination)
        writer.SetInputData(model)
        if writer.Write() != 1 or writer.GetErrorCode():
            raise RuntimeError(
                "VTK could not write the surface file "
                f"(error code {writer.GetErrorCode()})"
            )
    finally:
        if os.path.exists(padded_path):
            os.remove(padded_path)

    return labels


def _to_patient_space(model, image, reader_spacing):
    """Move the surface from VTK's index-scaled space into the image's own.

    `vtkNIFTIImageReader` deliberately does not apply the file's qform: it sets
    the output origin to (0, 0, 0), reports the spacing, and leaves the matrix
    for the caller. Upstream never applied it, so every surface it produced was
    written at the wrong place -- offset by the volume's origin, un-rotated, and
    shifted again by the padding. Inside the Slicer module nobody saw it,
    because the module loaded the segmentation rather than the `.vtk`; the
    moment the returned file is opened beside its scan, the mesh floats away
    from the anatomy it belongs to. The same shape of defect as ALI's
    `display.visibility: false`.

    A point `p` in the reader's space is the continuous index `p /
    reader_spacing`, so the physical point is
    `origin + direction @ (spacing * index)` -- one 4x4 matrix, no resampling.
    The spacing ratio is there only because the NIfTI header stores pixdim as
    float32, so the reader's spacing differs from the image's in the last bits.
    """
    import numpy as np
    import vtk

    direction = np.array(image.GetDirection()).reshape(3, 3)
    spacing = np.array(image.GetSpacing())
    origin = np.array(image.GetOrigin())
    linear = direction @ np.diag(spacing / np.array(reader_spacing))

    matrix = vtk.vtkMatrix4x4()
    for row in range(3):
        for column in range(3):
            matrix.SetElement(row, column, float(linear[row, column]))
        matrix.SetElement(row, 3, float(origin[row]))

    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)
    filter_ = vtk.vtkTransformPolyDataFilter()
    filter_.SetTransform(transform)
    filter_.SetInputData(model)
    filter_.Update()
    return filter_.GetOutput()


def _color_by_label(model, labels: list) -> None:
    """Give every cell the colour of the label it was contoured from.

    Upstream set EVERY cell to `LABEL_COLORS[np.max(img_arr)]`, so the per-label
    colouring its own table implies never happened: a five-structure
    segmentation came out uniformly the colour of structure 5.

    `vtkDiscreteMarchingCubes` labels its output with the contour value, and
    `vtkSmoothPolyDataFilter` moves points without changing the cells, so the
    scalars survive the smoothing and say which label each triangle belongs to.
    VTK 9.6 puts them on the CELLS; the point-scalar branch is there because
    older releases put them on the points, and a tool whose colours depend on
    the VTK build is worse than one that checks.
    """
    import vtk

    scalars = model.GetCellData().GetScalars()
    if scalars is None:
        scalars = model.GetPointData().GetScalars()
        point_scalars = True
    else:
        point_scalars = False

    color = vtk.vtkUnsignedCharArray()
    color.SetName("Colors")
    color.SetNumberOfComponents(3)
    color.SetNumberOfTuples(model.GetNumberOfCells())

    fallback = labels[0]
    for cell in range(model.GetNumberOfCells()):
        if scalars is None:
            label = fallback
        elif point_scalars:
            # The label of the cell's first point; every point of a contour
            # cell carries the same value.
            ids = model.GetCell(cell).GetPointIds()
            label = int(scalars.GetTuple1(ids.GetId(0))) if ids.GetNumberOfIds() else fallback
        else:
            label = int(scalars.GetTuple1(cell))
        color.SetTuple3(cell, *color_for_label(label))

    model.GetCellData().SetScalars(color)
