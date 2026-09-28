"""The CBCT read as a surface, so the registration has something to refine on.

The landmark fit places the arch from six points. Six points carry six points'
worth of error: when one of them is a landmark the network put on the opposing
arch, the whole arch follows it, and nothing downstream notices. What does
notice is the CBCT surface itself -- forty thousand crown points against it
instead of six -- and that is the only reason this module exists.

Contouring needs a level, and the level is read off the scan rather than fixed:
the same anatomy images at different intensities depending on the machine and
the reconstruction, so one number cannot serve every scan. The rule here is the
one upstream settled on after measuring it -- halfway between what the enamel
reads at the landmarks and what the tissue around the arch reads -- with a
fallback to 400, the value every run used before anybody measured it.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# How far around a landmark to look for the crown it is meant to sit on. Two
# voxels: a landmark a voxel off the occlusal surface still reads the tooth,
# while a window wide enough to reach the next tooth would read the brightest
# thing nearby rather than this one.
PROBE_RADIUS_VOXELS = 2

# What counts as soft tissue in Hounsfield units. Not a physical definition: it
# is the band that holds the mucosa and gingiva the crowns emerge through,
# whose median is the "surround" half of the level.
SOFT_TISSUE_BAND = (-300.0, 300.0)

# What to contour at when the scan cannot be read. Every run used this before
# the level was measured, so it is a known quantity rather than a guess.
FALLBACK_LEVEL = 400.0


def _to_voxel(positions, ijk_to_lps):
    """Patient coordinates to voxel indices, rounded to the nearest voxel."""
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    homogeneous = np.hstack([positions, np.ones((len(positions), 1))])
    return np.rint((homogeneous @ np.linalg.inv(ijk_to_lps).T)[:, :3]).astype(int)


def _peak_around(array, voxel):
    """The brightest voxel in a small window, or None when it falls outside.

    Brightest rather than the voxel itself: a landmark one voxel off the crown
    would otherwise read the gap beside the tooth and be judged as not on it.
    """
    i, j, k = voxel
    depth, height, width = array.shape
    if not (0 <= i < width and 0 <= j < height and 0 <= k < depth):
        return None
    radius = PROBE_RADIUS_VOXELS
    window = array[max(k - radius, 0):k + radius + 1,
                   max(j - radius, 0):j + radius + 1,
                   max(i - radius, 0):i + radius + 1]
    return float(window.max()) if window.size else None


def enamel_and_soft_levels(array, ijk_to_lps, landmarks):
    """The two intensities the crown boundary lies between.

    The enamel plateau is the median of the peaks read at the landmarks, which
    sit on the occlusal surfaces; the median carries no single bad landmark. The
    surround is the median of everything in the soft-tissue band across the box
    the landmarks span.

    Returns (None, None) when the scan cannot be probed, and the caller falls
    back rather than guessing.
    """
    if landmarks is None or len(landmarks) == 0:
        return None, None

    voxels = _to_voxel(landmarks, ijk_to_lps)
    peaks = [peak for peak in (_peak_around(array, v) for v in voxels)
             if peak is not None]
    if not peaks:
        logger.warning("No CBCT landmark falls inside the scan, so the surface "
                       "level cannot be read from the crowns.")
        return None, None

    low = np.clip(voxels.min(axis=0) - PROBE_RADIUS_VOXELS, 0, None)
    high = voxels.max(axis=0) + PROBE_RADIUS_VOXELS + 1
    region = array[low[2]:high[2], low[1]:high[1], low[0]:high[0]]
    in_band = region[(region > SOFT_TISSUE_BAND[0]) & (region < SOFT_TISSUE_BAND[1])]

    return (
        float(np.median(peaks)),
        float(np.median(in_band)) if in_band.size else float(region.min()),
    )


def surface_threshold(array, ijk_to_lps, landmarks, level=None):
    """The level to contour this CBCT at.

    An explicit `level` wins, for the scan this rule cannot read.
    """
    if level is not None:
        logger.info("CBCT contoured at %.0f, as asked.", float(level))
        return float(level)

    enamel, soft = enamel_and_soft_levels(array, ijk_to_lps, landmarks)
    if enamel is None or not enamel > soft:
        logger.warning(
            "The crowns are no brighter than what surrounds them in this scan, "
            "so the surface level cannot be read from it. Falling back to %.0f; "
            "if the registration comes out poor, pass a level explicitly.",
            FALLBACK_LEVEL)
        return FALLBACK_LEVEL

    level = 0.5 * (enamel + soft)
    logger.info("CBCT contoured at %.0f -- enamel reads %.0f at the landmarks, "
                "the soft tissue around the arch %.0f.", level, enamel, soft)
    return level


def landmarks_on_enamel(array, ijk_to_lps, landmarks, level):
    """Which CBCT landmarks actually sit on a tooth, as {label: bool}.

    A landmark whose neighbourhood never reaches the level the scan is contoured
    at is not on enamel. In a closed bite that is usually a point the network
    put on the opposing arch -- which the landmark fit cannot detect, because
    such a point is perfectly consistent with itself.

    An empty verdict lets every landmark through, which is what a caller with no
    volume to read gets.
    """
    verdict = {}
    for label, position in landmarks.items():
        peak = _peak_around(array, _to_voxel(position, ijk_to_lps)[0])
        if peak is None:
            logger.warning("%s falls outside the scan entirely.", label)
            verdict[label] = False
            continue
        verdict[label] = peak >= level
        if not verdict[label]:
            logger.warning(
                "%s reads %.0f at its brightest, under the %.0f the scan is "
                "contoured at, so it is not on a tooth. In a closed bite that is "
                "usually a point that landed on the opposing arch.",
                label, peak, level)
    return verdict


def read(scan_path, landmarks=None, level=None):
    """The CBCT as a surface mesh in patient coordinates, plus what it says.

    Returns the contoured surface and the per-label verdict on the landmarks.
    Both come from one read of the volume: the array is gigabytes, and it is
    dropped on the way out rather than carried to two separate callers.
    """
    import SimpleITK as sitk
    import pyvista as pv

    image = sitk.ReadImage(scan_path)
    array = sitk.GetArrayFromImage(image)

    ijk_to_lps = np.eye(4)
    ijk_to_lps[:3, :3] = (np.array(image.GetDirection()).reshape(3, 3)
                          @ np.diag(image.GetSpacing()))
    ijk_to_lps[:3, 3] = np.array(image.GetOrigin())

    landmarks = landmarks or {}
    positions = np.array(list(landmarks.values()), dtype=float) if landmarks else None
    level = surface_threshold(array, ijk_to_lps, positions, level)
    on_enamel = landmarks_on_enamel(array, ijk_to_lps, landmarks, level)

    # transpose because SimpleITK hands the array back as (k, j, i) while
    # pyvista reads it as (i, j, k); contouring the untransposed array gives a
    # surface that looks plausible and is indexed along the wrong axes.
    surface = pv.wrap(array.transpose(2, 1, 0)).contour(isosurfaces=[level])
    return surface.transform(ijk_to_lps, inplace=False), on_enamel
