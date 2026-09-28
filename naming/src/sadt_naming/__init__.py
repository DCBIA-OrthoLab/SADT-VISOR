"""Which file extensions are a volume, a surface, a set of landmarks.

**Shared rather than copied, and that is the exception this package is.**
`CONTRIBUTING.md` says small helpers are copied between tools on purpose --
`iter_scans` and `errors.py` stay duplicated, deliberately. What is shared is
what the same document calls a contract with the outside world, and it names
this table as one: ALI's two engines disagreed about `.stl` before the port,
so the UI accepted files the CLI silently ignored, and the fix was to put the
table in `tools/ALI/common/`. This is that fix, at the width the problem
actually has -- the disagreement was never between ALI's two halves alone.

Measured on 2026-09-24, before this existed: four byte-identical copies of the
scan vocabulary (AMASSS, ASO, Batch_Dental_Seg, AREG), and FIVE different
values for `SURFACE_EXTENSIONS` under one name, from `(".vtk",)` to
`(".vtk", ".vtp", ".stl", ".obj", ".off")`. Nothing in the code said which
difference was a decision and which was an oversight.

**Stdlib only, and no dependency, ever.** This installs into environments whose
pins are deliberately incompatible; anything imported here would have to be
satisfiable by all of them at once. The budget is `os`.

**A tool may read LESS than a format.** The table says what the format IS, not
what a tool handles, and advertising more than you read is the original bug:

    from sadt_naming import SURFACE
    SURFACE_EXTENSIONS = (".vtk",)   # this tool reads only this one

That narrowing is a fact about the tool and belongs beside it. What must not
be written beside it again is the list itself.
"""

import os

# A 3D image. Ordered longest-first, which `split_extension` depends on:
# `.nii.gz` has to be recognised before `.nii` or the stem keeps a `.nii`.
VOLUME = (".nii.gz", ".nrrd.gz", ".gipl.gz", ".nii", ".nrrd", ".gipl")

# A 3D surface mesh. Every extension here is one that the FORMAT covers -- a
# tool declaring it must be able to read it, or narrow (see the module
# docstring).
SURFACE = (".vtk", ".vtp", ".stl", ".obj", ".off")

# Slicer markups: landmarks, and anything else placed by hand. `.json` is in
# because that is what the pre-port CLIs wrote for identical content, and a
# reader who opens a folder from 2024 still has those.
MARKUPS = (".mrk.json", ".json")

# A spatial transform, as ITK and Slicer write them.
TRANSFORM = (".tfm", ".mat", ".h5", ".hdf5", ".txt")

# Free-text clinical material.
NOTES = (".txt", ".pdf", ".docx")

# Tabular measurements.
TABLE = (".csv", ".xlsx", ".ods")

# The compressed spelling ITK can WRITE for each volume extension. NIfTI and
# GIPL take an external .gz; NRRD compresses inside the file and ITK has no
# ".nrrd.gz" writer at all, so that spelling maps back down to ".nrrd".
_COMPRESSED = {".nii": ".nii.gz", ".gipl": ".gipl.gz", ".nrrd.gz": ".nrrd"}


def split_extension(filename: str, extensions=VOLUME) -> tuple:
    """`('scan.nii.gz')` -> `('scan', '.nii.gz')`, compound extensions kept.

    Not `os.path.splitext`, which answers `('scan.nii', '.gz')` and leaves a
    stem no patient is called. Falls back to it for anything unrecognised, so
    a name this table says nothing about still splits somewhere sensible.
    """
    lower = filename.lower()
    for extension in extensions:
        if lower.endswith(extension):
            return filename[: -len(extension)], filename[-len(extension):]
    return os.path.splitext(filename)


def compressed_extension(extension: str) -> str:
    """The compressed spelling ITK can write for a volume extension."""
    return _COMPRESSED.get(extension.lower(), extension)


def has_extension(filename: str, extensions) -> bool:
    """Whether `filename` ends in one of `extensions`, case-insensitively.

    Written once because every tool walks a folder and every tool got this
    subtly different: `.endswith` on a bare name, on a lowered name, on a
    tuple, or a `splitext` that cannot see `.nii.gz` at all.
    """
    lower = filename.lower()
    return any(lower.endswith(extension) for extension in extensions)


# ---------------------------------------------------------------------------
# Markers a run leaves on a name
# ---------------------------------------------------------------------------
# Every marker any tool of the catalogue appends, so that any tool can strip
# them and find the patient underneath. It is a UNION and it must stay one:
# this table is read to RECOGNISE, never to decide what to write, so an entry
# a given tool never produces costs that tool nothing -- while a missing entry
# costs a patient. ASO's own copy lacked `_Seg`, so a cohort AMASSS had
# segmented first read as twice as many patients as it had.
#
# Case-SENSITIVE, which is why "_Scan"/"_scan", "_Seg"/"_seg" and "_Or"/"_OR"
# are each listed twice and "_SEG" is not listed at all: these are the
# spellings the tools actually write, and matching case-insensitively would
# truncate a patient genuinely called `..._SEGMENT`.
#
# Matched as WHOLE TOKENS and truncated from -- never as a substring at any
# index. "_Seg" is a mark a previous run left; "_Seg1" is part of somebody's
# name. See `token_index`.
OUTPUT_MARKERS = (
    "_lm_Pred", "_Scanreg", "_MERGED", "_OutReg", "_SegOr", "_SegOut",
    "_scan", "_Scan", "_Seg", "_seg", "_Or", "_OR", "_lm",
)

# ---------------------------------------------------------------------------
# Tokens that are part of the identity, and must NEVER be stripped
# ---------------------------------------------------------------------------
UPPER = "Upper"
LOWER = "Lower"

# Which jaw a mesh is, from a token in its name. **The union of every spelling
# any tool ever accepted**, so a file that one tool reads is not refused by the
# next: before this, `P1_MX.vtk` was read by AREG and refused by ASO, and
# `P1_max.vtk` was read by both and refused by AREG_IOSCBCT, which knew four
# spellings out of thirteen. A clinician cannot be expected to know which tool
# learnt which word.
#
# Compared lowercased, and always as a whole TOKEN. That rules out the
# substring disaster -- `"max" in name` makes `MAXILLOFACIAL_03` a maxilla --
# and it does NOT rule out a patient whose name genuinely IS a jaw word:
# `MAX_01.vtk` reads as Upper here, exactly as it already does in ASO and
# AREG. Preserved on purpose rather than fixed: the rule that would catch it
# ("a jaw token must have something before it") refuses `Upper_gold.vtk`,
# which is the published reference's own file name. Pinned by a test so it is
# a known limit and not a surprise.
#
# `maxillaire` is here because `mandibule` was: one French spelling without the
# other is an oversight, not a decision.
JAW_TOKENS = {
    "u": UPPER, "up": UPPER, "upper": UPPER,
    "maxilla": UPPER, "maxillaire": UPPER, "max": UPPER, "mx": UPPER,
    "l": LOWER, "low": LOWER, "lower": LOWER,
    "mandible": LOWER, "mandibule": LOWER, "mand": LOWER, "md": LOWER,
}

# Two timepoints of one subject are two scans, not one. Dropped ONLY when a
# caller asks for the subject rather than the acquisition -- collapsing them by
# default dropped the second scan and merged both landmark sets into the
# survivor, which is the defect this constant is named after.
TIMEPOINT_TOKENS = ("t0", "t1", "t2")

# What separates tokens in every convention here.
_SEPARATORS = "_-."


def token_index(stem: str, marker: str) -> int:
    """Where `marker` starts in `stem`, if it sits on token boundaries. -1 if not.

    The rule the whole vocabulary depends on. A plain `find` is what collapsed
    `P_Seg1_T1` and `P_Seg2_T1` onto one patient: `_Seg` matches inside
    `_Seg1`, two subjects become one, and nothing says so.

    A marker that already begins with a separator only has to be followed by
    one, or be at the end of the stem.
    """
    start = 0
    while True:
        found = stem.find(marker, start)
        if found < 0:
            return -1
        after = found + len(marker)
        if after >= len(stem) or stem[after] in _SEPARATORS:
            return found
        start = found + 1


def tokens_of(stem: str) -> list:
    """The stem cut on every separator, empties dropped."""
    out, current = [], ""
    for character in stem:
        if character in _SEPARATORS:
            if current:
                out.append(current)
            current = ""
        else:
            current += character
    if current:
        out.append(current)
    return out


# ---------------------------------------------------------------------------
# A jaw run together with a timepoint
# ---------------------------------------------------------------------------
#
# Upstream's own AREG test set is named `A2_UpperT1.vtk` / `A2_UpperT2.vtk`:
# the jaw and the timepoint run together with no separator, so the whole thing
# is a single token, `uppert1`, matching neither table. `jaw_of` then found no
# jaw at all and the mesh was refused -- upstream's published IOS data could not
# be run.
#
# The rule is AREG's, verbatim rather than reinvented: the AREG family already
# split these names in `sadt_areg_common.pairing`, and this module did not, so
# one file was read by AREG_IOS and refused by ASO. A THIRD reading of the same
# name is the failure this shared vocabulary exists to prevent, so the fix is to
# adopt that one, not to write a better one.
#
# Deliberately narrow, and keyed on the two STATIC tables rather than on
# anything a caller passes: the split fires only when the prefix is a known jaw
# token AND the suffix is a known timepoint. `PAT1` is therefore untouched
# (`pa` is not a jaw), and so is any identifier that merely ends in something
# timepoint-shaped. Splitting on every camelCase boundary would start eating
# patient identifiers, which is what token matching exists to prevent.


def split_jaw_timepoint(part: str) -> list:
    """`['UpperT1']` -> `['Upper', 'T1']`, and everything else untouched."""
    lowered = part.lower()
    for timepoint in TIMEPOINT_TOKENS:
        if not lowered.endswith(timepoint) or len(lowered) <= len(timepoint):
            continue
        if lowered[: -len(timepoint)] in JAW_TOKENS:
            cut = len(lowered) - len(timepoint)
            return [part[:cut], part[cut:]]
    return [part]


def jaw_in_token(token: str):
    """The jaw one token names, or None -- a concatenated timepoint included."""
    for part in split_jaw_timepoint(token):
        jaw = JAW_TOKENS.get(part.lower())
        if jaw is not None:
            return jaw
    return None


def jaw_of(stem: str):
    """`Upper`, `Lower`, or None when no token names a jaw.

    None rather than a default, and that is the whole point: defaulting to
    Lower registered a maxillary mesh named `patient1.vtk` against the
    mandibular reference and returned it as a success.
    """
    for token in tokens_of(stem):
        jaw = jaw_in_token(token)
        if jaw is not None:
            return jaw
    return None


def strip_markers(stem: str, markers=OUTPUT_MARKERS) -> str:
    """The patient a stem belongs to: everything before the earliest marker.

    Truncated, not deleted -- everything a marker introduces goes with it, so
    `P1_Scan_reoriented` keys to `P1` and not to `P1_reoriented`. A stem that
    BEGINS with a marker is left alone: there is no patient in front of it to
    keep, and a file literally called `_Or.nii.gz` is a patient whose name we
    cannot read, not a patient with no name.
    """
    cut = len(stem)
    for marker in markers:
        found = token_index(stem, marker)
        if 0 < found < cut:
            cut = found
    return stem[:cut] or stem
