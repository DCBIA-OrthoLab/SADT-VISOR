# Convert

Converts 3D scans between file formats, and DICOM series to a volume. Any tool can
call it by name (`sup.run("Convert", ...)`) when the file a previous step wrote is
not a family it reads, so that tool needs no image library of its own. A clinician
can also run it alone.

## Provenance

Written for this repository by Jules Grivot Pélisson (UNC). New tool, not a port.
It does the same job as the DICOM conversion that `ASO` (`cbct/dicom.py`) and
`AREG_CBCT` (`dicom.py`) each carry in a slightly different copy; those two are
**not** changed by this tool and still use their own.
Pins: SimpleITK 2.5.6, the version `ASO` and `AREG_CBCT` lock.
Changes from those copies: no `dicom2nifti` fallback (see below).

## What it does

| | |
|---|---|
| Inputs | `scans`: one scan, or a folder holding scans (`.nii.gz`, `.nii`, `.nrrd`, `.gipl.gz`, `.gipl`; `.nrrd.gz` is not read) and DICOM series, at any depth. |
| Options | `to`: `nii.gz` (default), `nrrd` or `gipl.gz`. |
| Outputs | One `<name>.<to>` per scan, in the same sub-folder as its input. A DICOM series takes the name of its folder (a folder holding several series gives `series_1`, `series_2`, ... inside it). Sent through a server as a zip of one folder, the server unwraps that folder, so the file is called `scan`. `run_report.json` lists, per case, the input, the status and the file produced. |
| Model files | None. |

- A file already in the requested format is copied byte for byte, after its header
  has been read to prove it is a scan.
- Nothing is resampled. After writing, the file is read back: a case whose spacing, origin,
  direction or number of components changed is refused (GIPL keeps no direction and no vector
  components; NIfTI makes a skewed direction orthogonal). Voxel values keep their type.
- A DICOM series with a missing or unevenly spaced slice is refused rather than converted with
  an averaged spacing. A series is one DICOM series UID within one folder: one split over two
  folders gives two partial volumes, and is not detected.
- A `.nii.gz` that is not gzip is refused, not copied. A DICOM series with gantry tilt is refused (it would be stacked straight and stretched).
- One unreadable scan gets a short reason in the report and the batch goes on. A batch
  where nothing was converted raises.
- Two inputs that would write the same name (`a.nrrd` next to `a.nii.gz`): the first is
  written, the other is reported as skipped.
- No file or folder name appears in a log line, an error or the report's reasons: a failure is
  reported as its class and a fixed phrase.

## What was validated, and what was not

Validated on synthetic images built in the tests: voxels, spacing and origin identical
after NRRD to NIfTI, NIfTI to NRRD and NRRD to GIPL; a synthetic DICOM series; a folder
with sub-folders; a corrupt file; a name clash; empty and missing inputs.

**Not validated:** real clinical DICOM exports. `ASO` retries a series that GDCM refuses
with `dicom2nifti`; this tool does not, so such a series is reported as failed. Adding the
fallback means a second pinned dependency and a run against real exports. Meshes and
landmark files are not handled.
