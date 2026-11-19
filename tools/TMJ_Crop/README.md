# sadt-tmj-crop

Crops the temporomandibular joints of a CBCT into fixed-size volumes, so a
clinician can hand each joint to a viewer or a model without cutting the scan by
hand. It orients the scan, finds the mandible, and cuts a box around each
condylar head.

## Provenance

Written for this repository by Jules Grivot Pélisson (UNC), from
`scripts/crop_tmj.py` of a research project's TMJ pipeline (the `--oriented` branch; its older rigid-fit branch is dropped). That repository has no commit to cite, so the script is identified by its content: 317 lines, sha256 `0b8b8a96def409edb469becaf0cb6c30154f403c5a464fa0bc882cb486af0a27`, read on 2026-11-18. On a public scan, both crop arrays of this tool and of that script are identical.
No upstream module in `SlicerAutomatedDentalTools` corresponds to it.
Changes from that script: the geometry is a package with tests on synthetic
volumes, the orientation and the mandible are asked of ASO and AMASSS (and Convert, when ASO's file is not NIfTI) through
the supervisor instead of being run by a shell script, and the reference
landmarks come from ASO's report instead of a file the caller has to carry.
Validated by the script's authors on a clinical cohort of 198 CBCTs (396 of 396
sides found); this port has been driven end to end through a server on one
public test scan.

## What it does

| | |
|---|---|
| Inputs | `scans`: one CBCT (`.nrrd`, `.nii.gz`, any format ASO reads) or a folder of them. `output_dir`: where results go. |
| Options | `sides` (`Left and Right`, `Left` or `Right`: only the joints asked for are searched, cropped, drawn and reported), `outputs` ("Extra outputs", nothing ticked by default: `Oriented scan`, `Mandible segmentation`), `scale_by` (`none` or `mandible-volume`) and `scale` (a fixed factor, 0 = not given). The last two are hidden from the client panel on purpose: a clinician should not change the box size. |
| Outputs | Per scan: `<scan>_R.nii.gz` and `<scan>_L.nii.gz` (the crops), `<scan>.json` (the report), `<scan>_qc.png` (a quick view), and one `run_report.json` for the run (the catalogue's run report: for each case what it was given, its status and the files it produced, which is how the server relates a file to its case). On request, also `<scan>_oriented.nii.gz` (the whole scan in the Frankfurt horizontal frame the crops were cut from) and `<scan>_mandible.nii.gz` (the AMASSS mandible mask in that frame). |
| Model files | None of its own. ASO resolves its landmark weights and reference bundle, AMASSS its masks. |
| Calls | `ASO` (Fully-Automated, Frankfurt horizontal, which itself calls `ALI_CBCT`) then `Convert` (only when ASO wrote the oriented scan in another format than NIfTI, which it does for a `.nrrd` or `.gipl` input, because ASO keeps the input's format), then `AMASSS` (`MAND`, separate). Convert must be released before this tool. |
| Case status | `ok` when every requested joint was cropped; `partial: <side>: <why>` when one was not found or is flagged; a scan with no requested joint, or any failure, carries its reason. Every message reaches the clinician's panel. |

## The method

1. **Orientation.** ASO rotates the scan onto the Frankfurt horizontal frame. It
   does not translate it, so the oriented frame differs from the reference's by
   a translation only: the mean difference between the reference landmarks
   (published in ASO's report as `reference_positions`) and the landmarks ASO
   kept for this scan. Landmarks ASO dropped as outliers take no part, and are
   listed in the report.
2. **Condyle.** On each side the mandible mask is searched in a window around
   that side's porion (30 mm lateral, 10 mm behind, 25 mm ahead; the window stops
   short of the coronoid process). The condyle top is the most superior mask
   voxel there; the head is the mask within 8 mm below it and 12 mm of it.
   A side is `ok` when the top is within 25 mm of the porion, and says
   `check: condyle far from porion` otherwise.
3. **Crop.** A box of 96 x 112 x 80 voxels at 0.3 mm (28.8 x 33.6 x 24.0 mm) is
   resampled linearly in the oriented space, raised 10.8 mm above the top and
   placed 14.25 mm anterior of the head centre, so the fossa roof and part of
   the eminence are inside. **The left crop is mirrored** so every joint has the
   right-side orientation. With `scale_by = mandible-volume` the box follows the
   cube root of the mandible volume over 54.6 ml; the voxel count stays the same.

## What was validated

- Geometry, on a synthetic head with a known translation: the translation is
  recovered from the landmarks to 1e-3 mm, both condyles are found, the left crop
  is the mirror of the right, a dropped landmark takes no part, and an empty
  mask is refused. `uv run pytest` in this folder.
- The chain, through a fake supervisor: each tool is called once per batch, with
  no model path and no output folder; a scan ASO could not orient gets a report
  and the batch goes on; a batch that produced nothing raises.
- A study sent as one folder, through a fake supervisor: scans keep their own
  subfolder, so `T1/case` and `T2/case` do not collide or swap (when two scans
  share a file name, every output is named `<subfolder>_<name>`); names with
  brackets are found; a scan that breaks the crop (any error) is reported with
  its error class only and the batch goes on; the reason ASO gave for a failed
  scan reaches that scan's report with any path replaced by `<file>`; a
  `scale_by` or `scale` that means nothing is refused rather than ignored.
- The run report: every file of the output belongs to exactly one case of it, a
  failed case is listed with a reason that carries no path, and a batch that
  cropped nothing leaves no report. The tool does not declare `CASE_INPUT`, so
  the server replays the whole batch rather than narrowing it to marked cases;
  that wants a test through the viewer's review flow first.
- End to end, through the server, on one public test CBCT (about a minute on one
  card): both sides `ok`, the top 12 mm from each porion. **Not** diffed
  numerically against the original script on the cohort it was validated on.

## What to know before using the files

- **Frames.** The crops are written in the frame of the ORIENTED scan (the one
  `Oriented scan` returns), so they open on top of it. The original script wrote
  them in the reference frame, i.e. shifted by ASO's translation: the arrays are
  the same, the origin in the header is not. Crops from the two cannot be
  overlaid or pooled by their headers.
- **The left crop is mirrored in its array only.** Its affine is LAS, so a loader
  that reorients to RAS (`as_closest_canonical`, MONAI `Orientationd`, TorchIO
  `ToCanonical`) undoes the mirror. Read the raw array when both sides must look
  alike.
- **Nothing checks how much of a box falls outside the scan** or on the padding
  ASO adds when it rotates the volume. Measured on the public test scan: 5.5 %
  of the right box and 11.7 % of the left, both reported `ok`. The original
  script behaves the same; a threshold would need calibrating on a cohort.

## Needs

ASO's report must carry `reference_positions`; against an older ASO the tool
stops with a message saying so rather than guessing where the reference is.
