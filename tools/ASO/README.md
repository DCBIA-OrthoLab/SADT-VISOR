# sadt-aso

Orients CBCT volumes or intra-oral meshes onto a standard reference frame, so
that two timepoints of the same patient -- or two patients -- can be compared in
the same coordinate system. One tool, two engines, four modes:

|          | Semi-Automated | Fully-Automated |
|---|---|---|
| **CBCT** | landmarks you send, ICP onto a reference landmark set | landmarks predicted first, then the same ICP |
| **IOS**  | landmarks you send, ICP per jaw | tooth centroids of an already segmented mesh, ICP per jaw |

## Provenance

Ported from DCBIA-OrthoLab/SlicerAutomatedDentalTools, paths `ASO/`,
`ASO_CBCT/{PRE,SEMI}_ASO_CBCT/` and `ASO_IOS/{PRE,SEMI}_ASO_IOS/`, by way of
`slicer-remote-tool-server`'s `tools/ASO/` -- whose history this repository
carries, so `git log --follow` on `src/sadt_aso/cbct/pipeline.py` reaches back
through it.

> **The upstream commit is not recorded**, for the same reason as
> [ALI](../ALI/README.md#provenance): the server-side port landed with no
> upstream SHA in the message or the tree. The per-module mapping is exact and
> was verified by reading; only the revision is missing. See
> [PROVENANCE.md](../../PROVENANCE.md).

Upstream pins **not** kept: nothing is pinned upstream. This package pins
SimpleITK 2.5.6, vtk 9.6.2, numpy 2.3.2 and dicom2nifti 2.6.2 -- the first three
being what every sibling tool already locks.

Changes from upstream -- the algorithm is untouched, the envelope is not. The
first group were made during the server-side port and are unchanged here.

- **The whole Slicer envelope is gone**: no `<filter-progress>` prints, no
  `time.sleep` progress theatre, no `sys.exit`, no log file the client polls,
  no `<name>Error.txt` written beside the results, and nothing written into the
  caller's input tree. A patient that cannot be registered is a row in the run
  report saying why, not a stray text file in a folder that otherwise reads as
  a success.
- **Patients are keyed by path relative to the input root**, not by base name,
  so two scans called `scan.nii.gz` in different folders stay two patients.
- **Landmark files matching no scan are reported.** They used to be dropped in
  silence, and the caller was then told "no landmark file alongside this scan"
  about a file sitting next to it under a name that had put it in another
  bucket.
- **The reference is checked against the selection up front.** The two published
  reference bundles carry disjoint landmark sets, so picking the second without
  changing the selection made every patient fail separately with "0 usable
  landmarks" -- forty identical failures for one wrong choice.
- **The semi-automated CLI registered centred volumes against uncentred
  points**, because it recentred nothing and then read a `.tfm` only the
  fully-automated chain ever produced. `center_landmarks` fixes that.
- **`_no_landmarks_reason` tells three situations apart** that one message used
  to cover with the wording most likely to be wrong.

And this migration's:

- **Zip extraction removed.** The server unpacks archives before `run()` is
  called, with the bomb cap and `strip_single_root` that used to live in
  `_as_directory`. What survives of that function is linking a single *file*
  into a directory of its own, because its neighbours are not necessarily part
  of the same input.
- **`src/ali_client.py` is gone.** It reached into the server's `registry.TOOLS`
  to call ALI in-process. Replaced by one `sup.run("ALI", ...)` -- see below.
- **`landmarks` is a new argument**, and it is what makes the tool usable
  standalone.
- **`max_triplets` and `seed` are arguments**, not settings, with the defaults
  `icp.register` already declared (2500 and 0). `run()` must not read the
  environment, and both change the result, so they belong next to it.
- **The `section`, `visible_when` and `groups` metadata is gone** with the
  `ArgSpec` schema that carried it. See "What the client loses".

## Calling another tool

Fully-automated CBCT needs landmarks it does not place itself, and it needs them
**in the middle of the run**:

```
recentre every scan  →  predict landmarks on the CENTRED scans  →  register
```

That ordering is upstream's -- the Slicer chain ran `PRE_ASO_CBCT` before
`ALI_CBCT` -- and it is why this tool cannot be expressed as "run ALI first, then
run ASO on its output". Recentring is a pure metadata change, so the reordering
*ought* to be exact; ALI's `physical_position` takes the absolute value of the
origin, which does not commute with moving it. That is
[issue #11](https://github.com/DCBIA-OrthoLab/SADT-VISOR/issues/11), it is **not**
being fixed here, and the order is kept as it was rather than bet on.

So ASO receives a **supervisor** and calls through it, at the point ALI has
always run:

```python
predictions = sup.run("ALI", input=centered_root, model=..., output_dir=..., landmarks=[...])
```

- `sup` is **keyword-only and unannotated**. That is the marker `describe.py`
  reads to keep it out of the schema -- it is not data, and no client sends one.
  The schema instead publishes `"supervisor": true`, so a runner that cannot
  inject one refuses the tool rather than calling it and failing halfway.
- It is **duck-typed**. Nothing here imports a supervisor type; doing so would
  need a package shared with the server, which is the coupling this repository
  exists to remove. Three implementations are interchangeable: the server's, the
  `LocalSup` in `tests/test_integration.py`, and the `FakeSup` in
  `tests/test_run.py`.
- `sup.run("ALI", ...)`, never `sup.ALI(...)`: a typo in a string is greppable,
  and the call graph stays inspectable.
- ALI is asked for landmarks **by name**, not by region. ASO's seven points
  straddle two of ALI's regions, so asking by region would run 58 agents to use
  seven -- and one agent is a full two-scale walk of the volume.

**The server does not implement supervisors yet** -- re-checked 2026-08-14 against
`newArch`: one occurrence of the word in the whole repository, and it is a line
of documentation. Until that changes, fully-automated CBCT is not servable and
this package's other three modes are, including fully-automated **with
`landmarks` supplied**, which needs no supervisor at all. See
[docs/SERVER_CONTRACT.md](../../docs/SERVER_CONTRACT.md).

### From a checkout, with a supervisor

`scripts/run_tool.py` builds one and chains for you -- no server, no Docker:

```bash
python scripts/run_tool.py ASO \
    --input cohort/ --reference gold/ --output-dir out/ \
    --modality CBCT --automation Fully-Automated \
    --cbct-landmarks Ba S N --landmark-models /path/to/ALI_CBCT_Models
```

It runs ALI in `tools/ALI/.venv` as a subprocess, at the point in ASO's run
where ALI has always been called. See the repository README.

### Standalone, with no supervisor at all

Predict the landmarks yourself and pass the folder. Same registration, same
result, no supervisor and no repository checkout:

```bash
uv sync --project tools/ASO
uv run --project tools/ASO python -c "
from sadt_aso import run
run(input='cohort/', reference='gold/', output_dir='out/',
    modality='CBCT', automation='Fully-Automated', landmarks='predicted/')
"
```

`landmarks` also wins over the supervisor when both are available, so a caller
that already has the points does not spend a GPU re-predicting them.

## What it does

| | |
|---|---|
| Inputs | `input`: one scan (`.nii`/`.nii.gz`/`.nrrd`/`.nrrd.gz`/`.gipl`/`.gipl.gz`), one mesh (`.vtk`/`.stl`), or a folder of either. `reference`: the already-oriented case defining the target frame -- optional: left empty, ASO uses its own bundles from `DATA/ASO/models/`. `frame`: the CBCT frame by name (`Frankfurt horizontal`, `Occlusal plane`), which is how another tool asks for an orientation without holding ASO's references; `cbct_landmarks` sent empty then means the frame's own landmarks. `output_dir`: where results go. |
| Outputs | Per patient: the oriented scan or mesh, its landmarks (`_lm_Or.mrk.json`) and the transform (`_Or_transform.tfm`), mirroring the input tree, plus `ASO_report.json`. |
| Model files | None for three of the four modes -- a *reference bundle* is data, not weights. Fully-automated CBCT needs ALI's bundle, named in `landmark_model` and passed straight through. Both `reference` and `landmark_model` are named so the server publishes them as hosted names rather than uploads. |
| GPU | None. This is the one migrated tool with no torch in it; the venv is 1.2 GB. |

Three behaviours worth knowing before reading a result:

- **`modality` is never inferred from the file extension.** A folder can hold
  either kind, and guessing wrong means orienting a patient against the wrong
  reference and calling it a success.
- **The `.tfm` maps ORIENTED → ORIGINAL**, so a point measured on the result
  comes back onto the acquisition. Getting that direction backwards is silent:
  the file still loads and still transforms.
- **`max_triplets` and `seed` change the result.** The report records both.

## What the client loses

The old `ArgSpec` schema published presentation metadata `describe.py` has no
field for. ASO is the tool that used it most, because its four modes share one
schema:

- **`visible_when`** hid each mode's arguments when the other was selected --
  the Slicer module's four-page `QStackedWidget`, expressed as data. Without it
  a panel shows all 115 CBCT landmarks next to all 32 teeth, whichever mode is
  chosen.
- **`section`** grouped the panel into "Inputs", "Landmark Reference", "Teeth &
  Landmarks" and "Outputs".
- **`groups`** laid the teeth out as a dental chart rather than 32 stacked check
  boxes, and the CBCT landmarks as one tab per region.

None of this affects a result, and every cross-argument rule is still *enforced*
-- `_check_cbct` and `_check_ios` refuse an impossible combination before a file
is read. What is lost is the panel preventing it in the first place. Raised here
rather than worked around; the same question as
[ALI's](../ALI/README.md#what-the-client-loses).

## Versions

SimpleITK 2.5.6, vtk 9.6.2, numpy 2.3.2, dicom2nifti 2.6.2, Python 3.11 -- the
deployment image's interpreter, and the same imaging stack the sibling tools
lock, so this adds nothing new to the image.

Nothing upstream is pinned, so there is no upstream pin to keep or discard.
`dicom2nifti` is the one package no other tool here uses; it is a small
pure-python wheel and only the `dicom_input` path touches it.

`uv lock` resolves 26 packages. The venv is 1.2 GB -- a twentieth of a torch
tool's, which is what makes this the cheapest tool in the repository to deploy.

## Validated against

- **Schema**: `describe.py` accepts the signature and publishes 16 arguments,
  `returns: path`, `supervisor: true`, with `choices` on `cbct_landmarks` (115),
  `ios_teeth` (32), `ios_landmark_types` (8), `ios_jaws` (2), `ios_occlusion`
  (3), `modality` (2) and `automation` (2). `sup` is absent from `arguments`, as
  it must be. Asserted out of process against the real venv.
- **Tests**: 193 passing, 1 GPU test deselected. Six of them run the tool **out of
  process, in its own venv**, the way the server does -- including a complete
  semi-automated CBCT registration on synthetic data, so the registration itself
  is exercised for real rather than stubbed.
- **The supervisor seam**: covered in-process by a fake supervisor that writes
  the markups files the real tool would write, into the directory it is handed,
  so the whole seam (call → write → read back → merge per patient → recentre) is
  exercised. `test_the_landmark_tool_is_run_on_the_recentred_scans` asserts what
  ALI is actually handed is centred on the physical origin -- the ordering this
  whole design exists to preserve.
- **Geometry**: the registration is checked against known rotations --
  `test_registration_recovers_the_reference_frame`, and
  `test_the_transform_file_maps_the_result_back_to_the_original` inverts the
  written `.tfm` and lands back on the acquisition.
- **The registration maths is bit-identical to the implementation it replaced.**
  `geometry.py` was made about three times faster in 2026-09 by computing each
  pair's half-alignment once instead of once per triplet, and by calling
  numpy's own fast paths for `norm` and `cross` directly.
  `tests/test_geometry_equivalence.py` carries the pre-rewrite code as a
  reference implementation and asserts `np.array_equal` -- never `allclose` --
  on random inputs, because a coarse alignment differing in the last bit picks
  a different triplet on a near-tie, and a different triplet is a different
  orientation of a patient's skull. `tests/test_ios_file_cache.py` does the
  same for the IOS engine's file cache, by running a cohort with the cache and
  with a cache that re-reads, and comparing every written byte.
- **Patient keys are matched on token boundaries.** `PATIENT_SUFFIXES` used to
  be matched with `stem.find(suffix)` at any index, so `_OR` matched inside
  `_ORTHO` and `_scan` inside `_scanned`: `SMITH_ORTHO_Scan.nii.gz` and
  `SMITH_ORLANDO_Scan.nii.gz` both keyed to `SMITH`, and `discover` then kept
  one of the two scans and handed it both patients' landmarks. Fixed the way
  the shared `sadt_areg_common.pairing` was -- `_token_aligned_index` requires
  the match to close on the end of the stem or a separator, and checks the left
  edge rather than assuming it. Copied rather than imported: ASO does not
  depend on that package. 797 real basenames (the staged `DATA/ASO` bundles and
  this repository's own fixtures) give the same key as before; the 19 names
  whose key changed are all boundary cases where the old answer was wrong, and
  in every one the old key is a strict prefix of the new. Pinned by
  `tests/test_patient_stem_boundaries.py`.
- **On real data, end to end, four modes.** Semi- and fully-automated CBCT
  (`DATA/ASO/testfiles/CBCT_SemiAuto`, NIfTI and DICOM) and semi- and
  fully-automated IOS (`IOS_SemiAuto`, `IOS_FullyAuto`) were run against the
  code as it stood before those two changes and after, and every file produced
  -- the 247 MB oriented volume voxel by voxel, the `.tfm` element by element,
  the meshes point by point -- is byte-identical. Maximum difference: 0.
- **The real chain, on real weights and a real card.** ASO fully-automated CBCT
  was driven end to end through a real supervisor -- one that runs `ALI` in
  `tools/ALI/.venv` as a subprocess, through `sadt_testkit`'s driver, exactly as
  the server's runner will. Input `DATA/ALI/testfiles/MG_test_scan.nii.gz` with
  the 4.7 GB `ALI_CBCT_Models` bundle on an RTX 6000 Ada:

  ```
  landmark_source: ALI          supervisor calls: 1
  summary: {'patients': 1, 'oriented': 1, 'failed': 0}
  landmarks used: ['Ba', 'LOr', 'LPo', 'N', 'ROr', 'RPo', 'S']   dropped: {}
  outputs: MG_test_Or.nii.gz, MG_test_Or_transform.tfm, MG_test_lm_Or.mrk.json
  work dir removed: True
  ```

  All seven landmarks survived the round trip -- recentre, hand ALI the centred
  volumes, read its markups back, merge per patient, register -- none dropped as
  missing or as an outlier. Nothing was imported across the two tools; they have
  different dependency sets and different venvs.
- **ALI itself is bit-identical to its pre-port implementation** on the same
  scan and the same seven landmarks (see [ALI's README](../ALI/README.md)), so
  what this chain feeds on is the same data the in-process version fed on.
- **Not compared against the pre-port ASO.** The end-to-end registration has not
  been diffed against `slicer-remote-tool-server`'s own copy on real patient
  data. The registration is covered by geometry tests against known rotations,
  and the seam by the run above, but a numerical comparison of oriented volumes
  is still owed.

## Working on it

```bash
cd tools/ASO
uv sync                    # 1.2 GB, no CUDA
uv run pytest -m "not gpu" # 193 tests, no GPU and no model bundles needed
uv run pytest -m models    # the ALI chain, see tests/data/README.md
```

```bash
# The schema the server publishes
.venv/bin/python ../../scripts/describe.py .
```
