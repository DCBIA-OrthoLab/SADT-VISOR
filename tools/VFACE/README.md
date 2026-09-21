# sadt-vface

Classifies a patient's facial asymmetry from a CBCT, or measures the change
between two timepoints.

**A patient is compared against themselves.** An asymmetry assessment has no
second scan: the baseline is oriented, mirrored across the mid-sagittal plane,
registered back onto itself region by region, and what is measured is how far
each landmark has moved from its own reflection. A longitudinal study is the
same chain with a real follow-up in the mirror's place, which is why one tool
does both and why the mirroring is the single step where they part.

## Provenance

Ported from `VFACE/`, `VFACE_utils/` and `VFACE_CLI/` in
DCBIA-OrthoLab/SlicerAutomatedDentalTools at `84ef432` (2026-09-18).

Upstream pins **not** kept: pinned to the deployed stack, which is what every
sibling locks -- SimpleITK 2.5.6, vtk 9.6.2, numpy 2.3.2, Python 3.11 -- rather
than upstream's torch 2.2 / monai 1.3.2 / nnUNet 2.8.0, none of which this tool
touches. It runs no network of its own.

Changes from upstream. The algorithm is reproduced -- and checked against
upstream's own code on random input, see below -- and the envelope is not.

- **The Slicer orchestration is gone.** Upstream builds a list of ~40 CLI
  invocations and steps through them with Qt callbacks. Here the chain is
  `dispatch.py` and each other tool is one `sup.run(...)`.
- **The pause-and-review system is not ported.** Upstream stops the run at a
  dozen points to have a clinician check and drag landmarks
  (`VFACE_utils/review_steps.py`, `AREG_Method/Review.py`). A tool here runs to
  completion out of process; there is nobody to ask.
- **The resample is vendored.** See below.
- **`scikit-learn` is a declared dependency**, and upstream does not install it.
  The three classifier models are LightGBM's SKLEARN-API wrappers -- they answer
  to `feature_name_`, which a plain `lgb.Booster` does not have -- so unpickling
  one imports `lightgbm.sklearn`, which refuses to load without it. Upstream's
  Slicer happens to have it from another extension; a tool here has only what it
  declares, and a run would have died at the classification step.
- **The folder tree survives the resample.** Upstream derives each output path
  with `file_path.replace(os.path.dirname(file_path), output_folder)`, which
  flattens it: two sites holding a scan of the same name write to one file and
  the cohort silently loses a patient.
- **A patient is read off a name by whole tokens**, never by substring.
  Upstream cuts with `basename.split("_CB")[0]`, which also cuts `P1_CBrown`.
- **A feature column naming a landmark the run never produced is left empty**
  rather than ending the post-processing on a `KeyError` -- which is how one
  missing landmark turned into a failed classification and an Excel that was
  never written.
- **`AREG_LANDMARK_TOOL` and the rest are not settings.** `run()` does not read
  the environment.

### The one thing borrowed from a tool that is not here

Upstream's full pipeline begins by calling `MRI2CBCT`'s resample CLI, and
MRI2CBCT is not a tool in this repository. What VFACE asks it for is not
MRI2CBCT, though: every argument it sends turns off everything that makes it
what it is.

| VFACE sends | so this never runs |
|---|---|
| `input_folder_MRI = "None"` (and four more) | the MRI branches; the CLI tests `os.path.isdir("None")` |
| `resample_size = "None"` | the grid size derived from a target size |
| `mri = 0` | the direction-aware centring |
| `rightSide = 0` | the left/right mirror |
| `iso_spacing = False` | the max-spacing branch |
| `linear = True` | the nearest-neighbour path segmentations take |

What is left is fifteen lines of SimpleITK: keep the volume's grid, set 0.3 mm
isotropic spacing, recentre the origin, resample linearly with the volume's own
minimum as the padding value. Nothing of the MRI-to-CBCT approximation, the
registration, the TMJ and LR crops, the percentile normalisation or the condyle
segmentation is reachable from there. So it is a **copy**, which is what
CONTRIBUTING.md says two tools needing the same code usually get.

**The divergence that copy costs is real and worth naming.** Resampling
interpolates, so it changes voxel values, and a result therefore depends on this
copy rather than on MRI2CBCT. If MRI2CBCT is ever ported here, the two can drift
apart without anything failing. PROVENANCE.md records that too.

### One upstream oddity reproduced rather than repaired

`Measure.__SignMeaningDist` resolves a midpoint's side from the two landmarks it
joins -- and then overwrites its own answer. The block replaces `direction` with
a letter, which makes the guard below it (`if direction1 != "Mid" and
direction2 != "Mid"`) always true, and that guard re-reads `name[0]`. So
`Mid_ROr_LOr` resolves to `M`, never to `R` or `L`, and the midpoint logic never
reaches a result.

That label decides the SIGN of a feature the classifier was trained on, so
quietly making it read `R` where every model saw `M` would change
classifications on models already trained. It is a clinical decision, not a
port's, and it is written down in `measure._side_of` and covered by a test
rather than fixed.

## The six tools it drives

VFACE measures and classifies. It does not orient, segment, mirror, register or
place a landmark -- each of those is another tool here, reached through the
**supervisor**:

| Asked for | Tool | When |
|---|---|---|
| an orientation, per frame | `ASO` | the full pipeline |
| masks around the regions measured | `AMASSS` | anything that registers |
| the patient's own scan, mirrored | `AutoMatrix` | an asymmetry assessment |
| the mirror (or follow-up) registered onto the baseline | `AREG_CBCT` | anything that registers |
| the landmarks every measurement is computed from | `ALI_CBCT` | measurements |
| surfaces to draw a heat map on | `Batch_Dental_Seg` | heat maps |

**This is the widest call graph in the repository**, and two of the six are
themselves supervised: `ASO` reaches `ALI_CBCT`, `AREG_CBCT` reaches `AMASSS`
and `ASO`. A full run is four tools deep against the runner's cap of five.

Every call is in one file, by string, in `src/sadt_vface/tools.py` --
`sup.run("ASO", ...)`, never `sup.ASO(...)`.

**Without a supervisor, a mode that needs one refuses at the door** and names
the mode that works instead: send already-oriented scans and use `File already
Oriented`, send your own landmarks, ask for measurements alone. That is a real
answer where "deploy a tool" is not.

## What it does

| | |
|---|---|
| Inputs | `t1`: a folder of CBCT volumes. `t2` for a longitudinal study only. `measurements`: one measurement list per region. `feature_template`: the features the classifier was trained on. |
| Outputs | `Measurements/Measurements_{CB,MAND,MAX}.xlsx`, `Measurements/PostProcess_Measurements.xlsx`, `Classification/Classification.xlsx`, `Heat maps/<region>/`, plus `VFACE_report.json`. |
| Model files | `segmentation_model` (AMASSS), `landmark_model` (ALI_CBCT), `surface_model` (Batch_Dental_Seg), `classifier_model` (the three asymmetry models), `cranial_base_reference` / `maxilla_reference` (orientation), `mirror_reference` (the reflection). All named so the server publishes them as hosted names rather than uploads. |
| GPU | None of its own. Every network it needs belongs to another tool. |

Four things worth knowing before reading a result:

- **A scan is oriented TWICE**, into two frames, and each region is worked in
  the one it is defined in: the mandible against the cranial base, the maxilla
  against the occlusal plane. A measurement of the mandible read in the
  maxillary frame is a different question.
- **The landmarks are found once**, in the cranial base frame, and carried into
  the maxillary one by a rigid transform. Searching twice costs minutes per
  patient and makes the same anatomical point land in two slightly different
  places.
- **The measurement lists decide which landmarks are searched for.** ALI spawns
  one agent per landmark at about a minute each, so a list naming four points
  is four minutes and a region's whole catalogue is hours.
- **The classification needs a template and a bundle from the same training
  run.** A template alone gives the feature table, which is what somebody
  training a model would ask for; the verdict on top of it needs the models to
  find the columns they name.

## Versions

SimpleITK 2.5.6, vtk 9.6.2, numpy 2.3.2, scipy 1.16.2, pandas 2.3.3,
openpyxl 3.1.5, lightgbm 4.6.0, joblib 1.5.2, scikit-learn 1.7.2, Python 3.11.

Two packages no other tool here needs: **`lightgbm`** and **`scikit-learn`**,
which are the asymmetry classifier itself. There is no torch, no nnUNet and no
pytorch3d: VFACE runs no network of its own.

## Validated against

- **Upstream's own arithmetic, on random input.** `tests/test_matches_upstream.py`
  transcribes `Measure.__computeDistance`, `__computeLinePoint`,
  `__computeAngle`, `__SignMeaningDist`, the eight dental sign blocks and
  `reorganizeStat` line for line, and drives them against this port. The two
  agree on every case: 6 200 distances, 3 000 angles, 65 536 dental distance
  labels, 12 800 dental angle labels, 3 916 skeletal ones and 300 random stat
  tables. That is the repackaging claim, checked rather than asserted.
- **The whole chain, end to end**, with the six callees standing in for
  themselves -- each writing what the real one writes, named as it names it --
  and everything VFACE does running for real on top: the resample, the padding,
  the derivation between frames, the measurements, the signed features and a
  prediction from real LightGBM models.
- **The seam against the real schemas.** `tests/test_schema_seam.py` reads all
  six tools' published schemas out of process and checks every argument sent
  exists, every landmark asked for is one ALI catalogs, every region one AREG
  has and every structure one AMASSS offers. It skips per tool when that tool
  is not built.
- **Tests**: 170 passing, 1 skipped.
- **NOT run against another tool for real.** No supervised chain has been
  executed with the six; the calls are covered by a fake supervisor asserting
  the parameters, and the schemas by the test above.
- **NOT compared against upstream's output on a patient.** The arithmetic is
  checked function by function; nothing here has run upstream's pipeline and
  this one on the same CBCT and diffed the workbooks. That is the validation
  this tool still needs, and it needs a cohort, the model bundles and a Slicer.
- **NOT validated clinically.** No annotated case is staged here, so nothing
  measures whether a verdict is right -- only that it is the verdict upstream's
  code would have produced from the same numbers.

## Working on it

```bash
cd tools/VFACE
uv sync                     # no CUDA wheels; a minute
uv run pytest               # 170 tests, no GPU, no weights
```

```bash
# The schema the server publishes
.venv/bin/python ../../scripts/describe.py .

# And the CLI the signature implies
python ../../scripts/run_tool.py VFACE --help
```
