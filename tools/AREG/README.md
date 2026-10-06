# sadt-areg

Registers a follow-up scan onto its baseline, so two timepoints of the same
patient share one coordinate system and can be measured against each other.

- **CBCT** -- elastix, rigid, restricted to the anatomy that has *not* changed:
  the cranial base, the mandible or the maxilla, taken as masks.
- **IOS** -- a patch of the arch that does not move with growth or treatment
  (the palate, or the band around the mucogingival line), matched by ICP.

## Provenance

Ported from DCBIA-OrthoLab/SlicerAutomatedDentalTools by way of
`slicer-remote-tool-server`'s `tools/AREG/`, which lived on an unmerged `AREG`
branch and is now parked there as `server/tools/_AREG/`.

> **Unlike the other six, this history does not follow.** AREG was never part
> of the `git subtree split` that carried the tools into this repository -- it
> was on a branch at the time -- so `git log --follow` stops at the commit that
> copied these files in. The server-side source is preserved at the
> `archive/AREG` tag in that repository. **The upstream commit is unrecorded**,
> as it is for [ALI](../ALI/README.md#provenance) and [ASO](../ASO/README.md).

Upstream pins **not** kept: pinned to the deployed stack, which is what every
sibling tool locks -- torch 2.8.0+cu128, monai 1.6.0, itk 5.4.7, Python 3.11.

Changes from upstream -- the algorithm is untouched, the envelope is not. The
first group were made during the server-side port and are unchanged here.

- **The elastix centre of rotation is honoured.** `MatrixRetrieval` read
  elastix's three angles and its translation and *dropped* its
  `CenterOfRotationPoint`, so the transform it built rotated about the physical
  origin instead. The two differ by `(I − R)c` -- invisible on centred data, and
  a gross misregistration on anything else.
- **The masked image never reaches the disk.** It was written to
  `<temp>/fixed_image_masked.nii.gz`: one fixed name shared by every patient of
  a run and by every concurrent request.
- **A subject with no mask is reported, and the batch goes on.**
- **T1/T2 pairing is by patient**, with the timepoint and mask tokens stripped
  once, in `pairing.py`, rather than guessed at each call site.

And this migration's:

- **Zip extraction removed.** The server unpacks archives before `run()` is
  called.
- **`src/tools_client.py` is gone.** It reached into the server's
  `registry.TOOLS` to call four tools in-process. Replaced by
  [`tools.py`](src/sadt_areg/tools.py), one `sup.run(...)` per tool.
- **The GPU semaphore is gone.** It serialised inference when every tool shared
  one process; a tool is its own process now and the limit is the server's.
- **`AREG_LANDMARK_TOOL`, `AREG_MAX_GPU_JOBS` and the rest are no longer
  settings.** `run()` must not read the environment.
- **Intermediates live in `<output_dir>/.areg_work/`** and are removed before
  returning, whether or not the run succeeded.

## The four tools it drives

AREG registers. It does not segment, orient, label crowns or find a
mucogingival line -- each of those is another tool here, reached through the
**supervisor**:

| Asked for | Tool | When |
|---|---|---|
| T1 masks around the regions to register on | `AMASSS` | CBCT, both automated modes |
| an orientation both timepoints share | `ASO` | CBCT "Oriented + Fully-Automated", IOS Fully-Automated |
| tooth labels | `Crown_Seg` | IOS Fully-Automated |
| the 13 mucogingival landmarks per lower arch | `ALI` | IOS, mucogingival patch |

**This is the deepest chain in the family.** `ASO` is itself supervised for
CBCT, so a fully-automated CBCT run is `AREG → ASO → ALI` -- three tools, three
virtualenvs, three interpreters. The runner's supervisor handles that by
recursion and caps it at four deep; nothing here arranges it.

Every call is in one file, by string:

```python
predictions = sup.run("ALI", input=meshes, output_dir=..., ios_networks=["Mucogingival"])
```

`sup.run("ALI", ...)`, never `sup.ALI(...)` -- a typo in a string is greppable
and `tools.py` is the whole call graph; a typo in an attribute is an
`AttributeError` an hour into a job.

**Without a supervisor, an automated mode refuses at the door** and names the
mode that works instead: send your own masks and use Semi-Automated, or send
the landmarks in `mgl_landmarks`. That is a real answer, where "deploy a tool"
usually is not -- and it is what makes this usable standalone.

## What it does

| | |
|---|---|
| Inputs | `t1` and `t2`: folders of CBCT scans, or of intra-oral meshes, paired by patient name. `output_dir`: where results go. |
| Outputs | Per patient and per region: the registered T2 and the transform that produced it, plus `AREG_report.json`. |
| Model files | `registration_model` (the IOS patch network), resolved from `DATA/AREG/models/` when left empty. The masks and the orientation are other tools' work, and each owns its model: AMASSS segments with its own bundle, ASO orients into the frame AREG names with its own reference. `segmentation_model`, `reference` / `cbct_reference` / `ios_reference` remain as overrides a caller may name. All named so the server publishes them as hosted names rather than uploads. |
| GPU | The IOS patch network uses it. The CBCT engine is elastix on the CPU. |

Three behaviours worth knowing before reading a result:

- **Register on what has NOT changed.** The cranial base is the usual choice; a
  patient whose growth is elsewhere may want the mandible or maxilla. Choosing
  a region that moved between timepoints produces a confident, wrong answer.
- **Each region is a separate registration** with its own output folder.
- **`modality` is never inferred from the file extension.**

## Versions

torch 2.8.0+cu128, monai 1.6.0, itk 5.4.7, SimpleITK 2.5.6, vtk 9.6.2,
numpy 2.3.2, dicom2nifti 2.6.2, Python 3.11 -- the same stack every sibling
locks, so this adds no new runtime to the image.

One package no other tool here needs: **`itk-elastix`**. The CBCT engine reaches
elastix as `itk.ElastixRegistrationMethod`, which plain `itk` does not carry.

`pytorch3d` sits behind an `ios` extra, same tag as ALI and Crown_Seg so the
three share one build. The CBCT engine works without it.

## Validated against

- **The CBCT engine, end to end, for real.** `itk-elastix` is a wheel, so the
  registration itself is exercised rather than mocked: a 48³ synthetic phantom
  is moved by a known rigid transform and the recovered transform is checked
  against it. That covers the centre-of-rotation fix, the masked-image fix and
  the per-patient reporting.
- **Tests**: 88 passing. Pairing, mask discovery, elastix, the CBCT mode end to
  end, every argument rule, the checkpoint lookup, and the four supervisor
  calls -- each asserted on the parameters the callee actually publishes.
- **The seam against the real schemas**: `test_the_arguments_it_sends_are_the_arguments_they_publish`
  reads all four tools' published schemas out of process and checks every
  argument AREG sends exists -- including that `Mucogingival` is one of ALI's
  offered networks. It skips unless the four are built.
- **Not run end to end against another tool.** No supervised chain has been
  executed with the real four; the calls are covered by a fake supervisor
  asserting the parameters, and the schemas by the test above. **The IOS engine
  is unvalidated** -- it needs pytorch3d and a segmented lower arch, neither of
  which is staged here.
- **No comparison against the pre-port implementation**, on any modality.

## Working on it

```bash
cd tools/AREG
uv sync                     # ~5.8 GB, CUDA wheels, no pytorch3d
uv run pytest               # 88 tests, elastix real, no GPU needed
uv sync --extra ios         # compiles pytorch3d, needs nvcc
```

```bash
# The schema the server publishes
.venv/bin/python ../../scripts/describe.py .
```
