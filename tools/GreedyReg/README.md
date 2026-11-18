# GreedyReg

Register each patient's second CBCT onto their first, rigidly or affinely.

For every patient present at both timepoints, T2 is registered onto T1 with
[Greedy](https://greedy.readthedocs.io) and resampled into its frame; the
transform is written beside the resampled volume.

## Provenance

Ported from DCBIA-OrthoLab/SlicerAutomatedDentalTools, `GreedyReg/`. Of its
2 470 lines, **2 266 are the Qt panel and its logic**; the 204 that compute are
`GreedyReg_CLI/GreedyReg_CLI.py`, which is what this package holds.

## Changes from upstream

### Greedy is a Python package, not a binary to find

Upstream took the greedy executable's **path as an argument** (`greedyBinary`)
and made the Slicer module locate it on the user's machine. A client cannot
supply a path on the server, and a server should not depend on an executable
nobody installed.

`picsl_greedy` is greedy's own PyPI distribution, published by its authors. Its
`Greedy3D.execute(command: str)` takes the same argument string the command line
does, so the two invocations here are upstream's verbatim -- the affine search
with its `-n 100x100x50x25` schedule and `-search 100 10 20`, then the resample.
Nothing had to be reimplemented and nothing has to be installed into the image.

### The patient rule is the shared one

Upstream read a patient from a file name with `^([A-Za-z]+\d+)` -- letters, then
digits, with nothing between. Measured against the data this repository itself
ships:

| file | upstream | here |
|---|---|---|
| `C_0001_T1_Or.nii.gz` (AREG's test scans) | **skipped** | `C_0001` |
| `MAMP_0002_T1.nii.gz` (ASO's reference) | **skipped** | `MAMP_0002` |
| `IC_0005.nii.gz` (ASO's CBCT test file) | **skipped** | `IC_0005` |
| `A1_scan` and `A1_other` | both `A1`, one lost | kept apart |

A file that matched nothing was not reported: it simply never entered the dict,
so a run could register 3 of 40 patients and say nothing about the other 37.

This tool uses `sadt_areg_common.pairing` instead. `CONTRIBUTING.md` lists
identity derivation -- how a patient key is read from a filename -- as one of
the three things that belong in a shared package rather than being copied,
because two tools exchanging files by name have to agree on where the identifier
ends. It also carries what it could NOT pair, which is the half a caller can act
on.

### One patient failing costs one patient

Upstream's per-patient `except` called `sys.exit(1)`. Patient 3 failing lost
patients 4 to 40, and the batch reported nothing about any of them -- including
the two that had already succeeded. Failures are recorded per patient now, and
a run where NOTHING registered raises rather than returning an empty success.

### Each registration is bounded, at 1800 s

Upstream ran greedy with `subprocess.run(..., timeout=600)`. The bound is kept
-- greedy still runs as a child process, so one pair that will not converge
frees its slot instead of holding it for ever -- but at 1800 s
(`CASE_TIMEOUT_SECONDS`). Measured on a clinical CBCT pair with the default
NCC metric and a rigid transform: 534 s on an otherwise idle server, and over
600 s with three other runs alongside, so the default request failed under
ordinary load. A timed-out pair is recorded like any failed one and the batch
goes on.

### Not ported

The `<filter-progress>` / `<filter-comment>` stdout protocol, which is how the
panel drove its progress bar; and the nibabel import guard that told the user to
open the module and let it install the package.

## Kept deliberately

`write_identity_init` nudges the x translation by one micron, and the comment
saying why is upstream's: Greedy reads an exact identity as "no initialisation
given" and substitutes its own guess, so the nudge is what makes "start from
where the images already are" expressible.

The multi-resolution schedule and the random search are not exposed. They
describe how this registration was tuned, not a per-request choice.

## Working on it

```bash
cd tools/GreedyReg
uv sync
uv run pytest -m "not gpu"
.venv/bin/python ../../scripts/describe.py .
```

## Data

**None is declared for this tool.** The manifest has no `GreedyReg` entry, so the
run recorded in the pull request used `DATA/AREG/testfiles/CBCT_SemiAuto/{T1,T2}`
-- chosen because those file names are exactly the ones upstream's patient rule
dropped.
