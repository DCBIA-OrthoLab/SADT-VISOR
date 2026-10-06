# AREG: how the three tools are tested

Nothing is committed here. The suite builds its own 48³ synthetic phantoms with
SimpleITK, moves them by a known rigid transform, and checks the recovered
transform against it -- `itk-elastix` is a wheel and fast enough on a phantom
that the CBCT registration is exercised for real rather than mocked.

Each of the three tools now runs its own share of them in its own
virtualenv (AREG_CBCT 25, AREG_IOS 53, AREG_IOSCBCT 1). The IOS patch network is stubbed; it needs pytorch3d
and a checkpoint.

## What the supervised chain would need

The four tools AREG drives are stood in for by a fake supervisor, which asserts
the parameters each call sends. One test goes further and reads their **real**
published schemas out of process, skipping unless all four are built:

```bash
for t in AMASSS ASO Crown_Seg ALI; do (cd ../$t && uv sync); done
uv run pytest -k arguments_it_sends
```

Running an actual chain needs more than this repository holds: the AMASSS
bundle, an orientation reference, and -- for the IOS half -- the `ios` extra plus
a segmented lower arch. See `../../ALI/tests/data/README.md` for the container
recipe that avoids compiling pytorch3d.

## Staging the model bundles

Weights are never committed. The server fetches them into `/DATA/AREG/`; the
server's `scripts/data-manifest.yml` holds the URLs and sizes.

| Argument | What it names |
|---|---|
| `registration_model` | the network that finds the IOS patch |
| `segmentation_model` | an override of the AMASSS bundle; left empty, AMASSS uses its own |
| `cbct_reference` / `ios_reference` | an override of the orientation reference; left empty, ASO uses its own |

Only `registration_model` is AREG's own bundle. AMASSS and ASO resolve theirs
from `/DATA/AMASSS/` and `/DATA/ASO/`, so AREG keeps no copy of either.

All four are named `*_model` or `*_reference` on purpose: whatever serves the
tool publishes those as names picked from the data it already holds, never as a
file a clinician uploads. See CONTRIBUTING.md.
