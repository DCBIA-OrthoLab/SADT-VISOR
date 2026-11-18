# SADT-VISOR

One isolated Python project per SlicerAutomatedDentalTools tool. Each has its
own interpreter, its own virtualenv and its own lockfile, and they never import
each other.

This repository contains **no server code**: no HTTP, no FastAPI, no
authentication, no data store. A tool here does not know it is being served.

## Why the tools are isolated

Because the shared environment they came from does not resolve. In the upstream
extension every module installs into 3D Slicer's Python interpreter and into
ad-hoc conda environments, which produces conflicts that are reproducible today:

- `SurgMovPred` runs `pip_install("numpy==2.4.0")`; `AREG`, `MedX` and `CLIC`
  pin `numpy<2.0.0`. Whichever module runs last breaks the others.
- Two different conda environments are both called `shapeaxi`: Python 3.9 in
  `ALI`/`ASO`/`AREG`/`FlexReg`, Python 3.12 in `DOCShapeAXI`. Creating the
  second silently overwrites the first.
- `monai` resolves to 1.3.2 or 0.7.0 depending on the interpreter's Python.
- `torch` is pinned to 2.2.0 in AMASSS/ASO/AREG/MRI2CBCT/CLIC, unpinned
  elsewhere, and `MRI2CBCT` forces the cu118 build.

None of that is fixable by picking better versions, because the versions are
part of the science: the same weights on a different torch can produce different
masks. So each tool keeps the pins its results were produced with, and disk cost
is handled by deduplication at build time rather than by convergence.

## Where things live

| Repository | What is in it | Open it to |
|---|---|---|
| [VISOR-serve](https://github.com/DCBIA-OrthoLab/VISOR-serve) | The generic HTTP server: routes, auth, dispatch, the runner, the `/DATA` store. Knows nothing about dental tools. | Change the API, authentication, job handling or deployment. What it has to do to serve these tools is written down in [docs/SERVER_CONTRACT.md](docs/SERVER_CONTRACT.md) |
| **SADT-VISOR** (this one) | One isolated project per tool. Knows nothing about the server. | Change what a tool computes, or the versions it computes it with |
| [AutomatedDentalToolsRemote](https://github.com/DCBIA-OrthoLab/AutomatedDentalToolsRemote) | The thin 3D Slicer client. Discovers everything through `GET /tools`. | Change the user interface |
| [SlicerAutomatedDentalTools](https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools) | Upstream. The origin of every algorithm here. | Check what a port was ported from |

## The interface

A tool package exposes exactly one public callable, `run`. There is no base
class to subclass, no registry to edit and nothing to import from the server:

```python
from pathlib import Path

def run(
    scan: Path,
    model: Path,
    output_dir: Path,
    structures: list[str] = ["Mandible", "Maxilla"],
    crop: bool = False,
) -> Path:
    """Segment craniofacial structures on a CBCT scan."""
    import torch          # heavy imports go INSIDE run(), never at module level
    ...
    return output_path
```

The server runs it out of process, one interpreter per tool:

```
/tools/<name>/.venv/bin/python /opt/sadt/runner.py --job /jobs/<uuid>/job.json
```

`runner.py` ships with the server and is injected by absolute path -- never
installed into a tool venv -- so runner and server are always the same version
and there is no cross-repo skew to manage. That is also why there is no shared
`sadt-core` package: adding one would put a version of *ours* inside every tool
venv, and it would solve a problem that does not exist.

A tool that needs another tool **mid-run** declares `*, sup` and is handed a
supervisor; the call re-enters the same runner with the sibling's interpreter.
`ASO`, `AREG` and `VFACE` are the ones that do, and `VFACE` is the widest: six
tools, two of them supervised in their own right, so a full run is four deep
against the runner's cap of five. Everything the server has to hold up on its
side of that -- and everything else it took over when the tools stopped doing it
-- is in [docs/SERVER_CONTRACT.md](docs/SERVER_CONTRACT.md).

The full set of rules -- annotations, defaults, batch inputs, where output may be
written -- is in [CONTRIBUTING.md](CONTRIBUTING.md). `tools/_template/` is a
working example of all of them.

## Layout

```
tools/<name>/
├── pyproject.toml        # name, requires-python, dependencies, uv index
├── uv.lock               # committed
├── README.md             # provenance, inputs, outputs, model files, validation
├── src/sadt_<name>/
│   ├── __init__.py       # defines run()
│   └── ...               # the ported implementation
├── tests/
│   ├── test_run.py       # calls run() end to end
│   └── test_integration.py   # runs it out of process, and any tool it chains with
└── ...
```

`testkit/` holds `sadt-testkit`, a **development-only** package that lets one
tool's tests run another through that tool's own venv, as a subprocess. It is
how a tool whose input is another tool's output gets tested against the real
thing without importing it. See [testkit/README.md](testkit/README.md).

## Scripts

`scripts/describe.py` emits the JSON schema the server publishes for a tool,
read from `run()`'s signature -- so the schema cannot drift from the code. It
runs with the tool's own interpreter, because importing a tool needs the tool's
dependencies:

```console
$ tools/_template/.venv/bin/python scripts/describe.py tools/_template
{
  "name": "_template",
  "description": "Summarise the intensity distribution of a scan or a folder of scans.",
  "arguments": {
    "scans":      {"type": "path", "required": true},
    "output_dir": {"type": "path", "required": true},
    "metrics":    {"type": "list[str]", "required": false, "default": ["mean", "max"],
                   "choices": ["mean", "max", "min", "std"]},
    ...
  },
  "returns": "path",
  "source_hash": "96ab3611..."
}
```

An argument annotated `Literal[...]` publishes its options as `choices`, so the
client can render a picker without a second declaration to keep in step --
`list[Literal[...]]` for several-of, a bare `Literal[...]` for exactly-one.

It exits 2 on anything it cannot represent rather than emitting a schema that is
almost right; `source_hash` is what lets the server notice a cached schema has
gone stale.

`scripts/audit.py` reports the distinct torch and Python versions across the
tools and what each costs on disk:

```console
$ uv run scripts/audit.py
torch  2.2.0 (5 tools) | 2.4.1 (2) | unpinned (3)
python >=3.11 (4) | >=3.9,<3.10 (6)
```

It is a diagnostic and only ever reads. Aligning a pin is a human decision that
requires revalidating the model's outputs first.

`scripts/run_tool.py` runs a tool from the command line, in its own venv, and
chains to other tools when one asks for it. The parser is built from `run()`'s
signature, so there is no per-tool CLI to keep in step with anything:

```console
$ python scripts/run_tool.py ALI_CBCT --input scan.nii.gz --model bundle/ \
      --output-dir out/ --landmarks Ba S N
out/

$ python scripts/run_tool.py ASO --input cohort/ --reference gold/ \
      --output-dir out/ --automation Fully-Automated --landmark-model bundle/
[sup] 20% predicting landmarks with ALI_CBCT
[sup] running ALI_CBCT
out/
```

The second command is the whole point: ASO needs landmarks mid-run, so it is
given a **supervisor**, and `sup.run("ALI_CBCT", ...)` re-enters this same
script with that tool's interpreter. Chaining and nesting are the same
recursion -- `VFACE → AREG_CBCT → ASO → ALI_CBCT` is four levels of it with
no special case, and it is the deepest chain here.

**Developer convenience, not the deployment path.** In production the server's
`execution/runner.py` does this, and a tool cannot tell the two apart: five
members, duck-typed, nothing shared. It is still the shortest readable
reference for what a supervisor has to be -- and the place to reproduce a
chaining bug without standing a server up.

## Getting started

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you do not have uv

cd tools/_template
uv sync          # build .venv from uv.lock
uv run pytest    # runs run() end to end
```

## Provenance

[PROVENANCE.md](PROVENANCE.md) records, for every tool, the upstream path and
commit it was ported from and whether the algorithm was modified. That table --
not this repository's commit history -- is what tells you six months from now
whether a result came from upstream code or from something we changed.

## Licence and provenance

This repository is released under the MIT licence (`LICENSE`).

The clinical tools packaged here derive from
[SlicerAutomatedDentalTools](https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools),
which is distributed under the 3D Slicer licence, a BSD-style licence. That
licence continues to govern the parts of this code that come from it, and its
terms are met by keeping the upstream notice with any redistribution.
