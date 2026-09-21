# What the server has to do

Written for whoever works on
[VISOR-serve](https://github.com/DCBIA-OrthoLab/VISOR-serve).
It is the other half of the split: this repository holds tools that know nothing
about the server, so everything they stopped doing, the server now does.

## Status: re-checked against the server, 2026-08-18

Checked against `VISOR-serve` on **`main`** at `10be711`, with
uncommitted work in the tree (a per-tool `timeout_seconds`, a process-group
kill, and `peak_vram_bytes` instrumentation). Read-only inspection.

**Everything this document previously reported as missing is implemented.** The
three gaps in order of how much they mattered:

- **The server injects a supervisor.** `server/execution/runner.py` builds one
  when `run()` declares `*, sup` -- keyword-only and unannotated, the same rule
  `describe.py` uses -- and `sup.run(tool, **params)` re-enters that same file
  with the sibling's interpreter. Five members, duck-typed, exactly as §5 asked
  for. `ASO`'s fully-automated CBCT mode and all of `AREG` work under it.
- **`supervisor` is a recognised top-level key**, not an ignored one
  (`registry/schema_tool._TOP_LEVEL_KEYS`), so a tool declaring it is no longer
  loaded as though it had not.
- **A tool needs no server-side configuration at all.**
  `server/registry/conventions.py` derives from argument NAMES what
  `deployment.toml` used to have to state, `server/deployment.toml` is an empty
  file of comments, and `DATA/` is resolved by the tool's name with underscores
  stripped -- so `Batch_Dental_Seg` reads `DATA/BatchDentalSeg/` with nothing
  written down. The lowercase-vs-capitalised mismatch this document warned
  about no longer exists on either side.

**The presentation keys travel.** `label`, `section`, `ui`, `groups`,
`visible_when` and `hidden` are published through `GET /tools` and read by the
Slicer client. There was a period where the server dropped them silently --
this repository published them, the client read them, and nothing arrived -- so
each key is now named explicitly in `schema_tool.py` rather than passed through
wholesale. **Adding a seventh key needs a one-line change on the server**, and
until it lands the key does not exist as far as any client is concerned.

There is one such key already: **`options_when`**, which the server accepts and
the client renders, and which `describe.py`'s `LAYOUT_KEYS` does not emit. It
narrows a choice argument's own options instead of hiding the whole field --
`{"modality": {"IOS": ["Semi-Automated", "Fully-Automated"]}}` -- and it exists
because `AREG`'s three automation modes are all meaningful while IOS has no
"Oriented + Fully-Automated". Without it the combo box offers a mode that fails
at the end of a run. **This is the one live divergence between the two
repositories.**

### What the server still does not do

- **It does not provision the tools.** `scripts/setup-server.sh` and
  `server_ctl.py` start a service that serves `TOOLS_DIR`, and nothing fetches
  a tool folder into it: today it is either the deployment image (tools baked
  in through a named build context) or a developer's checkout via
  `run-local.sh`. A `tools.lock` + volume bootstrap is the intended answer and
  is not written.
- **It does not walk a grouping folder.** `registry/` and `execution/dispatch.py`
  look exactly one level below `TOOLS_DIR`, so `tools/ALI/ALI_CBCT` is
  discovered as neither `ALI` nor `ALI_CBCT`. Only the supervisor's own lookup
  descends into a group. **The ALI split therefore needs either a flattening
  step when the tools are staged, or two more lookups on the server** --
  whichever is chosen has to be chosen deliberately, because the failure is a
  tool that simply does not appear in `GET /tools`.
- **VRAM is counted, not budgeted.** `MAX_CONCURRENT_GPU_JOBS` is a job
  counter, and a supervised call is a subprocess of its parent rather than a
  new admission, so a chain is invisible to it. The runner records
  `peak_vram_bytes` per run precisely so a real budget can be set from
  measurements later.

## Naming, because the server reads it

A tool that hosts nothing gets this for free, but two names are load-bearing:

| Argument named | Published as |
|---|---|
| `model`, `*_model`, `*_reference` | a name picked from `DATA/<tool>/models/` |
| any other `path` | a file the caller may upload |

`ASO` shipped `landmark_models` -- plural -- which misses `*_model` by one letter
and would have put a file picker in front of a 4.7 GB weight bundle. It is
`landmark_model` now, with a test guarding it. Check a new tool's path
arguments against this table before opening the PR; nothing else will.

The rest of this document is the contract itself, kept as the reference the two
repositories are written against.

## 1. Discovery

`scripts/describe.py` in this repository turns a tool's `run()` signature into
the JSON the server publishes. Run it with the **tool's own interpreter** --
importing a tool needs the tool's dependencies:

```bash
/tools/AMASSS/.venv/bin/python /tools/scripts/describe.py /tools/AMASSS
```

It is dependency-free and stays Python 3.9-compatible so it can run inside any
tool venv, including one pinned to an old interpreter. It exits **2** with a
message on stderr for anything it cannot represent, and prints nothing -- treat a
non-zero exit as "this tool is not loadable" and say so, rather than serving a
partial schema.

```json
{
  "name": "AMASSS",
  "description": "Segment craniofacial structures on a CBCT scan.",
  "arguments": {
    "scans":      {"type": "path", "required": true},
    "model":      {"type": "path", "required": true},
    "output_dir": {"type": "path", "required": true},
    "structures": {"type": "list[str]", "required": false,
                   "default": ["MAND", "MAX", "CB", "CV", "UAW"],
                   "choices": ["MAND", "MAX", "CB", "CV", "UAW", "SKIN",
                               "CBMASK", "MANDMASK", "MAXMASK"],
                   "section": "Structures", "ui": "inline"},
    "device":     {"type": "str", "required": false, "default": "cuda",
                   "choices": ["cuda", "cpu"], "hidden": true}
  },
  "returns": "path",
  "source_hash": "966bf6d9…"
}
```

**`name` is the folder name**, spelled as a client sends it -- `AMASSS`,
`ALI_CBCT`, `Batch_Dental_Seg`. It is not lowercased anywhere: the server looks
up the interpreter at `<TOOLS_DIR>/<name>/.venv/bin/python`, and a Slicer
module holds the same string in `TOOL_NAME`.

Three things to build against:

- **`required` comes only from the absence of a default.** There is no second
  declaration that can disagree with the signature.
- **`choices` is optional and means a fixed set.** `list[...]` with `choices` is
  several-of (the old `multichoice`); a scalar with `choices` is exactly-one (the
  old `choice`). Absent `choices`, the value is free-form.
- **`source_hash` is a sha256 of the tool's `src/`.** Cache schemas against it
  and regenerate when it moves; that is what it is for.
- **`supervisor` is optional and means the tool can call another one.** Absent
  on every tool that cannot, so nothing written before supervisors existed
  changes. See §5.
- **The layout keys ride on the arguments they describe.** `label`, `section`,
  `ui`, `groups`, `visible_when` and `hidden` are merged in from the tool's
  `layout.py`; the server publishes them untouched and `validate()` ignores
  every one. A key the server does not name is silently dropped, so the set is
  a joint decision rather than something either side can extend alone --
  `options_when` is currently accepted by the server and not emitted here.

**Argument order is the signature's order.** Render forms in it.

## 2. Invocation

```
/tools/<name>/.venv/bin/python /opt/sadt/runner.py --job /jobs/<uuid>/job.json
```

`uv sync` installs each tool into its own venv, so `import sadt_<name>` works
directly -- no `sys.path` juggling needed, though adding `/tools/<name>/src`
first is harmless and makes the runner work against an unsynced checkout too.

`server/execution/runner.py` implements this, and
[`testkit/src/sadt_testkit/_driver.py`](../testkit/src/sadt_testkit/_driver.py)
in this repository is the same contract, ~60 lines, stdlib-only. The two were
written independently and agree -- same coercion by annotation, same result
file, same error-class-name convention. **Keep them in step deliberately**: the
driver is what every tool's integration tests run against, so if they drift, the
tests here pass while production fails.

Two details from it worth carrying over:

- **Coerce by annotation.** JSON has no path type, so paths arrive as strings and
  must become `Path` for parameters annotated `Path` (or `list[Path]`).
  `typing.get_type_hints(run)` is how the driver decides.
- **An empty string is ABSENCE and must stay a string.** `Path("")` is
  `PosixPath(".")` -- the current directory, and truthy -- so coercing the
  "not supplied" default of an optional path hands the tool a real directory.
  This is not hypothetical: it made `ASO` read an unset `landmarks=""` as a
  supplied landmark folder and walk the entire checkout, `.venv` included.
  Calling `run()` directly in Python keeps the `""` the signature declares, so
  coercing it is also what makes the two call paths disagree. One line:

  ```python
  return Path(value) if value != "" else value
  ```

  `run_tool.py` and the testkit driver both do this; the server's runner must
  too. It is the price of `describe.py` refusing `None` defaults -- an optional
  path has no other way to say "unset".
- **Never parse the result off stdout.** These tools print progress bars, nnUNet
  banners and shapeaxi chatter. The driver writes the result to a file whose path
  it is given. Anything scraped from stdout breaks the first time a dependency
  prints something new.

`run()` returns a `Path` or a `dict[str, Path]`; `returns` in the schema says
which. Today: `surgmovpred` returns `dict[str, path]` (`excel` and `csv`), the
others return a `path` -- the output directory they were given.

## 3. Work that moved to the server

Each of these used to happen inside a tool and now happens nowhere unless the
server does it. **All of them are already implemented** -- the file references
are in the status table above. They are kept here because they are the reasoning
behind the code, and the first person to touch that code will want it.

### Unpacking archives

No tool handles `.zip` any more. The server unpacks before calling `run()` and
passes a real file or directory. Two behaviours have to come with it:

- **the zip-bomb cap.** Tools used to apply `MAX_EXTRACTED_MB` themselves when
  they received the archive directly (AMASSS, ALI, BatchDentalSeg, CrownSeg all
  did). Nothing applies it now.
- **`strip_single_root`.** Zipping `patients/` on any OS produces
  `patients/<files>`, and callers want the files. AMASSS and ALI relied on this.

### Capping GPU work

**Implemented**: `settings.MAX_CONCURRENT_GPU_JOBS` (default 1), one counter
**across** tools, held in `execution/dispatch.py`. Every tool used to hold a
`threading.BoundedSemaphore` -- `AMASSS_MAX_GPU_JOBS`, `BATCHDENTALSEG_MAX_GPU_JOBS`,
`CROWNSEG_MAX_GPU_JOBS`, `ALI_MAX_GPU_JOBS`, all defaulting to 1 -- because every
tool shared one server process. A tool is now its own process, so an in-process
semaphore would cap nothing and they have all been removed. An AMASSS run and a
`Crown_Seg` run compete for the same device, which is why the counter cannot be
per tool.

**A run is assumed to want the card** unless it declares `device` and resolves
it to a CPU value. That default is deliberately the strict one: a tool that
imports torch without declaring `device` would otherwise never queue at all.
The consequence for this repository is concrete -- **a GPU tool must declare
`device`**, or every run of it, CPU or not, takes a GPU slot; and a CPU-only
tool that declares one gets to say so and never queues.

Two things it does not cover, both stated rather than hidden: a **supervised**
call is a subprocess of its parent and never re-enters the queue (that is what
makes nesting deadlock-free, and it means a chain can put two tools on the card
at once), and the counter is jobs rather than memory.

### Creating and owning the output directory

`output_dir` is a required argument on every tool. Create it (or let the tool --
they all `mkdir(parents=True, exist_ok=True)`), pass it, and archive what comes
back. Tools write **only** there; each has a test asserting it.

Intermediates go in a dotted subdirectory that the tool removes before returning
(`.amasss_work/`, `.batchdentalseg_work/`, `.crownseg_work/`). If one survives, a
run crashed.

### Resolving model weights

Tools no longer touch `data_store` or `settings.CROWNSEG_MODEL`. The server
resolves the name and passes a path -- but **what kind of path differs per tool**:

| Tool | `model` is | Note |
|---|---|---|
| `surgmovpred` | a folder | every `stacking_package.pkl` under it is loaded, recursively |
| `amasss` | the bundle root | one subfolder per structure code (`MAND/`, `MAX/`, …); a single wrapper folder is descended into |
| `batchdentalseg` | **the bundle folder itself** | its *name* selects the model and its label table -- see below |
| `crownseg` | a `.pth` file | not a folder |

**`batchdentalseg` is the sharp edge**: the folder's basename must equal a key in
its `catalogs.MODELS` (`DentalSegmentator`, `PediatricDentalSeg`,
`NasoMaxillaDentSeg`, `UniversalLab`), which must in turn equal the folder
`scripts/data-manifest.yml` downloads that bundle into. The server-side test that
enforced that has no home any more -- this repository cannot read the manifest.

**This no longer needs a `deployment.toml`.** Tool directories here are the
tool's name as a client sends it (`AMASSS`, `Batch_Dental_Seg`, `Crown_Seg`,
`Surg_Mov_Pred`), and the server resolves `DATA/` by that name with underscores
stripped, so `Batch_Dental_Seg` finds `DATA/BatchDentalSeg/` on its own. A
literal match wins wherever one exists, so a folder that really does carry
underscores is never mis-resolved. Only a genuinely different folder name needs
a `data_dir` line.

### Configuring logging

Tools use a plain module logger and attach no handlers -- a library that
configures logging takes the decision away from whatever runs it. The runner owns
handlers, levels and formatting. Without that, tool logs go nowhere.

### Settings that became arguments

`run()` must not read the environment, so these moved into the signature with
their previous defaults. The server passes them only to override:

| Was | Now | Default |
|---|---|---|
| `settings.DEVICE` | `device` on every GPU tool | `"cuda"`, falls back to CPU with a warning |
| `settings.AMASSS_TILE_STEP_SIZE` | `tile_step_size` (amasss) | `0.5` |
| `settings.AMASSS_GPU_RESAMPLING` | `gpu_resampling` (amasss) | `true` |
| `settings.BATCHDENTALSEG_TILE_STEP_SIZE` | `tile_step_size` (batchdentalseg) | `0.5` |
| `settings.CROWNSEG_NUM_WORKERS` | `num_workers` (Crown_Seg) | `2` |
| `settings.*_MAX_GPU_JOBS` | -- | gone; see "Capping GPU work" |

`tile_step_size` and `gpu_resampling` **change the segmentation**. The tools
record what was used in their run report, so a mask stays reproducible.

## 4. Errors

There is no shared exception type, because there is no shared package. Each tool
defines its own `ToolInputError(ValueError)`, and `surgmovpred` deliberately kept
upstream's `FileNotFoundError` / `RuntimeError` instead. So the server cannot
`isinstance`-check the way `base.ToolArgumentError` allowed.

**Map by exception class name.** The suggested convention, which the runner
should write into its result file:

```json
{"error": {"type": "ToolInputError", "message": "Unknown structure code(s): RC. Known: MAND, MAX, …"}}
```

- `ToolInputError`, `ValueError`, `FileNotFoundError` → **422**, message passed
  through: these always mean the request or the staged data is wrong, and every
  message is written to be read by whoever sent it.
- `ToolUnavailableError` → **503**: the tool is installed but its engine is not.
  Only `crownseg` raises it today, when the `segmentation` extra is missing.
- anything else → **500**, message not passed through.

If you would rather not match on names, the alternative is a `type` field the
tools set explicitly -- but that is a shared convention either way, and names are
already the convention.

## 5. Sequencing tools

Almost all sequencing is the server's: where one tool's output is another's
input, the server chains them and neither tool knows the other exists.

| Chain | Why | Handoff |
|---|---|---|
| `Crown_Seg` → `ALI` | ALI's IOS engine needs meshes carrying tooth labels. `ALILogic.ensure_segmented()` used to import CrownSeg directly. | CrownSeg's `run_report.json` has `segmented_meshes`: absolute paths of every mesh that now carries labels, whether this run produced them or found them already labelled. Feed those to ALI. |

CrownSeg passes an already-labelled mesh through untouched, so re-running the
chain on mixed input is cheap and safe. ALI refuses an unlabelled batch up front,
naming `Crown_Seg`, rather than failing per mesh.

### The one chain that cannot be a chain

**`ASO` fully-automated CBCT needs `ALI` from the middle of its own run.** It
recentres each scan, predicts landmarks **on the centred volumes**, then
registers -- the order the Slicer chain used (`PRE_ASO_CBCT` before `ALI_CBCT`).
Running ALI first and handing ASO the markups reorders those two steps.

That reordering *ought* to be exact: recentring resamples onto a grid shifted by
the same offset, so the voxel array is untouched and only the origin metadata
moves. But ALI's `physical_position` takes `abs(origin / spacing)`, which does
not commute with moving the origin -- [issue #11]. Rather than bet a clinical
result on it, ASO calls ALI where it always ran.

So ASO takes a **supervisor**, and the server provides one
(`server/execution/runner.py`, `_Supervisor`):

```python
predictions = sup.run("ALI_CBCT", input=centered_root, model=bundle, output_dir=..., landmarks=[...])
```

Every requirement this section used to list is met, and each was met the way it
asked:

- **Five members, duck-typed** -- `run(tool, **params)`, `out`, `tmp`,
  `progress(fraction, message)`, `log(message)` -- passed as the keyword-only
  `sup`. Nothing is imported across the two repositories.
  [`scripts/run_tool.py`](../scripts/run_tool.py) produces the same shape, and
  a tool cannot tell the three implementations apart.
- **The callee is selected by VENV.** `sup.run` looks up
  `<TOOLS_DIR>/<name>/.venv/bin/python` and execs it, so the callee gets its own
  interpreter and its own dependency set. Nesting is the same recursion:
  `VFACE → AREG_CBCT → ASO → ALI_CBCT` needs no special case.
- **Absolute paths and a neutral working directory.** Each nested call gets its
  own job directory under `<job>/sup/NN_<tool>/`, and runs with that as `cwd`.
- **The scratch is `sup.tmp`**, a sibling of `output/` inside the job directory,
  removed with it -- so the tool stays held to writing only under `output_dir`.
- **`sup.run` returns what the tool returned**, a `Path` or a
  `dict[str, Path]`, reconstructed from the callee's `result.json`.
- **Concurrency is answered by not queueing at all.** A nested call is a
  subprocess of its parent, so it never re-enters the server's admission queue
  and cannot wait for a slot the parent is holding -- the deadlock this section
  described is structurally impossible rather than mitigated. The cost is that
  nested work is invisible to `MAX_CONCURRENT_GPU_JOBS`.

Two guards the server added on its own, worth knowing because they produce
errors a tool author will read:

- **A cycle is refused by name.** The chain of tools already running above a
  call travels in the environment (`SADT_SUPERVISOR_CHAIN`), so a tool asking
  for one that is already above it fails immediately, naming the chain, rather
  than after starting four processes. `MAX_SUPERVISOR_DEPTH` (5) is only the
  backstop for a chain that grows without repeating.

  **It has one level of headroom left, and it used to have three.** `VFACE`
  drives six tools and two of them are supervised in their own right, so
  `VFACE → AREG_CBCT → ASO → ALI_CBCT` is four deep. A tool added above VFACE,
  or a fifth level inside any of those, hits the cap -- and hits it as a refusal
  partway through a run rather than at admission. Worth raising deliberately if
  another orchestrating tool arrives, rather than discovering it on a cohort.
- **A failing child carries its reason up.** The parent reads the child's
  `result.json` and raises with the child's own `type` and `message`, because
  "see the output above" is a promise a nested process cannot keep.

`describe.py` publishes `"supervisor": true` for a tool that takes one, and the
server now reads that key. Passing `landmarks` (a folder of `.mrk.json`) still
makes fully-automated CBCT work with no supervisor at all, which is what lets
ASO be used standalone -- keep that door open in every tool that takes a `sup`.

[issue #11]: https://github.com/DCBIA-OrthoLab/SADT-VISOR/issues/11

## 6. What was deleted, and what was not

This happened, tool by tool, each after its PR merged *here* -- never before,
because the server kept working off its own copy until this one was proven.

**Gone from the server**: every clinical tool. `server/tools/` holds
`Test_Tool` and `Example_Tool` (in-process demos of the old path, kept
deliberately), a `_dispatch_probe` fixture, and a parked `_AREG` kept only for
its history. The server's own suite went from 461 tests to 220 in the same
movement -- a packaged tool's tests belong to that tool and run in ITS
interpreter.

**Still there, and for a reason**: `base.py`'s `Tool`, `ArgSpec`, `Selection`,
`ResolvedPath`, `FILE_TYPES` and `ToolArgumentError`. A `.schema.json` is
turned into exactly one of those `Tool` objects
(`registry/schema_tool.SchemaTool`), so `GET /tools`, `validate()`, the upload
handling and the data-store resolution all work on a packaged tool unchanged.
That layer is not legacy; it is the shape everything becomes. What is legacy is
the *import* half of discovery, and the `SADT_DISPATCH_MODE` flag, both of which
now only concern the two demos.

`requirements-api.txt` is what the API actually needs -- fastapi, uvicorn,
python-multipart, pydantic-settings -- and a test asserts it stays that way.
`requirements.txt` is still the heavy one for a dev checkout, which is the last
piece of this list outstanding.

## 7. Two things to know before you start

- **CrownSeg does not currently run, and did not before the migration.**
  `shapeaxi.dental_model_seg` reads `DentalModelSeg` off `shapeaxi.saxi_nets`,
  but in the whole 2.0.x line the class lives in `shapeaxi.saxi_nets_lightning`.
  The pre-port tool fails the same way on the deployed image. This repository
  carries a two-line workaround (`_restore_moved_class`) guarded so it vanishes
  when upstream fixes it. Anything downstream of crown segmentation -- ALI's IOS
  half, the IOS modes of ASO/AREG/FlexReg -- has been broken on the current image.
- **Disk.** Each torch venv is 7.2-7.7 GB at cu128. Deduplication across them is
  not automatic and depends on `UV_CACHE_DIR` sitting on the same filesystem as
  the venvs; see the repository README.
