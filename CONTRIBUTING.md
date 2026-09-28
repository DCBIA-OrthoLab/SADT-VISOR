# Adding a tool

Every version choice in this repository is a scientific decision before it is a
packaging one. A tool that imports cleanly is not a working tool, and a pin that
resolves is not necessarily the pin the results were produced with. Most of what
follows exists to keep those two things apart.

## 1. Branch

One branch and one pull request per tool, off `main`, never on `main`:

- `tool/<name>` -- migrating or changing a tool
- `infra/<topic>` -- repository-level work
- `fix/<topic>` -- corrections

Commit messages are a single short sentence prefixed with `ADD :`, `FIX :`,
`CLEAN :` or `UPDATE :`, in English, with no body unless the change genuinely
needs one:

```
ADD : AMASSS tool package with uv lockfile
FIX : describe.py silently accepted unsupported annotations
```

No AI attribution of any kind -- no `Co-Authored-By`, no generated-with trailer,
no emoji. Never merge your own PR, and never force-push a branch that has one
open.

Several tools at once is expected. Use `git worktree` rather than stashing:

```bash
git worktree add ../sadt-amasss tool/amasss
```

## 2. Copy the template

```bash
cp -r tools/_template tools/AMASSS
cd tools/AMASSS
mv src/sadt_template src/sadt_amasss
```

Then edit `pyproject.toml` (`name`, `requires-python`, `dependencies`), and
delete the placeholder pipeline.

**Declare the tool name in `[tool.sadt]`.** It is the API identity: what a
client sends, what `deployment.toml` is keyed by, and what another tool passes
to `sup.run()`.

```toml
[tool.sadt]
tool = true
name = "Crown_Seg"
```

The section is also what makes the directory a tool at all -- a `pyproject.toml`
without it is a plain package, importable and installable but never discovered
or served, which is how `tools/ALI/common/` and `testkit/` sit beside tools
without becoming ones.

Resolution order, most explicit first:

1. `[tool.sadt] name`, when declared;
2. otherwise the directory name, with acronyms preserved as today -- `ALI`,
   `ASO`, `AMASSS`.

A new tool should declare it rather than lean on the fallback, so the API name
is a **decision** and not an accident of directory casing.

**The directory must still be NAMED after the tool**, even when the name is
declared: the interpreter is looked up at
`<TOOLS_DIR>/[<group>/]<tool name>/.venv/bin/python`, so a folder called
anything else registers a tool that cannot be run. What declaring the name buys
is that the tool's DEPTH may change -- `ALI_CBCT` moved under the `ALI/` grouping
folder and kept its name -- and that renaming a folder without meaning to change
the API name fails at startup instead of silently renaming the tool clients ask
for.

The naming convention:

- an acronym stays as it is -- `ALI`, `ASO`, `AMASSS`;
- anything else is capitalised words joined by underscores --
  `Batch_Dental_Seg`, `Crown_Seg`, `Surg_Mov_Pred`, `Example_Tool`;
- a client renders the name with the underscores as spaces where it can.

The **Python package** under `src/` stays lowercase -- `sadt_batch_dental_seg`,
not `sadt_Batch_Dental_Seg` -- because it is a Python identifier and PEP 8
applies. `describe.py` finds it as "the single package under src/", so the two
spellings never have to agree.

Renaming a tool is a **breaking change for every client**: the name is what a
client sends, and the Slicer modules hold it in `TOOL_NAME`. Change it in the
same breath as the client, or the tool disappears from the UI.

## 3. Write `run()`

One public callable. Nothing else is part of the contract.

**Annotations are stdlib only**: `Path`, `str`, `int`, `float`, `bool`,
`Literal[...]` and `list[...]` of those. `Path` means "a file or a directory";
everything else is a scalar. No `Optional`, no unions, no custom marker types,
nothing imported from the server. `scripts/describe.py` refuses anything else
rather than publish a schema the client will render wrongly.

**A fixed set of options is a `Literal`.** `list[Literal["MAND", "MAX", ...]]`
is several-of, a bare `Literal["MERGED", "SEPARATE"]` is exactly-one, and
describe.py publishes both as `choices` so the client can render a picker.
Saying it in the annotation is the point -- the `ArgSpec.choices` tables this
replaces were a second declaration, and they drifted. The default is checked
against the options, so a picker can always produce the value the tool starts
from.

`Literal` is published, not enforced: the runner calls `run(**params)` from a
JSON object, so a stale client or a direct caller can still send anything.
Validate the value in the tool and raise on a bad one.

**No default means required.** That is the only thing `required` in the schema
comes from, so there is no second declaration to contradict the signature.
`n: int = None` is refused: it makes a required argument look optional.

**Name a hosted path `*_model` or `*_reference`.** Whatever serves the tool
decides from the argument's NAME whether a `Path` is something it already holds
or something the caller uploads: `model`, `*_model` and `*_reference` are picked
from `DATA/<tool>/models/`, everything else gets a file picker. That is a safety
property -- a clinician must not be able to send model weights from a laptop --
and it is one letter wide: `ASO`'s `landmark_models`, plural, missed it and
would have asked for a 4.7 GB bundle as an upload. Check every `Path` argument
against that rule before opening the PR.

**Path arguments take a folder.** The server pays a process start-up cost per
call, so a folder of 40 scans has to be one call, not 40. Upstream is
inconsistent about this; standardise on batch-capable inputs. `iter_scans` in
the template is the shape to copy.

**A panel is presentation, and it lives in `layout.py`.** The signature says
what a tool accepts; it says nothing about how a clinician should be shown it,
and a schema that publishes "119 strings, pick some" is honest and unusable.
An optional `layout.py` beside the package fixes that:

```python
from .cbct import catalog

LAYOUT = {
    "landmarks": {
        "section": "CBCT landmarks",
        "ui": "tabs",
        "groups": {display: list(catalog.GROUP_LABELS[code])
                   for display, code in catalog.REGION_NAMES.items()},
    },
    "ios_teeth": {"visible_when": {"modality": "IOS"}},
}
```

Six keys, nothing else: `section`, `ui`, `groups`, `visible_when`, `label`,
`hidden`. `describe.py` merges them into the published arguments and **refuses**
any that names an argument the signature does not take, an option it does not
offer, or a value a condition could never match. Absent is fine -- the schema is
then exactly what it was before.

The set is a **joint** decision with the server: a key it does not name is
dropped silently on the way through -- that happened once, and was invisible
from both ends -- so adding a seventh here does nothing until it is added there
too. One is pending in that direction: **`options_when`**,
`{other_arg: {value: [options]}}`, which narrows a choice argument's own
options instead of hiding the whole field. The server accepts it and the client
renders it; `LAYOUT_KEYS` does not emit it. It is what `AREG` wants for its
three automation modes -- all meaningful, none of which offers "Oriented +
Fully-Automated" on IOS, so today the combo box offers a mode that fails at the
end of a run.

**Derive it, never restate it.** This replaces the `ArgSpec` tables, and those
drifted precisely because they listed options by hand: a landmark added to the
catalog was published by the schema and reachable through no tab at all.
Computing the groups from the catalog makes that impossible, and the validation
above is the backstop for whatever cannot be computed.

**Never unpack a `.zip`.** The server unpacks archives before `run()` is
called, so a tool always receives a real file or directory. The upstream tools
each carried their own extraction, zip-bomb cap and scratch directory for it;
that is exactly the duplication the split removes, and re-adding it in a tool
puts path resolution back on the wrong side of the line.

**Heavy imports go inside `run()`**, and that covers the whole import chain, not
just `__init__.py`. CI imports every tool on every PR to generate its schema; a
module-level `import torch` makes that cost a CUDA stack and fail on a machine
without one.

**The first docstring line is the tool description** shown to a clinician in the
client. Write it as a clinical action, not an implementation note. Everything
after it is for maintainers and never reaches the UI.

**Write only under `output_dir`.** Every tool takes it as a required `Path`
argument. `run()` returns a `Path`, or a `dict[str, Path]` when there are
several named outputs, and it must not write beside its inputs, into the working
directory or anywhere else. `output/` at the repository root is gitignored --
point a manual run at it rather than scattering results through the tree.

**`run()` does not read the environment and does not know about `/DATA`.** Path
resolution belongs to the server. Model weights are not packaged either: the
server fetches them into `/DATA/<tool>/models` and passes the path in.

**Tool sequencing belongs to the server too -- with one exception.** Where one
tool's output is another's input, the server chains them and neither tool knows
the other exists. That covers almost every case: `Crown_Seg → ALI` is two calls
with a folder in between. The exception is a tool that needs another *in the
middle of its own run*, where the output cannot simply be fed in beforehand.
`ASO` is the only one: fully-automated CBCT recentres its scans, predicts
landmarks **on the centred volumes**, and then registers. For that, take a
supervisor:

```python
def run(scans: Path, reference: Path, output_dir: Path, *, sup=None) -> Path:
    ...
    landmarks = sup.run("ALI", input=centered, model=bundle, output_dir=...)
```

- `sup` is **keyword-only and unannotated**, and that shape is the marker:
  `describe.py` keeps it out of the schema and publishes `"supervisor": true`
  instead, so a runner that cannot inject one refuses the tool rather than
  calling it and failing halfway. A `sup` that is positional or annotated is a
  hard error, not a schema entry.
- It is **duck-typed**. Never import a supervisor type -- that would need a
  package shared with the server, which is what the split removes. Three
  implementations produce the same shape and a tool cannot tell them apart: the
  server's (`server/execution/runner.py`), `scripts/run_tool.py`, and the dozen
  lines wrapping `sadt-testkit` in `tools/ASO/tests/`.
- **A cycle is refused by name, not by depth.** The server carries the chain of
  tools already running above a call, so asking for one of them fails at once
  and names the chain. The depth cap (5) is only a backstop for a chain that
  grows without repeating; the deepest real one is `AREG → ASO → ALI_CBCT`.
- Five members, nothing more: `sup.run(tool, **params)`, `sup.out`, `sup.tmp`,
  `sup.progress(fraction, message)`, `sup.log(message)`.
- **`sup.progress` is not a reason to take a supervisor.** It is one of the
  five members and it still works -- it writes to the same place a tool's own
  `progress.py` does -- but reporting progress needs no supervisor at all. See
  "Report where a batch has got to" below.
- `sup.run("ALI", ...)`, never `sup.ALI(...)`. A typo in a string is greppable;
  a typo in an attribute is an `AttributeError` fifteen minutes into a job, and
  the call graph stops being inspectable.
- **Give the caller a way in.** A tool that takes `sup` should also accept the
  dependency's output as an ordinary argument (`landmarks: Path = ""`) and skip
  the call when it is supplied. That is what keeps it usable standalone, and it
  is the only way it works with no supervisor at all.

Reach for this only when the ordering genuinely forbids chaining. Two calls with
a folder in between is simpler for everyone, and the server already does it.

### Report where a batch has got to

A forty-scan run takes an hour and the client shows an elapsed timer, which
says only that the connection is still open. A tool tells it where it has got
to by appending one JSON object per line to the file the server names in
`SADT_PROGRESS_FILE`:

```json
{"fraction": 0.325, "message": "scan 14 of 40"}
```

The server stamps the sequence number, the time and the state itself, and a
supervised child INHERITS the variable -- so a chain's events land in one file,
already ordered, with no plumbing.

Copy `progress.py` from `tools/_template/src/sadt_template/` into your package,
the way `iter_scans` is copied rather than shared, and call it at the top of
the cohort loop:

```python
from . import progress

for index, scan in enumerate(found, start=1):
    progress.report(index, len(found), "scan")
```

- `progress.report(index, total, what)` is the whole API for a batch. `index`
  is 1-based and names the item ABOUT to be processed, so the fraction is the
  share already behind it and never claims an item that is still running.
  `start=` and `end=` bound the slice of the run a loop occupies, for a tool
  whose batch is one phase among several -- without them the second phase
  sends the bar back to zero.
- `progress.emit(fraction, message)` is the raw form, for a waypoint that is
  not a position in a batch.
- **Never a file name, a patient name or an argument value in a message.**
  Position in the batch is progress; a file name is patient metadata, and
  these messages are stored on the server and shown to whoever is watching.
- **Best effort, and stdlib only.** Every failure to write is swallowed: a
  tool that cannot report its progress still has to finish the run. With the
  variable unset -- a checkout, a test, an older server -- it does nothing at
  all, which is why no tool needs a fallback path.
- **Report what is true.** `fraction` may be `None`, and that is the right
  answer for a phase the tool cannot see inside: `Batch_Dental_Seg` hands its
  whole cohort to one nnUNet call and can only say when the reading ends and
  the writing starts, while `AMASSS` loops per STRUCTURE and says "structure 3
  of 5". Interpolating a number a tool cannot know is worse than no bar.
- `run_tool.py` sets the variable itself and echoes what a tool appends, so
  running a tool from a checkout exercises the same channel the server uses.

Check the result before going further:

```bash
uv sync
.venv/bin/python ../../scripts/describe.py .          # the schema
python ../../scripts/run_tool.py <Name> --help        # the CLI it implies
```

`run_tool.py` builds its parser from the same signature, so `--help` is the
fastest way to see what you have actually declared -- and running it is the
fastest way to exercise a supervisor without a server.

## 4. Port the implementation

This is a repackaging exercise, not a rewrite. Keep the upstream algorithm
byte-for-byte identical wherever you can; the Slicer-specific scaffolding around
it (Qt widgets, queue tables, RAM watchdogs, progress dialogs) is not part of
the algorithm and does not come across. If you have to change logic, say so
explicitly in the PR description and in the tool's README.

### Enumerate, never pattern-match, when comparing against upstream

Comparing our port against an upstream module means listing what each side
declares. List it -- do not filter it through a regex you wrote from memory.

The cost is not a missed line, it is a **confident wrong answer**. An audit of
every tool's arguments used `^\s+([a-z_]+)\s*:` to pull parameter names out of
a `run()` signature. It silently dropped every argument containing a capital,
which is exactly `prediction_ID` -- and the report that came out of it named
AMASSS's missing `prediction_ID` as an API inconsistency across tools meant to
compose with each other. AMASSS has had it all along, in its signature,
documented and wired. So did Batch_Dental_Seg.

Upstream CLI parameters mix case freely (`DCMInput`, `SegmentInput`,
`save_in_folder`, `lm_type`), and so do ours. A filter that assumes otherwise
produces a table that looks like coverage and is not -- the same defect the
audit itself exists to find, turned on the instrument.

Use `[A-Za-z_][A-Za-z0-9_]*`, or better, parse rather than grep: `ast` for a
signature, an XML reader for a `.xml`. And when a finding is sharp enough to act
on, re-derive it a second way before reporting it.

### A guard counts what the tool produced, not what it walked past

Every tool here tolerates a partial failure -- one unreadable scan must not cost
the other 199 -- and every tool therefore ends with a guard that refuses to
return when nothing worked. Three of them counted the wrong noun, and all three
returned a clean report on a run that had produced nothing:

| tool | the guard counted | what it should have counted |
|---|---|---|
| `ALI_CBCT` | scans whose status was `ok` | landmarks actually placed |
| `Surg_Mov_Pred` | model packages successfully **loaded** | predictions actually **produced** |
| `Crown_Seg` | files written to the output tree | segmentations actually **performed** |

Each is one noun away from correct, and each failure is invisible from the
outside: the response is a 200, the report is full of successes, and the output
directory has files in it. `ALI_CBCT` shows what that costs -- a scan on which
every agent failed to converge was recorded `ok`, so a torch upgrade that left
two landmarks unplaced reported success, and the only thing that caught it was
comparing coordinates by hand against a reference.

Two habits follow:

- **Name the guard after the output.** `if not written`, `if not
  predictions_by_target` -- not `if not processed`. The noun you count is the
  claim you are making.
- **A guard at one stage does not cover the next.** `Surg_Mov_Pred` refused to
  continue when no model could be *loaded*, then predicted nothing perfectly
  happily. If a stage can produce zero, it needs its own guard however well
  guarded the stage before it is.

A fourth case is not a counting mistake and is worth recognising separately: a
guard that is never reached. `Crown_Seg` imports its segmentation engine inside
the branch that segments, so a batch of already-labelled meshes returned a clean
report on a deployment where the engine could not run at all. **A tool's
availability must not depend on its input data** -- one that can serve some
batches and not others is not available, it has only not been asked the right
question yet.

When a tool records that condition, name it with the vocabulary the server
already has: `ToolUnavailableError` is answered as a 501, so a per-item status
of `engine_unavailable` reads with that and `degraded` reads beside it. And
remember what recording buys and what it does not -- a field is only a signal if
something reads it. Putting the fact in the status rather than in a side field
is the right *place*; it is not a guarantee anyone sees it.

### Where an identifier ends, when a file name has been decorated twice

Tools here exchange files by NAME. `ALI` and `Crown_Seg` write beside a mesh,
`ASO` and `AREG` read what they wrote, and nothing but the file name says the
two belong to the same patient. So the rule that extracts a patient from a name
is a contract between tools, not a detail of whichever one you are porting.

A predictor appends what it did to the stem it was given, and what it appends
**names the jaw again**:

```
Upper_new_9.vtk                    <- the mesh
Upper_new_9_Upper_O_Pred.json      <- the landmarks ALI predicted from it
```

`ASO` reads a patient as "everything before the jaw token", and -- because the
published reference bundle is `Upper_gold.vtk` -- as "everything after it" when
the token comes first. Both rules are right. Applied to the pair above they give
`new_9` and `new_9_Upper_O_Pred`: two patients, so the mesh has no landmarks,
the landmarks have no mesh, and Semi-Automated IOS failed with *"no landmark
file for this jaw"* on the dataset this repository's own manifest ships.

**Your own output is decoration too**, and this is the half that is easy to
miss. `ASO` writes `Upper_new_9_Or.vtk`, which read back gives `new_9_Or`, while
its landmark file `Upper_new_9_Upper_O_Pred_Or.mrk.json` gives `new_9` -- so a
run fed its own results split one patient in two and failed the same way. A tool
that accepts a previous run's outputs has to read the names it writes with the
rule it reads inputs with. **Test the round trip**, not just the input.

Upstream never hit it because it compared names with `in` -- and that is exactly
what made patient `1` match patient `10`. Replacing the substring test with an
exact stem was right; it just has to be handed the right stem. **A second jaw
token ends the identifier**, because everything from it onwards is decoration a
later step added.

Two things to carry over:

- **Test the names your neighbours actually write**, not the names your tool
  writes. The regression was invisible to `ASO`'s own outputs and to every
  synthetic fixture; only the published test data exposed it.
- **A stricter rule inherits the burden of the loose one it replaces.** When you
  tighten a match to fix a false positive, enumerate the true positives the
  loose version was quietly carrying -- they are not in the tests, or the bug
  you are fixing would have been caught too.

### Duplicate the implementation, share the formats

Two tools that need the same code usually get **two copies**. That is
deliberate and it goes against the usual instinct: `nnunet_runner.py` exists
twice, in AMASSS and in BatchDentalSeg, because each tool is its own virtualenv
and importing across them would make one tool's missing dependency take both
out of the registry. A copy costs a divergence; a coupling costs an entire
class of failure. The copy usually wins.

**The exception is anything that defines the shape of bytes leaving this
repository.** File formats, extension vocabularies, on-disk layouts -- anything
a third party reads or writes. Those go in a shared package, because a
divergence there does not fail, it produces output that one consumer accepts
and another silently mis-reads.

The example that settles it. Before the split, ALI's two engines both wrote
Slicer markups files, and both set `display.visibility: false` in them. That
switches the markups display node off: Slicer loads the file, builds the node,
and draws nothing. The bug was invisible for as long as nobody opened a result
outside the module, and it was in **both** copies -- one mistake, written twice,
because the format was duplicated rather than shared. It is now in
`tools/ALI/common/`, imported by `ALI_CBCT` and `ALI_IOS` alike, along with the
table of which file extensions count as a CBCT volume and which as a surface
(the pre-port CLIs disagreed about `.stl`, so it was accepted by the UI and
silently ignored by the CLI).

What stays duplicated even between two halves of one tool: `errors.py`. Errors
cross the process boundary by exception class **name** -- the runner records the
name, the server maps it to an HTTP status -- so a shared base class is not
merely unnecessary, it is not the mechanism.

A shared package must declare **no dependencies**. It installs into several
tool environments whose pins are deliberately incompatible, and anything it
pulled in would have to be satisfiable by all of them at once -- which is the
constraint this repository exists to remove.

### A path dependency is installed as a COPY, not a link

`sadt-areg-common = { path = "../common" }` installs a snapshot. Editing
`common/src/.../pairing.py` changes nothing in a tool's virtualenv until:

```bash
uv sync --reinstall-package sadt-areg-common
```

Without it you edit, re-run, and see the OLD behaviour -- which reads as "my fix
did not work" and sends you rewriting a correct patch. It cost a cycle the first
time it came up. `editable = true` avoids it, and is why the dev-only
`sadt-testkit` entries carry it; a shared runtime package deliberately does not,
so what a tool runs against is a fixed copy rather than whatever the working
tree happens to hold.

### A `sup.run()` call must live in the tool's own `src/`

Never in a shared package, however much orchestration two tools appear to have
in common. `describe.py` derives the schema's `calls` field by **reading each
tool's own source** -- the call sites sit in branches only a real run reaches, so
there is nothing to introspect at import time -- and the server refuses to start
when a declared call names a tool it does not serve. Orchestration moved into a
shared package is invisible to both: the names never reach `calls`, and the
startup check silently has nothing to verify.

That is worse than no check. A tool would then declare `supervisor = true` with
an empty `calls`, and a renamed sibling would break it at run time again --
which is exactly what the check was added to prevent.

AREG is the case that settles it. Its two engines share four of the six helpers
in `tools.py` (`require`, `_output`, `_returned`, `orient_scans`), and only
`segment_masks` is CBCT-specific against `label_crowns` and
`predict_mucogingival` on the IOS side. Sharing the four would have been the
obvious move; it would have hidden `sup.run("ASO", ...)` from the generator for
both tools. Each engine carries its own copy, holding only what it calls, and
the two `calls` lists come out right:

    AREG_CBCT   calls = ["AMASSS", "ASO"]
    AREG_IOS    calls = ["ALI_IOS", "ASO", "Crown_Seg"]

This is the standing rule applied, not an exception to it: orchestration is
**implementation**, and it has to stay where the tooling can see it.

### `[tool.uv.sources]` only applies to DECLARED dependencies

A source says *where* a package comes from. It does not make the package a
dependency. Name it in a source and forget to name it in `dependencies`, and uv
resolves without complaint, installs nothing, and the failure arrives at
runtime as a `ModuleNotFoundError` or -- worse -- as a *different* build of the
package pulled in transitively from PyPI.

It has caught three different things in this repository, which is what makes it
a rule rather than an anecdote:

| package | left undeclared | what happened |
|---|---|---|
| `pytorch3d` | pulled transitively by `shapeaxi` | source ignored, *"no wheels with a matching Python version tag"* |
| `torchvision` | pulled transitively by the torch stack | came from PyPI, built against the default torch instead of cu128 -- imports fine, then `RuntimeError: operator torchvision::nms does not exist` |
| `sadt-ali-common` | a path dependency of ALI_CBCT/ALI_IOS | installed nothing at all; `ModuleNotFoundError` on first import |

The rule: **if it has a `[tool.uv.sources]` entry, it must also be in
`[project] dependencies` (or in an extra).** Transitive is not enough, and the
two failure modes it produces -- a missing module, and a right-version wrong-build
C extension -- look nothing like each other, so recognising one does not help you
recognise the next.

### A directory is a tool when its pyproject says so

Discovery keys on a `[tool.sadt]` section, not on the presence of a
`pyproject.toml` and not on where the directory sits:

```toml
[tool.sadt]
tool = true
```

A shared path dependency (`tools/ALI/common/`, `testkit/`) has a
`pyproject.toml` -- it must, to be installable -- and no `[tool.sadt]`, so it is
importable, installable, and never discovered or served. This is what lets a
grouping folder like `tools/ALI/` hold `ALI_CBCT/`, `ALI_IOS/` and `common/`
side by side. Single-engine tools stay flat: there is no `tools/AMASSS/AMASSS/`.

## 5. Pin what upstream actually used

**Do not bump torch, monai or numpy to "something newer that works".** Changing
them can change model outputs, and outputs must be revalidated against reference
data before any version moves. If upstream's pins are ambiguous or
contradictory, ask -- do not pick one yourself.

`requires-python` must be accurate: bound it by what the pins actually support,
or uv will pick an interpreter with no wheels for them and spend a quarter of an
hour building from source.

CUDA-variant wheels need their own index, per tool, and `explicit` is
load-bearing -- without it uv looks for *every* package on that index:

```toml
[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu118" }
```

`pytorch3d` needs torch present at build time and fails under uv's build
isolation. Install it with `--no-build-isolation` and an explicit order, and
write the exact working incantation into the tool's README -- deployment
precompiles it once into a local `/wheels` directory. If it will not build and
the only workaround is a different torch, stop and ask.

**Do not create a `[tool.uv.workspace]`.** A workspace forces one shared
lockfile across its members, which is precisely what must not happen here: uv
cannot resolve torch 1.13 and torch 2.4 in one lock. These tools live in one
repository for convenience and share no dependency resolution. For the same
reason there is no shared `sadt-core` package -- small helpers like `iter_scans`
are copied between tools on purpose.

**What IS shared is a contract, never a helper**, and the line between them is
the one drawn above: a helper is internal and may drift without anyone being
misled, while a contract is what the outside world sees. Two exist.
`tools/<family>/common/` holds what one family must not disagree about, and
`naming/` (`sadt-naming`) holds the file-format vocabulary -- which
extensions are a volume, a surface, a markups file -- because that is a
property of NIfTI and VTK rather than of any tool, and duplicating it is what
let ALI advertise `.stl` and silently ignore it. Before it existed there were
four byte-identical copies of the scan vocabulary and five different values of
`SURFACE_EXTENSIONS` under one name, with nothing saying which difference was
a decision.

A tool that reads LESS than a format still says so beside itself -- the table
states what the format is, not what a tool handles, and advertising more than
you read is the original bug. What a tool must not do is write the list again.

A shared package declares **no dependencies**; see below.

Commit `uv.lock`. CI runs `uv sync --frozen`, which fails if it is stale.

## 6. Validate

`tests/test_run.py` calls `run()` end to end on real input and asserts on the
output: the expected files exist, they are a plausible size, and where a
reference output exists, results match within a documented tolerance.

Test data lives in `tools/<name>/tests/data/` as **a download script plus
checksums** -- never large binaries, never patient data, and the fixtures must be
anonymised and public-domain. If no suitable public sample exists, ask before
committing anything.

**When a tool's input is another tool's output**, test against the real thing
rather than a stand-in. `sadt-testkit` runs the other tool through *its* venv as
a subprocess -- the way the server does -- so nothing is imported across tools:

```python
from sadt_testkit import is_built, run_tool

@pytest.mark.skipif(not is_built("Crown_Seg"), reason="run uv sync in tools/Crown_Seg")
def test_on_freshly_segmented_meshes(tmp_path):
    segmented = run_tool("Crown_Seg", meshes=..., model=..., output_dir=tmp_path / "seg")
    run(scans=segmented, model=..., output_dir=tmp_path / "out")
```

Skip, never fail, when the other tool is not built: CI builds each tool in its
own job. `tools/_template/tests/test_integration.py` is the file to copy, and
[testkit/README.md](testkit/README.md) covers the rest. This is a **development
dependency** and must never appear in `[project] dependencies`.

It is also what a supervisor is built out of in a test: `tools/ASO`'s `LocalSup`
is a dozen lines wrapping `run_tool`, and the tool cannot tell it from the
server's.

GPU tests carry `@pytest.mark.gpu`. CI skips them (`-m "not gpu"`) because the
runner has no CUDA device, which makes them your responsibility: run them by
hand and state in the PR that you did and what came out.

The tool's README records what you validated against -- which input, which model
weights, which reference, what tolerance.

You may use the **Slicer Cloud** application to exercise a tool end to end
against realistic data when you cannot validate locally. Ask for the endpoint
and credentials rather than guessing, and never commit them.

## 7. Record provenance

The tool README opens with the block from `tools/_template/README.md`, filled
in, and the same PR adds the tool's row to [PROVENANCE.md](PROVENANCE.md).
Upstream history is not grafted into this repository -- it is one history for
sixteen unrelated modules and the result would be unreadable -- so this table is
the only record of where an algorithm came from.

## 8. Open the pull request

State:

- the version pins chosen and why,
- what you validated against, and the result of the GPU tests,
- anything changed from upstream, and why.

Prefer an open question in the PR over a silent decision.

Once the tool is merged here, open a companion PR on
`VISOR-serve` deleting it there. **Never delete first** -- the
server keeps working off its copy until this one is proven.

## Stop and ask when

- upstream's pins are ambiguous or contradictory and you would have to choose,
- porting would mean changing the algorithm rather than repackaging it,
- you cannot find suitable non-patient test data,
- a tool needs model weights you cannot locate,
- `pytorch3d` will not build and the workaround means moving torch.

## Repository checks

```bash
uv run --no-project --python 3.12 --with pytest --with numpy -- pytest scripts/tests -q
uv run scripts/audit.py
uv run --no-project --python 3.11 --with pyflakes -- python -m pyflakes tools/*/src tools/*/*/src
```

**pyflakes, and specifically for undefined names.** An import proves a module
loads; it says nothing about a name referenced inside a branch only a real run
reaches, and that is exactly where this repository keeps finding them. Splitting
ALI left four such defects -- a constant whose import went with the block that
defined it, a module that moved to the other engine, a dependency dropped from a
pyproject, and a semaphore whose definition was removed while one use survived.
All four passed `import`, all four passed schema generation, and the first three
were found one per run until a single pyflakes sweep found the rest at once.

Run it across every tool, including the ones nested under a grouping folder --
`tools/*/src tools/*/*/src` covers both depths.

`audit.py` is read-only by design: it reports distinct torch and Python versions
and what dropping one would save, and it never edits a lockfile. Aligning two
tools onto one runtime means revalidating both against reference data first.
