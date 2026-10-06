# Agent

Route a free-text request to one of this server's tools.

Not an imaging tool. A local language model reads what a clinician typed, picks
one tool out of the server's own catalogue, and extracts that tool's parameters
from the same text. It writes a **proposal**; it runs the chosen tool only when
the caller asks it to.

```
"I have 40 cone beam CTs and I need the mandible segmented out of each,
 they are in /data/cohort_A"

    -> routing.json
       { "tool": "AMASSS",
         "arguments": {"scans": "/data/cohort_A", "structures": ["MAND"]},
         "confidence": 0.92,
         "reasoning": "the request asks for a craniofacial structure on CBCT",
         "alternatives": [{"tool": "Batch_Dental_Seg", "score": 4.11}, ...],
         "missing_required": ["model"] }
```

## What it does

1. read the catalogue -- by default the server's own `GET /tools`;
2. rank every tool in it against the request, and show the model the best few;
3. ask the model for one tool name and a reason, then check that the name is
   one it was actually offered;
4. ask the model for that tool's parameters, and check every value against the
   tool's real declared type and choices;
5. write `routing.json`, plus `Agent_report.json` saying how the decision was
   reached;
6. **only if `execute` was asked for**, run the chosen tool through the
   supervisor and put its outputs in `run/`.

In `Ask (Interactive)` mode it does none of steps 3-6: it answers with
methodology advice in `advice.md` and proposes nothing.

## What it requires to run

Two things, and it acquires neither of them:

| prerequisite | how it is satisfied |
|---|---|
| **A reachable Ollama endpoint** | Running already. `endpoint`, else `OLLAMA_HOST`, else `http://127.0.0.1:11434`. This tool does not download, install, unpack or start Ollama. |
| **The model pulled on that server** | `ollama pull qwen3:8b` on the machine Ollama runs on, once, by an operator. This tool does not pull models. |
| A tool catalogue | Nothing to do: `SADT_API` is set by the server and `GET /tools` needs no token. Offline, pass `catalog_file`. |

An absent endpoint or an unpulled model is a `ToolUnavailableError`, which the
server answers **503** -- a deployment problem, not the caller's. The message
names the endpoint or the exact model tag. A server that is up but too slow is
the same class with a different message, "router call exceeded
timeout_seconds=N", so a slow model is not mistaken for a stopped one. So is a
live registry that is unset, unreachable or too slow.

The server maps the exact class name, not its parents, so who is at fault
decides what is raised: a `catalog_file` the caller sent that cannot be used is
a plain `ToolInputError` (422); a live registry that answers with something
that is not a catalogue is a `CatalogError`, and a model answer that is not
JSON a `ModelAnswerError` -- both server faults, 500.

## Arguments

| argument | what it is |
|---|---|
| `prompt` | what the user wants, in their own words |
| `output_dir` | filled in by the server: the job's own output folder |
| `catalog_file` | a JSON file holding `GET /tools`. Empty = read the live registry |
| `mode` | `Agent (Automated)` or `Ask (Interactive)` |
| `folders` | the folders holding the data, as a **list** -- the only paths the model is shown |
| `history` | earlier turns, `[{"role", "content"}, ...]` as JSON |
| `candidates` | how many tools the model is shown, best first. `0` = all |
| `execute` | run the chosen tool as well as proposing it. **Off by default** |
| `endpoint` | the Ollama server. Empty = `OLLAMA_HOST`, then loopback |
| `model_tag` | the Ollama model, e.g. `qwen3:8b` |
| `temperature`, `seed` | 0 and 0, so the same request routes the same way twice |
| `timeout_seconds` | per model call, and for reading the registry |

`model_tag` is deliberately not called `model`: the server publishes a `path`
argument named `model`, `*_model` or `*_reference` as a name picked from
`DATA/<tool>/models/`. An Ollama tag is neither a file nor a bundle this server
hosts, and an argument that reads like one is a trap for the next person.

## The design change: the catalogue is the live registry

**Upstream kept the catalogue in a file.** `Agent_CLI/manifest.yaml`, 891 lines,
23 tools, each one's parameters, types, defaults, encodings and positional order
written by hand a second time. It had drifted:

- **11 of the 13 verifiable entries omit interior required positionals**, so the
  command built from them is rejected by the target's own `argparse` every
  single time;
- **two entries (`batchdentalseg`, `clic`) describe Slicer widgets as CLIs**,
  and die on `import slicer`;
- **`resolve_tool_path` searches for a `CLI files/` directory that exists
  nowhere in the repository** and then returns its first candidate anyway
  (`utils.py:177-179`), so every routed run resolves to a path that is not
  there.

Porting the YAML would have been porting the defect. On this server the
authoritative catalogue already exists and **cannot** drift: `GET /tools`
publishes every tool's name, every argument's type, whether it is required, its
choices and its description, generated from that tool's own `run()` signature by
that tool's own interpreter (`scripts/describe.py`). There is no second copy to
keep in step, because there is no second copy.

So the catalogue is an **input**, and its default source is the registry:

1. `catalog_file`, a JSON file. Explicit, offline, and what every test uses.
2. `SADT_API` + `/tools`. `GET /tools` needs no token -- the server strips
   `API_TOKEN` from a tool's environment on purpose -- and `SADT_API` exists
   precisely so a tool can reach the server it is running under
   (`server/config.py`). A loopback read of the server's own registry is not an
   outbound call.

Neither available is a refusal naming both. It never guesses.

Both shapes are read: the schema `scripts/describe.py` writes (`path`, `str`,
`int`, `float`, `bool`, `list[...]`, `vec2`), and what `GET /tools` publishes
after the server folds it into its own vocabulary (`choice`, `multichoice` with
an `{option: selected}` mapping, `initial` for a default, and server file types
such as `csv_file` or `folder`). An argument of any other type is kept
**opaque**: logged, never offered to the model, and still counted as missing
when required -- one new type never takes the whole catalogue down.

### Why not the supervisor

The obvious question is whether `sup` can enumerate the tools, which would make
the catalogue need no argument at all. **It cannot.** The supervisor's interface
is frozen at five members -- `run(tool, **params)`, `out`, `tmp`,
`progress(fraction, message)`, `log(message)` -- in all three implementations
(`VISOR-serve`'s `server/execution/runner.py`, this repository's
`scripts/run_tool.py`, and the fakes in every tool's tests). There is no
listing, no schema lookup, and adding one would be a change to a contract three
codebases agree on. Hence an argument, with the registry as its default source.

### What that costs: no startup check on the callee

`describe.py` reads a tool's source for `sup.run("<name>", ...)` and publishes
the names as `calls`, so the server can refuse at startup a tool that chains to
one it does not serve. It requires a literal or a module-level constant.

**A router has no such name.** Its callee is whatever the model picks out of the
catalogue, and the catalogue is the registry -- the set of names it might call
*is* the set the server serves, so there is nothing for a startup check to
verify. `execution.invoke` therefore takes its supervisor as a parameter named
`supervisor` rather than `sup`, which is what lets the schema generate; the
reasoning is written at length in `execution.py` and pinned by
`test_no_calls_are_published_and_that_is_the_design`.

What replaces the check is strictly narrower and happens earlier: the name is
resolved against the catalogue before any call, so only names the registry
published in this same run ever reach `supervisor.run`.

## It proposes; it does not execute

`execute` defaults to **False**. Upstream gated the same step behind a
`QMessageBox.question` showing the parameters and asking Yes/No
(`Agent/Agent.py:864-875`). A packaged tool has no dialog box, so that human
approval had to become a field a request carries rather than something that
disappeared with the Qt.

Turning it on is not enough. Execution is refused, with the reason recorded in
`routing.json`, when:

- no tool was chosen;
- a required argument is missing (it is a question to ask, not a value to make
  up);
- the router's own confidence is below 0.5 -- a floor against a model that has
  said in its own answer that it is guessing, not a quality bar;
- nothing injected a supervisor (`SupervisorRequired`, which names both ways
  forward).

`output_dir` is never the model's to propose: it is taken out of the schema the
model is shown, and filled with `output_dir/run/`.

## The ranker: measured, not assumed

Upstream ranked candidates with
`sentence_transformers.CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`.
That is ~1.2 GB of CPU torch wheels (7 GB at cu128) plus an 88 MB model, to
score a request against 23 short strings.

**Both rankers were measured**, on 30 realistic requests written against the
real 16-tool catalogue of this server (`AMASSS`, `ALI_CBCT`, `ALI_IOS`,
`AREG_CBCT/IOS/IOSCBCT`, `ASO`, `AutoCrop3D`, `AutoMatrix`, `Batch_Dental_Seg`,
`CLIC`, `CNE`, `Crown_Seg`, `DOCShapeAXI`, `GreedyReg`, `Surg_Mov_Pred`), each
schema generated by that tool's own interpreter. Requests are phrased as a
clinician would phrase them, and 11 of the 30 share no content word with their
target ("line up my second timepoint with the first", "cut everything outside
the region of interest I drew").

| ranker | recall@1 | recall@3 | recall@5 | MRR | per query | dependencies |
|---|---|---|---|---|---|---|
| **BM25 + synonyms (shipped)** | **0.867** | **1.000** | **1.000** | **0.928** | **4.3 ms** | **none** |
| BM25, synonyms disabled | 0.833 | 0.967 | 0.967 | 0.906 | 4.3 ms | none |
| CrossEncoder, name+description+arguments | 0.833 | 0.967 | 0.967 | 0.893 | 295 ms | ~1.2 GB |
| CrossEncoder, upstream's document text | 0.833 | 0.967 | 0.967 | 0.899 | 295 ms | ~1.2 GB |

Plus a 6.7 s model load in every process, and `Agent_CLI.py` ran as a fresh
subprocess per request.

**Recall@k is the metric that matters**: the ranker's job is to put the right
tool in front of the model, and the model picks. At the default `candidates=8`
both are perfect; at upstream's hardcoded `k=3` the pure-Python ranker retrieves
every target and the cross-encoder misses one outright -- for that request the
router could not have chosen correctly whatever it answered.

The obvious objection is that the synonym table was written by someone who had
read this catalogue. The ablation row answers it: with the table **disabled**,
plain BM25 still matches the cross-encoder on every metric. The table buys the
last 3 points of recall@1, not the result.

So: no `sentence-transformers`, no torch, and this tool's dependency list is
empty. Reproduce with `tests/` plus the request set in this README's history, or
re-run the comparison by pip-installing `sentence-transformers` in a scratch
venv -- nothing in the tool needs it.

## Changes from upstream

Each of these is a defect that lost, fabricated or misdirected a result, and
each has a test named after it.

- **A ranker that cannot rank no longer narrows silently.** `utils.py:90-94`
  caught every exception and returned `scripts[:k]` -- the first three tools in
  file order -- while the router prompt went on saying "choose ONLY from the
  candidates". With no network or a stale model cache, a registration request
  was offered landmarking and segmentation and nothing else. It is an error
  now (`RankingError`), and separately: a ranking with **no signal at all**
  hands over the whole catalogue rather than an arbitrary prefix. Narrowing is
  something a ranker earns. The `print` that announced the fallback is gone
  too; it went to the stdout the caller parsed as JSON.
- **The extraction path returns one shape.** `utils.py:220` annotates a 3-tuple;
  two paths return 4 values and one returns 3, and the caller unpacks 4
  (`utils.py:222-225`), so an unknown tool name raised
  `ValueError: not enough values to unpack` instead of the intended empty
  result.
- **A missing model is named with its whole tag.** `Agent.py:915` did
  `model_name.split(':')[0]`, so a missing `qwen3:8b` was reported as "run:
  ollama pull qwen3" -- a different model, several gigabytes, and still not the
  one the tool asks for.
- **A folder whose name contains a comma stays one folder.** `Agent_CLI.py:88`
  did `input.folders.split(",")` with no escaping, so `/data/Smith, John/T1`
  became two paths that do not exist. `folders` is `list[Path]`.
- **How many tools the model sees is an argument.** `Agent_CLI.py:102,182`
  hardcoded `k=3` at both call sites, so 3 of 23 tools were ever shown,
  whatever the request. `candidates` defaults to 8; `0` shows all of them.
- **Nothing is truncated in the router prompt.** `Agent_CLI.py:29,31` cut each
  description to 140 characters and each tag list to 8, mid-sentence and
  mid-word, on a prompt that then carried a whole manifest's worth of parameter
  definitions.
- **Conversation history is data, not text split on emoji.**
  `Agent.py:1126-1131` reloaded a transcript by rewriting every robot emoji as a
  person emoji, splitting on that one character and assigning speakers by index
  parity: a user who typed or pasted one inverted every role after it, and the
  model was then told its own answers were the clinician's instructions.
  History is a JSON array of `{"role", "content"}`, and a malformed one is an
  error rather than a silently forgotten conversation (`Agent_CLI.py:79-84`
  swallowed the decode error and continued with `[]`).
- **A nameless parameter cannot become a phantom argument.**
  `utils.complete_with_defaults` keyed defaults on `p.get("name", "")` over a
  list of parameter objects, so every parameter lacking a name collapsed into
  one `""` entry and was injected into the proposal. `arguments` is a mapping
  keyed by name; an empty key is refused at load.
- **Folder arguments are not rejected by an extension list.**
  `parameter_validator.py:191` rejected any argument whose *name* contained
  "folder" or "dir" if its value ended in one of seven extensions -- a list
  missing `.nrrd`, `.mha`, `.gipl`, `.dcm`, `.vtp`, `.obj` and `.off`, every one
  of which these tools read. A guess about the argument's meaning checked
  against an incomplete guess about the file's. The schema says `path` and does
  not distinguish a file from a directory; nothing here pretends otherwise.
- **`choices` is enforced, because it is real now.**
  `parameter_validator.py:206` read a `choices` field no manifest entry ever
  set, so the check never once fired. From a live schema it comes from a
  `Literal[...]` in the callee's own signature, and every item of a
  `list[Literal[...]]` is checked, not just the list.
- **`min` and `max` are gone.** `parameter_validator.py:233,242` read fields no
  manifest set and `describe.py` cannot emit. Checking a field that never exists
  is not a check.
- **Defaults are not copied into the proposal.** They are in the callee's own
  signature; restating them adds a copy that can only drift and hides which
  values the *user* actually asked for. What is proposed is what was extracted.
- **A hallucinated tool name is a refusal, not a `KeyError`.** Upstream passed
  the router's answer straight into `build_cli_args`, which raised `KeyError:
  Tool 'X' not found in manifest` at the user. It is checked against the
  candidate list and recorded as `rejected_tool_name`.
- **Ask mode keeps its Markdown.** `Agent_CLI.py:215-216` stripped every `*` and
  `#` from the answer so it would look like plain text in a Qt label; the answer
  is written to a `.md` file.

## Not ported, and why

- **The Ollama installation.** `Agent.py:101-207` downloads a 500 MB-1 GB binary
  over plain `urllib` with no checksum and no signature, `chmod 0o755`s it,
  strips the macOS Gatekeeper quarantine attribute (`xattr -dr
  com.apple.quarantine`), and `Popen`s it -- and on Linux it
  `tarfile.extractall`s the downloaded archive with no member sanitisation at
  all. A packaged tool declares a reachable endpoint as a prerequisite; it does
  not acquire a server, and it does not extract an untrusted tarball.
- **Pulling the model at run time.** `chat_with_auto_pull` treated a 404 as a
  cue to download ~5 GB from inside a request handler. A server holding
  confidential imaging does not make outbound calls mid-request, and a request
  that blocks for a quarter of an hour on its first run is not a request. Same
  reasoning for the unpinned cross-encoder download, which is moot now anyway.
- **`slicer.util.pip_install`.** `Agent.py:1141-1183` installed five packages,
  two of them version-pinned to work around other packages' regressions, into
  Slicer's **shared** site-packages -- mutating every other SADT module's
  environment to fix this one. A tool has its own virtualenv; this one has no
  dependencies at all.
- **Writing chat transcripts to `$HOME`.** `Agent.py:1078` wrote
  `~/Chat_LLM_<timestamp>.txt`, containing the patient folder paths the user
  typed, outside any job directory and outside any cleanup. Everything this tool
  writes is under `output_dir`.
- **The blocking `subprocess.run` on the Qt main thread**
  (`Agent.py:962, 1266`), all Qt, the `.ui`, the drop zone, the chat bubbles,
  and the `_suggestFixFor` remediation table. The tool is a function; the panel
  is the client's.
- **The post-failure repair loop** (`runToolWithRepair`, `_proposeRepair`,
  `build_repair_prompt`). It re-ran a failed imaging tool with parameters a
  model proposed from its stderr, up to `MAX_REPAIR_ATTEMPTS` times, each behind
  a dialog. It is a reasonable idea and it is not a routing decision: it belongs
  to whatever holds the conversation, which is where the Yes/No that gated it
  lives. Nothing here re-runs anything.
- **`build_cli_args`, `resolve_tool_path`, `cli_style`, `positional_order`,
  `encode`, `flag`.** Half the manifest existed to turn parameters into a
  command line for a script found on disk. A tool is reached through the
  supervisor by name, with keyword arguments; there is no command line to build
  and no path to resolve.
- **`tags` and `priority`.** Manifest fields with no counterpart in a published
  schema. The ranker reads the tool's own description and argument text
  instead, which is the text that cannot drift.

## Provenance

Ported from DCBIA-OrthoLab/SlicerAutomatedDentalTools:
`Agent_CLI/Agent_CLI.py` (234 lines), `Agent_CLI/Agent_CLI_utils/utils.py`
(404), `parameter_validator.py` (385), `parameter_extraction_improved.py` (90),
and `Agent_CLI/manifest.yaml` (891, replaced by the live registry).
`Agent/Agent.py` (1382 lines of Qt) is not ported; the changes and omissions
above cite it by line.

## Tests

```bash
uv run pytest                       # 203 tests, no Ollama, no network, no GPU
uv run pytest -m models -o addopts= # 7 more, against a real endpoint
```

The stubbed suite replaces `llm.chat` and bans `urllib.request.urlopen`
outright, so a test that reaches for a live endpoint fails rather than passing
on whatever happens to be running. The catalogue every test routes against is
fabricated in `conftest.py`; nothing here reads the real registry or a sibling
tool.
