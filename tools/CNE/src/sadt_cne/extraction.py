"""The model: finding it, loading it, asking it, and reading its answer.

Ported from `CNE_CLI/CNE_CLI.py`'s inference block (`:141-251`). What differs:

* **the note type is a fixed choice, so its two readings cannot disagree.**
  Upstream compared `notesType == "TMJ"` when picking the context length
  (`:150`) and `notesType.upper() == "TMJ"` when picking the system prompt
  (`:193`). `"tmj"` therefore got the TMJ instruction with the Ortho context of
  2048 tokens -- a long note silently truncated, from a spelling difference.
  Here the value is a `Literal["TMJ", "Ortho"]`, validated by the server before
  `run` is entered, and both readings come from the same table.
* **a truncated answer is a failure, not a result.** Upstream capped generation
  at 500 tokens and, when the JSON that ran off the end failed to parse, wrote
  the truncated blob as if it were the extraction. `finish_reason` says exactly
  when that happened, and it is treated as the failure it is.
* **temperature defaults to 0.** Upstream used 0.1, so the same note extracted
  twice gave two different answers. A clinical extraction that cannot be
  reproduced cannot be checked.
* **stderr is not surgically redirected.** Upstream dup2'd the process's stderr
  onto /dev/null around the model load with no `try/finally`
  (`CNE/CNE.py:155-169`): a load that raised left the whole process writing its
  traceback into the void, permanently. `quiet_stderr` is a context manager, so
  the restore happens on every path including the raising one.
"""

import contextlib
import io
import json
import re
import logging
import os
from pathlib import Path

from .dependencies import ToolUnavailableError, preload_cuda_runtime, require

logger = logging.getLogger("CNE")

# The system message for TMJ notes, verbatim from upstream's INSTRUCTION_TMJ.
# Ortho gets none: its model is fine-tuned to answer from the note alone, and
# adding an instruction it never saw in training changes what it emits.
INSTRUCTION_TMJ = (
    "Using the following note, extract structured key-value pairs about the "
    "patient's symptoms and diagnoses:"
)

# Context window per note type, upstream's numbers. TMJ notes are long enough to
# need 6144; the Ortho model was trained at 2048. One table, read once, so the
# prompt and the window can no longer be chosen by two different rules.
CONTEXT_TOKENS = {"TMJ": 6144, "Ortho": 2048}

SYSTEM_PROMPTS = {"TMJ": INSTRUCTION_TMJ, "Ortho": None}

# What a quantised llama.cpp model file is called.
MODEL_EXTENSION = ".gguf"

# The model's answer ran out of room. llama.cpp reports it, upstream ignored it.
TRUNCATED = "length"

# The wrapper key the fine-tunes sometimes put their answer under.
EXTRACTION_KEY = "extraction"


@contextlib.contextmanager
def quiet_stderr():
    """Swallow the loader's chatter, and give stderr back whatever happens.

    llama.cpp prints a screen of build and tensor information at load. Upstream
    silenced it with `os.dup2` on the process's real stderr and restored it with
    two more `dup2` calls placed AFTER the load -- so a load that raised jumped
    over the restore and left the interpreter's stderr pointing at /dev/null for
    the rest of the run, traceback included.

    A context manager cannot have that bug: the restore is in the `finally` the
    `with` statement compiles to. It redirects Python's `sys.stderr` only, which
    is deliberate -- what the C library writes to file descriptor 2 goes to the
    server's per-job stderr file, where a failed load is meant to be readable.
    Anything captured here is logged at DEBUG rather than dropped; when the
    load FAILS, `load_model` logs its last lines again at WARNING, where an
    operator can see them.
    """
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stderr(buffer):
            yield buffer
    finally:
        captured = buffer.getvalue().strip()
        if captured:
            logger.debug("llama.cpp: %s", captured)


def resolve_model_file(model) -> Path:
    """The `.gguf` to load, from a file or from a hosted bundle directory.

    The server hands over whatever `DATA/CNE/models/<name>` is -- the manifest
    stages the TMJ weights as a `TMJ/` folder holding one file -- so both shapes
    have to work. A directory holding exactly one `.gguf` resolves to it; one
    holding several is REFUSED by name rather than resolved by sort order.
    Which model vintage ran must never depend on file names.
    """
    model = Path(model)
    if model.is_file():
        if model.suffix.lower() != MODEL_EXTENSION:
            raise ValueError(
                f"'{model.name}' is not a {MODEL_EXTENSION} model file. CNE runs "
                f"quantised GGUF models."
            )
        return model

    if not model.is_dir():
        raise FileNotFoundError(
            f"No model at '{os.path.basename(str(model))}'. Stage one with "
            f"`setup-models.sh --tool CNE`."
        )

    candidates = sorted(
        path for path in model.rglob("*")
        if path.is_file() and path.suffix.lower() == MODEL_EXTENSION
    )
    if not candidates:
        raise FileNotFoundError(
            f"'{model.name}' holds no {MODEL_EXTENSION} file. Stage the model with "
            f"`setup-models.sh --tool CNE`."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"'{model.name}' holds {len(candidates)} {MODEL_EXTENSION} files "
            f"({', '.join(path.name for path in candidates)}). Name the one to "
            f"run rather than leaving the choice to sort order."
        )
    return candidates[0]


def model_type_hint(model_file) -> str:
    """"TMJ", "Ortho" or "" -- what the model's own path says it is for.

    A guess, and used only as a guess: it is reported as a warning when it
    disagrees with `notes_type`, never as a refusal. The two fine-tunes are
    different models with different prompts, and pairing the Ortho weights with
    the TMJ instruction produces an answer that looks right and is not -- but a
    file name is not a contract, and refusing on one would make a renamed bundle
    unusable.
    """
    text = str(model_file).lower()
    hits = [name for name in CONTEXT_TOKENS if name.lower() in text]
    return hits[0] if len(hits) == 1 else ""


def context_for(notes_type: str, requested: int) -> int:
    """The context window: what the caller asked for, else the type's default."""
    if requested < 0:
        raise ValueError(
            f"context_tokens must be 0 (the note type's default) or positive, "
            f"got {requested}."
        )
    return requested or CONTEXT_TOKENS[notes_type]


def build_messages(notes_type: str, text: str) -> list:
    """The chat the model is given: a system prompt where there is one, then
    the note. TMJ has an instruction, Ortho deliberately has none."""
    messages = []
    prompt = SYSTEM_PROMPTS[notes_type]
    if prompt:
        messages.append({"role": "system", "content": prompt})
    messages.append({"role": "user", "content": text})
    return messages


def engine_module():
    """The `llama_cpp` module, with the CUDA runtime opened first.

    Every import of llama.cpp in this tool goes through here, and the order is
    the whole point: the CUDA build links `libcudart.so.12` without shipping it,
    so `preload_cuda_runtime()` has to run BEFORE the import rather than before
    the first GPU call. It is a no-op on the CPU build and on any machine whose
    loader would have found the library by itself.
    """
    preload_cuda_runtime()
    return require("llama_cpp", "llama-cpp-python")


def supports_gpu_offload() -> bool:
    """Whether the llama.cpp build INSTALLED HERE can put layers on a GPU.

    A property of the wheel, not of the machine and not of the request. It is
    what makes `device="cuda"` checkable: the argument has always been declared
    and has always been passed through as `n_gpu_layers=-1`, but on the CPU
    wheel that offloads nothing at all, silently, and the run then reports
    `device: cuda` over a run that was entirely on the CPU. Read once and put
    in the report beside `device`, so the two can be compared.
    """
    return bool(engine_module().llama_supports_gpu_offload())


# How many trailing lines of the loader's own output are worth a log line.
# llama.cpp prints its reason for refusing a file last; the screen above it is
# tensor metadata.
_LOADER_TAIL_LINES = 5


def load_model(model_file, context_tokens: int, seed: int, n_gpu_layers: int):
    """The llama.cpp engine, loaded once for the whole batch.

    llama.cpp answers every failed load with the same `ValueError("Failed to
    load model from file: <path>")`, whether the file is corrupt, the card ran
    out of memory or the build cannot read the quantisation. That message
    carries nothing but a path, which redaction then removes, and its class is
    the one the server reads as "the caller's fault". So what was ATTEMPTED is
    logged here, and the error is re-raised as the server-side fault it is.
    """
    # Outside the `try` on purpose: a missing `llama_cpp` is already a
    # ToolUnavailableError, raised by `require` and naming the package.
    llama_cpp = engine_module()
    attempted = _load_settings(llama_cpp, model_file, context_tokens, n_gpu_layers)
    # Logged BEFORE the call as well as on failure: a load that kills the
    # process outright (a segfault in the CUDA backend) raises nothing, and
    # this line is then the only record of what was being tried.
    logger.info("loading the language model: %s", attempted)
    failure = None
    with quiet_stderr() as captured:
        try:
            engine = llama_cpp.Llama(
                model_path=str(model_file),
                n_ctx=context_tokens,
                n_gpu_layers=n_gpu_layers,
                seed=seed,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001 -- re-raised below, classified.
            # Handled after the `with`, once stderr is back: a logging handler
            # that writes to `sys.stderr` would otherwise log into the buffer.
            failure = exc
    if failure is None:
        return engine

    kind = type(failure).__name__
    reason = _last_line(str(failure)) or kind
    logger.error(
        "the language model could not be loaded (%s: %s); attempted %s",
        kind, reason, attempted,
    )
    tail = [line for line in captured.getvalue().splitlines()
            if line.strip()][-_LOADER_TAIL_LINES:]
    if tail:
        logger.warning("llama.cpp loader output: %s", " | ".join(tail))
    # Which class decides what the server answers. A model file that is no
    # longer there is a deployment that has lost its weights: nothing the
    # caller sends will bring them back, so ToolUnavailableError and a 503.
    # Anything else -- a corrupt or truncated file, a quantisation this build
    # cannot read, the card out of memory -- is this server crashing mid-run,
    # so a plain RuntimeError and a 500. Never the ValueError llama.cpp
    # raised: that would be answered 422 and blamed on the request.
    if not Path(model_file).is_file():
        raise ToolUnavailableError(
            f"the language model is no longer on this server ({kind}: "
            f"{reason}). Stage it again with `setup-models.sh --tool CNE`."
        ) from failure
    raise RuntimeError(
        f"loading the language model failed ({kind}: {reason}); "
        f"n_ctx={context_tokens}, n_gpu_layers={n_gpu_layers}"
    ) from failure


def _load_settings(llama_cpp, model_file, context_tokens: int,
                   n_gpu_layers: int) -> str:
    """What a load is about to ask of llama.cpp, in one loggable line.

    The size says whether the file is the whole model or a truncated copy; the
    GPU fields say whether `n_gpu_layers` could have meant anything on this
    build. No path: it is an argument value, and redaction would drop it.
    """
    try:
        size = f"{Path(model_file).stat().st_size / 1e6:.0f} MB"
    except OSError:
        size = "size unknown (the file cannot be read)"
    try:
        offload = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:  # noqa: BLE001 -- a diagnostic must not mask the load.
        offload = "unknown"
    return (
        f"model {size}, n_ctx={context_tokens}, n_gpu_layers={n_gpu_layers}, "
        f"gpu offload compiled in={offload}"
    )


def _last_line(text: str) -> str:
    """The last non-blank line of a message, which is where the cause sits."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def complete(model, messages: list, max_tokens: int, temperature: float) -> tuple:
    """One answer, as `(text, finish_reason)`.

    `finish_reason` is the half upstream threw away. "length" means the answer
    was cut off at `max_tokens` mid-sentence -- and, for a model asked for JSON,
    mid-object. The caller is what refuses to write that.
    """
    output = model.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    try:
        choice = output["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        # A RuntimeError, not a ValueError: the note was fine, the ENGINE
        # answered in a shape it should not have, and that is this server's
        # fault rather than the caller's.
        raise RuntimeError(
            f"the model returned an answer CNE cannot read "
            f"({type(error).__name__}: {error})"
        ) from error
    return (content or "").strip(), choice.get("finish_reason")


# What a `key: value` line looks like. The key is a field name, so it is
# lowercase words and underscores and nothing else -- deliberately narrow, so
# that a sentence containing a colon ("ASSESSMENT: right TMJ derangement") is
# not read as a field.
_FIELD_LINE = re.compile(r"^\s*([a-z][a-z0-9_ ]{0,60}?)\s*:\s*(.+?)\s*$")

# How much of the answer has to look like fields before it is treated as one.
# A model that answers in prose with one stray `note: ...` line has not
# produced an extraction.
_FIELD_LINE_RATIO = 0.6


def parse_extraction(answer: str):
    """The structured extraction in the model's answer, or None.

    Two shapes, because the fine-tunes produce two. JSON is tried first --
    the fine-tunes wrap it in a sentence often enough that upstream already
    sliced between the first `{` and the last `}`, and that is kept. Failing
    that, the answer is read as `key: value` lines, which is what the
    published TMJ fine-tune actually emits: 46 fields, no JSON anywhere.

    Upstream reached the same two shapes by accident -- it parsed JSON and
    wrote the RAW ANSWER when that failed -- which happened to work for the
    plain-text model and silently wrote a truncated blob for everything else.
    Here both shapes are parsed, and anything that is neither is None, which
    means "no extraction here" and the caller writes nothing.
    """
    if not answer:
        return None
    parsed = _parse_json_object(answer)
    if parsed is not None:
        return parsed
    return _parse_field_lines(answer)


def _parse_json_object(answer: str):
    """The JSON object in the answer, or None."""
    start = answer.find("{")
    end = answer.rfind("}") + 1
    if start == -1 or end == 0 or end <= start:
        return None
    try:
        data = json.loads(answer[start:end])
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    if EXTRACTION_KEY in data and isinstance(data[EXTRACTION_KEY], dict):
        data = data[EXTRACTION_KEY]
    return data or None


def _parse_field_lines(answer: str):
    """`key: value` lines, or None if the answer is not mostly fields.

    A repeated key keeps its FIRST value rather than its last. The published
    TMJ fine-tune repeats `pain_relieving_factors` in a single answer, and a
    plain dict build would silently keep whichever came last.
    """
    lines = [line for line in answer.splitlines() if line.strip()]
    if not lines:
        return None

    data = {}
    matched = 0
    for line in lines:
        found = _FIELD_LINE.match(line)
        if not found:
            continue
        matched += 1
        key = found.group(1).strip().replace(" ", "_")
        data.setdefault(key, found.group(2).strip())

    if not data or matched < _FIELD_LINE_RATIO * len(lines):
        return None
    return data


def render_extraction(data: dict) -> str:
    """The `key : value` text upstream produced, and this still writes.

    It is the readable half of the output; the JSON beside it is the
    machine-readable one. Upstream had only this, so every downstream consumer
    had to parse a format with no escaping -- a value containing " : " is
    already ambiguous, which is why the JSON is the canonical artifact now.
    """
    lines = []
    for key, value in data.items():
        lines.append(f"{key} : {_render_value(value)}")
    return "\n".join(lines) + "\n"


def _render_value(value) -> str:
    """One value, readably. A list is joined; anything nested stays JSON."""
    if isinstance(value, (list, tuple)):
        return ", ".join(_render_value(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


__all__ = [
    "CONTEXT_TOKENS",
    "INSTRUCTION_TMJ",
    "MODEL_EXTENSION",
    "SYSTEM_PROMPTS",
    "TRUNCATED",
    "ToolUnavailableError",
    "build_messages",
    "complete",
    "context_for",
    "engine_module",
    "load_model",
    "model_type_hint",
    "parse_extraction",
    "quiet_stderr",
    "render_extraction",
    "resolve_model_file",
    "supports_gpu_offload",
]
