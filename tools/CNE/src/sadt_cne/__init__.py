"""Extract structured findings from free-text clinical notes."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Literal

from . import progress
from .dependencies import ToolUnavailableError
from .extraction import (
    SYSTEM_PROMPTS,
    TRUNCATED,
    build_messages,
    complete,
    context_for,
    load_model,
    model_type_hint,
    parse_extraction,
    render_extraction,
    resolve_model_file,
    supports_gpu_offload,
)
from .readers import (
    NOTE_EXTENSIONS,
    discover_notes,
    looks_empty,
    read_note,
)

logger = logging.getLogger("CNE")

# What every output file is called: the source note's FULL name, extension
# included, under an `Extraction_` prefix. Upstream used the stem alone
# (`CNE_CLI.py:226`) while discovery accepted three extensions, so `B_001.txt`,
# `B_001.pdf` and `B_001.docx` in one folder all wrote `Extraction_B_001.txt`
# with `open(..., "w")`: two of the three extractions were destroyed and the run
# reported 3/3 processed.
OUTPUT_PREFIX = "Extraction_"

REPORT_NAME = "CNE_report.json"

__all__ = ["run"]


def run(
    notes: Path,
    notes_type: Literal["TMJ", "Ortho"],
    model: Path,
    output_dir: Path,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    context_tokens: int = 0,
    seed: int = 0,
    device: Literal["cpu", "cuda"] = "cpu",
) -> Path:
    """Extract structured findings from free-text clinical notes with a local LLM.

    Args:
        notes: One clinical note (.txt, .pdf or .docx), or a folder of them for
            a cohort. Folders are searched recursively and the extension is
            matched case-insensitively, so notes filed per patient and a
            `NOTE.PDF` off a Windows share are both found.
        notes_type: Which fine-tune is being run, "TMJ" or "Ortho". It selects
            the system prompt and the context window together, and it must match
            the model picked below -- they are two different models.
        model: The quantised GGUF model, named from what this server hosts. A
            hosted bundle folder holding exactly one .gguf resolves to it; one
            holding several is refused rather than resolved by sort order.
        output_dir: Where the extractions are written. Nothing is written
            outside it, and it may not be the notes folder itself.
        max_tokens: The longest answer the model may give per note. An answer
            cut off at this limit is reported as a failure and never written:
            truncated JSON parses as nothing, and upstream wrote the fragment as
            if it were the extraction.
        temperature: Sampling temperature. 0 makes the same note extract to the
            same answer twice, which is what lets a result be checked.
        context_tokens: The context window, in tokens. 0 means the note type's
            own default (TMJ 6144, Ortho 2048).
        seed: The sampler's seed, so a run above temperature 0 is still
            reproducible.
        device: "cpu" or "cuda". "cuda" offloads every layer of the model to
            the card, which is measured at 8-13x faster per note; it needs the
            CUDA build of llama.cpp, which this tool pins. Where that build is
            not the one installed, the request is reported as ignored --
            `gpu_offload` in the report is what actually happened -- rather than
            passing silently as it used to.

    Returns:
        The output directory, holding per note an `Extraction_<note>.json` and
        an `Extraction_<note>.txt`, plus `CNE_report.json`.
    """
    started = time.monotonic()
    notes = Path(notes)
    output_dir = Path(output_dir)

    if not notes.exists():
        raise FileNotFoundError(
            f"No notes at '{os.path.basename(str(notes))}'."
        )

    # Upstream let the two folders be the same, and its outputs matched its own
    # discovery glob, so a second run extracted from the first run's
    # extractions. Refused outright when they are the same directory; excluded
    # from discovery when the output merely sits inside the input.
    if notes.is_dir() and _same_directory(notes, output_dir):
        raise ValueError(
            "The output folder must not be the notes folder. Extractions written "
            "beside the notes are read back as notes by the next run."
        )

    # The engine creates its own output directory. Upstream's was created by the
    # Slicer panel (`CNE/CNE.py:578`) and never by the engine, so a headless run
    # raised FileNotFoundError on every single write -- inside the per-file
    # `except` that swallowed it -- and then reported the batch as complete.
    output_dir.mkdir(parents=True, exist_ok=True)

    found = discover_notes(notes, excluded=[output_dir])
    if not found:
        # Upstream exited 0 here (`CNE_CLI.py:135-139`), so a mistyped path was
        # reported to the user as a completed extraction. The server maps
        # ValueError to 422 with this message.
        raise ValueError(
            f"No note found in '{os.path.basename(str(notes))}'. CNE reads "
            f"{', '.join(NOTE_EXTENSIONS)} files, recursively."
        )

    root = notes.parent if notes.is_file() else notes
    destinations = plan_outputs(found, root, output_dir)

    model_file = resolve_model_file(model)
    warnings = []
    hint = model_type_hint(model_file)
    if hint and hint != notes_type:
        # Reported, not refused: a file name is not a contract. But pairing the
        # Ortho weights with the TMJ instruction yields an answer that looks
        # exactly like a good one, so it must not pass in silence.
        warnings.append(
            f"notes_type is '{notes_type}' but the model at "
            f"'{model_file.name}' names '{hint}'. Check they match."
        )
        logger.warning("%s", warnings[-1])
        # The clinician's copy says the same without the model's file name: it
        # is an argument value, and those stay out of the run's events.
        progress.log(
            f"the model chosen looks like the {hint} fine-tune, but the notes "
            f"were declared {notes_type}; check the extractions",
            "warning", user=True,
        )

    window = context_for(notes_type, context_tokens)
    wants_gpu = device.startswith("cuda")
    # Asked BEFORE the 4.4 GB load, so a deployment that cannot honour "cuda"
    # says so in the same second rather than after the weights are in memory.
    offloading = wants_gpu and supports_gpu_offload()
    if wants_gpu and not offloading:
        # The defect this closes: `device` has always been declared and always
        # been passed through as n_gpu_layers=-1, and on the CPU build of
        # llama.cpp that offloads nothing whatsoever. The run then reported
        # `device: cuda` over work done entirely on the CPU, and -- because the
        # server reads `device` to decide who takes the card -- held one of
        # MAX_CONCURRENT_GPU_JOBS while doing so. Reported, not refused: the
        # server fills `device` in from its own setting when the caller sends
        # none, so refusing would break every request to a CPU deployment
        # configured for a GPU one.
        warnings.append(
            "device is 'cuda' but the llama.cpp installed here has no GPU "
            "support compiled in, so this run was entirely on the CPU. Install "
            "the CUDA build of llama-cpp-python, or ask for device='cpu'."
        )
        logger.warning("%s", warnings[-1])
        # The operator's concern rather than the clinician's: the result is the
        # same, only slower, and the fix is in the deployment.
        progress.log(
            "a GPU was requested but this llama.cpp build has none; running "
            "on the CPU", "warning",
        )
    # The 4.4 GB load is one opaque call, so it is announced at the start of the
    # bar rather than given a share of it nothing here could measure.
    progress.emit(0.0, "loading the language model")
    engine = load_model(model_file, window, seed, -1 if wants_gpu else 0)
    logger.info(
        "CNE: %d note(s), type=%s, context=%d, max_tokens=%d, temperature=%.2f",
        len(found), notes_type, window, max_tokens, temperature,
    )

    report = {
        "tool": "CNE",
        "notes_type": notes_type,
        "model": model_file.name,
        "system_prompt": SYSTEM_PROMPTS[notes_type] is not None,
        "context_tokens": window,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "device": device,
        # What was ASKED FOR is `device`; this is what HAPPENED. They differ
        # exactly when a CUDA request met a CPU build, and the warning above
        # says so in words.
        "gpu_offload": offloading,
        "warnings": warnings,
        "notes": [],
    }
    written = []
    failures = []

    for index, note in enumerate(found, start=1):
        # One note is one call to the model, 2 to 176 s each as measured, so
        # the position in the batch is the only progress there is to report.
        progress.report(index, len(found), "note")
        relative = note.relative_to(root)
        entry = {"input": relative.as_posix()}
        try:
            _extract_one(
                engine, note, notes_type, destinations[note], output_dir,
                max_tokens, temperature, entry,
            )
            written.append(note)
        except Exception as exc:
            # One note failing costs one note. Upstream's blanket
            # `except Exception: sys.exit(1)` around the whole loop
            # (`:248-251`) meant the first bad file ended the cohort, and its
            # inner `except ... continue` counted failures into a tally that
            # was then printed over by an unconditional "All files processed
            # successfully!".
            #
            # Logged by position and by the KIND of failure only: the note's
            # path is patient metadata, and a reader's own exception can name
            # the file it could not open. The full reason goes to the report,
            # which returns to whoever sent the notes.
            logger.warning(
                "CNE could not extract from note %d of %d: %s",
                index, len(found), type(exc).__name__,
            )
            progress.log(
                f"note {index} of {len(found)} could not be extracted; the "
                f"report says why", "warning", user=True,
            )
            entry["status"] = "failed"
            entry["reason"] = f"{type(exc).__name__}: {exc}"
            failures.append(type(exc))
        report["notes"].append(entry)

    report["summary"] = f"{len(written)}/{len(found)} note(s) extracted"
    report["duration_seconds"] = round(time.monotonic() - started, 2)

    if not written:
        reasons = "; ".join(
            f"{note['input']}: {note.get('reason', 'unknown')}"
            for note in report["notes"]
        )
        # A batch that failed only because this deployment lacks a reader is a
        # deployment problem (503), not the caller's (422). Both messages name
        # what went wrong per note.
        if failures and all(kind is ToolUnavailableError for kind in failures):
            raise ToolUnavailableError(
                f"CNE could not read any of the notes it was given. {reasons}"
            )
        raise ValueError(
            f"CNE extracted nothing from any of the notes it was given. {reasons}"
        )

    (output_dir / REPORT_NAME).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return output_dir


def plan_outputs(found, root: Path, output_dir: Path) -> dict:
    """Where each note's two outputs go, checked to be distinct.

    The output tree mirrors the input tree and each name keeps its source
    extension, so neither `B_001.txt` beside `B_001.pdf` nor `a/x.txt` beside
    `b/x.txt` can write the same file. The uniqueness check is belt and braces:
    it costs nothing, and the failure it guards against destroys one patient's
    extraction while reporting success.
    """
    destinations = {}
    seen = {}
    for note in found:
        relative = Path(note).relative_to(root)
        stem = OUTPUT_PREFIX + relative.name
        pair = (
            output_dir / relative.parent / (stem + ".json"),
            output_dir / relative.parent / (stem + ".txt"),
        )
        key = tuple(str(path) for path in pair)
        if key in seen:
            raise ValueError(
                f"'{relative.as_posix()}' and '{seen[key]}' would both write "
                f"'{pair[1].name}'. Rename one of them."
            )
        seen[key] = relative.as_posix()
        destinations[note] = pair
    return destinations


def _extract_one(engine, note: Path, notes_type: str, destination, output_dir,
                 max_tokens: int, temperature: float, entry: dict) -> None:
    """One note, read to written. Raises rather than writing anything doubtful."""
    text, details = read_note(note)
    entry.update(details)
    entry["characters"] = len(text)

    if looks_empty(text):
        # THE defect this port exists for. A scanned, image-only PDF has no
        # text layer, so upstream sent an empty string to the model, which
        # answered with a plausible-looking extraction from nothing at all --
        # written to disk and counted as a success. There is no OCR here and
        # inventing one would be worse; the note is refused by name.
        raise ValueError(
            "no text could be extracted. A scanned, image-only PDF has no text "
            "layer, and CNE does not perform OCR."
        )

    answer, finish_reason = complete(
        engine, build_messages(notes_type, text), max_tokens, temperature
    )
    entry["finish_reason"] = finish_reason
    entry["answer_characters"] = len(answer)

    if finish_reason == TRUNCATED:
        raise ValueError(
            f"the model's answer was cut off at max_tokens={max_tokens}, so the "
            f"extraction is incomplete. Raise max_tokens and run this note again."
        )

    data = parse_extraction(answer)
    if data is None:
        raise ValueError(
            "the model's answer held no JSON object. Nothing was written: "
            "upstream saved the raw answer here, which reads exactly like a "
            "successful extraction."
        )

    json_path, text_path = destination
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    text_path.write_text(render_extraction(data), encoding="utf-8")

    entry["status"] = "ok"
    entry["fields"] = len(data)
    entry["output"] = json_path.relative_to(output_dir).as_posix()
    entry["output_text"] = text_path.relative_to(output_dir).as_posix()


def _same_directory(one: Path, other: Path) -> bool:
    """Whether two paths name the same directory, symlinks resolved."""
    return os.path.normcase(str(Path(one).resolve())) == os.path.normcase(
        str(Path(other).resolve())
    )
