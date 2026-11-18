"""Which input files go together, for a client that splits a cohort into batches.

A tool that takes two paired folders -- a baseline and a follow-up -- cannot be
split one folder at a time: cut T1 into five pieces and send T2 whole, and
every batch registers patients against whoever happens to share their name in
the other folder. So the split has to be decided by the pairing itself, and the
pairing is the TOOL's: each tool in this family names its patients its own way,
and a client re-implementing those rules would drift from them silently.

`pairs_by_name` runs the tool's real pairing on a tree of EMPTY files carrying
the names the client listed. Nothing is uploaded and nothing is read; the
answer is what `run()` would pair if it were handed those folders.

The unit a client moves is an ENTRY: a direct child of the folder the tool
actually reads (after it descends into `<cohort>/T1`, when it does). A patient
whose files sit in a subfolder moves with that subfolder, its masks and DICOM
slices with it. Entries holding several patients stay together; entries
holding none -- a README, a folder of unrelated files -- travel with every
batch, which costs bytes and never a wrong pairing.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Callable, Dict, Iterable, List

from .pairing import SCAN_EXTENSIONS

# A file with no extension, or a `.dcm`, is a DICOM slice: tools in this family
# convert a folder of them into ONE scan named after the folder, so that is
# what the placeholder tree shows the pairing.
_DICOM_EXTENSIONS = (".dcm", ".ima", "")
_PLACEHOLDER_SCAN = ".nii.gz"


def _is_dicom_slice(name: str, known: Iterable[str]) -> bool:
    lowered = name.lower()
    if any(lowered.endswith(ext) for ext in known):
        return False
    return os.path.splitext(lowered)[1] in _DICOM_EXTENSIONS


def _safe(relative: str) -> str:
    """A listed name as a path inside the placeholder tree, or "" to skip it."""
    parts = [p for p in relative.replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        return ""
    return os.path.join(*parts)


def _materialise(root: str, names: Iterable[str], known_extensions) -> Dict[str, str]:
    """Write empty files for `names` under `root`. Returns {placeholder path:
    the folder it stands for} for every DICOM series folder replaced by one
    scan, so the answer can be mapped back to the slices."""
    folders: Dict[str, List[str]] = {}
    for name in names:
        relative = _safe(name)
        if not relative:
            continue
        if _is_dicom_slice(os.path.basename(relative), known_extensions):
            folders.setdefault(os.path.dirname(relative), []).append(relative)
            continue
        path = os.path.join(root, relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").close()
    placeholders = {}
    for folder in folders:
        if not folder:
            # Slices loose at the top: the whole input is one series, which a
            # tool converts to a scan named "scan".
            target = os.path.join(root, "scan" + _PLACEHOLDER_SCAN)
        else:
            target = os.path.join(root, folder + _PLACEHOLDER_SCAN)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        open(target, "w").close()
        placeholders[target] = os.path.join(root, folder) if folder else root
    return placeholders


def _entry_of(path: str, base: str, root: str) -> str:
    """The direct child of `base` that holds `path`, relative to `root` --
    the unit a client zips."""
    relative = os.path.relpath(path, base)
    first = relative.split(os.sep)[0]
    return os.path.relpath(os.path.join(base, first), root).replace(os.sep, "/")


def pairs_by_name(inputs: Dict[str, List[str]], match: Callable,
                  known_extensions=SCAN_EXTENSIONS) -> dict:
    """Group the listed files of each input by the patient the tool pairs them as.

    `inputs` is `{argument: [relative file path, ...]}` -- names only.
    `match(roots)` is the tool's own pairing, given `{argument: folder}` and
    returning `(bases, matched, unmatched)`:

    * `bases`: `{argument: folder the tool actually reads}` -- the root, or the
      subfolder it descends into;
    * `matched`: `{key: {argument: [paths]}}` for every patient it would run;
    * `unmatched`: `{argument: {key: [paths]}}` for those it would skip --
      with their files, so those are left out rather than sent with every
      batch.

    Returns `{"groups": [{"key", "entries": {argument: [entry]}}],
    "shared": {argument: [entry]}, "unpaired": {argument: [key]}}`, every entry
    a path relative to that input's root, as the client listed it.
    """
    workspace = tempfile.mkdtemp(prefix="sadt-pairs-")
    try:
        roots, placeholders = {}, {}
        for argument, names in inputs.items():
            root = os.path.join(workspace, argument)
            os.makedirs(root, exist_ok=True)
            roots[argument] = root
            placeholders.update(_materialise(root, names or [], known_extensions))

        bases, matched, unmatched = match(dict(roots))

        # Every entry of every input, so what no patient claims can be shared.
        entries: Dict[str, set] = {}
        for argument, root in roots.items():
            base = bases.get(argument, root)
            found = set()
            if os.path.isdir(base):
                for child in os.listdir(base):
                    full = os.path.join(base, child)
                    if full in placeholders:
                        continue    # stands for a DICOM folder, which is listed itself
                    found.add(os.path.relpath(full, root).replace(os.sep, "/"))
            entries[argument] = found

        # Patients sharing an entry cannot be separated: merge them.
        owner: Dict[tuple, str] = {}
        parent: Dict[str, str] = {key: key for key in matched}

        def find(key):
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        claimed: Dict[str, Dict[str, set]] = {}
        for key, sides in matched.items():
            for argument, paths in sides.items():
                root = roots[argument]
                base = bases.get(argument, root)
                for path in paths:
                    source = placeholders.get(path, path)
                    entry = _entry_of(source, base, root)
                    claimed.setdefault(key, {}).setdefault(argument, set()).add(entry)
                    other = owner.setdefault((argument, entry), key)
                    if other != key:
                        parent[find(key)] = find(other)

        groups: Dict[str, dict] = {}
        for key in sorted(matched):
            head = find(key)
            group = groups.setdefault(head, {"key": head, "keys": [], "entries": {}})
            group["keys"].append(key)
            for argument, found in claimed.get(key, {}).items():
                group["entries"].setdefault(argument, set()).update(found)

        used = {(argument, entry) for (argument, entry) in owner}
        skipped = set()
        for argument, keyed in (unmatched or {}).items():
            root = roots[argument]
            base = bases.get(argument, root)
            for paths in keyed.values():
                for path in paths:
                    skipped.add((argument, _entry_of(placeholders.get(path, path), base, root)))
        used |= skipped

        # What a run wrote beside a scan is named after it -- `P1_T1_Or_SegOut/`
        # next to `P1_T1_Or.nii.gz` -- and goes where the scan goes: with its
        # patient, or nowhere when the patient is skipped.
        shared = {}
        for argument, found in entries.items():
            left = []
            for entry in sorted(found):
                if (argument, entry) in used:
                    continue
                owner_entry = _named_after(entry, [e for (a, e) in used if a == argument])
                if owner_entry is None:
                    left.append(entry)
                    continue
                key = owner.get((argument, owner_entry))
                if key is not None:
                    group = groups[find(key)]
                    group["entries"].setdefault(argument, set()).add(entry)
            shared[argument] = left

        return {
            "groups": [{"key": g["key"], "keys": g["keys"],
                        "entries": {a: sorted(e) for a, e in g["entries"].items()}}
                       for g in groups.values()],
            "shared": {a: e for a, e in shared.items() if e},
            "unpaired": {a: sorted(k) for a, k in (unmatched or {}).items() if k},
        }
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _named_after(entry: str, candidates) -> "str | None":
    """The candidate `entry` is named after -- `P1_T1_Or_SegOut` after
    `P1_T1_Or.nii.gz` -- or None. The longest such name wins, so `P1_T1` never
    claims what belongs to `P1_T1_Or`."""
    name = entry.rsplit("/", 1)[-1]
    folder = entry[: -len(name)]
    best, best_len = None, 0
    for candidate in candidates:
        if not candidate.startswith(folder) or candidate == entry:
            continue
        stem = candidate.rsplit("/", 1)[-1]
        for extension in sorted(SCAN_EXTENSIONS, key=len, reverse=True):
            if stem.lower().endswith(extension):
                stem = stem[: -len(extension)]
                break
        else:
            stem = os.path.splitext(stem)[0]
        if stem and name.startswith(stem) and len(name) > len(stem) and name[len(stem)] in "_-." \
                and len(stem) > best_len:
            best, best_len = candidate, len(stem)
    return best
