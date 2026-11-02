"""The modules every tool carries a copy of must stay byte-identical.

`progress.py` and `scans.py` are vendored into each tool, and
`metrics_table.py` into each that writes a metrics spreadsheet, rather than
imported from a shared package, because no package is common to every tool: they are
isolated projects on purpose, and an import between two of them would make one
tool's missing dependency take out the other. CONTRIBUTING.md sets that rule.

But what these files hold is a FORMAT, not an implementation. `progress.py`
writes the one-JSON-object-per-line stream the SERVER parses, and `scans.py`
names the extensions a caller's data is recognised by. A copy that drifts does
not fail loudly -- it reports progress the server silently ignores, or accepts a
scan its neighbour refuses, or a spreadsheet one tool writes that another
does not. Copying is the accepted cost; drifting is not, and
this is what stops it.

When a copy has to change, change every copy.
"""

import hashlib
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"

# Vendored by copy, and required to stay identical. Keyed by file name, since
# that is how each lands in its own tool's package.
VENDORED = ("progress.py", "scans.py", "metrics_table.py")


def copies(name):
    """Every vendored copy of `name`, sorted, virtualenvs excluded."""
    return sorted(
        path
        for path in TOOLS.rglob("src/*/" + name)
        if ".venv" not in path.parts
    )


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("name", VENDORED)
def test_every_copy_is_identical(name):
    found = copies(name)
    if len(found) < 2:
        pytest.skip("{} is vendored into fewer than two tools".format(name))

    by_digest = {}
    for path in found:
        by_digest.setdefault(digest(path), []).append(
            str(path.relative_to(TOOLS))
        )

    assert len(by_digest) == 1, (
        "{} has drifted between tools. It is vendored by copy, so a change to "
        "one copy has to be made to all of them. Versions found:\n".format(name)
        + "\n".join(
            "  {}\n    {}".format(short, "\n    ".join(paths))
            for short, paths in (
                (key[:12], value) for key, value in sorted(by_digest.items())
            )
        )
    )
