"""A tool another tool calls is not told a device, so its default is the device.

`sup.run(...)` from a parent that declares no `device` sends none, and the
callee then runs with the default of its own `run()`. On macOS a default of
`cuda` is a request nobody can honour: the engine falls back to the CPU, which
measured ~18x slower than MPS for a segmentation. Elsewhere it stays `cpu`: the llama.cpp build pinned there offloads nothing.
"""

import subprocess
import sys

import pytest

# The default is fixed when the package is imported, like the published
# choices, so each platform is simulated in a fresh interpreter. Reloading the
# package here would leave every other test holding a stale `run`.
_PROBE = (
    "import inspect, sys; sys.platform = {platform!r}; import sadt_cne; "
    "print(inspect.signature(sadt_cne.run).parameters['device'].default)"
)


@pytest.mark.parametrize("platform, expected", [
    ("linux", "cpu"),
    ("win32", "cpu"),
    ("darwin", "mps"),
])
def test_the_default_device_is_the_one_the_platform_can_use(platform, expected):
    finished = subprocess.run(
        [sys.executable, "-c", _PROBE.format(platform=platform)],
        capture_output=True, text=True, check=False,
    )
    assert finished.returncode == 0, finished.stderr
    assert finished.stdout.strip() == expected
