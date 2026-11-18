"""Importing a third-party package late, and saying so when it is not there.

Every heavy or optional import in this tool goes through `require`. Two reasons,
both of them defects upstream had:

* **the schema must cost nothing.** CI imports this package on every pull
  request to publish the schema, and the server regenerates it at startup with
  the tool's own interpreter. Neither may pay for loading an inference engine.
* **a missing reader must cost the notes that need it, not the run.** Upstream
  called `sys.exit(1)` when `pymupdf` OR `python-docx` was absent
  (`CNE_CLI.py:105-115`), so a folder of forty `.txt` notes could not be
  extracted because a PDF reader nobody was going to use was missing.

It also holds `preload_cuda_runtime`, which is the same idea one layer down: a
shared library the wheel links but does not ship, opened before the thing that
needs it, and never fatal when this deployment does not need it at all.
"""

import ctypes
import importlib
import importlib.util
import logging
import os

logger = logging.getLogger("CNE")


class ToolUnavailableError(RuntimeError):
    """This deployment cannot do it, and no request will change that.

    The server maps an exception by its class NAME -- there is no shared base
    class to inherit, because there is no package shared with the server -- and
    this name answers 503 with the message. That is the right answer for a
    missing dependency: the request was valid, the reason names a package, and
    nothing the caller sends will help.

    A `RuntimeError` rather than a bare `Exception` so that anything catching
    server-side faults by that class catches this one too: it is never the
    caller's fault, and must never be mistaken for a `ValueError`.
    """


def require(module_name: str, package_name: str):
    """Import `module_name`, or say which package to install.

    `importlib.import_module` rather than a bare `import` so a test can replace
    it: "a missing reader fails only the notes that need it" is a behaviour
    worth pinning, and it cannot be pinned by uninstalling a dependency.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise ToolUnavailableError(
            f"This server cannot read that note: the '{package_name}' package is not "
            f"installed (import {module_name} failed: {error})."
        ) from error


# The CUDA runtime this deployment's llama.cpp needs, in the order it needs it:
# cuBLAS is linked against cuBLASLt, so that one is opened first. Each name is
# the package that ships the library and the file inside its `lib/` directory.
CUDA_RUNTIME_LIBRARIES = (
    ("nvidia.cuda_runtime", "libcudart.so.12"),
    ("nvidia.cublas", "libcublasLt.so.12"),
    ("nvidia.cublas", "libcublas.so.12"),
)


def preload_cuda_runtime() -> list:
    """Open the CUDA runtime out of this virtualenv, before llama.cpp needs it.

    The CUDA build of `llama-cpp-python` LINKS `libcudart.so.12` and
    `libcublas.so.12` and SHIPS NEITHER -- auditwheel excludes the CUDA runtime
    from a wheel by policy, the way it excludes the driver. On a workstation
    with a CUDA toolkit installed the dynamic loader finds them anyway, which is
    exactly why this is easy to miss: the deployment image is `python:3.13-slim`
    and has no CUDA at all, and there `import llama_cpp` dies on

        OSError: libcudart.so.12: cannot open shared object file

    before a single line of this tool runs. Measured, in that image, with the
    card attached.

    So they are pinned in `pyproject.toml` as `nvidia-cuda-runtime-cu12` and
    `nvidia-cublas-cu12` -- the same wheels at the same versions every torch
    tool in the image already carries, so the image hardlinks them instead of
    keeping a second copy -- and opened here with `RTLD_GLOBAL`, so that the
    `dlopen` llama.cpp does moments later resolves against them. It is what
    `import torch` does, for the same reason.

    **Best effort, deliberately.** A CPU-wheel deployment has neither the
    packages nor any need of them, and must not fail here; nor must a
    workstation whose loader would have found the system copy regardless.
    Returns the paths it opened, which is both what the tests assert on and
    what says, in one value, which of the two deployments this is.
    """
    opened = []
    for package, library in CUDA_RUNTIME_LIBRARIES:
        try:
            spec = importlib.util.find_spec(package)
        except (ImportError, ValueError):
            continue
        locations = list(getattr(spec, "submodule_search_locations", None) or [])
        for location in locations:
            candidate = os.path.join(location, "lib", library)
            if not os.path.exists(candidate):
                continue
            try:
                ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            except OSError as error:
                # Not fatal: llama.cpp is about to try the system copy, and the
                # error it raises then names the library far more usefully than
                # anything that could be raised here. But logged at WARNING, not
                # DEBUG: the library was SHIPPED here and would not open, which
                # is a broken deployment, and the import error that follows is
                # unreadable without this line beside it.
                logger.warning(
                    "could not preload the CUDA runtime library %s "
                    "(OSError: %s); llama.cpp will look for a system copy",
                    library, error,
                )
                continue
            opened.append(candidate)
            break
    return opened
