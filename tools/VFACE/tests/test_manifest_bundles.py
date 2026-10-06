"""Every bundle VFACE resolves on its own is one the manifest stages for it.

`dispatch._BUNDLES` names folders under DATA/VFACE/models/, and the manifest
is what puts folders there. The two drifted silently: the classifier was
unpacked as `V_FACE_Models` while the code asked for `VFACE_classifier`, and
four bundles were never staged for this tool at all -- so every run refused at
the door on a deployment that had downloaded everything the manifest listed.
Read with the fetcher's own parser and path rule, so this checks the folder
names that land on disk, not a second reading of the file.
"""

import importlib.util
import os

import pytest

from sadt_vface import dispatch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FETCH_DATA = os.path.join(REPO, "scripts", "fetch_data.py")
MANIFEST = os.path.join(REPO, "scripts", "data-manifest.yml")


def _fetcher():
    if not (os.path.isfile(FETCH_DATA) and os.path.isfile(MANIFEST)):
        pytest.skip("the manifest ships with the repository, not with this package")
    spec = importlib.util.spec_from_file_location("fetch_data", FETCH_DATA)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_bundle_vface_resolves_is_staged_under_that_name():
    fetcher = _fetcher()
    manifest = fetcher._parse_manifest(MANIFEST)
    root = os.path.join("DATA", "VFACE", "models")
    staged = {os.path.relpath(fetcher._target_path("DATA", entry), root)
              for entry in fetcher._entries(manifest, "models", ["VFACE"])}

    missing = {argument: name for argument, name in dispatch._BUNDLES.items()
               if name not in staged}
    assert not missing, f"named in _BUNDLES but staged by no VFACE manifest entry: {missing}"
