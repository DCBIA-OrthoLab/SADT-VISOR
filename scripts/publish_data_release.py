#!/usr/bin/env python3
"""Republish every manifest file as ONE release of the organisation's data repository.

    python3 scripts/publish_data_release.py --dry-run
    python3 scripts/publish_data_release.py --draft --only ROI.mrk.zip --only Gold_file.zip
    python3 scripts/publish_data_release.py

Each file is downloaded from the address the manifest lists, ANONYMOUSLY: no
token, no cookie. A file only an authenticated client can read has never been
public, and is refused rather than published. Nothing is ever taken from a
local disk, so nothing that was not already distributed can end up in the
release. The bytes uploaded are the bytes downloaded; their sha256 is computed
on the way.

The release holds each distinct file once. Two different files that share a
name (four nnUNet bundles each ship a `checkpoint_final.pth`) are told apart by
a prefix naming their original release; the manifest's `name` is unchanged, so
every file still lands under DATA/ exactly where it did.

Beside the files the release carries:
  SHA256SUMS   one line per file, `sha256sum -c` format
  SOURCES.md   for each file: its original public address, the date it was
               first published there, and the tools that use it

and the run writes `data-manifest.yml` with every republished entry's `url`
pointed at the new release and its `sha256` pinned. Review it, then commit it.

Tags are the publication date plus a counter, `2026.10.09-1`, so several
releases can be cut on one day. Needs the GitHub CLI (`gh`), logged in with
the right to create releases in --repo.
"""

import argparse
import collections
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_data  # noqa: E402

_GITHUB_ASSET = re.compile(r"https://github\.com/([^/]+)/([^/]+)/releases/download/([^/]+)/([^/]+)$")
_HEADERS = {"User-Agent": "visor-data-publish/1.0"}


def _anonymous(url: str):
    # Deliberately no Authorization header, whatever the environment holds:
    # what this can read is what anybody can read.
    return urllib.request.urlopen(urllib.request.Request(url, headers=_HEADERS), timeout=60)


def _plan(manifest: dict, excluded_tools: set) -> list:
    """One item per distinct source URL, with the tools and kinds using it."""
    by_url = collections.OrderedDict()
    for kind in fetch_data.KINDS:
        for entry in fetch_data._entries(manifest, kind, None):
            if entry["tool"] in excluded_tools:
                continue
            item = by_url.setdefault(entry["url"], {"url": entry["url"], "name": entry["name"],
                                                    "tools": set(), "kinds": set()})
            item["tools"].add(entry["tool"])
            item["kinds"].add(kind)
    items = list(by_url.values())
    taken = collections.Counter(item["name"] for item in items)
    for item in items:
        item["asset"] = item["name"] if taken[item["name"]] == 1 else f"{_origin(item['url'])}_{item['name']}"
    clashes = [name for name, n in collections.Counter(i["asset"] for i in items).items() if n > 1]
    if clashes:
        raise SystemExit(f"two different files would share an asset name: {clashes}")
    return items


def _origin(url: str) -> str:
    """A short, stable name for where a file comes from: its release tag, or
    the repository when the tag says nothing (`v1.0.0`, `TestFiles`)."""
    match = _GITHUB_ASSET.match(url)
    if not match:
        return urllib.parse.urlparse(url).netloc.split(".")[0]
    _owner, repo, tag, _asset = match.groups()
    return repo if re.fullmatch(r"v?[\d.]+|TestFiles?|main", tag) else tag


_RELEASES = {}


def _first_published(url: str) -> str:
    """The date the source release was published.

    Read through `gh`, whose quota is the logged-in user's: the anonymous API
    allows 60 calls an hour. Only this metadata is read that way -- the file
    itself is still downloaded anonymously, and a draft release, which only a
    logged-in reader can see, is refused here.
    """
    match = _GITHUB_ASSET.match(url)
    if not match:
        return "in the source tree"
    owner, repo, tag, _asset = match.groups()
    key = (owner, repo, tag)
    if key not in _RELEASES:
        try:
            _RELEASES[key] = json.loads(_gh("api", f"repos/{owner}/{repo}/releases/tags/{tag}", capture=True))
        except subprocess.CalledProcessError:
            _RELEASES[key] = {}
    release = _RELEASES[key]
    if release.get("draft"):
        raise SystemExit(f"{url} belongs to a draft release: it was never public")
    return (release.get("published_at") or "unknown")[:10]


def _download(url: str, destination: str) -> tuple:
    digest, size = hashlib.sha256(), 0
    with _anonymous(url) as response, open(destination, "wb") as out:
        for chunk in iter(lambda: response.read(1 << 20), b""):
            out.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _gh(*args, capture=False) -> str:
    result = subprocess.run(["gh", *args], check=True, text=True,
                            stdout=subprocess.PIPE if capture else None)
    return result.stdout if capture else ""


def _next_tag(repo: str, day: str) -> str:
    """`<day>-<n>`, n one past the highest already used that day (drafts included)."""
    # Through the API rather than `gh release list`, whose --json is missing
    # from the gh that distributions still ship.
    listed = _gh("api", "--paginate", f"repos/{repo}/releases", "--jq", ".[].tag_name",
                 capture=True).split()
    used = [int(tag.rsplit("-", 1)[1]) for tag in listed
            if tag.startswith(day + "-") and tag.rsplit("-", 1)[1].isdigit()]
    return f"{day}-{max(used, default=0) + 1}"


def _rewrite_manifest(text: str, moved: dict) -> str:
    """Point each republished `url:` line at its new address and pin its sha256.

    Line-based on purpose: the manifest's comments are its documentation, and
    a YAML round trip would drop them.
    """
    out, lines, i = [], text.splitlines(keepends=True), 0
    while i < len(lines):
        line = lines[i]
        match = re.match(r"^(\s*)url:\s*(\S+)\s*$", line)
        if not match or match.group(2) not in moved:
            out.append(line)
            i += 1
            continue
        indent, old = match.groups()
        new_url, sha256 = moved[old]
        out.append(f"{indent}url: {new_url}\n")
        i += 1
        # The entry's other keys follow at the same indent; drop an old pin.
        block = []
        while i < len(lines) and re.match(rf"^{indent}(?!- )\w", lines[i]):
            if not re.match(rf"^{indent}sha256:", lines[i]):
                block.append(lines[i])
            i += 1
        out.extend(block)
        out.append(f"{indent}sha256: {sha256}\n")
    return "".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default="DCBIA-OrthoLab/VISOR-data")
    parser.add_argument("--manifest", default=fetch_data._DEFAULT_MANIFEST)
    parser.add_argument("--exclude-tool", action="append", default=[],
                        help="Leave a tool out entirely (repeatable), e.g. one not ported yet.")
    parser.add_argument("--only", action="append",
                        help="Republish only these source file names (repeatable): a trial run.")
    parser.add_argument("--draft", action="store_true", help="Create the release as a draft.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and check every source is public; download nothing.")
    parser.add_argument("--workdir", help="Where files are downloaded before upload. Default: a temp dir.")
    parser.add_argument("--manifest-out", default="data-manifest.yml",
                        help="Where to write the updated manifest. Default: ./data-manifest.yml")
    args = parser.parse_args(argv)

    with open(args.manifest, encoding="utf-8") as handle:
        manifest_text = handle.read()
    items = _plan(fetch_data._parse_manifest(args.manifest), set(args.exclude_tool))
    if args.only:
        items = [item for item in items if item["name"] in args.only]
        missing = set(args.only) - {item["name"] for item in items}
        if missing:
            raise SystemExit(f"--only names no manifest file: {sorted(missing)}")

    for item in items:
        item["published"] = _first_published(item["url"])
        print(f"  {item['asset']:<52} {item['published']:<18} {','.join(sorted(item['tools']))}")
    print(f"{len(items)} file(s)")
    unreachable = []
    for item in items:
        try:
            with _anonymous(item["url"]) as response:
                response.read(1)
        except OSError as exc:
            unreachable.append(f"{item['url']}: {exc}")
    if unreachable:
        print("not readable anonymously, so not provably public -- nothing published:")
        print("\n".join(f"  {line}" for line in unreachable))
        return 1
    print("every source answers an anonymous request: all are public")
    if args.dry_run:
        return 0

    day = datetime.date.today().strftime("%Y.%m.%d")
    tag = _next_tag(args.repo, day)
    workdir = args.workdir or tempfile.mkdtemp(prefix="visor-data-")
    os.makedirs(workdir, exist_ok=True)

    for item in items:
        path = os.path.join(workdir, item["asset"])
        item["path"] = path
        item["sha256"], item["size"] = _download(item["url"], path)
        print(f"  downloaded {item['asset']}  {item['size']} B  {item['sha256'][:12]}")

    # Identical bytes behind two addresses are one asset.
    by_sha = collections.OrderedDict()
    for item in items:
        keeper = by_sha.setdefault(item["sha256"], item)
        if keeper is not item:
            keeper["tools"] |= item["tools"]
            item["asset"] = keeper["asset"]
    assets = list(by_sha.values())

    sums = os.path.join(workdir, "SHA256SUMS")
    with open(sums, "w", encoding="utf-8") as handle:
        handle.writelines(f"{a['sha256']}  {a['asset']}\n" for a in assets)
    sources = os.path.join(workdir, "SOURCES.md")
    with open(sources, "w", encoding="utf-8") as handle:
        handle.write(f"# Sources of {tag}\n\nEvery file here was already public, and was "
                     "downloaded anonymously from the address below before being republished "
                     "unchanged.\n\n| File | sha256 | Bytes | First published | Original address | Used by |\n"
                     "|---|---|---|---|---|---|\n")
        for item in items:
            handle.write(f"| `{item['asset']}` | `{item['sha256'][:16]}…` | {item['size']} | "
                         f"{item['published']} | {item['url']} | {', '.join(sorted(item['tools']))} |\n")

    notes = os.path.join(workdir, "NOTES.md")
    with open(notes, "w", encoding="utf-8") as handle:
        handle.write(f"Models and test files for the VISOR tools, {len(assets)} files.\n\n"
                     "Every file was already public; `SOURCES.md` gives its original address and "
                     "date. `SHA256SUMS` verifies them (`sha256sum -c SHA256SUMS`).\n")
    # Every file handed to `create` itself: a separate `upload` looks the
    # release up by its tag, which a draft does not have yet, and failed.
    create = ["release", "create", tag, "-R", args.repo, "--title", tag, "--notes-file", notes,
              *(["--draft"] if args.draft else []), *(a["path"] for a in assets), sums, sources]
    _gh(*create)

    base = f"https://github.com/{args.repo}/releases/download/{tag}"
    moved = {item["url"]: (f"{base}/{urllib.parse.quote(item['asset'])}", item["sha256"]) for item in items}
    with open(args.manifest_out, "w", encoding="utf-8") as handle:
        handle.write(_rewrite_manifest(manifest_text, moved))
    print(f"\nrelease {tag}{' (draft)' if args.draft else ''}: {len(assets)} file(s)")
    print(f"manifest with the new addresses: {args.manifest_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
