#!/bin/sh
# Download the tools' AI models and test files, as scripts/data-manifest.yml
# lists them, into the server's DATA/ directory.
#
# From anywhere, without cloning:
#
#   curl -fsSL https://raw.githubusercontent.com/DCBIA-OrthoLab/SADT-VISOR/main/scripts/setup-data.sh | sh
#
# Arguments reach it through `sh -s --` when piping:
#
#   curl -fsSL .../setup-data.sh | sh -s -- --tool AMASSS --tool ALI
#   curl -fsSL .../setup-data.sh | sh -s -- --kind testfiles
#
# Options (passed straight to fetch_data.py):
#   --tool NAME     restrict to this tool, repeatable. Default: every tool.
#   --kind KIND     models | testfiles. Repeatable. Default: both.
#   --data-dir DIR  destination root. Default: ./DATA, or $DATA_DIR.
#   --force         re-download even what is already present.
#   --list          print what WOULD be fetched, and stop.
#
# Environment:
#   DATA_DIR   destination root. A server deployment points this at its own
#              DATA/, which is the folder server/data_store.py reads.
#   REPO/REF   where to fetch the engine + manifest from when this script is
#              piped rather than run from a checkout.
#
# WHY THIS LIVES HERE and not in the server repository. The manifest is a list
# of which bundles belong to AMASSS, which weights to ALI, which gold
# references to ASO -- 43 kB of dental knowledge. The server is built around
# knowing no dental tool (`scripts/domain_coupling.py` over there measures
# that claim), and a manifest naming every structure code is exactly the kind
# of knowledge that belongs beside the tools it describes. The server asks for
# data by tool name; what a tool's data IS, is the tool's business.
#
# Everything already on disk is skipped, so re-running is both "resume" and
# "add one more tool". The layout produced is DATA/<tool>/{models,testfiles}/.

set -eu

REPO="${REPO:-DCBIA-OrthoLab/SADT-VISOR}"
REF="${REF:-main}"
RAW="https://raw.githubusercontent.com/${REPO}/${REF}/scripts"

if ! command -v python3 >/dev/null 2>&1; then
    echo "setup-data: python3 is required but was not found in PATH." >&2
    echo "  Debian/Ubuntu: sudo apt-get install -y python3" >&2
    exit 1
fi

# From a checkout when there is one, so a local edit to the manifest is what
# takes effect; otherwise pull both files into a temp dir.
if [ -f "./scripts/fetch_data.py" ] && [ -f "./scripts/data-manifest.yml" ]; then
    exec python3 ./scripts/fetch_data.py "$@"
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "setup-data: curl is required but was not found in PATH." >&2
    exit 1
fi

work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT INT TERM

fetch() {
    if ! curl -fsSL "${RAW}/$1" -o "${work_dir}/$1"; then
        echo "setup-data: could not download $1 from ${REPO}@${REF}." >&2
        echo "  Check that the branch exists and carries scripts/$1," >&2
        echo "  or point elsewhere with: REF=<branch> REPO=<owner/repo>" >&2
        exit 1
    fi
}

echo "Fetching the download engine and the manifest from ${REPO}@${REF}..."
fetch fetch_data.py
fetch data-manifest.yml

python3 "${work_dir}/fetch_data.py" --manifest "${work_dir}/data-manifest.yml" "$@"
