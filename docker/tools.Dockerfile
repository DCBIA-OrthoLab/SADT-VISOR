# syntax=docker/dockerfile:1.7
#
# This library's tools, already installed: every tool's virtualenv and the
# Python they run on, at one fixed path. A VISOR server mounts the image there
# read-only (`--mount type=image`) instead of running `uv sync` for an hour.
#
#     docker build -f docker/tools.Dockerfile -t sadt-visor-tools .
#
# Why one fixed path: a virtualenv records absolute paths -- its interpreter in
# pyvenv.cfg and in every script's shebang. Built here at ROOT and mounted at
# ROOT, every one of them holds; mounted anywhere else, none does. The
# interpreter is inside ROOT for the same reason.
#
# The runtime contract with the server image: tools may count on the system
# libraries visor-serve installs (libgl1 libglib2.0-0 libgomp1 libx11-6
# libxext6 libxrender1) and on nothing else.

ARG ROOT=/opt/visor/libraries/sadt-visor

# ----------------------------------------------------------------------
# Manifests only
# ----------------------------------------------------------------------
# Every pyproject.toml and uv.lock, at their paths, and nothing else. This
# stage re-runs on any change to the repository, but its OUTPUT is content-
# addressed: as long as no manifest changed, it is byte-identical, so the
# dependency layer below is a cache hit -- and the layer a server already has.
FROM busybox:1.36 AS manifests
COPY . /ctx/
RUN mkdir -p /manifests && cd /ctx && \
    find . \( -name pyproject.toml -o -name uv.lock \) -not -path "*/.venv/*" | \
    while read -r manifest; do \
        mkdir -p "/manifests/$(dirname "$manifest")" && cp "$manifest" "/manifests/$manifest"; \
    done

FROM python:3.11-slim-bookworm
ARG ROOT
COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /usr/local/bin/uv
# The cache lives in the build's own filesystem, not in a cache mount: uv
# hardlinks every installed file to it, and a hardlink cannot cross into a
# mount. All nineteen virtualenvs then share one copy of torch and its CUDA
# wheels, and dropping the cache afterwards keeps the files the venvs link to.
# only-managed: never the base image's /usr/local/bin/python3.11. A venv built
# on it would work only where the same file sits at the same path -- which is
# true of the server image today by coincidence, and would break silently the
# day either base moves.
ENV UV_PYTHON_INSTALL_DIR=${ROOT}/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PYTHON_DOWNLOADS=automatic \
    UV_CACHE_DIR=/build-cache \
    UV_LINK_MODE=hardlink

# ----------------------------------------------------------------------
# Layer 1: every third-party dependency (~25 GB), from the lockfiles alone
# ----------------------------------------------------------------------
# Neither a tool's own code nor the shared packages beside it (AREG/common,
# ALI/common, naming): those are what change from one release to the next,
# and they go in the small layer below. A release that changes only code
# therefore rebuilds nothing here, and a server pulling it downloads nothing
# of this layer -- its digest is the one it already holds.
COPY --from=manifests /manifests/ ${ROOT}/
RUN set -eu; \
    cd "${ROOT}"; \
    for project in tools/*/pyproject.toml tools/*/*/pyproject.toml; do \
        grep -q "^\[tool.sadt\]" "$project" || continue; \
        tool=$(dirname "$project"); \
        echo "==> dependencies of $tool"; \
        ( cd "$tool" && uv sync --frozen --all-extras --no-dev --no-install-project --no-install-local ); \
    done; \
    rm -rf /build-cache

# ----------------------------------------------------------------------
# Layer 2: the code -- each tool and the shared packages it uses (a few MB)
# ----------------------------------------------------------------------
# The same sync, now with the sources present: everything above is already in
# place, so it only installs the tool itself and its local packages.
COPY . ${ROOT}/
RUN set -eu; \
    cd "${ROOT}"; \
    for project in tools/*/pyproject.toml tools/*/*/pyproject.toml; do \
        grep -q "^\[tool.sadt\]" "$project" || continue; \
        tool=$(dirname "$project"); \
        echo "==> $tool"; \
        ( cd "$tool" && uv sync --frozen --all-extras --no-dev ); \
    done; \
    rm -rf /build-cache

# Which release this is, readable two ways: as image labels, which a server's
# update agent reads with `docker image inspect` without starting anything,
# and as RELEASE.json beside the tools, which the server process can read
# through the mount. A local build that passes nothing is "local".
ARG RELEASE=local
ARG REVISION=unknown
ARG SOURCE=https://github.com/DCBIA-OrthoLab/SADT-VISOR
LABEL org.opencontainers.image.title="SADT-VISOR tools" \
      org.opencontainers.image.version="${RELEASE}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.source="${SOURCE}" \
      org.visor.library.root="${ROOT}"
RUN printf '{"library": "SADT-VISOR", "release": "%s", "revision": "%s", "source": "%s", "built": "%s"}\n' \
        "${RELEASE}" "${REVISION}" "${SOURCE}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${ROOT}/RELEASE.json"

# The build stage IS the image. Copying ROOT into an empty image would save the
# 150 MB Debian base, but costs a second full copy of the virtualenvs while it
# builds; the server never runs anything from this image's own filesystem
# anyway, it only mounts ROOT out of it.
