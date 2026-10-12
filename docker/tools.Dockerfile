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

# The build stage IS the image. Copying ROOT into an empty image would save the
# 150 MB Debian base, but costs a second full copy of the virtualenvs while it
# builds; the server never runs anything from this image's own filesystem
# anyway, it only mounts ROOT out of it.
