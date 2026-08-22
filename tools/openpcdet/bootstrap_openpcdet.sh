#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OPENPCDET_ROOT="$REPO_ROOT/third_party/OpenPCDet"
PINNED_COMMIT=233f849829b6ac19afb8af8837a0246890908755

# Print the pre-install state first, including the exact torch/CUDA versions.
python "$REPO_ROOT/tools/openpcdet/check_openpcdet_environment.py"
if [[ ! -d "$OPENPCDET_ROOT/.git" ]]; then
    mkdir -p "$REPO_ROOT/third_party"
    git clone https://github.com/open-mmlab/OpenPCDet.git "$OPENPCDET_ROOT"
fi
git -C "$OPENPCDET_ROOT" fetch origin "$PINNED_COMMIT"
git -C "$OPENPCDET_ROOT" checkout --detach "$PINNED_COMMIT"

ACTUAL_COMMIT=$(git -C "$OPENPCDET_ROOT" rev-parse HEAD)
if [[ "$ACTUAL_COMMIT" != "$PINNED_COMMIT" ]]; then
    echo "OpenPCDet checkout is $ACTUAL_COMMIT; expected $PINNED_COMMIT" >&2
    exit 1
fi

python -m pip install -r "$REPO_ROOT/requirements-openpcdet.txt"

# spconv-cu120 only publishes CPython wheels through 3.11.  The reconstruction
# environment currently uses Python 3.12, for which the maintained CUDA 12.6
# package publishes compatible Linux wheels.  CUDA 12 minor-version
# compatibility lets that package run with the newer CUDA driver/runtime used
# by the Pod.  Keep SPCONV_PACKAGE as an explicit escape hatch.
if [[ -z "${SPCONV_PACKAGE:-}" ]]; then
    PYTHON_MINOR=$(python -c 'import sys; print(sys.version_info.minor)')
    if (( PYTHON_MINOR >= 12 )); then
        SPCONV_PACKAGE="spconv-cu126>=2.3.8"
    else
        SPCONV_PACKAGE="spconv-cu120>=2.3"
    fi
fi
echo "Installing $SPCONV_PACKAGE"
python -m pip install "$SPCONV_PACKAGE"
# OpenPCDet's setup imports torch to build its CUDA extensions.  PEP 517's
# temporary build environment does not inherit the already-installed torch,
# so the editable build must deliberately use the active detector environment.
python -m pip install --no-build-isolation -e "$OPENPCDET_ROOT"
python "$REPO_ROOT/tools/openpcdet/check_openpcdet_environment.py" --strict
