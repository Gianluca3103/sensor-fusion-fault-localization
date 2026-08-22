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
python -m pip install "${SPCONV_PACKAGE:-spconv-cu120>=2.3}"
python -m pip install -e "$OPENPCDET_ROOT"
python "$REPO_ROOT/tools/openpcdet/check_openpcdet_environment.py" --strict
