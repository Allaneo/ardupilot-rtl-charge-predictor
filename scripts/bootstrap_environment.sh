#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARDUPILOT_DIR="$PROJECT_ROOT/external/ardupilot"
ARDUPILOT_COMMIT="d12a8f8997634cee977581cbaa5548c04facc1f5"
PAYLOAD_PATCH="$PROJECT_ROOT/patches/ardupilot_centered_payload.patch"
UNMODIFIED_BINARY="$PROJECT_ROOT/external/baselines/arducopter-unmodified-$ARDUPILOT_COMMIT"
PYTHON_BIN="/usr/local/bin/python3.11"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python 3.11 is required at $PYTHON_BIN." >&2
    echo "Install it with: brew install python@3.11" >&2
    exit 1
fi

if [[ ! -d "$ARDUPILOT_DIR/.git" ]]; then
    git clone --filter=blob:none --recurse-submodules \
        https://github.com/ArduPilot/ardupilot.git "$ARDUPILOT_DIR"
fi

PATCH_ALREADY_APPLIED=false
ARDUPILOT_STATUS="$(git -C "$ARDUPILOT_DIR" status --short)"
EXPECTED_PATCH_STATUS=$' M libraries/SITL/SIM_Aircraft.cpp\n M libraries/SITL/SITL.cpp\n M libraries/SITL/SITL.h'
if [[ -n "$ARDUPILOT_STATUS" ]]; then
    if [[ "$(git -C "$ARDUPILOT_DIR" rev-parse HEAD)" != "$ARDUPILOT_COMMIT" ]] || \
        [[ "$ARDUPILOT_STATUS" != "$EXPECTED_PATCH_STATUS" ]] || \
        ! git -C "$ARDUPILOT_DIR" apply --reverse --check "$PAYLOAD_PATCH"; then
        echo "ArduPilot checkout has changes other than the expected payload patch." >&2
        exit 1
    fi
    PATCH_ALREADY_APPLIED=true
else
    git -C "$ARDUPILOT_DIR" checkout --detach "$ARDUPILOT_COMMIT"
fi
git -C "$ARDUPILOT_DIR" submodule update --init --recursive

"$PYTHON_BIN" -m venv "$PROJECT_ROOT/.venv"

# A dot-named directory can carry macOS' UF_HIDDEN flag into its children.
# Clear it before installing, and use a normal wheel install: Python 3.11.16
# skips hidden .pth files, which makes editable installs unreliable here.
if [[ "$(uname -s)" == "Darwin" ]]; then
    chflags -R nohidden "$PROJECT_ROOT/.venv"
fi

"$PROJECT_ROOT/.venv/bin/python" -m pip install \
    "$PROJECT_ROOT[simulation,ml,test]"

if [[ "$(uname -s)" == "Darwin" ]]; then
    chflags -R nohidden "$PROJECT_ROOT/.venv"
fi

(
    cd "$ARDUPILOT_DIR"
    "$PROJECT_ROOT/.venv/bin/python" ./waf configure --board sitl

    if [[ "$PATCH_ALREADY_APPLIED" == false ]]; then
        "$PROJECT_ROOT/.venv/bin/python" ./waf copter
        mkdir -p "$(dirname "$UNMODIFIED_BINARY")"
        cp build/sitl/bin/arducopter "$UNMODIFIED_BINARY"
        git apply --check "$PAYLOAD_PATCH"
        git apply "$PAYLOAD_PATCH"
    fi

    "$PROJECT_ROOT/.venv/bin/python" ./waf copter
)

"$PROJECT_ROOT/.venv/bin/python" -m pytest "$PROJECT_ROOT/tests"

echo "Environment ready. Patched SITL binary:"
echo "$ARDUPILOT_DIR/build/sitl/bin/arducopter"
