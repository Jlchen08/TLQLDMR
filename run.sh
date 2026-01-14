#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/../.venv"
PY="${VENV_DIR}/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "Venv python not found: $PY" >&2
  echo "Create it with: python -m venv \"$VENV_DIR\"" >&2
  exit 1
fi

export PYTHONNOUSERSITE=1

TARGET="${SCRIPT_DIR}/wind_farm_experiment.py"
# TARGET="${SCRIPT_DIR}/real_data.py"
if [[ $# -ge 1 ]]; then
  TARGET="$1"
  shift
fi

if [[ ! -f "$TARGET" ]]; then
  TARGET="${SCRIPT_DIR}/${TARGET}"
fi

exec "$PY" "$TARGET" "$@"
