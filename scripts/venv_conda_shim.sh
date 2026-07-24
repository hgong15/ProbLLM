#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
VENV_BIN="$(dirname "$PYTHON_BIN")"

if [[ "${1:-}" == "run" ]]; then
  shift
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --no-capture-output)
        shift
        ;;
      -n|--name)
        shift 2
        ;;
      *)
        break
        ;;
    esac
  done
fi

if [[ "$#" -eq 0 ]]; then
  exit 0
fi

cmd="$1"
shift

case "$cmd" in
  python|python3)
    exec "$PYTHON_BIN" "$@"
    ;;
  *)
    if [[ -x "$VENV_BIN/$cmd" ]]; then
      exec "$VENV_BIN/$cmd" "$@"
    fi
    exec "$cmd" "$@"
    ;;
esac
