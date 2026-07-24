#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "run" ]]; then
  exec "$@"
fi
shift

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --no-capture-output)
      shift
      ;;
    -n|--name)
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -*)
      shift
      ;;
    *)
      break
      ;;
  esac
done

exec "$@"
