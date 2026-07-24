#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
This legacy convenience runner is not the current paper's main-table protocol.
It used one shared beta/gamma pair and 100-epoch from-scratch updates, whereas
the paper uses setting-specific beta/gamma values, warm-checkpoint initialization,
and the protocol in configs/main_table_protocol.json.

Use scripts/run_main_table_finalupdate.sh instead.  See README.md for the
required seed-specific pseudo-edge CSV, warm checkpoint, and optional MovieLens
score-prior input.
EOF
exit 2
