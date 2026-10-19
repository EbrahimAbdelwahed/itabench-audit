#!/usr/bin/env bash
# Clones SapienzaNLP/ita-bench at the commit audited here and installs the pinned evaluation stack.
set -euo pipefail
ITABENCH_COMMIT=c1d077f5af6670985ec797555fa949dba6ecf121
if [ ! -d ita-bench ]; then
  git clone https://github.com/SapienzaNLP/ita-bench.git
fi
git -C ita-bench checkout --quiet "$ITABENCH_COMMIT"
"${PYTHON:-python3.11}" -m venv .venv  # tested with Python 3.11
.venv/bin/pip install --quiet -r requirements.txt
echo "ready: ita-bench at $ITABENCH_COMMIT, environment in .venv"
