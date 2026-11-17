#!/usr/bin/env bash
# Equivalente Linux/macOS de scripts/run_all.ps1
set -euo pipefail
python -m src.cli run-all "$@"
