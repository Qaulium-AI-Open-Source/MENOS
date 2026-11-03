#!/usr/bin/env bash
# Cold hardware calibration, once per fresh process (Phase 21 step 1,
# docs/adr/0025 section 5). Usage: bench/run_calibration.sh <probe> [runs]
#   <probe>  path to menos_calibration_probe (CPU or CUDA build)
#   runs     number of fresh processes (default 20)
# Prints CSV: one header line, then one line per process.
set -euo pipefail
probe="$1"
runs="${2:-20}"
"$probe" --header
for _ in $(seq "$runs"); do
    "$probe"
done
