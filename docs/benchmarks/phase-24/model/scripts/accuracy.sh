#!/usr/bin/env bash
# Phase 24 model fixes (docs/adr/0028 section 20): the cost-model accuracy
# benchmark in the WSL clone, three separate processes, in build/cuda12-release
# (so the CUDA dense rows run on the GPU). Records accuracy_run{1,2,3}.{csv,err},
# the commit and the load and power state into build/phase24-model/<tag>/linux.
# Refuses to start, and marks the run invalid, unless on AC at every checkpoint.
#
#   bash docs/benchmarks/phase-24/model/scripts/accuracy.sh <tag>
set -e
export PATH="$HOME/.local/bin:$PATH"
TAG=${1:?usage: accuracy.sh <tag>}
cd ~/MENOS
OUT=build/phase24-model/$TAG/linux
power() {
    .venv/bin/python -c "
import importlib.util, json
spec = importlib.util.spec_from_file_location('d', 'tools/dashboard/menos_dashboard.py')
d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
print(json.dumps(d.power_state()))"
}
note() {
    local p; p=$(power)
    echo "$1 $(date -u +%Y-%m-%dT%H:%M:%SZ) loadavg=$(cut -d' ' -f1-3 /proc/loadavg) power: $p" | tee -a "$OUT/load.log"
    case "$p" in *'"ac_online": true'*) ;; *) echo "NOT ON AC at $1: the run is invalid"; exit 3;; esac
}
rm -rf "$OUT" && mkdir -p "$OUT"
git log --oneline -1 | tee "$OUT/commit.txt"
git status --short --untracked-files=no | tee -a "$OUT/commit.txt"
cmake --build build/cuda12-release --target menos_cost_model_accuracy_benchmark -j"$(nproc)" > "$OUT/build.log" 2>&1
for r in 1 2 3; do
    note "accuracy_run$r"
    build/cuda12-release/benchmarks/menos_cost_model_accuracy_benchmark > "$OUT/accuracy_run$r.csv" 2> "$OUT/accuracy_run$r.err" || echo "exit $?" >> "$OUT/accuracy_run$r.err"
done
note end
echo "== done $(date -u +%T)"
