#!/usr/bin/env bash
# Phase 24 item 7, step 1 in the WSL clone (docs/adr/0028 section 17.3):
# builds and runs benchmarks/dense_fusion_probe.cpp in build/bench (GCC,
# Release, CPU only) and records its CSV, its environment line, the commit
# and the machine's load and power state. Refuses to start, and marks the
# run invalid, unless the machine is on AC before and after the probe.
#
#   bash docs/benchmarks/phase-24/fusion-step1/scripts/step1.sh [max_qubits]
set -e
export PATH="$HOME/.local/bin:$PATH"
MAX_QUBITS=${1:-24}
cd ~/MENOS
OUT=build/phase24-fusion-step1/linux
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
cmake -S . -B build/bench -DCMAKE_BUILD_TYPE=Release -DMENOS_BUILD_BENCHMARKS=ON > /dev/null
cmake --build build/bench --target menos_dense_fusion_probe -j"$(nproc)" > /dev/null
note start
build/bench/benchmarks/menos_dense_fusion_probe "$MAX_QUBITS" > "$OUT/probe.csv" 2> "$OUT/probe.err" || echo "probe exit $?" | tee -a "$OUT/probe.err"
note end
echo "== done $(date -u +%T)"
