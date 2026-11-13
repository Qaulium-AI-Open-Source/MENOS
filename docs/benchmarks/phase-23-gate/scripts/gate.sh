#!/usr/bin/env bash
# Phase 23 step 5, gate run $1 (a, b, c) in the WSL clone: the disk tier's
# gate-scale runs (benchmarks/tiered_gate_benchmark.cpp) with the scratch file
# on this clone's ext4 volume. WSL's budget (50% of 23.5 GiB) stops the
# in-memory engine at n = 29, so the reach run is n = 30 here (a 16 GiB
# file); the overlap runs are n = 29 as on Windows. The WSL virtual disk grows
# by the file's size and does not shrink (docs/adr/0027 section 10.1), which
# is why the tier's home on this machine is drive C: from native Windows and
# this script is run once. Refuses to start, and marks the run invalid, unless
# the machine is on AC before and after every tool.
set -e
export PATH="$HOME/.local/bin:$PATH"
RUN=$1
cd ~/MENOS
OUT=build/phase23-gate/wsl-run-$RUN
power() {
    .venv/bin/python -c "
import importlib.util, json
spec = importlib.util.spec_from_file_location('d', 'tools/dashboard/menos_dashboard.py')
d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
print(json.dumps(d.power_state()))"
}
check_ac() {
    local p; p=$(power)
    echo "$(date -u +%T) $1 power: $p" | tee -a "$OUT/power.log"
    case "$p" in *'"ac_online": true'*) ;; *) echo "NOT ON AC at $1: run $RUN is invalid"; exit 3;; esac
}
rm -rf "$OUT" && mkdir -p "$OUT"
git log --oneline -1 | tee "$OUT/commit.txt"
git status --short --untracked-files=no | tee -a "$OUT/commit.txt"
SCRATCH=$HOME/MENOS/build/phase23-scratch
mkdir -p "$SCRATCH"
df -B1 --output=source,fstype,size,avail "$SCRATCH" | tee "$OUT/disk-before.txt"
check_ac start
cmake -S . -B build/bench -DCMAKE_BUILD_TYPE=Release -DMENOS_BUILD_BENCHMARKS=ON > /dev/null
cmake --build build/bench --target menos_tiered_gate_benchmark -j"$(nproc)" > /dev/null
check_ac after-build
echo "== gate $(date -u +%T)"
MENOS_TIERED_GATE_JSON=$OUT/gate.json \
MENOS_TIERED_SCRATCH_DIR=$SCRATCH \
MENOS_TIERED_GATE_REACH_QUBITS=30 \
MENOS_TIERED_GATE_OVERLAP_QUBITS=29 \
    build/bench/benchmarks/menos_tiered_gate_benchmark > "$OUT/gate.log" 2>&1 || echo "gate driver exit $?" | tee -a "$OUT/gate.log"
check_ac end
df -B1 --output=source,fstype,size,avail "$SCRATCH" | tee "$OUT/disk-after.txt"
echo "== done $(date -u +%T)"
