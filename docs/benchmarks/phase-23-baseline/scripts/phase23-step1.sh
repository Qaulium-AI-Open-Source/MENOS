#!/usr/bin/env bash
# Phase 23 step 1 run $1 (a, b, c): docs/adr/0027 section 6 step 1, in the WSL
# clone. Refuses to start, and marks the run invalid, unless the machine is on
# AC before and after every tool (the check of docs/benchmarks/phase-16-gate/
# scripts/gate.sh). The CUDA side first (shorter), then the CPU side, whose disk
# rows write the scratch files to this clone's ext4 volume (the WSL virtual
# disk; its size before and after is recorded by the Windows side).
set -e
export PATH="$HOME/.local/bin:$PATH"
RUN=$1
cd ~/MENOS
OUT=build/phase23-baseline/wsl-run-$RUN
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
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee "$OUT/gpu.txt"
df -B1 --output=size,avail . | tee "$OUT/disk-before.txt"
check_ac start
cmake --preset cuda12-release -DMENOS_BUILD_BENCHMARKS=ON > /dev/null
cmake --build build/cuda12-release -j"$(nproc)" --target menos_cuda_tiered_memory_benchmark > /dev/null
cmake -S . -B build/bench -DCMAKE_BUILD_TYPE=Release -DMENOS_BUILD_BENCHMARKS=ON > /dev/null
cmake --build build/bench -j"$(nproc)" --target menos_tiered_memory_benchmark > /dev/null
check_ac after-build
echo "== cuda side $(date -u +%T)"
MENOS_CUDA_TIERED_JSON=$OUT/cuda.json \
    build/cuda12-release/benchmarks/menos_cuda_tiered_memory_benchmark > "$OUT/cuda.log" 2>&1 \
    || echo "cuda benchmark exit $?"
check_ac after-cuda
echo "== cpu side $(date -u +%T)"
MENOS_TIERED_MEMORY_JSON=$OUT/cpu.json \
    build/bench/benchmarks/menos_tiered_memory_benchmark > "$OUT/cpu.log" 2>&1 \
    || echo "cpu benchmark exit $?"
check_ac end
df -B1 --output=size,avail . | tee "$OUT/disk-after.txt"
echo "== done $(date -u +%T)"
