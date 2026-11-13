#!/usr/bin/env bash
# Phase 23 step 1, WSL CPU side re-run $1 (a, b, c) with the scratch files on
# this clone's ext4 volume (the WSL virtual disk on the C: drive NVMe). The
# first three runs (wsl-run-*) let the benchmark default to /tmp, which this
# WSL mounts as a 12 GB tmpfs, so their disk rows measured RAM; they are kept
# as the RAM-backed reference. Same AC rule as phase23-step1.sh. CPU side only:
# the CUDA sides of wsl-run-* stand.
set -e
export PATH="$HOME/.local/bin:$PATH"
RUN=$1
cd ~/MENOS
OUT=build/phase23-baseline/wsl-ext4-run-$RUN
SCRATCH=$HOME/MENOS/build/phase23-scratch
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
rm -rf "$OUT" && mkdir -p "$OUT" "$SCRATCH"
git log --oneline -1 | tee "$OUT/commit.txt"
git status --short --untracked-files=no | tee -a "$OUT/commit.txt"
df -B1 --output=source,fstype,size,avail "$SCRATCH" | tee "$OUT/disk-before.txt"
check_ac start
cmake --build build/bench -j"$(nproc)" --target menos_tiered_memory_benchmark > /dev/null
check_ac after-build
echo "== cpu side, scratch on $SCRATCH $(date -u +%T)"
MENOS_TIERED_SCRATCH_DIR=$SCRATCH MENOS_TIERED_MEMORY_JSON=$OUT/cpu.json \
    build/bench/benchmarks/menos_tiered_memory_benchmark > "$OUT/cpu.log" 2>&1 \
    || echo "cpu benchmark exit $?"
check_ac end
df -B1 --output=source,fstype,size,avail "$SCRATCH" | tee "$OUT/disk-after.txt"
ls -la "$SCRATCH" | tee -a "$OUT/disk-after.txt"
echo "== done $(date -u +%T)"
