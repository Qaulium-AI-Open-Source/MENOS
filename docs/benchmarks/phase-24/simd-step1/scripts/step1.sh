#!/usr/bin/env bash
# Phase 24 item 8, step 1 in the WSL clone (docs/adr/0028 section 19.3).
# Builds benchmarks/dense_simd_probe.cpp and benchmarks/dense_fusion_probe.cpp
# in build/bench (GCC, Release, CPU only) and records:
#   kernels.csv           table A, the kernels as built (GCC's AVX2 clones on)
#   runs.csv              table B, the fused brickwork sweep through menos::run
#   kernels-noclones.csv  table A again from build/bench-noclones, the same
#                         tree with MENOS_KERNEL_CLONES defined empty (the
#                         source is patched for that build only and restored
#                         on exit)
# with the commit and the machine's load and power state. Refuses to start,
# and marks the run invalid, unless the machine is on AC at every checkpoint.
#
#   bash docs/benchmarks/phase-24/simd-step1/scripts/step1.sh [max_qubits]
set -e
export PATH="$HOME/.local/bin:$PATH"
MAX_QUBITS=${1:-24}
cd ~/MENOS
OUT=build/phase24-simd-step1/linux
KERNELS=src/kernels/cpu/dense_kernels.cpp
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
cmake --build build/bench --target menos_dense_simd_probe menos_dense_fusion_probe -j"$(nproc)" > /dev/null

note start
build/bench/benchmarks/menos_dense_simd_probe "$MAX_QUBITS" > "$OUT/kernels.csv" 2> "$OUT/kernels.err" || echo "simd probe exit $?" | tee -a "$OUT/kernels.err"
note kernels-done
build/bench/benchmarks/menos_dense_fusion_probe "$MAX_QUBITS" runs > "$OUT/runs.csv" 2> "$OUT/runs.err" || echo "fusion probe exit $?" | tee -a "$OUT/runs.err"
note runs-done

# The clones-off build: the same tree, MENOS_KERNEL_CLONES empty.
trap 'git checkout -- "$KERNELS"' EXIT
sed -i 's/^#define MENOS_KERNEL_CLONES \[\[gnu::target_clones("avx2", "default")\]\]$/#define MENOS_KERNEL_CLONES/' "$KERNELS"
grep -q '^#define MENOS_KERNEL_CLONES$' "$KERNELS" || { echo "the clones patch did not apply"; exit 4; }
grep -c 'target_clones("avx2"' "$KERNELS" | sed 's/^/remaining target_clones lines (comments only expected): /' | tee -a "$OUT/commit.txt"
cmake -S . -B build/bench-noclones -DCMAKE_BUILD_TYPE=Release -DMENOS_BUILD_BENCHMARKS=ON \
    -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST="$PWD/build/bench/_deps/googletest-src" \
    -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON="$PWD/build/bench/_deps/nlohmann_json-src" \
    -DFETCHCONTENT_SOURCE_DIR_GOOGLE_BENCHMARK="$PWD/build/bench/_deps/google_benchmark-src" > /dev/null
cmake --build build/bench-noclones --target menos_dense_simd_probe -j"$(nproc)" > /dev/null
git checkout -- "$KERNELS"
trap - EXIT
note noclones-start
build/bench-noclones/benchmarks/menos_dense_simd_probe "$MAX_QUBITS" > "$OUT/kernels-noclones.csv" 2> "$OUT/kernels-noclones.err" || echo "noclones probe exit $?" | tee -a "$OUT/kernels-noclones.err"
note end
echo "== done $(date -u +%T)"
