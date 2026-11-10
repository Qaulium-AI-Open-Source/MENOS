#!/usr/bin/env bash
# Phase 24 item 8 gate run in the WSL clone (docs/adr/0028 section 19.7):
# the kernels and the fused brickwork sweep on the current commit against the
# unchanged kernels of the reference commit (default a326851, the step-1
# commit), in the same session. The reference tree is a detached worktree at
# build/simd-gate-ref built with the same flags. The probes run in the order
# old, new, new, old, so drift during the run lands on both sides.
#   kernels-{old,new}-{1,2}.csv   menos_dense_simd_probe (table A)
#   runs-{old,new}-{1,2}.csv      menos_dense_fusion_probe runs (table B)
# Refuses to start, and marks the run invalid, unless on AC at every checkpoint.
#
#   bash docs/benchmarks/phase-24/simd-gate/scripts/gate.sh [max_qubits]
set -e
export PATH="$HOME/.local/bin:$PATH"
MAX_QUBITS=${1:-24}
REF=${REF:-a326851}
cd ~/MENOS
OUT=build/phase24-simd-gate/linux
OLD=build/simd-gate-ref
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
echo "reference $REF" | tee -a "$OUT/commit.txt"
deps=(-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST="$PWD/build/bench/_deps/googletest-src"
      -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON="$PWD/build/bench/_deps/nlohmann_json-src"
      -DFETCHCONTENT_SOURCE_DIR_GOOGLE_BENCHMARK="$PWD/build/bench/_deps/google_benchmark-src")
[ -d "$OLD" ] || git worktree add --detach "$OLD" "$REF" > /dev/null
git -C "$OLD" checkout -q --detach "$REF"
cmake -S "$OLD" -B "$OLD/build/bench" -DCMAKE_BUILD_TYPE=Release -DMENOS_BUILD_BENCHMARKS=ON "${deps[@]}" > /dev/null
cmake --build "$OLD/build/bench" --target menos_dense_simd_probe menos_dense_fusion_probe -j"$(nproc)" > /dev/null
cmake -S . -B build/bench -DCMAKE_BUILD_TYPE=Release -DMENOS_BUILD_BENCHMARKS=ON > /dev/null
cmake --build build/bench --target menos_dense_simd_probe menos_dense_fusion_probe -j"$(nproc)" > /dev/null

kernels() { "$1/benchmarks/menos_dense_simd_probe" "$MAX_QUBITS" > "$OUT/kernels-$2.csv" 2> "$OUT/kernels-$2.err" || echo "exit $?" >> "$OUT/kernels-$2.err"; }
runs() { "$1/benchmarks/menos_dense_fusion_probe" "$MAX_QUBITS" runs > "$OUT/runs-$2.csv" 2> "$OUT/runs-$2.err" || echo "exit $?" >> "$OUT/runs-$2.err"; }
note start
kernels "$OLD/build/bench" old-1
kernels build/bench new-1
kernels build/bench new-2
kernels "$OLD/build/bench" old-2
note kernels-done
runs "$OLD/build/bench" old-1
runs build/bench new-1
runs build/bench new-2
runs "$OLD/build/bench" old-2
note end
echo "== done $(date -u +%T)"
