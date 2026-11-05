#!/usr/bin/env bash
# Phase 22 gate run $1 (a, b, ...): docs/BUILD_PLAN.md, Phase 22 gate, criteria 2 and 6.
# Run in the WSL clone (the gate machine), cuda12-release with benchmarks. Refuses to
# start, and marks the run invalid, unless the machine is on AC before and after every
# tool (the check of docs/benchmarks/phase-16-gate/scripts/gate.sh).
#
# Criterion 6 (recalibration and crossover shift): menos_cuda_crossover_benchmark on the
# last commit before the pool (766f9df, worktree ~/MENOS-766f9df) and then on the gate
# commit (~/MENOS), in the same run, so the two calibration lines and the two forced
# CPU/CUDA crossovers are as close to one GPU clock state as a sequential run allows.
# Criterion 2: menos_cuda_memory_benchmark on the gate commit; its allocation table is a
# same-run ratio of the direct cycle against the pooled cycle.
set -e
export PATH="$HOME/.local/bin:$PATH"
RUN=$1
HEAD_DIR=$HOME/MENOS
BASE_DIR=$HOME/MENOS-766f9df
OUT=$HEAD_DIR/build/phase22-gate/run-$RUN
power() {
    (cd "$HEAD_DIR" && .venv/bin/python -c "
import importlib.util, json
spec = importlib.util.spec_from_file_location('d', 'tools/dashboard/menos_dashboard.py')
d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
print(json.dumps(d.power_state()))")
}
check_ac() {
    local p; p=$(power)
    echo "$(date -u +%T) $1 power: $p" | tee -a "$OUT/power.log"
    case "$p" in *'"ac_online": true'*) ;; *) echo "NOT ON AC at $1: run $RUN is invalid"; exit 3;; esac
}
rm -rf "$OUT" && mkdir -p "$OUT"
(cd "$HEAD_DIR" && git log --oneline -1 && git status --short --untracked-files=no) | tee "$OUT/commit.txt"
(cd "$BASE_DIR" && git log --oneline -1 && git status --short --untracked-files=no) | tee "$OUT/base-commit.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee "$OUT/gpu.txt"
check_ac start
DEPS=(
    "-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=$HEAD_DIR/build/cuda12-dev/_deps/googletest-src"
    "-DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=$HEAD_DIR/build/cuda12-dev/_deps/nlohmann_json-src"
    "-DFETCHCONTENT_SOURCE_DIR_GOOGLE_BENCHMARK=$HEAD_DIR/build/cuda-release/_deps/google_benchmark-src"
)
(cd "$HEAD_DIR" && cmake --preset cuda12-release -DMENOS_BUILD_BENCHMARKS=ON "${DEPS[@]}" > /dev/null \
    && cmake --build build/cuda12-release -j"$(nproc)" \
        --target menos_cuda_crossover_benchmark menos_cuda_memory_benchmark > /dev/null)
(cd "$BASE_DIR" && cmake --preset cuda12-release -DMENOS_BUILD_BENCHMARKS=ON "${DEPS[@]}" > /dev/null \
    && cmake --build build/cuda12-release -j"$(nproc)" --target menos_cuda_crossover_benchmark > /dev/null)
check_ac after-build

echo "== crossover before the pool (766f9df), 10 repeats $(date -u +%T)"
(cd "$BASE_DIR" && MENOS_CROSSOVER_REPEATS=10 MENOS_CROSSOVER_JSON="$OUT/crossover-before.json" \
    build/cuda12-release/benchmarks/menos_cuda_crossover_benchmark > "$OUT/crossover-before.log" 2>&1) \
    || echo "crossover-before exit $?"
check_ac after-crossover-before

echo "== crossover with the pool (gate commit), 10 repeats $(date -u +%T)"
(cd "$HEAD_DIR" && MENOS_CROSSOVER_REPEATS=10 MENOS_CROSSOVER_JSON="$OUT/crossover-after.json" \
    build/cuda12-release/benchmarks/menos_cuda_crossover_benchmark > "$OUT/crossover-after.log" 2>&1) \
    || echo "crossover-after exit $?"
check_ac after-crossover-after

echo "== memory benchmark (gate commit) $(date -u +%T)"
(cd "$HEAD_DIR" && MENOS_CUDA_MEMORY_JSON="$OUT/memory.json" \
    build/cuda12-release/benchmarks/menos_cuda_memory_benchmark > "$OUT/memory.log" 2>&1) \
    || echo "memory benchmark exit $?"
check_ac end
echo "== done $(date -u +%T)"
