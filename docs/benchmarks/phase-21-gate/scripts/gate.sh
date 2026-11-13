# Phase 21 gate criterion 4 (docs/adr/0025 section 9, Q9): no benchmark
# slowdown. Builds the Phase 15 dense benchmark and the Phase 11 monitor
# benchmark at the phase's base (1f89192) and at the current commit, and runs
# them interleaved in one session, so each ratio compares two builds under the
# same machine state. gate_criteria.py turns the JSON into the criterion.
#
#   gate.sh RUN [ROUNDS]     RUN names the output directory; ROUNDS defaults to 3
#
# Run in the WSL clone (the gate machine), on AC power: like the Phase 15 and 16
# gate scripts, it refuses to start, and marks the run invalid, unless the
# machine is on AC before and after every round.
set -e
export PATH="$HOME/.local/bin:$PATH"
cd ~/MENOS
RUN=$1
ROUNDS=${2:-3}
BASE=1f89192
BASE_TREE=$HOME/MENOS-phase21-base
OUT=build/phase21-gate/run-$RUN
power() {
    .venv/bin/python -c "
import importlib.util, json
spec = importlib.util.spec_from_file_location('d', 'tools/dashboard/menos_dashboard.py')
d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
print(json.dumps(d.power_state()))"
}
check_ac() {
    local p; p=$(power)
    echo "$(date -u +%T) $1 power: $p" | tee -a $OUT/power.log
    case "$p" in *'"ac_online": true'*) ;; *) echo "NOT ON AC at $1: run $RUN is invalid"; exit 3;; esac
}
rm -rf $OUT && mkdir -p $OUT
git log --oneline -1 | tee $OUT/commit.txt
git status --short --untracked-files=no | tee -a $OUT/commit.txt
echo "base $BASE" | tee -a $OUT/commit.txt
check_ac start

# The base in its own worktree, both built from the release preset with the
# benchmarks on, each in a build directory of its own.
if [ ! -d "$BASE_TREE" ]; then
    git worktree add -q --detach "$BASE_TREE" $BASE
fi
git -C "$BASE_TREE" checkout -q --detach $BASE
for TREE in "$BASE_TREE" "$PWD"; do
    (cd "$TREE" && cmake --preset release -B build/phase21-bench -DMENOS_BUILD_BENCHMARKS=ON > /dev/null &&
     cmake --build build/phase21-bench -j"$(nproc)" \
         --target menos_dense_gate_application_benchmark menos_monitor_overhead_benchmark > /dev/null)
done

run_tree() {  # $1 = base|head, $2 = tree, $3 = round
    local bin=$2/build/phase21-bench/benchmarks
    $bin/menos_dense_gate_application_benchmark --benchmark_report_aggregates_only=false \
        --benchmark_out=$OUT/dense-$1-$3.json --benchmark_out_format=json > $OUT/dense-$1-$3.log 2>&1
    $bin/menos_monitor_overhead_benchmark --benchmark_filter=Unmonitored \
        --benchmark_repetitions=10 --benchmark_report_aggregates_only=false \
        --benchmark_out=$OUT/monitor-$1-$3.json --benchmark_out_format=json > $OUT/monitor-$1-$3.log 2>&1
}

# Alternating which build goes first, so a drift during the session falls on
# both equally.
for ROUND in $(seq 1 "$ROUNDS"); do
    echo "== round $ROUND $(date -u +%T)"
    if [ $((ROUND % 2)) -eq 1 ]; then
        run_tree base "$BASE_TREE" $ROUND
        run_tree head "$PWD" $ROUND
    else
        run_tree head "$PWD" $ROUND
        run_tree base "$BASE_TREE" $ROUND
    fi
    check_ac after-round-$ROUND
done
.venv/bin/python docs/benchmarks/phase-21-gate/scripts/gate_criteria.py $OUT | tee $OUT/criteria.txt
echo "== done $(date -u +%T)"
