#!/usr/bin/env bash
# Phase 22 criterion 6: the planner's CPU/CUDA choice for the crossover circuit
# family (statevector output) before the pool (766f9df, worktree ~/MENOS-766f9df)
# and after (~/MENOS at the gate commit), three times each, alternating, so both
# see the same machine state. Calibration runs inside each process, so each
# line is a fresh calibration. The program is Phase 16's, unchanged:
# docs/benchmarks/phase-16-gate/scripts/planner_choice.cpp.txt.
set -e
export PATH="$HOME/.local/bin:$PATH"
SRC=~/MENOS/docs/benchmarks/phase-16-gate/scripts/planner_choice.cpp.txt
for root in ~/MENOS-766f9df ~/MENOS; do
    cd $root
    B=build/cuda12-release
    cmake --preset cuda12-release > /dev/null
    cmake --build $B -j"$(nproc)" > /dev/null
    LIBS=$(find $B/src -name '*.a' | grep -v _deps | tr '\n' ' ')
    g++-13 -std=c++20 -O2 -pthread -Iinclude -Isrc -x c++ $SRC -x none \
        -Wl,--start-group $LIBS -Wl,--end-group -llapacke -llapack -lblas -lcudart \
        -o /tmp/p22_planner_$(basename $root)
done
for rep in 1 2 3; do
    for root in ~/MENOS-766f9df ~/MENOS; do
        echo "== $(basename $root) $(git -C $root log --oneline -1 | cut -c1-7) rep $rep"
        /tmp/p22_planner_$(basename $root)
    done
done
