#!/usr/bin/env bash
# Phase 24 item 8 gate, the one-thread rows pinned to one core (docs/adr/0028
# section 19.8): the reference and current menos_dense_simd_probe built by
# gate.sh, alternated four times on CPU $CPU (default 2, a performance core),
# so a single thread cannot land on an efficiency core. Prints, per one-thread
# row, the four medians of each binary and the ratio of their medians.
#
#   bash docs/benchmarks/phase-24/simd-gate/scripts/pinned.sh
cd ~/MENOS
OUT=build/phase24-simd-gate/linux-pinned
rm -rf $OUT; mkdir -p $OUT
lscpu -e=CPU,CORE,MAXMHZ | head -24 > $OUT/lscpu.txt
CPU=${CPU:-2}
for r in 1 2 3 4; do
  taskset -c $CPU build/simd-gate-ref/build/bench/benchmarks/menos_dense_simd_probe 20 > $OUT/old-$r.csv 2>/dev/null
  taskset -c $CPU build/bench/benchmarks/menos_dense_simd_probe 20 > $OUT/new-$r.csv 2>/dev/null
done
python3 - "$OUT" <<'EOF'
import csv, sys, statistics as st
d = sys.argv[1]
def load(p):
    return {(r['qubits'], r['kind'], r['operands']): float(r['median_s']) for r in csv.DictReader(open(p)) if r['threads'] == '1'}
old = [load(f'{d}/old-{i}.csv') for i in range(1, 5)]
new = [load(f'{d}/new-{i}.csv') for i in range(1, 5)]
for k in old[0]:
    o = [x[k] for x in old]; n = [x[k] for x in new]
    print(k, 'old', ' '.join(f'{v*1e6:.0f}' for v in o), '| new', ' '.join(f'{v*1e6:.0f}' for v in n), '| ratio %.2f' % (st.median(o) / st.median(n)))
EOF
