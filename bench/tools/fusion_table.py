#!/usr/bin/env python3
"""Tables for ADR 0028 section 17.5 from menos_dense_fusion_probe's CSV.

    python bench/tools/fusion_table.py docs/benchmarks/phase-24/fusion-step1/linux/probe.csv

Prints, as Markdown:
  1. one pass by size and operand count: the median in microseconds, its
     ratio to the 1-qubit pass of the same size and thread count, and the
     median over the bandwidth floor (2 * 16 * 2^n bytes at the calibrated
     streaming bandwidth), for dense and diagonal matrices;
  2. the circuit rows: passes, speed-up over the unfused run and the largest
     amplitude difference, by size, window and thread count.
"""

import csv
import sys
from collections import defaultdict


def load(path):
    passes, circuits = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0] == "kind":
                continue
            if row[0].startswith("pass_"):
                passes.append(row)
            elif row[0] == "circuit":
                circuits.append(row)
    return passes, circuits


def pass_tables(passes):
    # kind, qubits, k, threads, median_s, min_s, max_s, bytes_per_s, bandwidth_floor_s, amplitudes_per_s
    by = defaultdict(dict)
    for kind, n, k, threads, median, _mn, _mx, _bps, floor, _aps in passes:
        by[(kind, int(threads))][(int(n), int(k))] = (float(median), float(floor))
    for (kind, threads), rows in sorted(by.items()):
        sizes = sorted({n for n, _ in rows})
        ks = sorted({k for _, k in rows})
        print(f"\n**{kind}, {threads} thread{'s' if threads != 1 else ''}** — median µs per pass "
              f"(× the k = 1 pass of the same size; × the bandwidth floor)\n")
        print("| n | " + " | ".join(f"k = {k}" for k in ks) + " |")
        print("|---|" + "---|" * len(ks))
        for n in sizes:
            base = rows[(n, 1)][0]
            cells = []
            for k in ks:
                if (n, k) not in rows:
                    cells.append("")
                    continue
                median, floor = rows[(n, k)]
                over_floor = f"; {median / floor:.2f}× floor" if floor > 0 else ""
                cells.append(f"{median * 1e6:.0f} ({median / base:.1f}×{over_floor})")
            print(f"| {n} | " + " | ".join(cells) + " |")


def circuit_table(circuits):
    # kind, qubits, window, threads, gates, passes, gates_absorbed, median_s, min_s, max_s, speedup, max_abs_diff
    print("\n**Brickwork circuit, gate by gate (window 1) and fused (windows 2–5)**\n")
    print("| n | threads | window | gates | passes | median s | min..max s | speed-up | max abs diff |")
    print("|---|---|---|---|---|---|---|---|---|")
    for _kind, n, window, threads, gates, passes, _absorbed, median, mn, mx, speedup, diff in circuits:
        print(f"| {n} | {threads} | {window} | {gates} | {passes} | {float(median):.3f} | "
              f"{float(mn):.3f}..{float(mx):.3f} | {float(speedup):.2f} | {float(diff):.1e} |")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    passes, circuits = load(sys.argv[1])
    pass_tables(passes)
    circuit_table(circuits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
