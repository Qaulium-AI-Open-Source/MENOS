#!/usr/bin/env python3
"""The item 8 pinned one-thread comparison (ADR 0028 section 19.8).

    python bench/tools/simd_pinned_table.py DIR

Reads DIR/old-{1..4}.csv and DIR/new-{1..4}.csv (menos_dense_simd_probe, the
reference and current kernels, alternated on one pinned core by
docs/benchmarks/phase-24/simd-gate/scripts/pinned.{sh,ps1}) and prints, for
every one-thread row, the four medians of each binary (microseconds) and the
ratio of the median of old's four to the median of new's four.
"""

import csv
import statistics
import sys


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["qubits"], r["kind"], r["operands"]): float(r["median_s"])
                for r in csv.DictReader(f) if r["threads"] == "1"}


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    old = [load(f"{argv[1]}/old-{i}.csv") for i in range(1, 5)]
    new = [load(f"{argv[1]}/new-{i}.csv") for i in range(1, 5)]
    print("| n | kernel | operands | old µs (4 rounds) | new µs (4 rounds) | speed-up |")
    print("|---|---|---|---|---|---|")
    for key in old[0]:
        o = [x[key] * 1e6 for x in old]
        n = [x[key] * 1e6 for x in new]
        ratio = statistics.median(o) / statistics.median(n)
        print(f"| {key[0]} | {key[1]} | {key[2]} | {' '.join(f'{v:.0f}' for v in o)} | "
              f"{' '.join(f'{v:.0f}' for v in n)} | {ratio:.2f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
