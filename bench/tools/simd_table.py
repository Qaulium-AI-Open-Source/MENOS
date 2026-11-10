#!/usr/bin/env python3
"""Table A for ADR 0028 section 19 from menos_dense_simd_probe's CSV.

    python bench/tools/simd_table.py LINUX.csv LINUX_NOCLONES.csv [WINDOWS.csv]

Prints, as Markdown, one table per (qubits, threads): for each kernel and
operand placement, the Linux median in microseconds and its ratio to the
streaming-bandwidth floor; the Linux median with GCC's AVX2 clones off and
the clones' gain (clones-off median over clones-on median); and, if given,
the Windows (MSVC) median, its floor ratio and Windows over Linux.
"""

import csv
import sys


def load(path):
    rows = {}
    order = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (int(row["qubits"]), int(row["threads"]), row["kind"], row["operands"])
            rows[key] = row
            order.append(key)
    return rows, order


def us(row):
    return float(row["median_s"]) * 1e6


def main(argv):
    if len(argv) not in (3, 4):
        print(__doc__)
        return 2
    linux, order = load(argv[1])
    noclones, _ = load(argv[2])
    windows = load(argv[3])[0] if len(argv) == 4 else None
    groups = []
    for n, threads, _, _ in order:
        if (n, threads) not in groups:
            groups.append((n, threads))
    for n, threads in groups:
        print(f"\n**n = {n}, threads = {threads}**\n")
        head = "| kernel | operands | Linux µs | floor× | clones off µs | clones gain |"
        rule = "|---|---|---|---|---|---|"
        if windows is not None:
            head += " Windows µs | floor× | Win/Linux |"
            rule += "---|---|---|"
        print(head)
        print(rule)
        for key in order:
            if key[0] != n or key[1] != threads:
                continue
            row = linux[key]
            off = noclones.get(key)
            line = (f"| {key[2]} | {key[3]} | {us(row):.1f} | {float(row['floor_ratio']):.2f} | "
                    f"{us(off):.1f} | {us(off) / us(row):.2f} |" if off else
                    f"| {key[2]} | {key[3]} | {us(row):.1f} | {float(row['floor_ratio']):.2f} | - | - |")
            if windows is not None:
                w = windows.get(key)
                line += (f" {us(w):.1f} | {float(w['floor_ratio']):.2f} | {us(w) / us(row):.2f} |"
                         if w else " - | - | - |")
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
