#!/usr/bin/env python3
"""The item 8 gate tables (ADR 0028 section 19.8) from a gate run directory.

    python bench/tools/simd_gate_table.py docs/benchmarks/phase-24/simd-gate/linux [--windows]

Reads kernels-{old,new}-{1,2}.csv and runs-{old,new}-{1,2}.csv (written by
docs/benchmarks/phase-24/simd-gate/scripts/gate.{sh,ps1} in the order old,
new, new, old) and prints, as Markdown:
  1. per kernel row: the old and new medians (mean of the two rounds' medians,
     microseconds), the speed-up old/new, and whether new is a slowdown
     outside the run-to-run spread (new's faster round slower than old's
     slower round);
  2. the run rows: fused and unfused sweeps, old and new, with the speed-ups;
  3. the criterion 3 checks: 3a (2q, one thread, n 16-20, >= 2.0), 3b (1q,
     one thread, n 16-20, >= 1.8 on Windows, >= 0.95 on Linux), 3c (fused
     sweep, one thread, n 16-20, >= 1.5), and criterion 4 (no slowdown).
"""

import csv
import os
import sys


def load_kernels(path):
    rows = {}
    order = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (int(r["qubits"]), int(r["threads"]), r["kind"], r["operands"])
            rows[key] = float(r["median_s"])
            order.append(key)
    return rows, order


def load_runs(path):
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (int(r["qubits"]), int(r["threads"]))
            rows[key] = (float(r["unfused_median_s"]), float(r["fused_median_s"]))
    return rows


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    d = argv[1]
    windows = "--windows" in argv
    ko1, order = load_kernels(os.path.join(d, "kernels-old-1.csv"))
    ko2, _ = load_kernels(os.path.join(d, "kernels-old-2.csv"))
    kn1, _ = load_kernels(os.path.join(d, "kernels-new-1.csv"))
    kn2, _ = load_kernels(os.path.join(d, "kernels-new-2.csv"))
    ro1 = load_runs(os.path.join(d, "runs-old-1.csv"))
    ro2 = load_runs(os.path.join(d, "runs-old-2.csv"))
    rn1 = load_runs(os.path.join(d, "runs-new-1.csv"))
    rn2 = load_runs(os.path.join(d, "runs-new-2.csv"))

    speed = {}
    slow = []
    print("| n | threads | kernel | operands | old µs | new µs | speed-up | slowdown outside spread |")
    print("|---|---|---|---|---|---|---|---|")
    for key in order:
        old = (ko1[key] + ko2[key]) / 2
        new = (kn1[key] + kn2[key]) / 2
        s = old / new
        speed[key] = s
        outside = min(kn1[key], kn2[key]) > max(ko1[key], ko2[key])
        if outside:
            slow.append(key)
        print(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {old * 1e6:.1f} | {new * 1e6:.1f} | "
              f"{s:.2f} | {'YES' if outside else 'no'} |")

    print()
    print("| n | threads | unfused old s | unfused new s | unfused speed-up | fused old s | fused new s | fused speed-up |")
    print("|---|---|---|---|---|---|---|---|")
    run_speed = {}
    for key in sorted(ro1):
        uo = (ro1[key][0] + ro2[key][0]) / 2
        un = (rn1[key][0] + rn2[key][0]) / 2
        fo = (ro1[key][1] + ro2[key][1]) / 2
        fn = (rn1[key][1] + rn2[key][1]) / 2
        run_speed[key] = (uo / un, fo / fn)
        print(f"| {key[0]} | {key[1]} | {uo:.4f} | {un:.4f} | {uo / un:.2f} | {fo:.4f} | {fn:.4f} | {fo / fn:.2f} |")

    def worst(kind):
        vals = [(v, k) for k, v in speed.items() if k[2] == kind and k[1] == 1 and k[0] in (16, 20)]
        return min(vals)

    print()
    v, k = worst("2q")
    print(f"3a 2q one thread n16-20: min {v:.2f} at {k} -> {'MET' if v >= 2.0 else 'MISSED'} (>= 2.0)")
    v, k = worst("1q")
    bar = 1.8 if windows else 0.95
    print(f"3b 1q one thread n16-20: min {v:.2f} at {k} -> {'MET' if v >= bar else 'MISSED'} (>= {bar})")
    fused = [(run_speed[k][1], k) for k in run_speed if k[1] == 1 and k[0] in (16, 20)]
    v, k = min(fused)
    print(f"3c fused sweep one thread n16-20: min {v:.2f} at {k} -> {'MET' if v >= 1.5 else 'MISSED'} (>= 1.5)")
    print(f"4 slowdowns outside the spread: {len(slow)} {slow}")
    worst_all = min((v, k) for k, v in speed.items())
    print(f"  smallest kernel speed-up anywhere: {worst_all[0]:.2f} at {worst_all[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
