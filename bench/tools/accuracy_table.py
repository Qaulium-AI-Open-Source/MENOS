#!/usr/bin/env python3
"""Per-engine accuracy table from menos_cost_model_accuracy_benchmark CSV.

Phase 24 step 1 (docs/adr/0028 sections 16.7 Q2 and 16.9). For every engine
(the CSV's `engine` column) over its measured rows:

  n          rows with a measurement and a positive prediction
  med|log|   median of |ln(actual / predicted)|, the symmetric error
  p90|log|   90th percentile of the same (nearest rank)
  med ratio  median of actual / predicted
  min..max   range of actual / predicted
  under      rows predicted too fast (actual > predicted)
  over       rows predicted too slow (actual < predicted)
  bias       median of ln(actual / predicted): > 0 underpredicts, < 0 over

|ln r| is reported with its equivalent factor, exp(|ln r|): 0.182 is a
factor of 1.2 (20%), 0.405 a factor of 1.5 (50%).

Usage: accuracy_table.py RUN.csv [RUN.csv ...]
Several files (one per process) are pooled; each row's actual is its median.
"""

import csv
import math
import statistics
import sys
from collections import defaultdict


def nearest_rank(values, q):
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[rank - 1]


def main(paths):
    rows = defaultdict(list)
    other = []
    for path in paths:
        with open(path, encoding="utf-8-sig") as f:
            lines = [line for line in f if "," in line and not line.startswith("environment")]
        for r in csv.DictReader(lines):
            if r.get("status") == "measured" and float(r["predicted_s"]) > 0:
                ratio = float(r["actual_median_s"]) / float(r["predicted_s"])
                rows[r["engine"]].append((r["workload"], ratio))
            elif r.get("status"):
                other.append(r)

    print("| Engine | n | med abs(ln r) (factor) | p90 abs(ln r) (factor) | median ratio | "
          "min..max ratio | under / over | bias ln r |")
    print("|---|---|---|---|---|---|---|---|")
    for engine in sorted(rows):
        ratios = [ratio for _, ratio in rows[engine]]
        logs = [math.log(ratio) for ratio in ratios]
        abs_logs = [abs(x) for x in logs]
        med = statistics.median(abs_logs)
        p90 = nearest_rank(abs_logs, 0.9)
        under = sum(1 for x in logs if x > 0)
        print(f"| {engine} | {len(ratios)} | {med:.3f} ({math.exp(med):.2f}x) | "
              f"{p90:.3f} ({math.exp(p90):.2f}x) | {statistics.median(ratios):.3g} | "
              f"{min(ratios):.3g}..{max(ratios):.3g} | {under} / {len(ratios) - under} | "
              f"{statistics.median(logs):+.3f} |")

    print()
    print("Per workload (actual / predicted):")
    for engine in sorted(rows):
        for workload, ratio in rows[engine]:
            print(f"  {engine:16s} {workload:44s} {ratio:8.3g}")
    if other:
        print()
        print("Not measured:")
        for r in other:
            note = r.get("note", "")
            print(f"  {r['engine']:16s} {r['workload']:44s} {r['status']}: {note[:140]}")


if __name__ == "__main__":
    main(sys.argv[1:])
