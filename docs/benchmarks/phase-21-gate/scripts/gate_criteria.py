"""Phase 21 gate criterion 4 (docs/adr/0025 section 9, Q9): no benchmark
slowdown, as same-session ratios of the current commit against the phase's
base, 1f89192, from gate.sh's JSON.

For every benchmark, the median of every repetition in every round is taken
for each build, and the ratio is head / base (above 1 is slower). The owner's
figures: every dense benchmark (the Phase 15 dense benchmark) within 2%, and
BM_MpsUnmonitored within 5%. The dense and stabilizer monitor benchmarks are
printed beside them for the record. Exits 1 if any gated ratio fails.

    gate_criteria.py OUTPUT_DIRECTORY
"""

import glob
import json
import statistics
import sys

DENSE_LIMIT = 1.02
MPS_LIMIT = 1.05


def times(out, kind, build):
    """Benchmark name -> every repetition's real time, over all rounds."""
    result = {}
    for path in sorted(glob.glob(f"{out}/{kind}-{build}-*.json")):
        for row in json.load(open(path))["benchmarks"]:
            if row.get("run_type") != "iteration":
                continue
            result.setdefault(row["run_name"], []).append(row["real_time"])
    return result


def main(out):
    ok_all = True
    for kind, gated, limit in (("dense", lambda name: True, DENSE_LIMIT),
                               ("monitor", lambda name: name == "BM_MpsUnmonitored", MPS_LIMIT)):
        base = times(out, kind, "base")
        head = times(out, kind, "head")
        print(f"{kind}: head / base median real time")
        for name in sorted(base):
            if name not in head:
                print(f"  {name}: missing in head")
                ok_all = False
                continue
            b = statistics.median(base[name])
            h = statistics.median(head[name])
            ratio = h / b
            if gated(name):
                ok = ratio <= limit
                ok_all &= ok
                verdict = f"{'PASS' if ok else 'FAIL'} (limit {limit:.2f})"
            else:
                verdict = "(reported)"
            print(f"  {name}: {b:.4f} -> {h:.4f}  ratio {ratio:.3f}  "
                  f"n={len(base[name])}/{len(head[name])}  {verdict}")
    print("criterion 4:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
