#!/usr/bin/env python3
"""Compare Google Benchmark JSON runs of a before and an after binary.

usage: compare.py BEFORE.json[,BEFORE2.json...] AFTER.json[,AFTER2.json...]

Each side's per-repetition real times are pooled across its files (runs made
alternately, before/after/before/after, so that drift in the machine's state
falls on both sides). Prints, per benchmark: the pooled median of each side,
the speed-up (before / after), the before side's spread (max/min - 1 over its
repetitions) and the regression threshold of the Phase 24 gate,
max(5%, 2 x spread) (owner, 2026-11-03), flagging any job slower than that.
"""
import json
import statistics
import sys


def load(paths):
    times = {}
    for path in paths.split(","):
        with open(path) as f:
            data = json.load(f)
        for b in data["benchmarks"]:
            if b.get("run_type") != "iteration":
                continue
            times.setdefault(b["run_name"], []).append(b["real_time"])
    return times


def main():
    before, after = load(sys.argv[1]), load(sys.argv[2])
    print(f"{'benchmark':58} {'before':>11} {'after':>11} {'speed-up':>9} {'spread':>7} {'limit':>6}")
    flagged = []
    for name in before:
        if name not in after:
            continue
        b, a = statistics.median(before[name]), statistics.median(after[name])
        spread = max(before[name]) / min(before[name]) - 1.0
        limit = max(0.05, 2.0 * spread)
        slower = a / b - 1.0
        flag = "  SLOWER" if slower > limit else ""
        if flag:
            flagged.append(name)
        print(f"{name:58} {b:9.1f}us {a:9.1f}us {b / a:8.2f}x {spread * 100:6.1f}% {limit * 100:5.0f}%{flag}")
    print(f"\n{len(flagged)} job(s) slower than the gate's limit: {', '.join(flagged) or 'none'}")


if __name__ == "__main__":
    main()
