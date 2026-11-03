#!/usr/bin/env python3
"""Phase 23 step 1: the tables of docs/adr/0027 section 10.1 from the run JSON.

Reads every run directory under docs/benchmarks/phase-23-baseline/ (wsl-run-*,
wsl-ext4-run-*, win-run-*) and analysis/passes-gate-sizes.json, prints Markdown
tables of the measured figures (medians within a run; the spread across runs as
min-max), then the predictions the owner's Q4 and Q7 rules ask for, computed as
same-run ratios: the host-offloaded tier at n = 29 against the in-memory CPU
engine, and the disk tier at n = 31 against the 30-minute rule for 100 gates.

Usage: python3 summarise.py [record-dir]   (default: the parent of this file)
"""

from __future__ import annotations

import glob
import json
import os
import statistics
import sys

GIB = 1 << 30
MINUTE = 60.0
Q7_LIMIT_SECONDS = 30 * MINUTE
GATE_SIZE_HOST_OFFLOADED = 29  # 8 GiB state against 7.4 GB of free VRAM
GATE_SIZE_DISK = 31  # 32 GiB state against the 16 GiB host budget
WINDOW_LIMIT_HOST_OFFLOADED = 28  # L + w: a 4 GiB window is the largest that fits the device
WINDOW_LIMIT_DISK = 29  # L + w: an 8 GiB window leaves room in the 16 GiB budget


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def runs(record_dir, pattern, name):
    out = []
    for d in sorted(glob.glob(os.path.join(record_dir, pattern))):
        p = os.path.join(d, name)
        if os.path.exists(p):
            out.append((os.path.basename(d), load(p)))
    return out


def fmt_range(values, digits=2):
    if not values:
        return "-"
    lo, hi = min(values), max(values)
    if abs(hi - lo) < 10 ** -digits:
        return f"{statistics.median(values):.{digits}f}"
    return f"{lo:.{digits}f}-{hi:.{digits}f}"


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")
    print()


# ---------------------------------------------------------------------------
# (i) disk
# ---------------------------------------------------------------------------


def disk_tables(sets):
    for label, cpu_runs in sets:
        if not cpu_runs:
            continue
        facts = cpu_runs[0][1]["facts"]
        print(f"**{label}** ({len(cpu_runs)} runs): scratch `{facts['scratch_directory']}`, "
              f"capacity {facts['volume_capacity_bytes'] / GIB:.1f} GiB, "
              f"host budget {facts['host_budget_bytes'] / GIB:.1f} GiB, "
              f"{facts['hardware_threads']} threads; a 2 GiB file; medians over "
              f"{cpu_runs[0][1]['disk_repetitions']} repetitions per run, min-max across runs.\n")
        keys = []
        for row in cpu_runs[0][1]["disk"]:
            keys.append((row["chunk_bytes"], row["path"]))
        out = []
        for chunk, path in keys:
            w, wf, r = [], [], []
            for _, cpu in cpu_runs:
                for row in cpu["disk"]:
                    if row["chunk_bytes"] == chunk and row["path"] == path and row["measured"]:
                        w.append(row["write_gb_per_second"])
                        wf.append(row["write_and_flush_gb_per_second"])
                        r.append(row["read_gb_per_second"])
            out.append((f"{chunk // (1 << 20)} MiB", path, fmt_range(w), fmt_range(wf), fmt_range(r)))
        table(["chunk", "path", "write GB/s", "write+flush GB/s", "read GB/s (cache dropped)"], out)


def disk_figures(cpu, chunk_bytes, path):
    for row in cpu["disk"]:
        if row["chunk_bytes"] == chunk_bytes and row["path"] == path and row["measured"]:
            return row["write_and_flush_gb_per_second"] * 1e9, row["read_gb_per_second"] * 1e9
    return None


# ---------------------------------------------------------------------------
# (ii), (iii) CUDA
# ---------------------------------------------------------------------------


def cuda_tables(cuda_runs):
    if not cuda_runs:
        return
    gpu = cuda_runs[0][1]["gpu"]
    facts = cuda_runs[0][1]["facts"]
    print(f"**CUDA (WSL)** ({len(cuda_runs)} runs): {gpu['name']}, {gpu['total_memory_bytes'] / 1e9:.2f} GB total, "
          f"{gpu['free_memory_bytes'] / 1e9:.2f} GB free; runtime {facts['runtime_version']}, driver {facts['driver_version']}; "
          f"managed memory {facts['managed_memory']}, concurrent managed access {facts['concurrent_managed_access']}, "
          f"pageable memory access {facts['pageable_memory_access']}; medians over {cuda_runs[0][1]['repetitions']} "
          f"repetitions per run, min-max across runs.\n")
    print("Transfers (GB/s of the median):\n")
    out = []
    for row0 in cuda_runs[0][1]["transfer"]:
        h, d = [], []
        for _, c in cuda_runs:
            for row in c["transfer"]:
                if row["chunk_bytes"] == row0["chunk_bytes"] and row["source"] == row0["source"] and row["measured"]:
                    h.append(row["h2d_gb_per_second"])
                    d.append(row["d2h_gb_per_second"])
        out.append((f"{row0['chunk_bytes'] // GIB} GiB", row0["source"], fmt_range(h), fmt_range(d)))
    table(["chunk", "source", "host to device", "device to host"], out)

    print("A window copied in, k one-qubit gates applied, copied out (same run): the copies in units of one gate on the resident window.\n")
    out = []
    for row0 in cuda_runs[0][1]["window"]:
        v = []
        for _, c in cuda_runs:
            for row in c["window"]:
                if (row["chunk_bytes"] == row0["chunk_bytes"] and row["gates"] == row0["gates"]
                        and row["source"] == row0["source"] and row["measured"]):
                    v.append(row["copies_in_gates"])
        out.append((f"{row0['chunk_bytes'] // GIB} GiB", row0["gates"], row0["source"], fmt_range(v, 1)))
    table(["window", "k", "source", "copies / one gate"], out)

    print("Overlap: a 1 GiB pinned copy on a second stream against eight memsets on the first (milliseconds).\n")
    out = []
    for name, c in cuda_runs:
        for row in c["overlap"]:
            out.append((name, f"{row['work_only_seconds']['median'] * 1e3:.1f}", f"{row['copy_only_seconds']['median'] * 1e3:.1f}",
                        f"{row['sequential_seconds']['median'] * 1e3:.1f}", f"{row['concurrent_seconds']['median'] * 1e3:.1f}",
                        f"{row['overlap_gain_seconds'] * 1e3:.1f}"))
    table(["run", "work only", "copy only", "sequential", "concurrent", "gain"], out)

    print("`cudaMallocManaged` beyond device memory, one-qubit sweeps (seconds); the resident reference is the 2^28 sweep scaled by 2^(n-28).\n")
    out = []
    for name, c in cuda_runs:
        for row in c["managed"]:
            if row["measured"]:
                out.append((name, row["n"], row["outcome"], f"{row['sweep_seconds']['median']:.3f}",
                            f"{row['resident_sweep_scaled_seconds']:.3f}",
                            f"{row['sweep_seconds']['median'] / row['resident_sweep_scaled_seconds']:.1f}x"))
            else:
                out.append((name, row["n"], row["outcome"], "-", "-", "-"))
    table(["run", "n", "outcome", "sweep", "resident, scaled", "ratio"], out)


# ---------------------------------------------------------------------------
# (v) in-memory engines
# ---------------------------------------------------------------------------


def per_gate(section, n):
    for row in section:
        if row["n"] == n and row["measured"]:
            return row["per_gate_seconds"]
    return None


def circuit_seconds(section, n):
    for row in section:
        if row["n"] == n and row["measured"]:
            return row["circuit_seconds"]["median"], row["gates"]
    return None


def in_memory_tables(cpu_sets, cuda_runs):
    print("Per-gate seconds of the crossover circuit, (circuit - empty circuit) / gates, medians per run, min-max across runs.\n")
    ns = [26, 27, 28, 29, 30]
    header = ["n"]
    cols = []
    for label, cpu_runs in cpu_sets:
        if cpu_runs:
            header.append(f"CPU, {label}")
            cols.append([[per_gate(c["in_memory"], n) for _, c in cpu_runs] for n in ns])
    if cuda_runs:
        header.append("CUDA in VRAM (WSL)")
        cols.append([[per_gate(c["in_vram"], n) for _, c in cuda_runs] for n in ns])
    out = []
    for i, n in enumerate(ns):
        row = [n]
        for col in cols:
            vals = [v for v in col[i] if v is not None]
            row.append(fmt_range(vals, 4) if vals else "not run")
        out.append(row)
    table(header, out)
    if cuda_runs:
        wsl = dict(cpu_sets).get("WSL (tmpfs scratch)") or dict(cpu_sets).get("WSL (ext4 scratch)")
        if wsl:
            print("Same-run CPU / CUDA per-gate ratio on WSL (each CUDA run against the CPU side of the same run):\n")
            out = []
            for (name, cuda), (_, cpu) in zip(cuda_runs, wsl):
                row = [name]
                for n in (26, 27, 28):
                    a, b = per_gate(cpu["in_memory"], n), per_gate(cuda["in_vram"], n)
                    row.append(f"{a / b:.2f}x" if a and b else "-")
                out.append(row)
            table(["run", "n = 26", "n = 27", "n = 28"], out)


# ---------------------------------------------------------------------------
# predictions
# ---------------------------------------------------------------------------


def best_passes(pass_rows, family, n, window_limit, key="passes_window_gmax"):
    """The fewest passes over every (L, w) with L + w <= window_limit, w >= gmax."""
    best = None
    for row in pass_rows:
        if row["family"] != family or row["n"] != n:
            continue
        L = row["local_qubits"]
        gmax = row["max_global_operands"]
        for extra, field in ((0, "passes_window_gmax"), (1, "passes_window_gmax_plus_1"), (2, "passes_window_gmax_plus_2")):
            w = gmax + extra
            if L + w > window_limit:
                continue
            src = row if key == "passes_window_gmax" else row["first_100_ops"]
            p = src[field]
            if best is None or p < best[0]:
                best = (p, L, w, row["gates"], row["global_gates"])
    return best


def host_offloaded_prediction(pass_rows, cuda_runs, wsl_cpu_runs):
    print(f"### Q4: the host-offloaded tier at n = {GATE_SIZE_HOST_OFFLOADED} against the in-memory CPU engine\n")
    print("Model: seconds = passes x (state bytes / host-to-device + state bytes / device-to-host, 2 GiB pageable, "
          "same run) + gates x device per-gate seconds at n = 29 (2 x the measured n = 28 in-VRAM figure, same run). "
          "The window is the largest that fits the device (L + w <= 28, a 4 GiB window); passes are the fewest over L and w. "
          "The CPU figure is the measured crossover circuit at n = 29 of the same run, and for the other families gates x "
          "the measured CPU per-gate seconds at n = 29. Wins if the ratio is below 1.\n")
    state = 16 * (1 << GATE_SIZE_HOST_OFFLOADED)
    families = ["crossover", "ladder", "random_pairs", "qft_like"]
    out = []
    verdict = []
    for (name, cuda), (_, cpu) in zip(cuda_runs, wsl_cpu_runs):
        t = [r for r in cuda["transfer"] if r["chunk_bytes"] == 2 * GIB and r["source"] == "pageable"][0]
        h2d, d2h = t["h2d_gb_per_second"] * 1e9, t["d2h_gb_per_second"] * 1e9
        pass_seconds = state / h2d + state / d2h
        dev_gate = 2 * per_gate(cuda["in_vram"], 28)
        cpu_gate = per_gate(cpu["in_memory"], 29)
        for fam in families:
            p, L, w, gates, globals_ = best_passes(pass_rows, fam, GATE_SIZE_HOST_OFFLOADED, WINDOW_LIMIT_HOST_OFFLOADED)
            predicted = p * pass_seconds + gates * dev_gate
            if fam == "crossover":
                cpu_seconds, _ = circuit_seconds(cpu["in_memory"], 29)
                cpu_note = "measured"
            else:
                cpu_seconds = gates * cpu_gate
                cpu_note = "gates x per-gate"
            ratio = predicted / cpu_seconds
            verdict.append(ratio)
            out.append((name, fam, gates, f"L={L} w={w}", p, f"{pass_seconds:.2f}", f"{p * pass_seconds:.0f}",
                        f"{gates * dev_gate:.0f}", f"{predicted:.0f}", f"{cpu_seconds:.0f} ({cpu_note})", f"{ratio:.2f}x"))
    table(["run", "family", "gates", "window", "passes", "s / pass", "transfer s", "compute s", "predicted s",
           "CPU in memory s", "ratio"], out)
    print(f"Gates a pass must carry to break even (transfer per pass / (CPU gate - device gate), same run): "
          + ", ".join(f"{name}: {(state / ([r for r in cuda['transfer'] if r['chunk_bytes'] == 2 * GIB and r['source'] == 'pageable'][0]['h2d_gb_per_second'] * 1e9) + state / ([r for r in cuda['transfer'] if r['chunk_bytes'] == 2 * GIB and r['source'] == 'pageable'][0]['d2h_gb_per_second'] * 1e9)) / (per_gate(cpu['in_memory'], 29) - 2 * per_gate(cuda['in_vram'], 28)):.0f}"
                      for (name, cuda), (_, cpu) in zip(cuda_runs, wsl_cpu_runs)) + ".\n")
    print(f"Best ratio over every run and family: {min(verdict):.2f}x "
          + ("(the tier wins somewhere)" if min(verdict) < 1 else "(the tier never beats the in-memory CPU engine)") + ".\n")


def disk_prediction(pass_rows, label, cpu_runs, chunk_bytes, path):
    print(f"### Q7: the disk tier at n = {GATE_SIZE_DISK} on {label} ({path}, {chunk_bytes // (1 << 20)} MiB chunks)\n")
    print("Model: seconds = passes x (state bytes / write+flush + state bytes / read, same run) + gates x CPU per-gate "
          "seconds at n = 31 (4 x the measured n = 29 figure, same run). The window is the largest the 16 GiB budget "
          "holds beside the outputs (L + w <= 29, an 8 GiB window); passes are the fewest over L and w. The 100-gate "
          "rule is applied three ways: the first 100 operations of each family, the family's passes scaled to 100 gates, "
          "and the worst case of one pass per gate. Limit: 30 minutes.\n")
    state = 16 * (1 << GATE_SIZE_DISK)
    families = ["crossover", "ladder", "random_pairs", "qft_like"]
    out = []
    worst = 0.0
    for name, cpu in cpu_runs:
        fig = disk_figures(cpu, chunk_bytes, path)
        if fig is None:
            continue
        write, read = fig
        pass_seconds = state / write + state / read
        cpu_gate = 4 * per_gate(cpu["in_memory"], 29)
        for fam in families:
            p, L, w, gates, _ = best_passes(pass_rows, fam, GATE_SIZE_DISK, WINDOW_LIMIT_DISK)
            q, Lq, wq, _, _ = best_passes(pass_rows, fam, GATE_SIZE_DISK, WINDOW_LIMIT_DISK, key="first_100_ops")
            whole = p * pass_seconds + gates * cpu_gate
            prefix = q * pass_seconds + 100 * cpu_gate
            scaled = p * 100 / gates * pass_seconds + 100 * cpu_gate
            worst = max(worst, prefix, scaled)
            out.append((name, fam, f"{pass_seconds:.1f}", f"{cpu_gate:.2f}", f"{gates} / {p} (L={L} w={w})", f"{whole / MINUTE:.1f}",
                        f"{q} (L={Lq} w={wq})", f"{prefix / MINUTE:.1f}", f"{p * 100 / gates:.1f}", f"{scaled / MINUTE:.1f}"))
        worst_case = 100 * pass_seconds + 100 * cpu_gate
        out.append((name, "worst case", f"{pass_seconds:.1f}", f"{cpu_gate:.2f}", "-", "-", "100", f"{worst_case / MINUTE:.1f}", "100", f"{worst_case / MINUTE:.1f}"))
    table(["run", "family", "s / pass", "s / gate", "whole circuit: gates / passes", "whole min",
           "first 100 ops: passes", "min", "passes per 100 gates", "min"], out)
    print(f"Largest family figure for 100 gates: {worst / MINUTE:.1f} min against the 30-minute limit "
          + ("(passes)" if worst <= Q7_LIMIT_SECONDS else "(FAILS)") + "; the one-pass-per-gate worst case is in the last row of each run.\n")


def main():
    record_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wsl_cuda = runs(record_dir, "wsl-run-*", "cuda.json")
    wsl_cpu = runs(record_dir, "wsl-run-*", "cpu.json")
    wsl_ext4 = runs(record_dir, "wsl-ext4-run-*", "cpu.json")
    win_cpu = runs(record_dir, "win-run-*", "cpu.json")
    pass_rows = load(os.path.join(record_dir, "analysis", "passes-gate-sizes.json"))
    cpu_sets = [("Windows (NTFS, drive C:)", win_cpu), ("WSL (ext4 scratch)", wsl_ext4), ("WSL (tmpfs scratch)", wsl_cpu)]

    print("## (i) Disk\n")
    disk_tables(cpu_sets)
    print("## (ii), (iii) CUDA transfers, windows, overlap, managed memory\n")
    cuda_tables(wsl_cuda)
    print("## (iv) Pass counts at the gate sizes\n")
    print("From analysis/passes-gate-sizes.txt (the benchmark's own table covers n = 28, 30, 32, 34). "
          "Fewest passes over L + w within the window limit, whole circuit and the first 100 operations:\n")
    out = []
    for fam in ["crossover", "ladder", "random_pairs", "qft_like"]:
        for n, limit in ((GATE_SIZE_HOST_OFFLOADED, WINDOW_LIMIT_HOST_OFFLOADED), (GATE_SIZE_DISK, WINDOW_LIMIT_DISK)):
            p, L, w, gates, globals_ = best_passes(pass_rows, fam, n, limit)
            q, Lq, wq, _, _ = best_passes(pass_rows, fam, n, limit, key="first_100_ops")
            out.append((fam, n, gates, globals_, f"{p} (L={L} w={w})", f"{q} (L={Lq} w={wq})"))
    table(["family", "n", "gates", "global gates", "passes, whole circuit", "passes, first 100 ops"], out)
    print("## (v) In-memory engines\n")
    in_memory_tables(cpu_sets, wsl_cuda)
    print("## Predictions\n")
    if wsl_cuda and wsl_cpu:
        host_offloaded_prediction(pass_rows, wsl_cuda, wsl_cpu)
    for label, cpu_runs in cpu_sets:
        if cpu_runs and "tmpfs" not in label:
            # The best windows at n = 31 use 512 MiB chunks (L = 25); the 1 GiB
            # row is the nearest measured chunk size.
            for path in ("explicit buffered", "explicit unbuffered"):
                disk_prediction(pass_rows, label, cpu_runs, GIB, path)


if __name__ == "__main__":
    main()
