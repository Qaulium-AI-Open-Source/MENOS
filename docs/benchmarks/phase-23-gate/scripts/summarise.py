"""Phase 23 step 5: the gate runs' table (docs/adr/0027 section 10.3).

Reads every win-run-*/gate.json and wsl-run-*/gate.json beside this script's
record directory and prints the Markdown tables of the ADR: the overlap runs
(criterion 2) and the reach runs (criteria 1 and 3).

    python docs/benchmarks/phase-23-gate/scripts/summarise.py
"""

from __future__ import annotations

import json
import pathlib
import sys

RECORD = pathlib.Path(__file__).resolve().parent.parent


def load(prefix: str) -> list[tuple[str, dict]]:
    runs = []
    for directory in sorted(RECORD.glob(f"{prefix}-run-*")):
        path = directory / "gate.json"
        if path.exists():
            runs.append((directory.name, json.loads(path.read_text(encoding="utf-8"))))
    return runs


def seconds(value: float) -> str:
    return f"{value:.1f}"


def main() -> int:
    runs = load("win") + load("wsl")
    if not runs:
        print("no gate.json found under", RECORD, file=sys.stderr)
        return 1
    print("## Overlap: the disk tier forced against the in-memory dense engine (criterion 2)\n")
    print("| run | n | in-memory s | disk s | passes | chunk / window | counts | expectation | file gone |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, report in runs:
        o = report.get("overlap")
        if not o:
            continue
        m, d = o["in_memory"], o["disk"]
        print(
            f"| {name} | {o['qubits']} | {seconds(m['wall_seconds'])} | {seconds(d['wall_seconds'])} "
            f"| {d['passes']} | {d['chunk_qubits']} / {d['window_qubits']} "
            f"| {'bitwise' if o['bitwise_counts'] else 'DIFFERENT'} "
            f"| {'bitwise' if o['bitwise_expectation'] else 'DIFFERENT'} "
            f"| {'yes' if o['scratch_files_after'] == 0 else 'NO'} |"
        )
    print("\n## Reach: the planner left to itself over the budget (criteria 1 and 3)\n")
    print("| run | n | tier | passes | chunk / window | file bytes | predicted s | measured s | measured / predicted | within 2x | peak host | peak disk | file gone |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, report in runs:
        r = report.get("reach", {})
        if "run" not in r:
            print(f"| {name} | {r.get('qubits', '?')} | refused: {r.get('error', '?')} |")
            continue
        run = r["run"]
        print(
            f"| {name} | {r['qubits']} | {run['tier']} | {run['passes']} "
            f"| {run['chunk_qubits']} / {run['window_qubits']} | {run['file_bytes']} "
            f"| {seconds(run['predicted_seconds'])} | {seconds(run['wall_seconds'])} "
            f"| {run['measured_over_predicted']:.2f} | {'yes' if run['within_factor_two'] else 'NO'} "
            f"| {run['peak_host_bytes']} | {run['peak_disk_bytes']} "
            f"| {'yes' if run['scratch_files_after'] == 0 else 'NO'} |"
        )
    print("\n## Scratch bandwidth the plans used (measured once per run)\n")
    print("| run | write GB/s | read GB/s | available bytes |")
    print("|---|---|---|---|")
    for name, report in runs:
        run = report.get("reach", {}).get("run") or report.get("overlap", {}).get("disk")
        if run and "disk_write_bytes_per_second" in run:
            print(
                f"| {name} | {run['disk_write_bytes_per_second'] / 1e9:.2f} "
                f"| {run['disk_read_bytes_per_second'] / 1e9:.2f} | {run.get('disk_available_bytes', '?')} |"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
