"""Benchmark dashboard: MENOS (CPU and CUDA) against pure Qiskit.

Two subcommands:

  run     Sweeps qubit counts over one circuit family, timing MENOS
          (through the Qiskit converter and menos.run) and Qiskit's reference
          simulator, qiskit.quantum_info.Statevector, on the same circuit, and
          checks that their statevectors agree. Optionally merges the JSON that
          benchmarks/cuda_crossover_benchmark.cpp writes when
          MENOS_CROSSOVER_JSON is set, which is the only source of a forced CPU
          versus CUDA comparison: the Python API never names a backend, so the
          MENOS series here records whichever backend the planner chose.
  render  Embeds a results JSON in template.html and writes one self-contained
          HTML file.

The default family, "baseline", is the crossover benchmark's own (6 layers of
H on every qubit, T on q0, CX on alternating pairs), so all three series measure
the same work. --family picks "clifford" or "product" instead, which have no
CUDA crossover counterpart, so --crossover-json is refused with them. Nothing measured is ever filled in: a value that was not measured is
null in the JSON and blank on the page.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

import menos
from menos.qiskit.convert import convert_circuit

# 2: each timed series carries its samples and spread; environment records
# core counts, WSL, the CPU governor and the power state.
SCHEMA_VERSION = 2
CROSSOVER_SCHEMA_VERSION = 1
LAYERS = 6  # benchmarks/cuda_crossover_benchmark.cpp kLayers

# Same absolute tolerance as src/common/tolerance.h (kDefaultAbsTol); one
# tolerance policy across the whole project.
ABS_TOL = 1e-12

TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOL_DIR.parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "build" / "dashboard"
TEMPLATE = TOOL_DIR / "template.html"
DATA_PLACEHOLDER = "/*__MENOS_DASHBOARD_DATA__*/null"

# Bounds each external power query (powershell.exe, powercfg.exe), so a
# missing or slow WSL interop costs at most this long and records null.
PROBE_TIMEOUT_SECONDS = 10


def build_circuit(num_qubits: int) -> QuantumCircuit:
    """The crossover benchmark's circuit (make_circuit there), as Qiskit."""
    qc = QuantumCircuit(num_qubits)
    for layer in range(LAYERS):
        for q in range(num_qubits):
            qc.h(q)
        qc.t(0)
        for q in range(layer % 2, num_qubits - 1, 2):
            qc.cx(q, q + 1)
    return qc


def build_clifford(num_qubits: int) -> QuantumCircuit:
    """6 layers of H and S on every qubit, CX on alternating pairs."""
    qc = QuantumCircuit(num_qubits)
    for layer in range(LAYERS):
        for q in range(num_qubits):
            qc.h(q)
            qc.s(q)
        for q in range(layer % 2, num_qubits - 1, 2):
            qc.cx(q, q + 1)
    return qc


def build_product(num_qubits: int) -> QuantumCircuit:
    """6 layers of H and RZ on every qubit, no two-qubit gate: the planner can
    split it into independent one-qubit regions."""
    qc = QuantumCircuit(num_qubits)
    for layer in range(LAYERS):
        for q in range(num_qubits):
            qc.h(q)
            qc.rz(0.3 * (layer + 1), q)
    return qc


# name -> (builder, description of what one layer does)
FAMILIES = {
    "baseline": (build_circuit, "H on all qubits, T on q0, CX on alternating pairs, per layer"),
    "clifford": (build_clifford, "H and S on all qubits, CX on alternating pairs, per layer"),
    "product": (build_product, "H and RZ on all qubits, no entangling gate, per layer"),
}


def planner_backend(result: menos.Result) -> str:
    """The backend the plan recorded for region 0 (as examples/cuda.py reads it)."""
    region = next(line for line in result.explain().splitlines() if line.startswith("region 0"))
    for token in region.split():
        if token.startswith("backend="):
            return token.removeprefix("backend=")
    raise ValueError(f"no backend= in plan line: {region!r}")


def _run_menos(qc: QuantumCircuit) -> tuple[float, menos.Result]:
    start = time.perf_counter()
    options = menos.RunOptions()
    options.want_statevector = True
    result = menos.run(convert_circuit(qc), options)
    return time.perf_counter() - start, result


def _run_qiskit(qc: QuantumCircuit) -> tuple[float, Statevector]:
    start = time.perf_counter()
    state = Statevector(qc)
    return time.perf_counter() - start, state


def spread(samples: list[float]) -> dict:
    """Median, quartiles and range of one series' timed repeats. With a
    single repeat every statistic is that sample."""
    if len(samples) == 1:
        q1 = q3 = samples[0]
    else:
        q1, _, q3 = statistics.quantiles(samples, n=4, method="inclusive")
    return {
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "q1": q1,
        "q3": q3,
    }


def measure_point(num_qubits: int, repeats: int, family: str = "baseline") -> dict:
    """One warm-up run of each, then `repeats` timed runs. Each series keeps
    its raw samples and their spread; its *_seconds key is the median."""
    qc = FAMILIES[family][0](num_qubits)
    _run_menos(qc)
    _run_qiskit(qc)

    menos_wall: list[float] = []
    menos_execute: list[float] = []
    qiskit_wall: list[float] = []
    result = None
    state = None
    for _ in range(repeats):
        seconds, result = _run_menos(qc)
        menos_wall.append(seconds)
        menos_execute.append(result.execute_seconds)
        seconds, state = _run_qiskit(qc)
        qiskit_wall.append(seconds)

    deviation = float(np.max(np.abs(np.asarray(result.statevector) - state.data)))
    point = {
        "n": num_qubits,
        "gates": len(qc.data),
        "menos_backend": planner_backend(result),
    }
    for name, samples in (
        ("menos_wall", menos_wall),
        ("menos_execute", menos_execute),
        ("qiskit_wall", qiskit_wall),
    ):
        summary = spread(samples)
        point[f"{name}_seconds"] = summary["median"]
        point[f"{name}_spread"] = summary
        point[f"{name}_samples"] = samples
    point["max_abs_deviation"] = deviation
    point["agrees"] = deviation <= ABS_TOL
    return point


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _git_commit() -> str | None:
    """HEAD, marked -dirty when tracked files differ from it, so a result is
    never attributed to a commit that did not produce it."""
    commit = _git("rev-parse", "--short", "HEAD")
    if not commit:
        return None
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    return f"{commit}-dirty" if dirty else commit


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _physical_cpus() -> int | None:
    """Distinct (physical id, core id) pairs in /proc/cpuinfo; None where
    that file or those fields do not exist."""
    text = _read(Path("/proc/cpuinfo"))
    if text is None:
        return None
    cores = set()
    physical = core = None
    for line in [*text.splitlines(), ""]:
        if not line.strip():
            if physical is not None and core is not None:
                cores.add((physical, core))
            physical = core = None
        elif line.startswith("physical id"):
            physical = line.split(":", 1)[1].strip()
        elif line.startswith("core id"):
            core = line.split(":", 1)[1].strip()
    return len(cores) or None


def _is_wsl() -> bool:
    version = _read(Path("/proc/version"))
    return version is not None and "microsoft" in version.lower()


def _probe(*command: str) -> str | None:
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            check=True,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _linux_power() -> dict:
    ac_online = None
    battery_percent = None
    try:
        supplies = sorted(Path("/sys/class/power_supply").iterdir())
    except OSError:
        supplies = []
    for supply in supplies:
        kind = _read(supply / "type")
        if kind == "Mains":
            online = _read(supply / "online")
            if online in ("0", "1"):
                ac_online = online == "1"
        elif kind == "Battery":
            capacity = _read(supply / "capacity")
            if capacity is not None and capacity.isdigit():
                battery_percent = int(capacity)
    return {"ac_online": ac_online, "battery_percent": battery_percent, "power_scheme": None}


# Prints "<PowerLineStatus> <BatteryLifePercent>", e.g. "Online 0.87".
_POWER_STATUS_COMMAND = (
    "Add-Type -AssemblyName System.Windows.Forms; "
    "$s = [System.Windows.Forms.SystemInformation]::PowerStatus; "
    "'{0} {1}' -f $s.PowerLineStatus, "
    "$s.BatteryLifePercent.ToString([System.Globalization.CultureInfo]::InvariantCulture)"
)


def _windows_power() -> dict:
    """AC state and battery level from .NET's PowerStatus, and the active
    power scheme from powercfg; works natively and through WSL interop."""
    ac_online = None
    battery_percent = None
    status = _probe(
        "powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _POWER_STATUS_COMMAND
    )
    parts = status.split() if status is not None else []
    if len(parts) == 2:
        if parts[0] in ("Online", "Offline"):
            ac_online = parts[0] == "Online"
        try:
            fraction = float(parts[1])
        except ValueError:
            fraction = None
        # Windows reports 255 (not a fraction) when there is no battery or
        # its level is unknown.
        if fraction is not None and 0.0 <= fraction <= 1.0:
            battery_percent = round(fraction * 100)
    # "Power Scheme GUID: <guid>  (<name>)"
    scheme = _probe("powercfg.exe", "/getactivescheme")
    scheme_name = None
    if scheme is not None and scheme.endswith(")") and "(" in scheme:
        scheme_name = scheme[scheme.rindex("(") + 1 : -1].strip() or None
    return {"ac_online": ac_online, "battery_percent": battery_percent, "power_scheme": scheme_name}


def power_state() -> dict:
    """Mains/battery state of the machine. A value that cannot be read is
    None, never assumed: an unknown power state is recorded as unknown."""
    if sys.platform == "win32" or _is_wsl():
        return _windows_power()
    if sys.platform.startswith("linux"):
        return _linux_power()
    return {"ac_online": None, "battery_percent": None, "power_scheme": None}


def environment() -> dict:
    return {
        # datetime.timezone.utc, not datetime.UTC: the alias is Python 3.11+, and
        # MENOS supports 3.10 (docs/adr/0022).
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "cpu": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "physical_cpus": _physical_cpus(),
        "wsl": _is_wsl(),
        "cpu_governor": _read(Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")),
        "power": power_state(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "qiskit": qiskit.__version__,
        "menos": menos.__version__,
        "git_commit": _git_commit(),
    }


def load_crossover(path: Path) -> dict:
    """Reads and checks the crossover benchmark's MENOS_CROSSOVER_JSON output."""
    document = json.loads(path.read_text())
    if document.get("schema_version") != CROSSOVER_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: crossover schema_version {document.get('schema_version')!r}, "
            f"expected {CROSSOVER_SCHEMA_VERSION}"
        )
    if "device_present" not in document:
        raise ValueError(f"{path}: not a crossover benchmark report (no device_present)")
    for mode in ("statevector", "expectation"):
        for row in document.get(mode, []):
            missing = {"n", "cpu_seconds", "cuda_seconds", "vram_ok"} - row.keys()
            if missing:
                raise ValueError(f"{path}: {mode} row missing {sorted(missing)}")
    return document


def sweep(
    qubit_counts: list[int],
    repeats: int,
    crossover: dict | None = None,
    progress=None,
    family: str = "baseline",
) -> dict:
    # The environment (including power) is read before the timed runs, and
    # the power state again after them, so a sweep that was unplugged
    # part-way through shows it.
    env = environment()
    points = []
    for n in qubit_counts:
        point = measure_point(n, repeats, family)
        if progress is not None:
            progress(point)
        points.append(point)
    env["power_after"] = power_state()
    return {
        "schema_version": SCHEMA_VERSION,
        "environment": env,
        "circuit": {
            "family": family,
            "layers": LAYERS,
            "description": FAMILIES[family][1],
        },
        "repeats": repeats,
        "abs_tol": ABS_TOL,
        "menos_vs_qiskit": points,
        "crossover": crossover,
    }


def render(document: dict, template: Path = TEMPLATE) -> str:
    """Embeds `document` in the template. `</` is escaped so no string in the
    data can close the <script> element it is embedded in."""
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"results schema_version {document.get('schema_version')!r}")
    html = template.read_text(encoding="utf-8")
    if html.count(DATA_PLACEHOLDER) != 1:
        raise ValueError(f"{template}: expected exactly one {DATA_PLACEHOLDER}")
    data = json.dumps(document, allow_nan=False).replace("</", "<\\/")
    return html.replace(DATA_PLACEHOLDER, data)


def _print_point(point: dict) -> None:
    m = point["menos_wall_spread"]
    q = point["qiskit_wall_spread"]
    print(
        f"n={point['n']:>2}  menos[{point['menos_backend']}] "
        f"{m['median']:.4f} s ({m['min']:.4f}-{m['max']:.4f})  "
        f"qiskit {q['median']:.4f} s ({q['min']:.4f}-{q['max']:.4f})  "
        f"max|dev| {point['max_abs_deviation']:.1e}  {'ok' if point['agrees'] else 'MISMATCH'}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="measure and write results JSON")
    run_p.add_argument("--min-qubits", type=int, default=4)
    run_p.add_argument("--max-qubits", type=int, default=22)
    run_p.add_argument("--step", type=int, default=2)
    run_p.add_argument("--repeats", type=int, default=10)
    run_p.add_argument("--family", choices=sorted(FAMILIES), default="baseline")
    run_p.add_argument("--crossover-json", type=Path, help="MENOS_CROSSOVER_JSON output")
    run_p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR / "results.json")

    render_p = sub.add_parser("render", help="write the HTML dashboard from results JSON")
    render_p.add_argument("--results", type=Path, default=DEFAULT_OUT_DIR / "results.json")
    render_p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR / "dashboard.html")

    args = parser.parse_args(argv)
    if args.command == "run":
        if args.min_qubits < 1 or args.max_qubits < args.min_qubits or args.step < 1:
            parser.error("need 1 <= --min-qubits <= --max-qubits and --step >= 1")
        if args.repeats < 1:
            parser.error("--repeats must be >= 1")
        if args.crossover_json and args.family != "baseline":
            parser.error("--crossover-json only matches the baseline family")
        crossover = load_crossover(args.crossover_json) if args.crossover_json else None
        counts = list(range(args.min_qubits, args.max_qubits + 1, args.step))
        document = sweep(counts, args.repeats, crossover, progress=_print_point, family=args.family)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
        print(f"wrote {args.out}")
        mismatches = [p["n"] for p in document["menos_vs_qiskit"] if not p["agrees"]]
        if mismatches:
            print(f"statevector MISMATCH against Qiskit at n = {mismatches}", file=sys.stderr)
            return 1
        return 0

    document = json.loads(args.results.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(document), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
