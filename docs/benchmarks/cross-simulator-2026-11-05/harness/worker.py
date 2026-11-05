"""Run one (simulator, case) in a fresh process and write a JSON result.

usage: worker.py SIM CASE_JSON OUT_JSON
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import json
import os
import statistics
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402


class PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


def memory():
    k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    k32.GetCurrentProcess.restype = wt.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
    psapi.GetProcessMemoryInfo.restype = wt.BOOL
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    ok = psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    if not ok:
        return 0, 0
    return pmc.WorkingSetSize, pmc.PeakWorkingSetSize


def main():
    sim, case_path, out_path = sys.argv[1:4]
    with open(case_path) as f:
        case = json.load(f)
    case["gates"] = [(g[0], tuple(g[1]), tuple(g[2])) for g in case["gates"]]
    if "gates_list" in case:
        case["gates_list"] = [[(g[0], tuple(g[1]), tuple(g[2])) for g in gl] for gl in case["gates_list"]]
    if case.get("observable"):
        case["observable"] = [(c, {int(k): v for k, v in d.items()}) for c, d in case["observable"]]
    result = {"sim": sim, "case": case["name"], "status": "ok"}
    try:
        from builders import make_runner

        t0 = time.perf_counter()
        runner = make_runner(sim, case)
        result["t_init"] = time.perf_counter() - t0
        ws0, _ = memory()
        if case["task"] == "variational":
            # K evaluations with different angles, each rebuilt from the IR: the
            # typical variational loop. One warm-up evaluation first.
            options = runner.options()
            c = runner.build(case["gates_list"][0])
            r = runner.run(c, options)
            t0 = time.perf_counter()
            values = []
            for gl in case["gates_list"]:
                c = runner.build(gl)
                r = runner.run(c, options)
                values.append(runner.extract(r)["expectation"])
            result["t_loop"] = time.perf_counter() - t0
            result["t_per_eval"] = result["t_loop"] / len(case["gates_list"])
            result["values"] = values
            ex = runner.extract(r)
        else:
            t0 = time.perf_counter()
            circuit = runner.build(case["gates"])
            options = runner.options()
            result["t_build"] = time.perf_counter() - t0
            reps = case.get("reps", 3)
            times = []
            t0 = time.perf_counter()
            r = runner.run(circuit, options)
            result["t_first"] = time.perf_counter() - t0
            for _ in range(reps):
                t0 = time.perf_counter()
                r = runner.run(circuit, options)
                times.append(time.perf_counter() - t0)
            result["t_run"] = statistics.median(times)
            result["t_runs"] = times
            ex = runner.extract(r)
        ws1, peak = memory()
        result["ws_before_bytes"] = ws0
        result["peak_ws_bytes"] = peak
        result["peak_delta_bytes"] = max(0, peak - ws0)
        for k, v in ex.items():
            if isinstance(v, np.ndarray):
                path = os.path.splitext(out_path)[0] + f".{k}.npy"
                np.save(path, v)
                result[k + "_path"] = path
                if k == "statevector":
                    result["norm"] = float(np.vdot(v, v).real)
                # marginals from a probability vector
                if k == "probabilities":
                    n = case["n"]
                    pr = v.reshape([2] * n)
                    result["marginals"] = [float(pr.sum(axis=tuple(a for a in range(n) if a != n - 1 - q))[1]) for q in range(n)]
            elif k == "counts":
                n = case["n"]
                total = sum(v.values())
                marg = [0.0] * n
                for s, cnt in v.items():
                    for q in range(n):
                        if s[n - 1 - q] == "1":
                            marg[q] += cnt
                result["marginals"] = [m / total for m in marg]
                result["counts_total"] = total
                result["counts_top"] = sorted(v.items(), key=lambda kv: -kv[1])[:8]
                if n <= 22:
                    path = os.path.splitext(out_path)[0] + ".counts.json"
                    with open(path, "w") as f:
                        json.dump(v, f)
                    result["counts_path"] = path
            elif k == "explain":
                result["explain"] = v
            else:
                result[k] = v
    except MemoryError as e:
        result["status"] = "oom"
        result["error"] = repr(e)[:500]
    except Exception as e:  # noqa: BLE001
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        result["traceback"] = traceback.format_exc()[-2000:]
    with open(out_path, "w") as f:
        json.dump(result, f, default=str)


if __name__ == "__main__":
    main()
