"""Benchmark driver: builds the case matrix, runs one worker process per
(simulator, case), scores every output against the reference, and writes
results.jsonl + summary.md.

usage: driver.py [--scale smoke|full] [--only FAMILY,...] [--sims SIM,...] [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ir  # noqa: E402

PY = os.path.join(os.path.dirname(HERE), "benchvenv", "Scripts", "python.exe")
ALL_SIMS = ["menos", "menos-fused", "menos-1t", "aer", "aer-mps", "aer-dm", "lightning", "default.qubit",
            "default.mixed", "cirq", "cirq-clifford", "cirq-dm", "qsim"]


def case(name, family, n, gates, task, sims, *, shots=0, observable=None, noise=None, menos_noise=None,
         reference="aer", reference_method="statevector", reference_output=None, timeout=600, reps=3, extra=None):
    if task == "counts" and reference_output is None:
        reference_output = "probabilities" if n <= 22 else "marginals"
    d = {"name": name, "family": family, "n": n, "gates": gates, "task": task, "sims": sims, "shots": shots,
         "observable": observable, "noise": noise, "menos_noise": menos_noise, "reference": reference,
         "reference_method": reference_method, "reference_output": reference_output, "timeout": timeout,
         "reps": reps, "gate_count": len([g for g in gates if not g[0].startswith("dep")])}
    if extra:
        d.update(extra)
    return d


def build_cases(scale):
    S = scale == "smoke"
    cases = []
    dense = ["menos", "aer", "lightning", "qsim"]

    # A. dense random brickwork, statevector
    for n in ([6, 8] if S else [16, 20, 22]):
        sims = dense + ["cirq"] + (["default.qubit"] if n <= 16 else []) + (["menos-1t"] if n in (16, 20) else [])
        cases.append(case(f"dense_random_sv_n{n}", "dense_random", n, ir.dense_random(n, 20, 1), "statevector", sims))
    # dense random, counts at larger sizes
    for n in ([10] if S else [24, 26]):
        cases.append(case(f"dense_random_counts_n{n}", "dense_random", n, ir.dense_random(n, 20, 1), "counts", dense,
                          shots=1024, timeout=900))
    # P. deep, small: per-gate overhead
    n, depth = (6, 50) if S else (14, 400)
    cases.append(case(f"deep_dense_sv_n{n}_d{depth}", "deep_dense", n, ir.dense_random(n, depth, 7), "statevector",
                      dense + ["cirq", "default.qubit"]))
    # O. sampling throughput
    n = 6 if S else 20
    cases.append(case(f"sampling_n{n}_100k", "sampling", n, ir.dense_random(n, 20, 3), "counts",
                      dense + ["cirq"], shots=100000))

    # B. QFT
    for n in ([6, 8] if S else [16, 20, 22]):
        sims = ["menos", "menos-fused", "aer", "lightning", "qsim"] + (["cirq"] if n <= 20 else [])
        cases.append(case(f"qft_sv_n{n}", "qft", n, ir.qft(n), "statevector", sims))
    n = 10 if S else 24
    cases.append(case(f"qft_counts_n{n}", "qft", n, ir.qft(n), "counts", ["menos", "menos-fused", "aer", "lightning", "qsim"],
                      shots=1024, timeout=900))

    # C. Grover
    for n, iters in ([(4, None), (6, None)] if S else [(8, None), (12, None), (16, 20)]):
        gates, marked, k = ir.grover(n, 5, iters)
        sims = dense + (["cirq"] if n <= 12 else [])
        cases.append(case(f"grover_n{n}", "grover", n, gates, "counts", sims, shots=4096,
                          extra={"marked": marked, "iterations": k}))

    # D. QPE
    for m in ([4, 6] if S else [11, 15, 19]):
        gates, expected = ir.qpe(m)
        n = m + 1
        sims = ["menos", "menos-fused", "aer", "lightning", "qsim"] + (["cirq"] if n <= 16 else [])
        cases.append(case(f"qpe_n{n}", "qpe", n, gates, "counts", sims, shots=4096, extra={"expected": expected, "m": m}))

    # E. QAOA MaxCut expectation
    for n in ([6, 8] if S else [16, 20, 24]):
        gates, obs = ir.qaoa_maxcut(n, 4, 11)
        sims = ["menos", "menos-fused", "aer", "lightning", "qsim"] + (["cirq"] if n <= 20 else [])
        cases.append(case(f"qaoa_ev_n{n}", "qaoa", n, gates, "expectation", sims, observable=obs))

    # F. Hardware-efficient ansatz + Heisenberg expectation
    for n in ([6, 8] if S else [12, 16, 20, 24]):
        sims = dense + (["cirq"] if n <= 20 else [])
        cases.append(case(f"hea_heisenberg_ev_n{n}", "vqe", n, ir.hea_ansatz(n, 4, 13), "expectation", sims,
                          observable=ir.heisenberg(n)))

    # G. Trotterised TFIM expectation
    for n in ([6, 8] if S else [16, 20, 24]):
        gates, obs = ir.trotter_tfim(n, 10, 17)
        sims = ["menos", "menos-fused", "aer", "lightning", "qsim"] + (["cirq"] if n <= 20 else [])
        cases.append(case(f"trotter_tfim_ev_n{n}", "trotter", n, gates, "expectation", sims, observable=obs))

    # H. Variational loop: K evaluations, angles rebound each time
    n, K = (6, 5) if S else (16, 20)
    gl = [ir.hea_ansatz(n, 4, 100 + k) for k in range(K)]
    c = case(f"variational_hea_n{n}_K{K}", "variational", n, gl[0], "variational", dense + ["cirq"],
             observable=ir.heisenberg(n), reference=None)
    c["gates_list"] = gl
    cases.append(c)

    # I. GHZ (Clifford)
    for n in ([6, 30] if S else [24, 100, 1000]):
        sims = ["menos", "aer", "cirq-clifford"] + (["lightning", "qsim"] if n <= 24 else [])
        cases.append(case(f"ghz_counts_n{n}", "clifford", n, ir.ghz(n), "counts", sims, shots=1024,
                          reference_method="stabilizer", reference_output="marginals"))
    # J. random Clifford
    for n in ([6, 30] if S else [24, 50, 200, 1000]):
        sims = ["menos", "aer"] + (["cirq-clifford"] if n <= 200 else []) + (["lightning", "qsim"] if n <= 24 else [])
        cases.append(case(f"random_clifford_counts_n{n}", "clifford", n, ir.random_clifford(n, 50, 19), "counts", sims,
                          shots=1024, reference_method="stabilizer", reference_output="marginals", timeout=900))

    # K. low-entanglement 1D (MPS territory)
    for n in ([8, 12] if S else [28, 40, 60]):
        gates = ir.low_entanglement_1d(n, 4, 23)
        obs = [(1.0, {q: "Z", q + 1: "Z"}) for q in range(n - 1)]
        sims = ["menos", "aer", "aer-mps"] + (["lightning", "qsim"] if n <= 28 else [])
        cases.append(case(f"low_entanglement_ev_n{n}", "mps", n, gates, "expectation", sims, observable=obs,
                          reference_method="matrix_product_state", timeout=900))

    # L. factorized: independent blocks
    for blocks, size in ([(2, 3), (3, 2)] if S else [(2, 13), (4, 7)]):
        n = blocks * size
        gates = ir.factorized(blocks, size, 20, 29)
        cases.append(case(f"factorized_{blocks}x{size}_counts_n{n}", "factorized", n, gates, "counts", dense,
                          shots=1024, reference_output="marginals", timeout=900))

    # M. noisy, exact density matrix
    for n in ([4] if S else [8, 10]):
        gates = ir.with_noise(ir.dense_random(n, 10, 31), 0.001, 0.01)
        cases.append(case(f"noisy_exact_counts_n{n}", "noise_exact", n, gates, "counts",
                          ["menos", "aer-dm", "aer", "cirq-dm", "default.mixed"], shots=20000, noise=(0.001, 0.01),
                          menos_noise="exact", reference_method="density_matrix"))
    # N. noisy, trajectories
    for n in ([6] if S else [16, 20]):
        gates = ir.with_noise(ir.dense_random(n, 10, 37), 0.001, 0.01)
        sims = ["menos", "aer", "qsim"]
        cases.append(case(f"noisy_traj_counts_n{n}", "noise_traj", n, gates, "counts", sims, shots=1000,
                          noise=(0.001, 0.01), menos_noise="auto", reference=None, timeout=900))
    return cases


# --------------------------------------------------------------------------

def run_worker(sim, c, out_dir, timeout):
    case_path = os.path.join(out_dir, "case.json")
    out_path = os.path.join(out_dir, f"{sim}.json")
    if not os.path.exists(case_path):
        with open(case_path, "w") as f:
            json.dump(c, f)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run([PY, "-I", os.path.join(HERE, "worker.py"), sim, case_path, out_path],
                              capture_output=True, text=True, timeout=timeout, cwd=HERE)
        wall = time.perf_counter() - t0
        if os.path.exists(out_path):
            with open(out_path) as f:
                res = json.load(f)
        else:
            res = {"sim": sim, "case": c["name"], "status": "crash",
                   "error": (proc.stderr or proc.stdout)[-800:], "returncode": proc.returncode}
    except subprocess.TimeoutExpired:
        wall = time.perf_counter() - t0
        res = {"sim": sim, "case": c["name"], "status": "timeout", "error": f"exceeded {timeout}s"}
    res["wall"] = wall
    with open(out_path, "w") as f:
        json.dump(res, f, default=str)
    return res


def score(c, results):
    ref = results.get("reference")
    if c["task"] == "variational":
        base = results.get("aer")
        for sim, r in results.items():
            if r.get("status") == "ok" and base and base.get("status") == "ok":
                r["max_abs_err_vs_aer"] = float(np.max(np.abs(np.array(r["values"]) - np.array(base["values"]))))
        return
    if ref is None or ref.get("status") != "ok":
        base = results.get("aer")
        if base and base.get("status") == "ok" and "marginals" in base:
            for sim, r in results.items():
                if r.get("status") == "ok" and "marginals" in r:
                    r["max_marginal_diff_vs_aer"] = float(np.max(np.abs(np.array(r["marginals"]) - np.array(base["marginals"]))))
        return
    ref_sv = np.load(ref["statevector_path"]) if "statevector_path" in ref else None
    ref_pr = np.load(ref["probabilities_path"]) if "probabilities_path" in ref else None
    for sim, r in results.items():
        if sim == "reference" or r.get("status") != "ok":
            continue
        if c["task"] == "statevector" and ref_sv is not None and "statevector_path" in r:
            sv = np.load(r["statevector_path"])
            r["fidelity"] = float(abs(np.vdot(ref_sv, sv)) ** 2 / (np.vdot(sv, sv).real * np.vdot(ref_sv, ref_sv).real))
        elif c["task"] == "counts":
            if "marginals" in ref and "marginals" in r:
                r["max_marginal_diff"] = float(np.max(np.abs(np.array(r["marginals"]) - np.array(ref["marginals"]))))
            if ref_pr is not None and "counts_path" in r:
                with open(r["counts_path"]) as f:
                    counts = json.load(f)
                total = sum(counts.values())
                emp = np.zeros_like(ref_pr)
                for s, v in counts.items():
                    emp[int(s, 2)] += v / total
                r["tvd"] = float(0.5 * np.abs(emp - ref_pr).sum())
                # expected TVD of a sample of this size from the exact distribution
                pc = np.clip(ref_pr, 0, 1)
                r["tvd_expected"] = float(np.sum(np.sqrt(pc * (1 - pc) / total)) / 2 * np.sqrt(2 / np.pi))
                if c.get("marked"):
                    r["success_prob"] = counts.get(c["marked"], 0) / total
                    r["success_prob_exact"] = float(ref_pr[int(c["marked"], 2)])
                if c.get("expected"):
                    top = max(counts.items(), key=lambda kv: kv[1])[0]
                    m = c["m"]
                    r["top_outcome"] = top
                    r["qpe_ok"] = top[-m:] == c["expected"] or top[-m:][::-1] == c["expected"]
        elif c["task"] == "expectation" and "expectation" in ref and "expectation" in r:
            r["abs_err"] = abs(r["expectation"] - ref["expectation"])
            r["ref_expectation"] = ref["expectation"]


def fmt_time(r):
    if r.get("status") != "ok":
        return r.get("status", "?")
    t = r.get("t_run", r.get("t_per_eval"))
    return f"{t:.3g}"


def fmt_check(r):
    if r.get("status") != "ok":
        return (r.get("error") or "")[:60]
    for k in ("fidelity", "tvd", "max_marginal_diff", "abs_err", "max_abs_err_vs_aer", "max_marginal_diff_vs_aer"):
        if k in r:
            return f"{k}={r[k]:.2e}"
    return ""


def write_summary(cases, all_results, path, env):
    lines = ["# MENOS cross-simulator benchmark", "", f"Environment: {env}", ""]
    lines.append("Times are the median of 3 timed runs after one warm-up (seconds); `t_first` is the first run, "
                 "which includes planning / compilation. Memory is the process peak working set minus the working set "
                 "before the circuit was built.")
    for c in cases:
        if c["name"] not in all_results:
            continue
        res = all_results[c["name"]]
        lines += ["", f"## {c['name']}  (family={c['family']}, n={c['n']}, gates={c['gate_count']}, task={c['task']}"
                  + (f", shots={c['shots']}" if c["shots"] else "") + ")", ""]
        lines.append("| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |")
        lines.append("|---|---|---|---|---|---|---|")
        ref = res.get("reference")
        if ref:
            lines.append(f"| reference (Aer {c['reference_method']}) | {fmt_time(ref)} | | | | status={ref.get('status')} | {ref.get('error','')[:60]} |")
        for sim in c["sims"]:
            r = res.get(sim)
            if r is None:
                continue
            notes = []
            if "plan" in r:
                notes.append(r["plan"])
                if r.get("regions"):
                    notes.append("; ".join(sorted(set(r["regions"]))))
            if "aer_method" in r:
                notes.append(f"aer method={r['aer_method']}")
            if "success_prob" in r:
                notes.append(f"P(marked)={r['success_prob']:.3f} (exact {r['success_prob_exact']:.3f})")
            if "qpe_ok" in r:
                notes.append(f"qpe top={r['top_outcome'][-c['m']:]} ok={r['qpe_ok']}")
            if "tvd_expected" in r:
                notes.append(f"tvd_expected~{r['tvd_expected']:.2e}")
            if r.get("exact") is False:
                notes.append("sampled (trajectories)")
            mem = f"{r['peak_delta_bytes'] / 2**20:.0f}" if "peak_delta_bytes" in r else ""
            lines.append(f"| {sim} | {fmt_time(r)} | {r.get('t_first', 0):.3g} | {r.get('t_build', 0):.3g} | {mem} | {fmt_check(r)} | {'; '.join(notes)} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="full")
    ap.add_argument("--only", default="")
    ap.add_argument("--sims", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--no-reference", action="store_true")
    args = ap.parse_args()
    out_root = os.path.abspath(args.out or os.path.join(os.path.dirname(HERE), f"results-{args.scale}"))
    os.makedirs(out_root, exist_ok=True)
    cache = os.path.join(out_root, "cases.json")
    if os.path.exists(cache):
        with open(cache) as f:
            cases = json.load(f)
        for c in cases:
            c["gates"] = [(g[0], tuple(g[1]), tuple(g[2])) for g in c["gates"]]
            if "gates_list" in c:
                c["gates_list"] = [[(g[0], tuple(g[1]), tuple(g[2])) for g in gl] for gl in c["gates_list"]]
    else:
        cases = build_cases(args.scale)
        with open(cache, "w") as f:
            json.dump(cases, f)
    if args.only:
        fams = args.only.split(",")
        cases = [c for c in cases if c["family"] in fams or c["name"] in fams]
    sims_filter = args.sims.split(",") if args.sims else None
    env = f"{platform.processor()}, {os.cpu_count()} threads, {platform.platform()}"
    all_results = {}
    log = open(os.path.join(out_root, "driver.log"), "a", encoding="utf-8")
    for c in cases:
        if args.reps is not None:
            c["reps"] = args.reps
        if args.no_reference:
            c["reference"] = None
        out_dir = os.path.join(out_root, c["name"])
        os.makedirs(out_dir, exist_ok=True)
        results = {}
        sims = ([ "reference"] if c["reference"] else []) + c["sims"]
        for sim in sims:
            if sims_filter and sim not in sims_filter and sim != "reference":
                continue
            out_path = os.path.join(out_dir, f"{sim}.json")
            if os.path.exists(out_path) and not args.force:
                with open(out_path) as f:
                    results[sim] = json.load(f)
                continue
            msg = f"[{time.strftime('%H:%M:%S')}] {c['name']} :: {sim} ..."
            print(msg, flush=True)
            log.write(msg + "\n"); log.flush()
            r = run_worker(sim, c, out_dir, c["timeout"])
            results[sim] = r
            msg = f"    -> {r.get('status')} t_run={r.get('t_run', r.get('t_per_eval', '-'))} wall={r['wall']:.1f}s {r.get('error','')[:120]}"
            print(msg, flush=True)
            log.write(msg + "\n"); log.flush()
        score(c, results)
        for sim, r in results.items():
            with open(os.path.join(out_dir, f"{sim}.json"), "w") as f:
                json.dump(r, f, default=str)
        all_results[c["name"]] = results
        with open(os.path.join(out_root, "results.jsonl"), "w") as f:
            for name, res in all_results.items():
                for sim, r in res.items():
                    slim = {k: v for k, v in r.items() if k not in ("explain", "traceback", "values", "counts_top")}
                    f.write(json.dumps(slim, default=str) + "\n")
        write_summary(cases, all_results, os.path.join(out_root, "summary.md"), env)
    print("done", flush=True)


if __name__ == "__main__":
    main()
