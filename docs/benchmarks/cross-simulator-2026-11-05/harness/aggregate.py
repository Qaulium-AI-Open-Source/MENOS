"""Compact cross-simulator table: per case, each simulator's time and its ratio
to MENOS (ratio > 1 means MENOS was faster). usage: aggregate.py RESULTS_DIR [MEM_DIR]"""

import json
import os
import sys

root = sys.argv[1]
mem_root = sys.argv[2] if len(sys.argv) > 2 else None
cases = json.load(open(os.path.join(root, "cases.json")))
SIMS = ["menos", "menos-fused", "menos-1t", "aer", "aer-mps", "aer-dm", "lightning", "qsim", "cirq", "cirq-clifford",
        "cirq-dm", "default.qubit", "default.mixed"]


def load(d, sim):
    p = os.path.join(d, sim + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def t(r):
    if r is None:
        return None
    if r.get("status") != "ok":
        return r.get("status")
    return r.get("t_run", r.get("t_per_eval"))


rows = []
print(f"{'case':34s} {'n':>5s} {'gates':>7s} | " + " ".join(f"{s:>13s}" for s in SIMS))
for c in cases:
    d = os.path.join(root, c["name"])
    if not os.path.isdir(d):
        continue
    res = {s: load(d, s) for s in SIMS}
    base = t(res["menos"])
    cells = []
    for s in SIMS:
        v = t(res[s])
        if v is None:
            cells.append(f"{'':>13s}")
        elif isinstance(v, str):
            cells.append(f"{v:>13s}")
        else:
            ratio = (v / base) if isinstance(base, float) and base > 0 else float("nan")
            cells.append(f"{v:8.3g}({ratio:4.1f}x)" if s != "menos" else f"{v:13.3g}")
    plan = (res["menos"] or {}).get("plan", "")
    print(f"{c['name']:34s} {c['n']:5d} {c['gate_count']:7d} | " + " ".join(cells) + f"   plan={plan}")

if mem_root:
    print("\nPeak working-set delta (MiB), one run each:")
    print(f"{'case':34s} | " + " ".join(f"{s:>10s}" for s in ["menos", "aer", "lightning", "qsim"]))
    for c in cases:
        for d in (os.path.join(mem_root, c["name"]), os.path.join(root, c["name"])):
            if not os.path.isdir(d):
                continue
            vals = []
            for s in ["menos", "aer", "lightning", "qsim"]:
                r = load(d, s)
                if r and r.get("status") == "ok" and r.get("peak_delta_bytes", 0) > 0:
                    extra = f"/{r['menos_peak_host_bytes'] / 2**20:.0f}" if s == "menos" and "menos_peak_host_bytes" in r else ""
                    vals.append(f"{r['peak_delta_bytes'] / 2**20:7.0f}{extra:>3s}")
                else:
                    vals.append(f"{'':>10s}")
            if any(v.strip() for v in vals):
                print(f"{c['name']:34s} | " + " ".join(vals))
                break
