# How Planning Stays Fast

Before simulating anything, MENOS prices every candidate strategy and picks one ([Plan](/execution/plan)). This page explains why that step costs milliseconds and not seconds, what it costs the first time a process runs, and how a budget changes the choice. Everything here leaves the chosen plan unchanged; it only changes how quickly it is found.

## Planning cost, measured

The planner benchmark corpus (45 fixed jobs, every candidate family) was measured on one core of the development machine before and after the Phase 24 planner work (`bench/BASELINE.md`, `docs/adr/0028` section 13):

| Job | Before | After |
|---|---|---|
| Expectation, budgeted Ising, large | 1992 ms | 10.8 ms (185×) |
| Expectation, factorized, large | 14 041 ms | 4.44 ms (3 164×) |

No job in the corpus got slower. A 20-qubit, depth-10 expectation of 50 Pauli terms that used to spend 4.4 s planning now plans in 2–6 ms. Most jobs plan in well under 11 ms.

Every result reports its own planning time: `Result.plan_seconds`, plus `Result.replan_seconds` after a replan, printed as `plan_s=` and `replan_s=` on the `actual` line of `explain()`.

## Branch-and-bound pruning

Costing a **tensor-network** or **cut-and-recombine** candidate is itself a search, and the engine choice for each **factorized** region is a small planning problem of its own. The planner first costs the cheap candidates (dense through factorized), then stops each expensive search **as soon as its running total passes the cheapest feasible candidate already found**. A candidate stopped this way could never have won, so the winner, its cost, every region and every refusal are the same as if everything had been costed in full. The fixed golden planner corpus pins this.

`explain()` shows a stopped candidate as `reason="pruned by branch-and-bound: …"`, with the figure it had reached. See [Pruned candidates](/planning/explain#pruned-candidates).

Pruning is **off**, and every candidate is costed in full, when:
- the job has a time budget (it ranks by seconds, below, and a work-unit bound no longer proves a candidate cannot win);
- the CUDA backend is forced, or CPU and CUDA are compared by calibrated seconds;
- the disk tier is forced.

## Faster searches inside the cost model

These change how fast an estimate is computed, never its value:

- **Bitset elimination search.** The tensor-network candidate prices its contraction by a min-fill elimination over the interaction graph. The graph is kept as bitsets with a heap of candidate vertices, and the fill is updated incrementally instead of being recomputed. The search reports its own peak memory like the earlier search did, so the tensor-network estimate stays an exact upper bound.
- **Cached Schmidt ranks.** The MPS candidate bounds its bond dimensions by the Schmidt rank of each multi-qubit gate across each cut. Single-qubit gates are skipped without building a matrix, and each gate's rank is computed once and reused.

## The plan cache

A variational loop runs the same circuit many times with new angles. `run` keeps a **plan cache** so such a loop plans once:
- the key is the circuit's **structure** and the job's options, never the angles, together with the hardware figures the plan was checked against (memory, device memory, calibrated rates);
- only a **Direct, in-memory** plan is reused. A job with a time budget or a noise channel, a two-qubit rotation within `1e-8` of an angle where it degenerates (such as `rzz(0)`), and a `factorized` or `cut_and_recombine` winner are planned afresh and counted as a bypass. A disk-tier plan is never stored;
- the reused plan is the plan the planner would compute for the new angles, so results and `explain()` are unchanged.

`menos._core.plan_cache_stats()` returns `hits`, `misses` and `bypasses`, and `menos._core.plan_cache_clear()` empties the cache. The circuit builder's `bind` produces circuits of identical structure, so a rebinding loop hits the cache ([Writing Circuits](/build/circuit-builder)).

## Calibration: once per process

Seconds come from a calibration that measures this machine on the code that will really run: memory bandwidth, the dense work rate, each CPU engine's cost per **time count** (since Phase 24), and on a CUDA build the transfer rates, launch overhead and device rate ([Cost Model](/planning/cost-model#calibration)).

It runs **the first time a job in the process needs it**, is kept in memory, and then costs microseconds. It is **not saved to disk**, so a new process measures again. On the development machine the first calibration took 1.4–1.5 s on Windows and about 2.1 s on Linux with CUDA (it was about 0.1 s before the per-engine time model).

Who pays it:

| Build and job | Calibration |
|---|---|
| CPU build, no time budget | Never. The planner uses detected figures |
| CPU build, with a time budget | On the first budgeted job of the process |
| CUDA build | On the first job of the process, budgeted or not, because the planner compares CPU with CUDA |
| Windows | CPU builds only: native Windows has no CUDA backend ([Supported Platforms](/introduction/supported-platforms)) |

What that means in practice:

| How MENOS is used | Cost |
|---|---|
| One script or notebook kernel running many circuits | Once, on the first job that needs it; later jobs are fast |
| A new Python process, or a restarted notebook kernel | Once again |
| The command line | Once per command that needs it, since every command is a new process |

## Ranking: work units, or seconds under a budget

- **No time budget:** the planner picks the feasible candidate with the **fewest work units**, ties to the earlier candidate. This choice does not depend on the machine's speed, and these plans and their `explain()` output did not change in Phase 24.
- **With a time budget:** it picks the candidate with the **fewest predicted seconds** (since Phase 24, `docs/adr/0028` section 21), costing every candidate in full. A budgeted job can therefore run a different exact plan than the same job without a budget, and draw different counts for the same seed. Because predictions come from this process's calibration, two candidates predicted close can be chosen differently in another process.

On the development machine every CPU engine's typical prediction is within 2× of the measured time, and its 90th percentile within 4×, on Linux and Windows ([Cost Model](/planning/cost-model)).

## Related

[Plan](/execution/plan) · [Cost Model](/planning/cost-model) · [Plan Explanation](/planning/explain) · [Time Budgets](/runtime/time-budgets)
