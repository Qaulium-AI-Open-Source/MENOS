# Replanning

When a monitored run overruns its time budget, MENOS does not give up and does not guess. It stops, records what it learned, asks the **same planner** again, and restarts under the answer.

```text
Plan predicts runtime
        ↓
Execution begins
        ↓
Runtime monitors progress
        ↓
Budget threshold crossed?  ── no ──→  finish normally
        ↓ yes
Abandon the attempt at a safe boundary
        ↓
Record the observed time for the engine that overran
        ↓
Ask the existing planner again, with that observation
        ↓
A different plan fits?  ── no ──→  ResourceExceeded
        ↓ yes
Restart under the new plan
```

## The observation is a floor

What MENOS records is one fact: *engine E on backend B, on this job, has already spent this many seconds.* It is a **lower bound**. The planner takes the larger of its own estimate and that floor, so an observation can only ever make a candidate look more expensive. A stale or understated observation cannot make the planner optimistic.

A rate extrapolated from partial progress was measured and rejected: the MPS engine's cost per unit of work varies about 30× within a single run, so a rate taken early projected the total at 0.16× to 0.40× of the truth.

## One planner

There is no second planner and no second cost model in the runtime. A replan is the ordinary planning function called again with one extra argument. The runtime never plans and never interprets the monitor; the decision belongs to the layer above it.

## A real run

A 16-qubit, 8-layer circuit with a 0.2 s budget. The model predicted the MPS engine would need under a millisecond; it needed far more:

```text
replan 0   threshold=0.2 s crossed at elapsed=0.224734 s
           abandoned after 194 safe boundaries on engine=mps backend=cpu
           was  strategy=direct_mps predicted=0.000835369 s
           reason="observed runtime exceeded the time budget; mps on cpu is excluded by an observed floor of 0.224734 s"
           now  strategy=direct_dense
strategy   direct_dense (exact)
```

MPS was abandoned, dense was estimated to fit, and the job finished correctly in about 0.44 s in total, which is more than the 0.2 s budget, because the budget bounds one attempt. The [replanning example](/examples/replanning) runs this. **Which outcome you see depends on your hardware.**

## The four outcomes

| Situation | What happens |
|---|---|
| The current plan fits | It runs to completion; no replan; no monitor output in the report |
| The current plan exceeds the budget, and another plan is feasible | The attempt is abandoned and the job restarts under the new plan; the report shows every replan |
| The overrunning backend had an alternative | On a dense region, an overrun on one backend steers the choice to the other: a GPU that overran is not chosen again and the CPU is considered, and the reverse |
| Every alternative exceeds the budget | `ResourceExceeded`, with a message naming the observation, the budget, the engine, and why nothing else fits |

## Why replanning always terminates

An overrun records a floor above the budget for exactly the plan that ran, so that plan is infeasible next time. A job has finitely many plans, and each replan removes one of them for good, so the job either finishes or is refused after a bounded number of attempts. There is no tuned retry limit; the loop checks on every replan that it never returns a plan that already ran (the proof is in `docs/adr/0028` section 16.8).

### Factorized plans (since Phase 24)

A factorized plan runs its regions one after another, and MENOS times each region's engine on the attempt's own clock. When such a plan overruns, MENOS records:

- for **each region that started**, a floor at the time that region took (the whole region, or up to the crossing for the one that was running), keyed by the region's qubits and the engine, backend and tier that ran it. A region's floor never applies to any other region, even one of the same size;
- for **the plan as a whole**, a floor at the elapsed time of the attempt.

The plan floor makes that exact plan infeasible for the rest of the job. The region floors price any other plan that reuses one of those regions on the same implementation, and steer a region's CPU/CUDA choice away from a backend that overran.

## What is reported

Each abandoned attempt appears in the plan report: the threshold, the observed time, how long replanning took (`replanned_in`), how many safe boundaries were crossed, the old strategy and its prediction, the reason, and the new strategy. For a factorized plan, one line per region that started follows, with its qubits, engine, backend, its own prediction, the observed time, and whether it completed or was running. After the `replan` blocks comes the usual `actual` line, which times **only the attempt that succeeded**. See [Plan Explanation](/planning/explain).

## Correctness across a restart

A restarted run is identical to running the new plan from a cold start: the same counts for the same seed, bit for bit. Nothing from the abandoned attempt carries over. This holds across backends too: a run abandoned on the GPU and restarted on the CPU matches a cold CPU run exactly.

## Limits

- **A cut plan's overrun is refused.** Its fragments are not timed one by one, so the overrun cannot be attributed soundly, and MENOS refuses rather than guess, with a message that says so. Factorized plans replan (since Phase 24).
- **No checkpointing.** Restart repeats the work. Measurement showed restart is cheap for executions of about a second or more, so migration was not built; it would need a measurement showing otherwise.
- **Tensor-network** runs with one observable term have a single boundary, so they cannot be interrupted mid-contraction.
- **Planning is not charged against the budget.** Its cost is reported separately: `Result.plan_seconds` for the initial plan and `Result.replan_seconds` for the replans (since Phase 24).

## Related

[Monitoring](/runtime/monitoring) · [Time Budgets](/runtime/time-budgets) · [ResourceExceeded](/runtime/resource-exceeded) · [Backend Selection](/planning/backend-selection)
