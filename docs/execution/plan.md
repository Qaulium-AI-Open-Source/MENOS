# Plan

Planning turns the estimates into one decision.

## Candidates, in a fixed order

The planner considers these strategies, in this order. The order matters because ties go to the earlier candidate. It is the order `explain()` lists them in. The tensor network is *costed* after the cheaper candidates (dense through factorized), so that its search can be stopped early; see [Pruned candidates](/planning/explain#pruned-candidates).

| # | Strategy | Enumerated when |
|---|---|---|
| 1 | `direct_dense` | always |
| 2 | `direct_stabilizer` | always (so its refusal reason is always visible) |
| 3 | `direct_pauli` | only for an expectation request |
| 4 | `direct_mps` | always |
| 5 | `direct_tensor_network` | only for an expectation request |
| 6 | `factorized` | only when the circuit has two or more components |
| 7 | `cut_and_recombine` | only for an expectation request, and only if a cut exists |

Candidates that cannot be useful for the request (for example Pauli propagation for a counts request) are not listed at all, because they would always be infeasible and add nothing to the report.

## Selection

1. Apply the memory budget and, if one was set, the time budget. Candidates that fail are marked infeasible with a reason.
2. Among the feasible candidates, choose the one with the **fewest work units**.
3. **Ties go to the earlier candidate** in the table above.
4. If no candidate is feasible, planning throws `ResourceExceeded` and the message lists every candidate's reason.

A tensor-network or cut-and-recombine candidate whose costing passes a cheaper feasible candidate's work units is stopped and reported as pruned. It could not have won, so the choice is the same as if it had been costed in full.

There is no special rule such as "Clifford circuits use the stabilizer engine". A 200-qubit Clifford GHZ circuit is planned onto MPS because MPS costs 3,400 work units against the stabilizer engine's 361,000.

## Backend selection

Only after the algorithm is chosen, and only for dense, the planner chooses CPU or CUDA by calibrated estimated *seconds*. See [Backend Selection](/planning/backend-selection).

## The plan

The result is a `Plan`: the strategy, whether it is exact, its regions (each with qubits, engine, backend, operations), the estimated peak bytes and work units, the full list of candidates, and summaries of the circuit and of light-cone reduction. `explain()` renders it. See [Execution Plans](/planning/execution-plans).

## Determinism

Planning is deterministic. The same circuit, options and (for time budgets) calibrated profile produce the same plan. The choice of *algorithm* depends only on the circuit and the request, never on timing; only the backend and the time-budget check depend on measured speed.

## Planning cost

Planning is cheap but not free. In measurements it ranged from about 0.05 ms to about 17 ms on average across workloads, with one worst sample of 85 ms. That is negligible next to a second-scale simulation and comparable to the run itself for millisecond-scale jobs.
