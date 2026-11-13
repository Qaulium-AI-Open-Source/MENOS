# ResourceExceeded

`ResourceExceeded` (Python: `menos.exceptions.ResourceExceededError`) means **no plan fits the constraints**. It is a refusal, not a crash: MENOS will not run something it knows it cannot run within budget.

## When it is raised

| Cause | Stage |
|---|---|
| Every candidate needs more memory than the budget | Plan time |
| Every candidate is estimated to exceed the time budget | Plan time |
| A run overran its time budget and no other plan fits | Run time |
| A device allocation would exceed GPU memory | Plan time (the region falls back to CPU first) |
| Another process took device memory between the plan and the allocation | Run time, naming the bytes (since Phase 23) |
| The disk tier's scratch file plus its reserve would not fit the scratch directory, or its window would not fit the budget | Plan time (since Phase 23) |

It is raised **before** any allocation for plan-time causes, so a refusal never leaves a half-built state.

## Reading the message

The message lists every candidate's reason, separated by semicolons, so you can see exactly what blocked each one:

```text
plan_job: no feasible candidate -- direct dense simulation requires 17592186044416 bytes,
budget is 7752519680 bytes; the stabilizer engine does not track global phase -- only the
dense engine can produce a statevector; …
```

A run-time refusal names the observation:

```text
menos::run: observed runtime of 0.25 s exceeded the job's time budget of 1e-06 s after 3 safe
boundaries on engine=stabilizer backend=cpu, and no different plan meets that budget -- …
```

## Handling it

```python
from menos import Circuit, Opcode, RunOptions, run
from menos.exceptions import ResourceExceededError

circuit = Circuit(40, 0)
circuit.append(Opcode.H, [0])
for q in range(39):
    circuit.append(Opcode.CX, [q, q + 1])

options = RunOptions()
options.want_statevector = True
try:
    run(circuit, options)
except ResourceExceededError as error:
    print("cannot run:", str(error)[:80])
else:
    raise AssertionError("a 40-qubit dense statevector cannot fit in memory")
```

## What to try

| If the cause was | Try |
|---|---|
| memory | Ask for less: counts or an expectation value instead of a statevector; a circuit that splits into parts; a smaller circuit; a larger memory budget if the machine has the RAM |
| time, at plan time | A larger budget, or an output that is cheaper to compute |
| time, at run time | A larger budget; the abandoned plan's engine was slower than predicted |

## Hierarchy

`ResourceExceededError` derives from `MenosError`. Catch `MenosError` to handle any MENOS failure. See [Exceptions](/reference/exceptions).

## Related

[Memory Constraints](/planning/memory-constraints) · [Time Budgets](/runtime/time-budgets) · [Memory Errors](/troubleshooting/memory-errors)
