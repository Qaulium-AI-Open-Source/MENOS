# Command Line

The repository builds a small command-line tool, `menos`, that reads an **OpenQASM 2 or OpenQASM 3** file and either runs it or shows how it would be planned. It is an inspection tool, linked only to the public API.

::: info Not installed by pip
The command-line tool is a CMake target and is **not** part of the Python package. Build it with the CMake presets; the binary is `build/<preset>/cli/menos`.
:::

```bash
cmake --preset release
cmake --build --preset release --target menos_cli
build/release/cli/menos explain examples/bell.qasm
```

## Commands

```text
menos explain <file.qasm> [--shots N] [--seed N] [--time-budget SECONDS] [--backend cpu|cuda] [--tier disk|in_memory] [--scratch-dir PATH] [--fusion on|off] [--input NAME=VALUE]...
menos run     <file.qasm> --shots N --seed N [--statevector] [--time-budget SECONDS] [--backend cpu|cuda] [--tier disk|in_memory] [--scratch-dir PATH] [--fusion on|off] [--input NAME=VALUE]...
```

### `explain`

Plans the job and prints the report **without executing it**. `--shots` defaults to 1 (it only selects the counts output for planning).

```text
$ menos explain examples/bell.qasm
MENOS EXECUTION REPORT
circuit    qubits=2 clbits=2 operations=4 depth=3 global_phase=0
light_cone qubits 2->2 operations 4->4
candidate 0strategy=direct_dense feasible=true winner=false peak_bytes=64 work_units=32
candidate 1strategy=direct_stabilizer feasible=true winner=true peak_bytes=40 work_units=16
candidate 2strategy=direct_mps feasible=true winner=false peak_bytes=128 work_units=34
strategy   direct_stabilizer (exact)
region 0   qubits=[0,1] engine=stabilizer backend=cpu
estimated  state_bytes=40 work_units=16
```

### `run`

Runs the job, prints the full report including the `actual` line, then the counts (highest classical bit left):

```text
$ menos run examples/bell.qasm --shots 100 --seed 5
…
actual     state_bytes=40 execute_s=0.000006 sample_s=0.000000
00: 49
11: 51
```

`--shots` and `--seed` are both **required** for `run`. `--statevector` also prints every amplitude, and is limited to circuits of **10 qubits or fewer**.

### `--time-budget SECONDS`

Both commands accept a [time budget](/runtime/time-budgets), the CLI form of `RunOptions.time_budget_seconds` (since Phase 17). It must be a finite number greater than 0; anything else is a usage error that names the flag. With `explain` it only decides which plans fit, since nothing runs. With `run`, an attempt that overruns is abandoned and replanned, and the report says so; if no plan fits, the tool exits with `1` and an error naming the budget. With a budget, the report also prints machine-dependent time estimates.

### `--input NAME=VALUE`

Binds an `input` declaration of an OpenQASM 3 program (since Phase 19), for example `--input theta=0.5` for `input float theta;`. Repeat it once per input. Every `input` in the program needs a binding and every binding must name an `input`; a missing or unknown name, a value that is not a finite number, or a name given twice is an error. OpenQASM 2 has no inputs, so `--input` with an OpenQASM 2 file is a usage error.

### `--backend cpu|cuda`

Forces the backend instead of letting the planner choose, the CLI form of `RunOptions.backend` (since Phase 17). The report then contains a `backend    cpu (requested)` (or `cuda`) line. A CUDA request that cannot be honoured exits with `1` and an error, and never runs on the CPU. See [Forcing a backend](/planning/backend-selection#forcing-a-backend). Any value other than `cpu` or `cuda` is a usage error.

### `--tier disk|in_memory` and `--scratch-dir PATH`

The CLI forms of `RunOptions.tier` and `RunOptions.scratch_directory` (since Phase 23). `--tier disk` runs the dense engine from a scratch file in `--scratch-dir` (default: the system's temporary directory) even when the state would fit in memory, and never falls back: with `--statevector`, or with `--backend cuda`, the tool exits with `1` and an error. `--tier in_memory` forbids the disk tier. Without `--tier`, the planner uses the disk tier only when the dense state does not fit the memory budget and no statevector is wanted. The report then contains a `tier       disk (requested)` line, the candidate's `tier=disk passes=… chunk_qubits=… disk_bytes=…`, the region's `tier=disk`, and the scratch directory and its measured bandwidth on the `estimated` line; after a run, `peak_disk_bytes=` on the `actual` line. Any other value for `--tier` is a usage error; see [Memory Constraints](/planning/memory-constraints#the-disk-tier).

## Exit status

`0` on success. On any error the tool writes `error: …` to standard error and exits with `1`:

```text
error: --shots is required for 'run'
error: usage: menos <explain|run> <file.qasm> [--shots N] [--seed N] [--statevector] [--time-budget SECONDS] [--backend cpu|cuda] [--tier disk|in_memory] [--scratch-dir PATH] [--fusion on|off] [--input NAME=VALUE]...
```

## OpenQASM support

The tool reads the program's header to choose the language: `OPENQASM 2.0;` or `OPENQASM 3;`/`OPENQASM 3.0;`. A file without a header is refused, because the two languages read some programs differently:

```text
error: the program has no 'OPENQASM 2.0;' or 'OPENQASM 3.0;' header; add one so its language version is known
```

| | OpenQASM 2 | OpenQASM 3 |
|---|---|---|
| Gates | `qelib1.inc` gates that map to MENOS gates | `U`, `gphase`, every `stdgates.inc` gate, user `gate` definitions |
| Modifiers | — | `inv @`, `pow(k) @` with an integer `k`, `ctrl @`, `negctrl @` |
| Registers | one `qreg`, at most one `creg` | any number of `qubit`/`bit` (or `qreg`/`creg`) |
| Parameters | constant expressions | constant expressions, `const`, and `input` bound with `--input` |
| Measurement | `measure q[i] -> c[j];` bit by bit | bit by bit or a whole register: `c = measure q;`, `c[0] = measure q[0];`, `measure q -> c;` |
| Not supported | `gate` blocks, `if`, `opaque`, `reset` | `reset`, control flow, subroutines, timing (`delay`, `box`), calibrations, classical variables, applying a gate to a whole register |

Both importers keep operations in source order, measurements included. As everywhere in MENOS, measurements must be terminal: a gate after measuring the same qubit is refused. (Until Phase 19 the OpenQASM 2 importer moved every measurement to the end, which could silently change such a program's result.)

The `stdgates.inc` gates have Qiskit's matrices, global phase included, so `ctrl @` of any of them is what Qiskit means; the official file's own definitions are not used, because read literally they disagree with the gates they name (`docs/adr/0023` section 4.1). Anything outside the supported subset is refused with its line and column:

```text
error: qasm3: 'reset' is not supported (deferred to dynamic circuits) at line 3, column 1
error: qasm2: expected '[' after a register name at line 7
```

## Limits

### `--fusion on|off`

The CLI form of `RunOptions.fuse_gates` (since Phase 24). Gate fusion is on by default: the dense engines sweep the state once per fused block instead of once per gate, and the report's region line says what it did (`fusion=containment blocks=… absorbed=…`, printed only when a gate was absorbed). `--fusion off` applies every gate on its own; the results and the report are then byte-identical to a run from before fusion existed. Any other value is a usage error; see [Dense Engine](/engines/dense#gate-fusion).

The tool has no option for a memory budget; use the [native API](/build/native-api) for that. A time budget is `--time-budget`; the disk tier is `--tier disk` with `--scratch-dir`.
