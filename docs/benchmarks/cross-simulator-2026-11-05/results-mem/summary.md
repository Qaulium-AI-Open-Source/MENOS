# MENOS cross-simulator benchmark

Environment: Intel64 Family 6 Model 170 Stepping 4, GenuineIntel, 22 threads, Windows-11-10.0.26300-SP0

Times are the median of 3 timed runs after one warm-up (seconds); `t_first` is the first run, which includes planning / compilation. Memory is the process peak working set minus the working set before the circuit was built.

## dense_random_sv_n22  (family=dense_random, n=22, gates=650, task=statevector)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| menos | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |
| aer | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |
| lightning | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |
| qsim | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |

## dense_random_counts_n24  (family=dense_random, n=24, gates=710, task=counts, shots=1024)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| menos | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |
| aer | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |
| lightning | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |
| qsim | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |

## sampling_n20_100k  (family=sampling, n=20, gates=590, task=counts, shots=100000)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| menos | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |
| aer | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |
| lightning | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |
| qsim | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |

## qft_counts_n24  (family=qft, n=24, gates=312, task=counts, shots=1024)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| menos | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |
| aer | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |
| lightning | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |
| qsim | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |

## grover_n12  (family=grover, n=12, gates=152112, task=counts, shots=4096)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| menos | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |
| aer | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |
| lightning | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |
| qsim | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |

## qaoa_ev_n24  (family=qaoa, n=24, gates=256, task=expectation)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| menos | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |
| aer | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |
| lightning | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |
| qsim | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |

## hea_heisenberg_ev_n24  (family=vqe, n=24, gates=332, task=expectation)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| menos | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |
| aer | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |
| lightning | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |
| qsim | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |

## trotter_tfim_ev_n24  (family=trotter, n=24, gates=494, task=expectation)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| menos | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |
| aer | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |
| lightning | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |
| qsim | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |

## ghz_counts_n1000  (family=clifford, n=1000, gates=1000, task=counts, shots=1024)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| menos | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |
| aer | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |

## random_clifford_counts_n1000  (family=clifford, n=1000, gates=74975, task=counts, shots=1024)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| menos | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |
| aer | crash | 0 | 0 |  | Traceback (most recent call last):
  File "C:\Users\janak\Ap |  |
