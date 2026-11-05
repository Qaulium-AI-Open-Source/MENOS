# MENOS cross-simulator benchmark

Environment: Intel64 Family 6 Model 170 Stepping 4, GenuineIntel, 22 threads, Windows-11-10.0.26300-SP0

Times are the median of 3 timed runs after one warm-up (seconds); `t_first` is the first run, which includes planning / compilation. Memory is the process peak working set minus the working set before the circuit was built.

## dense_random_sv_n22  (family=dense_random, n=22, gates=650, task=statevector)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| menos | 1.37 | 1.32 | 0.000235 | 130 |  | direct_dense (exact); engine=dense backend=cpu |
| aer | 2.57 | 1.34 | 0.00243 | 132 |  | aer method=statevector |
| lightning | 25.8 | 23.3 | 0.000461 | 194 |  |  |
| qsim | 0.489 | 0.582 | 0.0661 | 99 |  |  |

## dense_random_counts_n24  (family=dense_random, n=24, gates=710, task=counts, shots=1024)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| menos | 13.9 | 11.4 | 0.000515 | 259 | max_marginal_diff_vs_aer=0.00e+00 | direct_dense (exact); engine=dense backend=cpu |
| aer | 18 | 17.1 | 0.00391 | 262 | max_marginal_diff_vs_aer=0.00e+00 | aer method=statevector |
| lightning | 114 | 113 | 0.00047 | 996 | max_marginal_diff_vs_aer=4.10e-02 |  |
| qsim | 2 | 2.08 | 0.08 | 131 | max_marginal_diff_vs_aer=4.10e-02 |  |

## sampling_n20_100k  (family=sampling, n=20, gates=590, task=counts, shots=100000)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| menos | 0.555 | 0.567 | 0.000536 | 36 | max_marginal_diff_vs_aer=0.00e+00 | direct_dense (exact); engine=dense backend=cpu |
| aer | 1.88 | 1.11 | 0.00316 | 58 | max_marginal_diff_vs_aer=0.00e+00 | aer method=statevector |
| lightning | 8.77 | 8.4 | 0.000405 | 115 | max_marginal_diff_vs_aer=3.84e-03 |  |
| qsim | 0.963 | 1.15 | 0.0818 | 52 | max_marginal_diff_vs_aer=4.14e-03 |  |

## qft_counts_n24  (family=qft, n=24, gates=312, task=counts, shots=1024)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| menos | 4.87 | 3.08 | 0.000237 | 258 | max_marginal_diff_vs_aer=0.00e+00 | direct_dense (exact); engine=dense backend=cpu |
| aer | 3.78 | 2.87 | 0.002 | 260 | max_marginal_diff_vs_aer=0.00e+00 | aer method=statevector |
| lightning | 14.2 | 14.4 | 0.000425 | 1091 | max_marginal_diff_vs_aer=3.52e-02 |  |
| qsim | 2.58 | 2.63 | 0.0064 | 131 | max_marginal_diff_vs_aer=4.30e-02 |  |

## grover_n12  (family=grover, n=12, gates=152112, task=counts, shots=4096)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| menos | 4.79 | 2.86 | 0.0783 | 119 | max_marginal_diff_vs_aer=0.00e+00 | direct_dense (exact); engine=dense backend=cpu |
| aer | 10.7 | 10.8 | 1.17 | 274 | max_marginal_diff_vs_aer=0.00e+00 | aer method=statevector |
| lightning | 22.2 | 21.5 | 0.000518 | 318 | max_marginal_diff_vs_aer=0.00e+00 |  |
| qsim | 10.9 | 12.3 | 3.64 | 300 | max_marginal_diff_vs_aer=0.00e+00 |  |

## qaoa_ev_n24  (family=qaoa, n=24, gates=256, task=expectation)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| menos | 6.79 | 5.4 | 0.000297 | 258 |  | direct_dense (exact); engine=dense backend=cpu |
| aer | 2.02 | 2.07 | 0.00379 | 259 |  | aer method=statevector |
| lightning | 14.8 | 14.2 | 0.000266 | 1025 |  |  |
| qsim | 1.02 | 1.05 | 0.00248 | 258 |  |  |

## hea_heisenberg_ev_n24  (family=vqe, n=24, gates=332, task=expectation)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| menos | 0.929 | 0.942 | 0.000402 | 2 |  | direct_mps (exact); engine=mps backend=cpu |
| aer | 1.76 | 2.03 | 0.00598 | 260 |  | aer method=statevector |
| lightning | 31.3 | 24.7 | 0.000189 | 1025 |  |  |
| qsim | 1.6 | 1.55 | 0.00721 | 258 |  |  |

## trotter_tfim_ev_n24  (family=trotter, n=24, gates=494, task=expectation)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| menos | 10.1 | 9.51 | 0.000411 | 259 |  | direct_dense (exact); engine=dense backend=cpu |
| aer | 9.83 | 7.17 | 0.00313 | 260 |  | aer method=statevector |
| lightning | 55.9 | 53.5 | 0.000189 | 1025 |  |  |
| qsim | 2.76 | 2.83 | 0.0163 | 258 |  |  |

## ghz_counts_n1000  (family=clifford, n=1000, gates=1000, task=counts, shots=1024)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| menos | 0.129 | 0.128 | 0.000679 | 3 | max_marginal_diff_vs_aer=2.25e-02 | direct_mps (exact); engine=mps backend=cpu |
| aer | 213 | 207 | 0.00765 | 10 | max_marginal_diff_vs_aer=0.00e+00 | aer method=stabilizer |

## random_clifford_counts_n1000  (family=clifford, n=1000, gates=74975, task=counts, shots=1024)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| menos | 2.45 | 1.7 | 0.035 | 58 | max_marginal_diff_vs_aer=5.17e-01 | direct_stabilizer (exact); engine=stabilizer backend=cpu |
| aer | 154 | 156 | 0.327 | 146 | max_marginal_diff_vs_aer=0.00e+00 | aer method=stabilizer |
