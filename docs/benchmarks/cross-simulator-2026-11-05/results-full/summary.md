# MENOS cross-simulator benchmark

Environment: Intel64 Family 6 Model 170 Stepping 4, GenuineIntel, 22 threads, Windows-11-10.0.26300-SP0

Times are the median of 3 timed runs after one warm-up (seconds); `t_first` is the first run, which includes planning / compilation. Memory is the process peak working set minus the working set before the circuit was built.

## dense_random_sv_n16  (family=dense_random, n=16, gates=470, task=statevector)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer statevector) | 0.0407 | | | | status=ok |  |
| menos | 0.066 | 0.0735 | 0.000268 | 0 | fidelity=1.00e+00 | direct_dense (exact); engine=dense backend=cpu |
| aer | 0.047 | 0.228 | 0.00232 | 0 | fidelity=1.00e+00 | aer method=statevector |
| lightning | 0.222 | 0.244 | 0.000256 | 0 | fidelity=1.00e+00 |  |
| qsim | 0.0246 | 0.0352 | 0.0293 | 0 | fidelity=1.00e+00 |  |
| cirq | 0.285 | 0.303 | 0.0343 | 0 | fidelity=1.00e+00 |  |
| default.qubit | 0.747 | 0.83 | 0.000287 | 0 | fidelity=1.00e+00 |  |
| menos-1t | 0.0698 | 0.14 | 0.000232 | 0 | fidelity=1.00e+00 | direct_dense (exact); engine=dense backend=cpu |

## dense_random_sv_n20  (family=dense_random, n=20, gates=590, task=statevector)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer statevector) | 0.723 | | | | status=ok |  |
| menos | 0.285 | 0.377 | 0.000314 | 0 | fidelity=1.00e+00 | direct_dense (exact); engine=dense backend=cpu |
| aer | 0.764 | 0.327 | 0.00242 | 0 | fidelity=1.00e+00 | aer method=statevector |
| lightning | 4.48 | 4.41 | 0.000279 | 0 | fidelity=1.00e+00 |  |
| qsim | 0.115 | 0.158 | 0.0393 | 0 | fidelity=1.00e+00 |  |
| cirq | 6.71 | 9.02 | 0.0746 | 0 | fidelity=1.00e+00 |  |
| menos-1t | 1.58 | 1.48 | 0.000247 | 0 | fidelity=1.00e+00 | direct_dense (exact); engine=dense backend=cpu |

## dense_random_sv_n22  (family=dense_random, n=22, gates=650, task=statevector)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer statevector) | 1.48 | | | | status=ok |  |
| menos | 1.54 | 1.47 | 0.000302 | 0 | fidelity=1.00e+00 | direct_dense (exact); engine=dense backend=cpu |
| aer | 1.57 | 1.54 | 0.00228 | 0 | fidelity=1.00e+00 | aer method=statevector |
| lightning | 8.72 | 8.96 | 0.000156 | 0 | fidelity=1.00e+00 |  |
| qsim | 0.178 | 0.154 | 0.017 | 0 | fidelity=1.00e+00 |  |
| cirq | 29.3 | 20.2 | 0.0195 | 0 | fidelity=1.00e+00 |  |

## dense_random_counts_n24  (family=dense_random, n=24, gates=710, task=counts, shots=1024)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer statevector) | 12.3 | | | | status=ok |  |
| menos | 9.67 | 11 | 0.000334 | 0 | max_marginal_diff=2.57e-02 | direct_dense (exact); engine=dense backend=cpu |
| aer | 12.9 | 9.82 | 0.00235 | 0 | max_marginal_diff=2.57e-02 | aer method=statevector |
| lightning | 81 | 76.8 | 0.000157 | 0 | max_marginal_diff=3.10e-02 |  |
| qsim | 1.53 | 1.61 | 0.0571 | 0 | max_marginal_diff=3.21e-02 |  |

## dense_random_counts_n26  (family=dense_random, n=26, gates=770, task=counts, shots=1024)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer statevector) | 44.7 | | | | status=ok |  |
| menos | 49.1 | 58.6 | 0.000794 | 0 | max_marginal_diff=3.16e-02 | direct_dense (exact); engine=dense backend=cpu |
| aer | 16.2 | 22.8 | 0.00283 | 0 | max_marginal_diff=3.16e-02 | aer method=statevector |
| lightning | timeout | 0 | 0 |  | exceeded 900s |  |
| qsim | 6.79 | 6.88 | 0.101 | 0 | max_marginal_diff=4.84e-02 |  |

## deep_dense_sv_n14_d400  (family=deep_dense, n=14, gates=8200, task=statevector)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer statevector) | 0.269 | | | | status=ok |  |
| menos | 0.243 | 0.245 | 0.00238 | 0 | fidelity=1.00e+00 | direct_dense (exact); engine=dense backend=cpu |
| aer | 0.284 | 0.269 | 0.0231 | 0 | fidelity=1.00e+00 | aer method=statevector |
| lightning | 1.77 | 1.81 | 0.000502 | 0 | fidelity=1.00e+00 |  |
| qsim | 0.287 | 0.193 | 0.708 | 0 | fidelity=1.00e+00 |  |
| cirq | 2.97 | 2.59 | 0.748 | 0 | fidelity=1.00e+00 |  |
| default.qubit | 4.14 | 3.84 | 0.000502 | 0 | fidelity=1.00e+00 |  |

## sampling_n20_100k  (family=sampling, n=20, gates=590, task=counts, shots=100000)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer statevector) | 0.301 | | | | status=ok |  |
| menos | 0.348 | 0.322 | 0.00024 | 0 | tvd=8.17e-01 | direct_dense (exact); engine=dense backend=cpu; tvd_expected~1.13e+00 |
| aer | 1.2 | 0.892 | 0.00225 | 0 | tvd=8.17e-01 | aer method=statevector; tvd_expected~1.13e+00 |
| lightning | 7.3 | 6.76 | 0.000541 | 0 | tvd=8.16e-01 | tvd_expected~1.13e+00 |
| qsim | 0.697 | 0.537 | 0.0139 | 0 | tvd=8.16e-01 | tvd_expected~1.13e+00 |
| cirq | 4.65 | 11 | 0.015 | 0 | tvd=8.16e-01 | tvd_expected~1.13e+00 |

## qft_sv_n16  (family=qft, n=16, gates=144, task=statevector)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer statevector) | 0.016 | | | | status=ok |  |
| menos | 0.0141 | 0.0138 | 9.94e-05 | 0 | fidelity=1.00e+00 | direct_dense (exact); engine=dense backend=cpu |
| menos-fused | 0.00785 | 0.0092 | 9.37e-05 | 0 | fidelity=1.00e+00 | direct_dense (exact); engine=dense backend=cpu |
| aer | 0.016 | 0.0144 | 0.000857 | 0 | fidelity=1.00e+00 | aer method=statevector |
| lightning | 0.0247 | 0.0306 | 0.000277 | 0 | fidelity=1.00e+00 |  |
| qsim | 0.0117 | 0.0156 | 0.000995 | 0 | fidelity=1.00e+00 |  |
| cirq | 0.0186 | 0.0223 | 0.00169 | 0 | fidelity=1.00e+00 |  |

## qft_sv_n20  (family=qft, n=20, gates=220, task=statevector)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer statevector) | 0.102 | | | | status=ok |  |
| menos | 0.078 | 0.086 | 0.00015 | 0 | fidelity=1.00e+00 | direct_dense (exact); engine=dense backend=cpu |
| menos-fused | 0.119 | 0.121 | 0.000137 | 0 | fidelity=1.00e+00 | direct_dense (exact); engine=dense backend=cpu |
| aer | 0.0836 | 0.0754 | 0.00116 | 0 | fidelity=1.00e+00 | aer method=statevector |
| lightning | 0.415 | 0.257 | 0.000361 | 0 | fidelity=1.00e+00 |  |
| qsim | 0.0902 | 0.118 | 0.00128 | 0 | fidelity=1.00e+00 |  |
| cirq | 0.139 | 0.146 | 0.0014 | 0 | fidelity=1.00e+00 |  |

## qft_sv_n22  (family=qft, n=22, gates=264, task=statevector)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer statevector) | 0.302 | | | | status=ok |  |
| menos | 0.522 | 0.515 | 0.000154 | 0 | fidelity=1.00e+00 | direct_dense (exact); engine=dense backend=cpu |
| menos-fused | 0.244 | 0.248 | 0.000132 | 0 | fidelity=1.00e+00 | direct_dense (exact); engine=dense backend=cpu |
| aer | 0.305 | 0.293 | 0.00116 | 0 | fidelity=1.00e+00 | aer method=statevector |
| lightning | 1.93 | 1.97 | 0.000179 | 0 | fidelity=1.00e+00 |  |
| qsim | 0.367 | 0.36 | 0.00329 | 0 | fidelity=1.00e+00 |  |

## qft_counts_n24  (family=qft, n=24, gates=312, task=counts, shots=1024)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer statevector) | 2.63 | | | | status=ok |  |
| menos | 3.55 | 2.76 | 0.000158 | 0 | max_marginal_diff=1.95e-02 | direct_dense (exact); engine=dense backend=cpu |
| menos-fused | 1.38 | 1.12 | 0.000161 | 0 | max_marginal_diff=1.95e-02 | direct_dense (exact); engine=dense backend=cpu |
| aer | 2.26 | 1.34 | 0.00131 | 0 | max_marginal_diff=1.95e-02 | aer method=statevector |
| lightning | 10.9 | 11.1 | 0.000418 | 0 | max_marginal_diff=2.73e-02 |  |
| qsim | 1.91 | 1.85 | 0.00196 | 0 | max_marginal_diff=3.42e-02 |  |

## grover_n8  (family=grover, n=8, gates=12800, task=counts, shots=4096)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer statevector) | 0.133 | | | | status=ok |  |
| menos | 0.0147 | 0.0366 | 0.00508 | 0 | tvd=1.13e-02 | direct_dense (exact); engine=dense backend=cpu; P(marked)=0.986 (exact 0.986); tvd_expected~1.24e-02 |
| aer | 0.219 | 0.168 | 0.0303 | 0 | tvd=1.13e-02 | aer method=statevector; P(marked)=0.986 (exact 0.986); tvd_expected~1.24e-02 |
| lightning | 1.3 | 1.2 | 0.00021 | 0 | tvd=1.14e-02 | P(marked)=0.988 (exact 0.986); tvd_expected~1.24e-02 |
| qsim | 0.801 | 0.766 | 0.13 | 0 | tvd=1.20e-02 | P(marked)=0.985 (exact 0.986); tvd_expected~1.24e-02 |
| cirq | 1.64 | 1.58 | 0.0645 | 0 | tvd=1.15e-02 | P(marked)=0.989 (exact 0.986); tvd_expected~1.24e-02 |

## grover_n12  (family=grover, n=12, gates=152112, task=counts, shots=4096)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer statevector) | 9.85 | | | | status=ok |  |
| menos | 3.84 | 1.49 | 0.0413 | 0 | tvd=5.47e-05 | direct_dense (exact); engine=dense backend=cpu; P(marked)=1.000 (exact 1.000); tvd_expected~3.00e-03 |
| aer | 9.63 | 5.33 | 0.359 | 0 | tvd=5.47e-05 | aer method=statevector; P(marked)=1.000 (exact 1.000); tvd_expected~3.00e-03 |
| lightning | 13.4 | 12.9 | 0.000236 | 0 | tvd=5.47e-05 | P(marked)=1.000 (exact 1.000); tvd_expected~3.00e-03 |
| qsim | 6.59 | 7.04 | 2.25 | 0 | tvd=5.47e-05 | P(marked)=1.000 (exact 1.000); tvd_expected~3.00e-03 |
| cirq | 15.5 | 16.1 | 1.94 | 0 | tvd=5.47e-05 | P(marked)=1.000 (exact 1.000); tvd_expected~3.00e-03 |

## grover_n16  (family=grover, n=16, gates=122136, task=counts, shots=4096)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer statevector) | 9.83 | | | | status=ok |  |
| menos | 35.2 | 37.9 | 0.0727 | 0 | tvd=9.17e-01 | direct_dense (exact); engine=dense backend=cpu; P(marked)=0.027 (exact 0.025); tvd_expected~1.58e+00 |
| aer | 9.22 | 5.04 | 0.284 | 0 | tvd=9.17e-01 | aer method=statevector; P(marked)=0.027 (exact 0.025); tvd_expected~1.58e+00 |
| lightning | 29.8 | 29.1 | 0.00036 | 0 | tvd=9.17e-01 | P(marked)=0.027 (exact 0.025); tvd_expected~1.58e+00 |
| qsim | 12.6 | 11.3 | 2.71 | 0 | tvd=9.18e-01 | P(marked)=0.024 (exact 0.025); tvd_expected~1.58e+00 |

## qpe_n12  (family=qpe, n=12, gates=94, task=counts, shots=4096)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer statevector) | 0.0031 | | | | status=ok |  |
| menos | 0.00149 | 0.00217 | 9.88e-05 | 0 | tvd=8.88e-16 | direct_dense (exact); engine=dense backend=cpu; qpe top=01010000000 ok=True; tvd_expected~2.43e-15 |
| menos-fused | 0.0014 | 0.00172 | 7.77e-05 | 0 | tvd=8.88e-16 | direct_dense (exact); engine=dense backend=cpu; qpe top=01010000000 ok=True; tvd_expected~2.43e-15 |
| aer | 0.0297 | 0.0302 | 0.000546 | 0 | tvd=8.88e-16 | aer method=statevector; qpe top=01010000000 ok=True; tvd_expected~2.43e-15 |
| lightning | 0.0486 | 0.0735 | 0.000472 | 0 | tvd=8.88e-16 | qpe top=01010000000 ok=True; tvd_expected~2.43e-15 |
| qsim | 0.00627 | 0.00746 | 0.000932 | 0 | tvd=8.88e-16 | qpe top=01010000000 ok=True; tvd_expected~2.43e-15 |
| cirq | 0.015 | 0.0168 | 0.000861 | 0 | tvd=8.88e-16 | qpe top=01010000000 ok=True; tvd_expected~2.43e-15 |

## qpe_n16  (family=qpe, n=16, gates=158, task=counts, shots=4096)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer statevector) | 0.0162 | | | | status=ok |  |
| menos | 0.0147 | 0.0152 | 0.000134 | 0 | tvd=2.44e-15 | direct_dense (exact); engine=dense backend=cpu; qpe top=010100000000000 ok=True; tvd_expected~5.24e-14 |
| menos-fused | 0.0105 | 0.0109 | 9.15e-05 | 0 | tvd=2.44e-15 | direct_dense (exact); engine=dense backend=cpu; qpe top=010100000000000 ok=True; tvd_expected~5.24e-14 |
| aer | 0.0372 | 0.0321 | 0.000752 | 0 | tvd=2.44e-15 | aer method=statevector; qpe top=010100000000000 ok=True; tvd_expected~5.24e-14 |
| lightning | 0.0735 | 0.0612 | 0.00017 | 0 | tvd=2.44e-15 | qpe top=010100000000000 ok=True; tvd_expected~5.24e-14 |
| qsim | 0.0204 | 0.0246 | 0.00112 | 0 | tvd=2.44e-15 | qpe top=010100000000000 ok=True; tvd_expected~5.24e-14 |
| cirq | 0.0298 | 0.0283 | 0.00115 | 0 | tvd=2.44e-15 | qpe top=010100000000000 ok=True; tvd_expected~5.24e-14 |

## qpe_n20  (family=qpe, n=20, gates=238, task=counts, shots=4096)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer statevector) | 0.0877 | | | | status=ok |  |
| menos | 0.0759 | 0.088 | 0.000117 | 0 | tvd=3.11e-15 | direct_dense (exact); engine=dense backend=cpu; qpe top=0101000000000000000 ok=True; tvd_expected~1.06e-12 |
| menos-fused | 0.148 | 0.148 | 0.00011 | 0 | tvd=3.11e-15 | direct_dense (exact); engine=dense backend=cpu; qpe top=0101000000000000000 ok=True; tvd_expected~1.06e-12 |
| aer | 0.126 | 0.121 | 0.00202 | 0 | tvd=3.11e-15 | aer method=statevector; qpe top=0101000000000000000 ok=True; tvd_expected~1.06e-12 |
| lightning | 0.615 | 0.639 | 0.000326 | 0 | tvd=3.11e-15 | qpe top=0101000000000000000 ok=True; tvd_expected~1.06e-12 |
| qsim | 0.0987 | 0.123 | 0.00789 | 0 | tvd=3.11e-15 | qpe top=0101000000000000000 ok=True; tvd_expected~1.06e-12 |

## qaoa_ev_n16  (family=qaoa, n=16, gates=168, task=expectation)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer statevector) | 0.0153 | | | | status=ok |  |
| menos | 0.0284 | 0.0278 | 0.000197 | 0 | abs_err=3.61e-15 | direct_dense (exact); engine=dense backend=cpu |
| menos-fused | 0.0149 | 0.0179 | 0.000176 | 0 | abs_err=3.28e-15 | direct_dense (exact); engine=dense backend=cpu |
| aer | 0.016 | 0.0168 | 0.00345 | 0 | abs_err=0.00e+00 | aer method=statevector |
| lightning | 0.0448 | 0.0462 | 0.000323 | 0 | abs_err=8.33e-16 |  |
| qsim | 0.0165 | 0.0475 | 0.00266 | 0 | abs_err=3.28e-06 |  |
| cirq | 0.131 | 0.0631 | 0.00119 | 0 | abs_err=1.14e-15 |  |

## qaoa_ev_n20  (family=qaoa, n=20, gates=216, task=expectation)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer statevector) | 0.0974 | | | | status=ok |  |
| menos | 0.123 | 0.126 | 0.000158 | 0 | abs_err=2.30e-14 | direct_dense (exact); engine=dense backend=cpu |
| menos-fused | 0.16 | 0.171 | 0.000198 | 0 | abs_err=2.50e-14 | direct_dense (exact); engine=dense backend=cpu |
| aer | 0.0948 | 0.108 | 0.00242 | 0 | abs_err=0.00e+00 | aer method=statevector |
| lightning | 1.34 | 1.39 | 0.000449 | 0 | abs_err=3.46e-14 |  |
| qsim | 0.084 | 0.12 | 0.00238 | 0 | abs_err=3.28e-06 |  |
| cirq | 3.08 | 3.3 | 0.00129 | 0 | abs_err=4.55e-15 |  |

## qaoa_ev_n24  (family=qaoa, n=24, gates=256, task=expectation)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer statevector) | 2.84 | | | | status=ok |  |
| menos | 6.37 | 3.61 | 0.000196 | 0 | abs_err=1.12e-13 | direct_dense (exact); engine=dense backend=cpu |
| menos-fused | 5.26 | 2.2 | 0.000237 | 0 | abs_err=1.07e-13 | direct_dense (exact); engine=dense backend=cpu |
| aer | 2.93 | 1.65 | 0.00237 | 0 | abs_err=0.00e+00 | aer method=statevector |
| lightning | 29.9 | 26.9 | 0.000543 | 0 | abs_err=1.20e-14 |  |
| qsim | 1.62 | 1.65 | 0.00457 | 0 | abs_err=9.72e-08 |  |

## hea_heisenberg_ev_n12  (family=vqe, n=12, gates=164, task=expectation)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer statevector) | 0.0039 | | | | status=ok |  |
| menos | 0.0733 | 0.0825 | 0.000192 | 0 | abs_err=1.11e-15 | direct_mps (exact); engine=mps backend=cpu |
| aer | 0.00394 | 0.00591 | 0.00188 | 0 | abs_err=0.00e+00 | aer method=statevector |
| lightning | 0.0144 | 0.0168 | 0.00032 | 0 | abs_err=1.89e-15 |  |
| qsim | 0.00569 | 0.0153 | 0.00117 | 0 | abs_err=1.86e-07 |  |
| cirq | 0.0113 | 0.0129 | 0.00104 | 0 | abs_err=1.05e-15 |  |

## hea_heisenberg_ev_n16  (family=vqe, n=16, gates=220, task=expectation)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer statevector) | 0.0189 | | | | status=ok |  |
| menos | 0.183 | 0.192 | 0.000215 | 0 | abs_err=1.67e-15 | direct_mps (exact); engine=mps backend=cpu |
| aer | 0.0184 | 0.0247 | 0.00229 | 0 | abs_err=0.00e+00 | aer method=statevector |
| lightning | 0.0628 | 0.0585 | 0.0003 | 0 | abs_err=1.44e-15 |  |
| qsim | 0.0114 | 0.0138 | 0.00154 | 0 | abs_err=3.27e-07 |  |
| cirq | 0.393 | 0.193 | 0.0013 | 0 | abs_err=1.44e-15 |  |

## hea_heisenberg_ev_n20  (family=vqe, n=20, gates=276, task=expectation)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer statevector) | 0.101 | | | | status=ok |  |
| menos | 0.353 | 0.352 | 0.000222 | 0 | abs_err=3.66e-15 | direct_mps (exact); engine=mps backend=cpu |
| aer | 0.0985 | 0.0952 | 0.00253 | 0 | abs_err=0.00e+00 | aer method=statevector |
| lightning | 1.89 | 1.84 | 0.000245 | 0 | abs_err=1.51e-14 |  |
| qsim | 0.0682 | 0.0404 | 0.00154 | 0 | abs_err=1.07e-07 |  |
| cirq | 4.58 | 4.27 | 0.00149 | 0 | abs_err=4.83e-15 |  |

## hea_heisenberg_ev_n24  (family=vqe, n=24, gates=332, task=expectation)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer statevector) | 3.04 | | | | status=ok |  |
| menos | 0.56 | 0.551 | 0.000276 | 0 | abs_err=8.99e-15 | direct_mps (exact); engine=mps backend=cpu |
| aer | 3.11 | 1.61 | 0.00294 | 0 | abs_err=0.00e+00 | aer method=statevector |
| lightning | 34.2 | 36.7 | 0.000305 | 0 | abs_err=8.10e-14 |  |
| qsim | 2.13 | 2.11 | 0.00255 | 0 | abs_err=1.27e-07 |  |

## trotter_tfim_ev_n16  (family=trotter, n=16, gates=326, task=expectation)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer statevector) | 0.0267 | | | | status=ok |  |
| menos | 0.028 | 0.0314 | 0.000231 | 0 | abs_err=1.78e-14 | direct_dense (exact); engine=dense backend=cpu |
| menos-fused | 0.0239 | 0.0238 | 0.00025 | 0 | abs_err=3.38e-14 | direct_dense (exact); engine=dense backend=cpu |
| aer | 0.026 | 0.0267 | 0.00297 | 0 | abs_err=0.00e+00 | aer method=statevector |
| lightning | 0.0825 | 0.12 | 0.000187 | 0 | abs_err=2.66e-14 |  |
| qsim | 0.0259 | 0.0429 | 0.00192 | 0 | abs_err=1.01e-04 |  |
| cirq | 0.398 | 0.107 | 0.00172 | 0 | abs_err=4.41e-13 |  |

## trotter_tfim_ev_n20  (family=trotter, n=20, gates=410, task=expectation)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer statevector) | 0.182 | | | | status=ok |  |
| menos | 0.192 | 0.191 | 0.000248 | 0 | abs_err=5.33e-14 | direct_dense (exact); engine=dense backend=cpu |
| menos-fused | 0.233 | 0.243 | 0.000233 | 0 | abs_err=1.42e-13 | direct_dense (exact); engine=dense backend=cpu |
| aer | 0.183 | 0.18 | 0.00254 | 0 | abs_err=0.00e+00 | aer method=statevector |
| lightning | 3.44 | 2.57 | 0.0002 | 0 | abs_err=8.67e-13 |  |
| qsim | 0.0516 | 0.0496 | 0.00206 | 0 | abs_err=1.56e-04 |  |
| cirq | 7.57 | 7.52 | 0.00343 | 0 | abs_err=6.64e-13 |  |

## trotter_tfim_ev_n24  (family=trotter, n=24, gates=494, task=expectation)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer statevector) | 6.02 | | | | status=ok |  |
| menos | 9.71 | 9.25 | 0.000271 | 0 | abs_err=3.91e-13 | direct_dense (exact); engine=dense backend=cpu |
| menos-fused | 7.95 | 7.71 | 0.000265 | 0 | abs_err=5.44e-13 | direct_dense (exact); engine=dense backend=cpu |
| aer | 5.56 | 4.37 | 0.00322 | 0 | abs_err=0.00e+00 | aer method=statevector |
| lightning | 49.4 | 52.9 | 0.000393 | 0 | abs_err=4.90e-12 |  |
| qsim | 2.37 | 2.4 | 0.00455 | 0 | abs_err=2.21e-04 |  |

## variational_hea_n16_K20  (family=variational, n=16, gates=220, task=variational)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| menos | 0.352 | 0 | 0 | 0 | max_abs_err_vs_aer=1.15e-14 | direct_mps (exact); engine=mps backend=cpu |
| aer | 0.0218 | 0 | 0 | 0 | max_abs_err_vs_aer=0.00e+00 | aer method=statevector |
| lightning | 0.0575 | 0 | 0 | 0 | max_abs_err_vs_aer=1.22e-14 |  |
| qsim | 0.0195 | 0 | 0 | 0 | max_abs_err_vs_aer=9.53e-07 |  |
| cirq | 0.352 | 0 | 0 | 0 | max_abs_err_vs_aer=2.28e-15 |  |

## ghz_counts_n24  (family=clifford, n=24, gates=24, task=counts, shots=1024)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer stabilizer) | 0.000775 | | | | status=ok |  |
| menos | 0.00204 | 0.00263 | 8.59e-05 | 0 | max_marginal_diff=1.17e-02 | direct_mps (exact); engine=mps backend=cpu |
| aer | 0.0203 | 0.0173 | 0.000305 | 0 | max_marginal_diff=3.81e-02 | aer method=stabilizer |
| cirq-clifford | 3.92 | 3.38 | 0.00052 | 0 | max_marginal_diff=9.77e-04 |  |
| lightning | 3.08 | 1.73 | 0.00026 | 0 | max_marginal_diff=9.77e-04 |  |
| qsim | 0.211 | 0.18 | 0.000512 | 0 | max_marginal_diff=3.91e-03 |  |

## ghz_counts_n100  (family=clifford, n=100, gates=100, task=counts, shots=1024)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer stabilizer) | 0.00203 | | | | status=ok |  |
| menos | 0.0076 | 0.0101 | 9.62e-05 | 0 | max_marginal_diff=9.77e-04 | direct_mps (exact); engine=mps backend=cpu |
| aer | 0.169 | 0.123 | 0.000674 | 0 | max_marginal_diff=5.86e-03 | aer method=stabilizer |
| cirq-clifford | 12 | 46.1 | 0.000946 | 0 | max_marginal_diff=5.00e-01 |  |

## ghz_counts_n1000  (family=clifford, n=1000, gates=1000, task=counts, shots=1024)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer stabilizer) | 0.0216 | | | | status=ok |  |
| menos | 0.076 | 0.0751 | 0.000414 | 0 | max_marginal_diff=8.79e-03 | direct_mps (exact); engine=mps backend=cpu |
| aer | 151 | 129 | 0.00485 | 0 | max_marginal_diff=1.37e-02 | aer method=stabilizer |
| cirq-clifford | timeout | 0 | 0 |  | exceeded 600s |  |

## random_clifford_counts_n24  (family=clifford, n=24, gates=1775, task=counts, shots=1024)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer stabilizer) | 0.0176 | | | | status=ok |  |
| menos | 0.00156 | 0.00559 | 0.000912 | 0 | max_marginal_diff=3.52e-02 | direct_stabilizer (exact); engine=stabilizer backend=cpu |
| aer | 0.064 | 0.0353 | 0.00434 | 0 | max_marginal_diff=3.81e-02 | aer method=stabilizer |
| cirq-clifford | 2.15 | 2 | 0.00845 | 0 | max_marginal_diff=2.54e-02 |  |
| lightning | 69.4 | 28 | 0.000136 | 0 | max_marginal_diff=4.10e-02 |  |
| qsim | 9.11 | 9.25 | 0.0184 | 0 | max_marginal_diff=3.52e-02 |  |

## random_clifford_counts_n50  (family=clifford, n=50, gates=3725, task=counts, shots=1024)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer stabilizer) | 0.0399 | | | | status=ok |  |
| menos | 0.0034 | 0.555 | 0.00142 | 0 | max_marginal_diff=5.00e-01 | direct_stabilizer (exact); engine=stabilizer backend=cpu |
| aer | 0.124 | 0.0808 | 0.00941 | 0 | max_marginal_diff=4.30e-02 | aer method=stabilizer |
| cirq-clifford | 6.79 | 7.01 | 0.0222 | 0 | max_marginal_diff=4.49e-02 |  |

## random_clifford_counts_n200  (family=clifford, n=200, gates=14975, task=counts, shots=1024)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer stabilizer) | 0.146 | | | | status=ok |  |
| menos | 0.0288 | 0.0375 | 0.00402 | 0 | max_marginal_diff=5.00e-01 | direct_stabilizer (exact); engine=stabilizer backend=cpu |
| aer | 1.38 | 0.818 | 0.0356 | 0 | max_marginal_diff=3.91e-02 | aer method=stabilizer |
| cirq-clifford | 238 | 175 | 0.188 | 0 | max_marginal_diff=5.00e-01 |  |

## random_clifford_counts_n1000  (family=clifford, n=1000, gates=74975, task=counts, shots=1024)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer stabilizer) | 0.825 | | | | status=ok |  |
| menos | 2.41 | 2.15 | 0.0228 | 0 | max_marginal_diff=5.00e-01 | direct_stabilizer (exact); engine=stabilizer backend=cpu |
| aer | 45.6 | 70.6 | 0.634 | 0 | max_marginal_diff=5.76e-02 | aer method=stabilizer |

## low_entanglement_ev_n28  (family=mps, n=28, gates=166, task=expectation)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer matrix_product_state) | 0.00841 | | | | status=ok |  |
| menos | 0.159 | 0.0914 | 0.000168 | 0 | abs_err=2.17e-10 | direct_mps (exact); engine=mps backend=cpu |
| aer | 18.4 | 19.8 | 0.00213 | 0 | abs_err=2.67e-10 | aer method=statevector |
| aer-mps | 0.00362 | 0.00582 | 0.00389 | 0 | abs_err=0.00e+00 | aer method=matrix_product_state |
| lightning | 190 | 166 | 0.000295 | 0 | abs_err=1.95e-10 |  |
| qsim | 25.5 | 25.5 | 0.00482 | 4098 | abs_err=8.93e-06 |  |

## low_entanglement_ev_n40  (family=mps, n=40, gates=238, task=expectation)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer matrix_product_state) | 0.0102 | | | | status=ok |  |
| menos | 0.269 | 0.951 | 0.000491 | 256 | abs_err=2.75e-10 | direct_mps (exact); engine=mps backend=cpu |
| aer | error | 0.0128 | 0.00368 |  | QiskitError: 'ERROR:  [Experiment 0] Insufficient memory to  |  |
| aer-mps | 0.00956 | 0.0109 | 0.00377 | 2 | abs_err=0.00e+00 | aer method=matrix_product_state |

## low_entanglement_ev_n60  (family=mps, n=60, gates=358, task=expectation)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer matrix_product_state) | 0.017 | | | | status=ok |  |
| menos | 0.713 | 1.34 | 0.000411 | 257 | abs_err=3.98e-10 | direct_mps (exact); engine=mps backend=cpu |
| aer | error | 0.0137 | 0.00499 |  | QiskitError: 'ERROR:  [Experiment 0] Insufficient memory to  |  |
| aer-mps | 0.0145 | 0.0166 | 0.00459 | 2 | abs_err=0.00e+00 | aer method=matrix_product_state |

## factorized_2x13_counts_n26  (family=factorized, n=26, gates=760, task=counts, shots=1024)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer statevector) | 20.5 | | | | status=ok |  |
| menos | 0.0253 | 0.0224 | 0.000456 | 2 | max_marginal_diff=2.60e-02 | factorized (exact); engine=dense backend=cpu |
| aer | 18.3 | 20.1 | 0.00413 | 1030 | max_marginal_diff=3.28e-02 | aer method=statevector |
| lightning | timeout | 0 | 0 |  | exceeded 900s |  |
| qsim | 5.61 | 5.51 | 0.083 | 516 | max_marginal_diff=3.43e-02 |  |

## factorized_4x7_counts_n28  (family=factorized, n=28, gates=800, task=counts, shots=1024)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer statevector) | 116 | | | | status=ok |  |
| menos | 0.0424 | 0.0461 | 0.000991 | 2 | max_marginal_diff=4.23e-02 | direct_mps (exact); engine=mps backend=cpu |
| aer | 108 | 107 | 0.00794 | 4103 | max_marginal_diff=3.06e-02 | aer method=statevector |
| lightning | timeout | 0 | 0 |  | exceeded 900s |  |
| qsim | 17.6 | 17.6 | 0.0205 | 2052 | max_marginal_diff=4.11e-02 |  |

## noisy_exact_counts_n8  (family=noise_exact, n=8, gates=115, task=counts, shots=20000)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer density_matrix) | 0.0162 | | | | status=ok |  |
| menos | 0.0829 | 0.0804 | 0.00068 | 4 | tvd=4.28e-02 | direct_density_matrix (exact); engine=density_matrix backend=cpu; tvd_expected~4.27e-02 |
| aer-dm | 0.123 | 0.127 | 0.00051 | 9 | tvd=4.28e-02 | aer method=density_matrix; tvd_expected~4.27e-02 |
| aer | 0.124 | 0.135 | 0.000455 | 9 | tvd=4.28e-02 | aer method=density_matrix; tvd_expected~4.27e-02 |
| cirq-dm | 1.28 | 1.31 | 0.00531 | 14 | tvd=4.32e-02 | tvd_expected~4.27e-02 |
| default.mixed | 3.2 | 3.24 | 0.000125 | 8 | tvd=3.73e-02 | tvd_expected~4.27e-02 |

## noisy_exact_counts_n10  (family=noise_exact, n=10, gates=145, task=counts, shots=20000)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| reference (Aer density_matrix) | 0.169 | | | | status=ok |  |
| menos | 0.324 | 0.337 | 0.000684 | 20 | tvd=8.29e-02 | direct_density_matrix (exact); engine=density_matrix backend=cpu; tvd_expected~8.22e-02 |
| aer-dm | 0.262 | 0.251 | 0.000581 | 25 | tvd=8.29e-02 | aer method=density_matrix; tvd_expected~8.22e-02 |
| aer | 0.267 | 0.267 | 0.000577 | 24 | tvd=8.29e-02 | aer method=density_matrix; tvd_expected~8.22e-02 |
| cirq-dm | 29.5 | 29.5 | 0.00595 | 91 | tvd=8.57e-02 | tvd_expected~8.22e-02 |
| default.mixed | 63.9 | 64.1 | 0.000127 | 61 | tvd=8.25e-02 | tvd_expected~8.22e-02 |

## noisy_traj_counts_n16  (family=noise_traj, n=16, gates=235, task=counts, shots=1000)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| menos | 97 | 96.6 | 0.000986 | 5 | max_marginal_diff_vs_aer=3.70e-02 | direct_trajectory (approximate); engine=trajectory backend=cpu; sampled (trajectories) |
| aer | 5.93 | 6.07 | 0.000832 | 44 | max_marginal_diff_vs_aer=0.00e+00 | aer method=statevector |
| qsim | 3.37 | 2.75 | 0.00928 | 6 | max_marginal_diff_vs_aer=5.50e-02 |  |

## noisy_traj_counts_n20  (family=noise_traj, n=20, gates=295, task=counts, shots=1000)

| simulator | time (s) | first (s) | build (s) | peak mem (MiB) | check | notes |
|---|---|---|---|---|---|---|
| menos | timeout | 0 | 0 |  | exceeded 900s |  |
| aer | 121 | 349 | 0.000981 | 378 | max_marginal_diff_vs_aer=0.00e+00 | aer method=statevector |
| qsim | 15.8 | 15.6 | 0.0114 | 13 | max_marginal_diff_vs_aer=6.70e-02 |  |
