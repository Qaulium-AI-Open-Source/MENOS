## (i) Disk

**Windows (NTFS, drive C:)** (3 runs): scratch `C:\Users\janak\menos\build\phase23-scratch`, capacity 722.8 GiB, host budget 15.9 GiB, 22 threads; a 2 GiB file; medians over 5 repetitions per run, min-max across runs.

| chunk | path | write GB/s | write+flush GB/s | read GB/s (cache dropped) |
|---|---|---|---|---|
| 256 MiB | explicit buffered | 1.44-1.57 | 1.41-1.53 | 4.80-4.97 |
| 256 MiB | explicit unbuffered | 4.69-4.74 | 4.68-4.72 | 6.07-6.09 |
| 256 MiB | mapped | 2.20 | 1.46 | 15.74-16.56 |
| 1024 MiB | explicit buffered | 1.52-1.55 | 1.51-1.54 | 4.86-5.03 |
| 1024 MiB | explicit unbuffered | 4.60-4.71 | 4.59-4.70 | 5.98-6.03 |
| 2048 MiB | explicit buffered | 1.51-1.55 | 1.50-1.54 | 4.83-5.05 |
| 2048 MiB | explicit unbuffered | 4.62-4.66 | 4.60-4.64 | 5.98-6.00 |

**WSL (ext4 scratch)** (3 runs): scratch `/home/janakiram/MENOS/build/phase23-scratch`, capacity 1006.9 GiB, host budget 11.7 GiB, 22 threads; a 2 GiB file; medians over 5 repetitions per run, min-max across runs.

| chunk | path | write GB/s | write+flush GB/s | read GB/s (cache dropped) |
|---|---|---|---|---|
| 256 MiB | explicit buffered | 4.51-4.67 | 2.34-2.39 | 4.33-4.40 |
| 256 MiB | explicit unbuffered | 4.47-4.56 | 4.42-4.50 | 6.41-6.49 |
| 256 MiB | mapped | 4.48-4.63 | 2.32-2.37 | 18.63-19.03 |
| 1024 MiB | explicit buffered | 4.48-4.56 | 2.33-2.35 | 3.71-4.29 |
| 1024 MiB | explicit unbuffered | 4.77-4.81 | 4.71-4.75 | 6.58-6.75 |
| 2048 MiB | explicit buffered | 4.50-4.55 | 2.33-2.36 | 3.15-4.39 |
| 2048 MiB | explicit unbuffered | 4.86-4.93 | 4.81-4.87 | 6.78-6.81 |

**WSL (tmpfs scratch)** (3 runs): scratch `/tmp`, capacity 11.7 GiB, host budget 11.7 GiB, 22 threads; a 2 GiB file; medians over 5 repetitions per run, min-max across runs.

| chunk | path | write GB/s | write+flush GB/s | read GB/s (cache dropped) |
|---|---|---|---|---|
| 256 MiB | explicit buffered | 3.88-4.04 | 3.88-4.04 | 5.06-5.34 |
| 256 MiB | explicit unbuffered | 3.92-4.09 | 3.92-4.09 | 5.01-5.29 |
| 256 MiB | mapped | 2.61-2.68 | 2.61-2.68 | 18.55-18.89 |
| 1024 MiB | explicit buffered | 4.05-4.13 | 4.05-4.13 | 5.27-5.33 |
| 1024 MiB | explicit unbuffered | 3.98-4.08 | 3.98-4.08 | 5.21-5.33 |
| 2048 MiB | explicit buffered | 3.74-4.10 | 3.74-4.10 | 4.96-5.33 |
| 2048 MiB | explicit unbuffered | 3.84-4.09 | 3.84-4.09 | 5.08-5.29 |

## (ii), (iii) CUDA transfers, windows, overlap, managed memory

**CUDA (WSL)** (3 runs): NVIDIA GeForce RTX 4060 Laptop GPU, 8.59 GB total, 7.44 GB free; runtime 12040, driver 12070; managed memory 1, concurrent managed access 0, pageable memory access 0; medians over 10 repetitions per run, min-max across runs.

Transfers (GB/s of the median):

| chunk | source | host to device | device to host |
|---|---|---|---|
| 1 GiB | pageable | 3.05 | 3.14 |
| 1 GiB | pinned | 3.05-3.13 | 3.18-3.19 |
| 1 GiB | registered | 2.99-3.02 | 3.17 |
| 2 GiB | pageable | 3.08-3.09 | 3.17-3.19 |
| 2 GiB | pinned | 3.10-3.19 | 3.23 |
| 2 GiB | registered | 3.04-3.06 | 3.22 |

A window copied in, k one-qubit gates applied, copied out (same run): the copies in units of one gate on the resident window.

| window | k | source | copies / one gate |
|---|---|---|---|
| 1 GiB | 1 | pageable | 19.7-19.9 |
| 1 GiB | 1 | pinned | 19.3-19.5 |
| 1 GiB | 10 | pageable | 35.4-35.9 |
| 1 GiB | 10 | pinned | 36.0-36.4 |
| 1 GiB | 25 | pageable | 53.0-54.1 |
| 1 GiB | 25 | pinned | 51.9-53.7 |
| 2 GiB | 1 | pageable | 21.1-21.3 |
| 2 GiB | 1 | pinned | 20.8 |
| 2 GiB | 10 | pageable | 48.5-49.6 |
| 2 GiB | 10 | pinned | 47.5-48.6 |
| 2 GiB | 25 | pageable | 49.7-51.0 |
| 2 GiB | 25 | pinned | 47.8-49.1 |

Overlap: a 1 GiB pinned copy on a second stream against eight memsets on the first (milliseconds).

| run | work only | copy only | sequential | concurrent | gain |
|---|---|---|---|---|---|
| wsl-run-a | 50.4 | 346.9 | 396.1 | 346.7 | 49.4 |
| wsl-run-b | 50.4 | 343.0 | 393.5 | 343.7 | 49.8 |
| wsl-run-c | 50.5 | 342.9 | 393.2 | 344.2 | 49.1 |

`cudaMallocManaged` beyond device memory, one-qubit sweeps (seconds); the resident reference is the 2^28 sweep scaled by 2^(n-28).

| run | n | outcome | sweep | resident, scaled | ratio |
|---|---|---|---|---|---|
| wsl-run-a | 29 | ok | 4.261 | 0.227 | 18.7x |
| wsl-run-a | 30 | exceeds host budget | - | - | - |
| wsl-run-b | 29 | ok | 4.402 | 0.235 | 18.8x |
| wsl-run-b | 30 | exceeds host budget | - | - | - |
| wsl-run-c | 29 | ok | 4.333 | 0.228 | 19.0x |
| wsl-run-c | 30 | exceeds host budget | - | - | - |

## (iv) Pass counts at the gate sizes

From analysis/passes-gate-sizes.txt (the benchmark's own table covers n = 28, 30, 32, 34). Fewest passes over L + w within the window limit, whole circuit and the first 100 operations:

| family | n | gates | global gates | passes, whole circuit | passes, first 100 ops |
|---|---|---|---|---|---|
| crossover | 29 | 264 | 45 | 13 (L=24 w=4) | 5 (L=24 w=4) |
| crossover | 31 | 282 | 54 | 18 (L=25 w=4) | 6 (L=25 w=4) |
| ladder | 29 | 342 | 60 | 18 (L=24 w=4) | 4 (L=24 w=4) |
| ladder | 31 | 366 | 72 | 20 (L=25 w=4) | 5 (L=25 w=4) |
| random_pairs | 29 | 200 | 46 | 5 (L=24 w=4) | 2 (L=24 w=4) |
| random_pairs | 31 | 200 | 43 | 6 (L=25 w=4) | 2 (L=25 w=4) |
| qft_like | 29 | 435 | 135 | 33 (L=24 w=4) | 4 (L=24 w=4) |
| qft_like | 31 | 496 | 171 | 42 (L=25 w=4) | 5 (L=25 w=4) |

## (v) In-memory engines

Per-gate seconds of the crossover circuit, (circuit - empty circuit) / gates, medians per run, min-max across runs.

| n | CPU, Windows (NTFS, drive C:) | CPU, WSL (ext4 scratch) | CPU, WSL (tmpfs scratch) | CUDA in VRAM (WSL) |
|---|---|---|---|---|
| 26 | 0.0338-0.0344 | 0.0337-0.0345 | 0.0340-0.0360 | 0.0176-0.0180 |
| 27 | 0.0658-0.0676 | 0.0667-0.0690 | 0.0680-0.0759 | 0.0351-0.0481 |
| 28 | 0.1300-0.1343 | 0.1309-0.1342 | 0.1397-0.1546 | 0.0701-0.0971 |
| 29 | 0.2631-0.2692 | 0.2663-0.2703 | 0.2683-0.2974 | not run |
| 30 | not run | not run | not run | not run |

Same-run CPU / CUDA per-gate ratio on WSL (each CUDA run against the CPU side of the same run):

| run | n = 26 | n = 27 | n = 28 |
|---|---|---|---|
| wsl-run-a | 2.00x | 1.58x | 1.59x |
| wsl-run-b | 1.98x | 1.98x | 2.00x |
| wsl-run-c | 1.93x | 1.94x | 1.99x |

## Predictions

### Q4: the host-offloaded tier at n = 29 against the in-memory CPU engine

Model: seconds = passes x (state bytes / host-to-device + state bytes / device-to-host, 2 GiB pageable, same run) + gates x device per-gate seconds at n = 29 (2 x the measured n = 28 in-VRAM figure, same run). The window is the largest that fits the device (L + w <= 28, a 4 GiB window); passes are the fewest over L and w. The CPU figure is the measured crossover circuit at n = 29 of the same run, and for the other families gates x the measured CPU per-gate seconds at n = 29. Wins if the ratio is below 1.

| run | family | gates | window | passes | s / pass | transfer s | compute s | predicted s | CPU in memory s | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| wsl-run-a | crossover | 264 | L=24 w=4 | 13 | 5.50 | 71 | 51 | 123 | 79 (measured) | 1.54x |
| wsl-run-a | ladder | 342 | L=24 w=4 | 18 | 5.50 | 99 | 66 | 165 | 98 (gates x per-gate) | 1.68x |
| wsl-run-a | random_pairs | 200 | L=24 w=4 | 5 | 5.50 | 27 | 39 | 66 | 57 (gates x per-gate) | 1.15x |
| wsl-run-a | qft_like | 435 | L=24 w=4 | 33 | 5.50 | 181 | 84 | 266 | 125 (gates x per-gate) | 2.13x |
| wsl-run-b | crossover | 264 | L=24 w=4 | 13 | 5.48 | 71 | 37 | 108 | 82 (measured) | 1.32x |
| wsl-run-b | ladder | 342 | L=24 w=4 | 18 | 5.48 | 99 | 48 | 147 | 102 (gates x per-gate) | 1.44x |
| wsl-run-b | random_pairs | 200 | L=24 w=4 | 5 | 5.48 | 27 | 28 | 55 | 59 (gates x per-gate) | 0.93x |
| wsl-run-b | qft_like | 435 | L=24 w=4 | 33 | 5.48 | 181 | 61 | 242 | 129 (gates x per-gate) | 1.87x |
| wsl-run-c | crossover | 264 | L=24 w=4 | 13 | 5.48 | 71 | 37 | 108 | 74 (measured) | 1.46x |
| wsl-run-c | ladder | 342 | L=24 w=4 | 18 | 5.48 | 99 | 48 | 147 | 92 (gates x per-gate) | 1.60x |
| wsl-run-c | random_pairs | 200 | L=24 w=4 | 5 | 5.48 | 27 | 28 | 55 | 54 (gates x per-gate) | 1.03x |
| wsl-run-c | qft_like | 435 | L=24 w=4 | 33 | 5.48 | 181 | 61 | 242 | 117 (gates x per-gate) | 2.07x |

Gates a pass must carry to break even (transfer per pass / (CPU gate - device gate), same run): wsl-run-a: 59, wsl-run-b: 35, wsl-run-c: 43.

Best ratio over every run and family: 0.93x (the tier wins somewhere).

### Q7: the disk tier at n = 31 on Windows (NTFS, drive C:) (explicit buffered, 1024 MiB chunks)

Model: seconds = passes x (state bytes / write+flush + state bytes / read, same run) + gates x CPU per-gate seconds at n = 31 (4 x the measured n = 29 figure, same run). The window is the largest the 16 GiB budget holds beside the outputs (L + w <= 29, an 8 GiB window); passes are the fewest over L and w. The 100-gate rule is applied three ways: the first 100 operations of each family, the family's passes scaled to 100 gates, and the worst case of one pass per gate. Limit: 30 minutes.

| run | family | s / pass | s / gate | whole circuit: gates / passes | whole min | first 100 ops: passes | min | passes per 100 gates | min |
|---|---|---|---|---|---|---|---|---|---|
| win-run-a | crossover | 29.7 | 1.05 | 282 / 18 (L=25 w=4) | 13.8 | 6 (L=25 w=4) | 4.7 | 6.4 | 4.9 |
| win-run-a | ladder | 29.7 | 1.05 | 366 / 20 (L=25 w=4) | 16.3 | 5 (L=25 w=4) | 4.2 | 5.5 | 4.5 |
| win-run-a | random_pairs | 29.7 | 1.05 | 200 / 6 (L=25 w=4) | 6.5 | 2 (L=25 w=4) | 2.7 | 3.0 | 3.2 |
| win-run-a | qft_like | 29.7 | 1.05 | 496 / 42 (L=25 w=4) | 29.5 | 5 (L=25 w=4) | 4.2 | 8.5 | 5.9 |
| win-run-a | worst case | 29.7 | 1.05 | - | - | 100 | 51.2 | 100 | 51.2 |
| win-run-b | crossover | 29.7 | 1.08 | 282 / 18 (L=25 w=4) | 14.0 | 6 (L=25 w=4) | 4.8 | 6.4 | 5.0 |
| win-run-b | ladder | 29.7 | 1.08 | 366 / 20 (L=25 w=4) | 16.5 | 5 (L=25 w=4) | 4.3 | 5.5 | 4.5 |
| win-run-b | random_pairs | 29.7 | 1.08 | 200 / 6 (L=25 w=4) | 6.6 | 2 (L=25 w=4) | 2.8 | 3.0 | 3.3 |
| win-run-b | qft_like | 29.7 | 1.08 | 496 / 42 (L=25 w=4) | 29.7 | 5 (L=25 w=4) | 4.3 | 8.5 | 6.0 |
| win-run-b | worst case | 29.7 | 1.08 | - | - | 100 | 51.3 | 100 | 51.3 |
| win-run-c | crossover | 29.2 | 1.08 | 282 / 18 (L=25 w=4) | 13.8 | 6 (L=25 w=4) | 4.7 | 6.4 | 4.9 |
| win-run-c | ladder | 29.2 | 1.08 | 366 / 20 (L=25 w=4) | 16.3 | 5 (L=25 w=4) | 4.2 | 5.5 | 4.5 |
| win-run-c | random_pairs | 29.2 | 1.08 | 200 / 6 (L=25 w=4) | 6.5 | 2 (L=25 w=4) | 2.8 | 3.0 | 3.3 |
| win-run-c | qft_like | 29.2 | 1.08 | 496 / 42 (L=25 w=4) | 29.3 | 5 (L=25 w=4) | 4.2 | 8.5 | 5.9 |
| win-run-c | worst case | 29.2 | 1.08 | - | - | 100 | 50.5 | 100 | 50.5 |

Largest family figure for 100 gates: 6.0 min against the 30-minute limit (passes); the one-pass-per-gate worst case is in the last row of each run.

### Q7: the disk tier at n = 31 on Windows (NTFS, drive C:) (explicit unbuffered, 1024 MiB chunks)

Model: seconds = passes x (state bytes / write+flush + state bytes / read, same run) + gates x CPU per-gate seconds at n = 31 (4 x the measured n = 29 figure, same run). The window is the largest the 16 GiB budget holds beside the outputs (L + w <= 29, an 8 GiB window); passes are the fewest over L and w. The 100-gate rule is applied three ways: the first 100 operations of each family, the family's passes scaled to 100 gates, and the worst case of one pass per gate. Limit: 30 minutes.

| run | family | s / pass | s / gate | whole circuit: gates / passes | whole min | first 100 ops: passes | min | passes per 100 gates | min |
|---|---|---|---|---|---|---|---|---|---|
| win-run-a | crossover | 13.2 | 1.05 | 282 / 18 (L=25 w=4) | 8.9 | 6 (L=25 w=4) | 3.1 | 6.4 | 3.2 |
| win-run-a | ladder | 13.2 | 1.05 | 366 / 20 (L=25 w=4) | 10.8 | 5 (L=25 w=4) | 2.9 | 5.5 | 3.0 |
| win-run-a | random_pairs | 13.2 | 1.05 | 200 / 6 (L=25 w=4) | 4.8 | 2 (L=25 w=4) | 2.2 | 3.0 | 2.4 |
| win-run-a | qft_like | 13.2 | 1.05 | 496 / 42 (L=25 w=4) | 17.9 | 5 (L=25 w=4) | 2.9 | 8.5 | 3.6 |
| win-run-a | worst case | 13.2 | 1.05 | - | - | 100 | 23.7 | 100 | 23.7 |
| win-run-b | crossover | 13.1 | 1.08 | 282 / 18 (L=25 w=4) | 9.0 | 6 (L=25 w=4) | 3.1 | 6.4 | 3.2 |
| win-run-b | ladder | 13.1 | 1.08 | 366 / 20 (L=25 w=4) | 10.9 | 5 (L=25 w=4) | 2.9 | 5.5 | 3.0 |
| win-run-b | random_pairs | 13.1 | 1.08 | 200 / 6 (L=25 w=4) | 4.9 | 2 (L=25 w=4) | 2.2 | 3.0 | 2.4 |
| win-run-b | qft_like | 13.1 | 1.08 | 496 / 42 (L=25 w=4) | 18.0 | 5 (L=25 w=4) | 2.9 | 8.5 | 3.6 |
| win-run-b | worst case | 13.1 | 1.08 | - | - | 100 | 23.6 | 100 | 23.6 |
| win-run-c | crossover | 13.2 | 1.08 | 282 / 18 (L=25 w=4) | 9.0 | 6 (L=25 w=4) | 3.1 | 6.4 | 3.2 |
| win-run-c | ladder | 13.2 | 1.08 | 366 / 20 (L=25 w=4) | 11.0 | 5 (L=25 w=4) | 2.9 | 5.5 | 3.0 |
| win-run-c | random_pairs | 13.2 | 1.08 | 200 / 6 (L=25 w=4) | 4.9 | 2 (L=25 w=4) | 2.2 | 3.0 | 2.5 |
| win-run-c | qft_like | 13.2 | 1.08 | 496 / 42 (L=25 w=4) | 18.1 | 5 (L=25 w=4) | 2.9 | 8.5 | 3.7 |
| win-run-c | worst case | 13.2 | 1.08 | - | - | 100 | 23.8 | 100 | 23.8 |

Largest family figure for 100 gates: 3.7 min against the 30-minute limit (passes); the one-pass-per-gate worst case is in the last row of each run.

### Q7: the disk tier at n = 31 on WSL (ext4 scratch) (explicit buffered, 1024 MiB chunks)

Model: seconds = passes x (state bytes / write+flush + state bytes / read, same run) + gates x CPU per-gate seconds at n = 31 (4 x the measured n = 29 figure, same run). The window is the largest the 16 GiB budget holds beside the outputs (L + w <= 29, an 8 GiB window); passes are the fewest over L and w. The 100-gate rule is applied three ways: the first 100 operations of each family, the family's passes scaled to 100 gates, and the worst case of one pass per gate. Limit: 30 minutes.

| run | family | s / pass | s / gate | whole circuit: gates / passes | whole min | first 100 ops: passes | min | passes per 100 gates | min |
|---|---|---|---|---|---|---|---|---|---|
| wsl-ext4-run-a | crossover | 22.8 | 1.07 | 282 / 18 (L=25 w=4) | 11.9 | 6 (L=25 w=4) | 4.1 | 6.4 | 4.2 |
| wsl-ext4-run-a | ladder | 22.8 | 1.07 | 366 / 20 (L=25 w=4) | 14.1 | 5 (L=25 w=4) | 3.7 | 5.5 | 3.9 |
| wsl-ext4-run-a | random_pairs | 22.8 | 1.07 | 200 / 6 (L=25 w=4) | 5.9 | 2 (L=25 w=4) | 2.5 | 3.0 | 2.9 |
| wsl-ext4-run-a | qft_like | 22.8 | 1.07 | 496 / 42 (L=25 w=4) | 24.8 | 5 (L=25 w=4) | 3.7 | 8.5 | 5.0 |
| wsl-ext4-run-a | worst case | 22.8 | 1.07 | - | - | 100 | 39.7 | 100 | 39.7 |
| wsl-ext4-run-b | crossover | 22.9 | 1.08 | 282 / 18 (L=25 w=4) | 12.0 | 6 (L=25 w=4) | 4.1 | 6.4 | 4.2 |
| wsl-ext4-run-b | ladder | 22.9 | 1.08 | 366 / 20 (L=25 w=4) | 14.2 | 5 (L=25 w=4) | 3.7 | 5.5 | 3.9 |
| wsl-ext4-run-b | random_pairs | 22.9 | 1.08 | 200 / 6 (L=25 w=4) | 5.9 | 2 (L=25 w=4) | 2.6 | 3.0 | 2.9 |
| wsl-ext4-run-b | qft_like | 22.9 | 1.08 | 496 / 42 (L=25 w=4) | 25.0 | 5 (L=25 w=4) | 3.7 | 8.5 | 5.0 |
| wsl-ext4-run-b | worst case | 22.9 | 1.08 | - | - | 100 | 40.0 | 100 | 40.0 |
| wsl-ext4-run-c | crossover | 23.9 | 1.07 | 282 / 18 (L=25 w=4) | 12.2 | 6 (L=25 w=4) | 4.2 | 6.4 | 4.3 |
| wsl-ext4-run-c | ladder | 23.9 | 1.07 | 366 / 20 (L=25 w=4) | 14.5 | 5 (L=25 w=4) | 3.8 | 5.5 | 3.9 |
| wsl-ext4-run-c | random_pairs | 23.9 | 1.07 | 200 / 6 (L=25 w=4) | 5.9 | 2 (L=25 w=4) | 2.6 | 3.0 | 3.0 |
| wsl-ext4-run-c | qft_like | 23.9 | 1.07 | 496 / 42 (L=25 w=4) | 25.5 | 5 (L=25 w=4) | 3.8 | 8.5 | 5.1 |
| wsl-ext4-run-c | worst case | 23.9 | 1.07 | - | - | 100 | 41.6 | 100 | 41.6 |

Largest family figure for 100 gates: 5.1 min against the 30-minute limit (passes); the one-pass-per-gate worst case is in the last row of each run.

### Q7: the disk tier at n = 31 on WSL (ext4 scratch) (explicit unbuffered, 1024 MiB chunks)

Model: seconds = passes x (state bytes / write+flush + state bytes / read, same run) + gates x CPU per-gate seconds at n = 31 (4 x the measured n = 29 figure, same run). The window is the largest the 16 GiB budget holds beside the outputs (L + w <= 29, an 8 GiB window); passes are the fewest over L and w. The 100-gate rule is applied three ways: the first 100 operations of each family, the family's passes scaled to 100 gates, and the worst case of one pass per gate. Limit: 30 minutes.

| run | family | s / pass | s / gate | whole circuit: gates / passes | whole min | first 100 ops: passes | min | passes per 100 gates | min |
|---|---|---|---|---|---|---|---|---|---|
| wsl-ext4-run-a | crossover | 12.5 | 1.07 | 282 / 18 (L=25 w=4) | 8.8 | 6 (L=25 w=4) | 3.0 | 6.4 | 3.1 |
| wsl-ext4-run-a | ladder | 12.5 | 1.07 | 366 / 20 (L=25 w=4) | 10.7 | 5 (L=25 w=4) | 2.8 | 5.5 | 2.9 |
| wsl-ext4-run-a | random_pairs | 12.5 | 1.07 | 200 / 6 (L=25 w=4) | 4.8 | 2 (L=25 w=4) | 2.2 | 3.0 | 2.4 |
| wsl-ext4-run-a | qft_like | 12.5 | 1.07 | 496 / 42 (L=25 w=4) | 17.6 | 5 (L=25 w=4) | 2.8 | 8.5 | 3.6 |
| wsl-ext4-run-a | worst case | 12.5 | 1.07 | - | - | 100 | 22.6 | 100 | 22.6 |
| wsl-ext4-run-b | crossover | 12.3 | 1.08 | 282 / 18 (L=25 w=4) | 8.8 | 6 (L=25 w=4) | 3.0 | 6.4 | 3.1 |
| wsl-ext4-run-b | ladder | 12.3 | 1.08 | 366 / 20 (L=25 w=4) | 10.7 | 5 (L=25 w=4) | 2.8 | 5.5 | 2.9 |
| wsl-ext4-run-b | random_pairs | 12.3 | 1.08 | 200 / 6 (L=25 w=4) | 4.8 | 2 (L=25 w=4) | 2.2 | 3.0 | 2.4 |
| wsl-ext4-run-b | qft_like | 12.3 | 1.08 | 496 / 42 (L=25 w=4) | 17.6 | 5 (L=25 w=4) | 2.8 | 8.5 | 3.5 |
| wsl-ext4-run-b | worst case | 12.3 | 1.08 | - | - | 100 | 22.3 | 100 | 22.3 |
| wsl-ext4-run-c | crossover | 12.4 | 1.07 | 282 / 18 (L=25 w=4) | 8.7 | 6 (L=25 w=4) | 3.0 | 6.4 | 3.1 |
| wsl-ext4-run-c | ladder | 12.4 | 1.07 | 366 / 20 (L=25 w=4) | 10.6 | 5 (L=25 w=4) | 2.8 | 5.5 | 2.9 |
| wsl-ext4-run-c | random_pairs | 12.4 | 1.07 | 200 / 6 (L=25 w=4) | 4.8 | 2 (L=25 w=4) | 2.2 | 3.0 | 2.4 |
| wsl-ext4-run-c | qft_like | 12.4 | 1.07 | 496 / 42 (L=25 w=4) | 17.5 | 5 (L=25 w=4) | 2.8 | 8.5 | 3.5 |
| wsl-ext4-run-c | worst case | 12.4 | 1.07 | - | - | 100 | 22.5 | 100 | 22.5 |

Largest family figure for 100 gates: 3.6 min against the 30-minute limit (passes); the one-pass-per-gate worst case is in the last row of each run.

