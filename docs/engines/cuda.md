# CUDA

The CUDA engine is the **same dense algorithm** running on an NVIDIA GPU. It is not a separate simulator: in a plan it appears as `engine=dense backend=cuda`. This page describes how it works; for setup, see [CUDA Installation](/gpu/cuda-installation).

## What runs on the device

| Step | Where |
|---|---|
| Initialize `|0…0⟩` with the global phase | Device (no upload of a `16·2ⁿ`-byte state) |
| Apply gates | Device |
| Pauli expectation reduction | Device, returns one scalar per term |
| Counts sampling | **Host**, from the copied-back state |
| Statevector | Copied back to the host once, only if requested |

An expectation-only job transfers almost nothing back, which is where the GPU's advantage is largest.

## Results are identical to the CPU's

Device kernels are compiled with `-fmad=false`, which disables the compiler's default fusing of `a·b + c` into a fused multiply-add. With that off, the device sums its products in exactly the same order as the CPU code, so:

- **Statevectors are bitwise identical** to the CPU's.
- **Counts are identical** for a given seed, because they are sampled on the host by the same sampler from that identical state.
- Expectation values agree to within `1e-12` (the device sums a reduction in a different but fixed order, so bitwise equality is not claimed).

This is what lets the planner move a region between CPU and GPU without changing the answer.

## When it is chosen

Backend choice is made by the planner from **measurements**, not by a rule. The dense algorithm is chosen first, by work units. Then the CPU and CUDA estimates are compared in calibrated seconds, including host-device transfers and a per-launch overhead, and CUDA is chosen only if it is strictly faster. See [Backend Selection](/planning/backend-selection).

On the development GPU (an RTX 3050 Laptop, 4 GB) the measured end-to-end crossover was **about 12 qubits**, and the GPU reached roughly **14 to 16 times** the CPU's speed by 24 qubits with transfers included. Below the crossover the GPU loses, because each kernel launch costs a fixed amount regardless of how little it does. These numbers are for that one GPU; yours will differ.

## Memory

Device memory is budgeted **separately** from host memory, from the free memory the driver reports rather than the marketing capacity. A region that does not fit is refused before any allocation, and the plan falls back to the CPU (unless CUDA is forced, which then raises instead). States come from MENOS's own device memory pool, which keeps a finished job's memory for the next one; `release_device_memory()` returns it to the driver. See [GPU Memory](/gpu/gpu-memory) and [Memory Architecture](/planning/memory-architecture).

## Scope

- **One device.** Multiple GPUs are not used.
- **Dense only.** No other engine runs on the GPU.
- **Per region, not per gate.** The backend is chosen per dense region. In a `factorized` plan each dense region is decided on its own (since Phase 17), so a large region can run on the GPU beside small ones on the CPU. MENOS does not split one region's gates between CPU and GPU, and `cut_and_recombine` plans keep every fragment on the CPU, because fragments are small and slower on the GPU.
- **NVIDIA only.** No ROCm or Metal backend exists.
- **Specialised one- and two-qubit kernels.** One- and two-qubit gates run on specialised kernels, which take the gate matrix as a kernel argument instead of copying it to the device before every gate. Larger gates run on a generic kernel that handles up to five qubits. Both give bitwise-identical results. On consumer GPUs, double-precision arithmetic runs at a small fraction of the single-precision rate, and exact results rule out fused multiply-add. So large circuits gain little from specialisation (about 1.2× end to end at 22–26 qubits on an RTX 4060 Laptop), while small circuits gain the most (about 1.6–2×).

## Related

[CPU vs CUDA](/gpu/cpu-vs-cuda) · [CUDA Requirements](/gpu/cuda-requirements) · [Troubleshooting CUDA](/gpu/troubleshooting-cuda)
