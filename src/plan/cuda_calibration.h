#pragma once

#include <cstdint>

namespace menos::plan {

// Measured CUDA device figures, all of them consumed by the cost model
// (docs/adr/0014). Nothing here is a datasheet number or a constant.
struct CudaCalibration {
    bool present = false;

    // From cudaMemGetInfo. The runtime figures, materially below the
    // nameplate capacity -- see kernels::cuda::DeviceInfo.
    std::uint64_t device_total_bytes = 0;
    std::uint64_t device_available_bytes = 0;

    // PCIe transfer rates, measured with pageable host memory because that is
    // what the engine actually transfers (a std::vector), not with pinned
    // memory, which would flatter the measurement by testing a path the
    // backend does not take.
    double host_to_device_bytes_per_second = 0.0;
    double device_to_host_bytes_per_second = 0.0;

    // The fixed cost of one kernel launch, as the backend actually pays it:
    // the symbol upload, the launch itself and the synchronize. This is the
    // term that makes a small circuit slower on the GPU than on the CPU, so
    // without it the cost model would pick CUDA for circuits where it loses.
    double kernel_launch_seconds = 0.0;

    // The fixed cost of one execution beyond its launches: allocating the
    // state, the synchronisations and the engine's setup. Timed through
    // CudaDenseEngine on a one-qubit circuit with no gates, minus its
    // launches (docs/adr/0028 section 23).
    double execution_seconds = 0.0;

    // Marginal device throughput in cost-model work units per second, with the
    // launch overhead already subtracted out so the two terms do not
    // double-count:
    //
    //     measured_time = execution_seconds
    //                     + launches * kernel_launch_seconds
    //                     + work_units / device_work_units_per_second
    //
    // Calibrated at a qubit count where the GPU is genuinely saturated, not at
    // the small size used for the host calibration -- at 13 qubits a GPU is
    // entirely launch-overhead-bound and the rate would be meaningless.
    double device_work_units_per_second = 0.0;
};

// Runs the measurements. Returns a CudaCalibration with present == false and
// every field zero when no device is available. Never throws.
[[nodiscard]] CudaCalibration measure_cuda_calibration();

}  // namespace menos::plan
