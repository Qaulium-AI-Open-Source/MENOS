#include "plan/cuda_calibration.h"

#include <algorithm>

#include "engines/dense_cuda/cuda_dense_engine.h"
#include "kernels/cuda/device_benchmarks.h"
#include "kernels/cuda/device_info.h"
#include "menos/circuit.h"
#include "menos/simulation_engine.h"

namespace menos::plan {
namespace {

using kernels::cuda::seconds_per_repetition;

// The device work rate is calibrated at 20 qubits. At the 13 qubits the host
// calibration uses, a GPU is entirely launch-bound and the resulting rate would
// describe overhead rather than throughput. 20 qubits is a 16 MiB device state,
// which saturates this class of GPU while costing only milliseconds.
constexpr std::uint32_t kWorkProbeQubits = 20;
constexpr std::uint32_t kWorkProbeLayers = 2;

// Takes no parameters on purpose: it has exactly one caller and both figures
// are the constants above, so passing them would only create two adjacent
// std::uint32_t arguments that could be transposed silently. The host
// calibration's make_calibration_circuit() is shaped the same way.
Circuit make_probe_circuit() {
    Circuit circuit(kWorkProbeQubits, 0);
    for (std::uint32_t layer = 0; layer < kWorkProbeLayers; ++layer) {
        for (std::uint32_t q = 0; q < kWorkProbeQubits; ++q) {
            circuit.append(Opcode::H, {static_cast<QubitId>(q)});
        }
        for (std::uint32_t q = layer % 2; q + 1 < kWorkProbeQubits; q += 2) {
            circuit.append(Opcode::CX, {static_cast<QubitId>(q), static_cast<QubitId>(q + 1)});
        }
    }
    return circuit;
}

}  // namespace

CudaCalibration measure_cuda_calibration() {
    CudaCalibration calibration;
    const kernels::cuda::DeviceInfo& info = kernels::cuda::device_info();
    if (!info.present) {
        return calibration;
    }
    calibration.present = true;
    calibration.device_total_bytes = info.total_memory_bytes;
    calibration.device_available_bytes = info.free_memory_bytes;

    // Transfers and per-launch overhead are pure kernel-layer measurements and
    // live there (kernels/cuda/device_benchmarks.h), so this file needs no CUDA
    // header at all. Only the work rate below has to be here, because it times
    // a real circuit through CudaDenseEngine, which sits above the kernels
    // layer.
    const kernels::cuda::TransferRates rates = kernels::cuda::measure_transfer_rates();
    calibration.host_to_device_bytes_per_second = rates.host_to_device_bytes_per_second;
    calibration.device_to_host_bytes_per_second = rates.device_to_host_bytes_per_second;
    calibration.kernel_launch_seconds = kernels::cuda::measure_kernel_launch_seconds();

    // --- fixed cost per execution --------------------------------------------
    // What every execution pays beyond its launches: allocating the state,
    // the synchronisations and the engine's setup. Timed through the engine
    // on one qubit with no gates, so nothing else is in it. Measured at
    // 0.19 ms on the development GPU, where a whole 4-qubit circuit takes 8 us
    // on the CPU (docs/adr/0028 section 23).
    {
        const Circuit circuit(1, 0);
        const engines::dense_cuda::CudaDenseEngine engine;
        const EngineEstimateRequest estimate_request{
            circuit.num_qubits(),
            &circuit.operations(),
            /*want_statevector=*/false,
            /*want_counts=*/false,
            /*want_expectation=*/false,
            /*observable=*/nullptr,
        };
        const CostEstimate estimate = engine.estimate(estimate_request);
        if (estimate.feasible) {
            const EngineExecutionRequest request{
                .num_qubits = circuit.num_qubits(),
                .operations = circuit.operations(),
                .want_statevector = false,
                .want_counts = false,
                .want_expectation = false,
                .shots = 0,
                .measure_targets = {},
                .bitstring_width = 0,
                .seed = 0,
                .global_phase = 0.0,
                .budget_bytes = estimate.estimated_peak_bytes,
                .is_sole_region = true,
                .observable = nullptr,
            };
            const double seconds =
                seconds_per_repetition([&] { static_cast<void>(engine.execute(request)); });
            const double launch_time = static_cast<double>(estimate.device.kernel_launches) *
                                       calibration.kernel_launch_seconds;
            calibration.execution_seconds = std::max(0.0, seconds - launch_time);
        }
    }

    // --- marginal device work rate ----------------------------------------
    {
        const Circuit circuit = make_probe_circuit();
        const engines::dense_cuda::CudaDenseEngine engine;
        const EngineEstimateRequest estimate_request{
            circuit.num_qubits(),
            &circuit.operations(),
            /*want_statevector=*/false,
            /*want_counts=*/false,
            /*want_expectation=*/false,
            /*observable=*/nullptr,
        };
        const CostEstimate estimate = engine.estimate(estimate_request);
        if (estimate.feasible && estimate.estimated_work_units > 0) {
            const EngineExecutionRequest request{
                .num_qubits = circuit.num_qubits(),
                .operations = circuit.operations(),
                .want_statevector = false,
                .want_counts = false,
                .want_expectation = false,
                .shots = 0,
                .measure_targets = {},
                .bitstring_width = 0,
                .seed = 0,
                .global_phase = 0.0,
                .budget_bytes = estimate.estimated_peak_bytes,
                .is_sole_region = true,
                .observable = nullptr,
            };
            const double seconds =
                seconds_per_repetition([&] { static_cast<void>(engine.execute(request)); });

            // Subtract the fixed and launch terms so they do not double-count. If
            // those already explain the whole measurement, the probe was
            // launch-bound and no meaningful throughput can be extracted from
            // it: leaving the rate at zero makes the profile uncalibrated,
            // which the planner refuses to plan a time budget against rather
            // than silently using a wrong number.
            const double launch_time = static_cast<double>(estimate.device.kernel_launches) *
                                       calibration.kernel_launch_seconds;
            const double work_time = seconds - calibration.execution_seconds - launch_time;
            if (work_time > 0.0) {
                calibration.device_work_units_per_second =
                    static_cast<double>(estimate.estimated_work_units) / work_time;
            }
        }
    }

    return calibration;
}

}  // namespace menos::plan
