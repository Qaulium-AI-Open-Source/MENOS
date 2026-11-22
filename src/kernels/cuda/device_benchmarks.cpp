#include "kernels/cuda/device_benchmarks.h"

#include <cuda_runtime.h>

#include <complex>
#include <cstddef>
#include <vector>

#include "kernels/cuda/device_buffer.h"
#include "kernels/cuda/device_info.h"
#include "kernels/dense_kernels.h"

namespace menos::kernels::cuda {
namespace {

// 16 MiB: comfortably past the per-transfer fixed overhead, and small enough
// that the whole calibration stays well under a second.
constexpr std::size_t kTransferBytes = std::size_t{16} << 20;

// Two qubits: a kernel's actual work here is negligible against the cost of
// launching it, so the measured per-call time is the fixed overhead.
constexpr std::size_t kLaunchProbeAmplitudes = 4;
constexpr int kLaunchProbeCalls = 200;

}  // namespace

TransferRates measure_transfer_rates() {
    TransferRates rates;
    const DeviceInfo& info = device_info();
    if (!info.present) {
        return rates;
    }

    const std::size_t amplitudes = kTransferBytes / sizeof(std::complex<double>);
    DeviceBuffer buffer(amplitudes, info.free_memory_bytes);
    std::vector<std::complex<double>> host(amplitudes);
    void* device = buffer.state().data;

    const double h2d = seconds_per_repetition([&] {
        static_cast<void>(cudaMemcpy(device, host.data(), kTransferBytes, cudaMemcpyHostToDevice));
        static_cast<void>(cudaDeviceSynchronize());
    });
    const double d2h = seconds_per_repetition([&] {
        static_cast<void>(cudaMemcpy(host.data(), device, kTransferBytes, cudaMemcpyDeviceToHost));
        static_cast<void>(cudaDeviceSynchronize());
    });

    if (h2d > 0.0) {
        rates.host_to_device_bytes_per_second = static_cast<double>(kTransferBytes) / h2d;
    }
    if (d2h > 0.0) {
        rates.device_to_host_bytes_per_second = static_cast<double>(kTransferBytes) / d2h;
    }
    return rates;
}

double measure_kernel_launch_seconds() {
    const DeviceInfo& info = device_info();
    if (!info.present) {
        return 0.0;
    }

    DeviceBuffer buffer(kLaunchProbeAmplitudes, info.free_memory_bytes);
    // The identity, so the probe measures launch cost and not arithmetic.
    const std::vector<std::complex<double>> identity{
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}};
    const std::vector<std::uint32_t> operands{0};

    const double per_batch = seconds_per_repetition([&] {
        for (int i = 0; i < kLaunchProbeCalls; ++i) {
            apply_matrix(buffer.state(), identity, operands);
        }
    });
    return per_batch / kLaunchProbeCalls;
}

}  // namespace menos::kernels::cuda
