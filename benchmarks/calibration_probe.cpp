// Cold hardware calibration probe (Phase 24 step 1, docs/adr/0028 section 5).
//
// Times plan::calibrate_hardware() twice in one process: the first call is the
// cold figure a user's first time-budgeted (or, in a CUDA build, any) job pays;
// the second is the same work warm. Prints one CSV line with both times and
// every derived figure, so that running it in many fresh processes
// (bench/run_calibration.sh) gives both the cost of calibration and the
// run-to-run spread of the constants it derives -- the baseline any later
// accuracy bound on those constants is set against (owner, 2026-11-03).

#include <chrono>
#include <cstdio>
#include <string_view>

#include "plan/hardware_profile.h"

int main(int argc, char** argv) {
    using Clock = std::chrono::steady_clock;
    const bool header = argc > 1 && std::string_view(argv[1]) == "--header";
    if (header) {
        std::puts(
            "cold_s,warm_s,work_units_per_s,bandwidth_B_per_s,cuda_present,"
            "cuda_h2d_B_per_s,cuda_d2h_B_per_s,cuda_launch_s,cuda_work_units_per_s");
        return 0;
    }
    const Clock::time_point t0 = Clock::now();
    const menos::plan::HardwareProfile cold = menos::plan::calibrate_hardware();
    const Clock::time_point t1 = Clock::now();
    const menos::plan::HardwareProfile warm = menos::plan::calibrate_hardware();
    const Clock::time_point t2 = Clock::now();
    static_cast<void>(warm);
    std::printf("%.6f,%.6f,%.6e,%.6e,%d,%.6e,%.6e,%.6e,%.6e\n",
                std::chrono::duration<double>(t1 - t0).count(),
                std::chrono::duration<double>(t2 - t1).count(),
                cold.work_units_per_second,
                cold.memory_bandwidth_bytes_per_second,
                cold.cuda_present ? 1 : 0,
                cold.cuda_host_to_device_bytes_per_second,
                cold.cuda_device_to_host_bytes_per_second,
                cold.cuda_kernel_launch_seconds,
                cold.cuda_device_work_units_per_second);
    return 0;
}
