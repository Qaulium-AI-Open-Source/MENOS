// Phase 24 step 1, item 9 (docs/adr/0028 sections 16.5, 16.7 Q7 and 16.9):
// the dense CPU work-unit rate as a function of state size and thread count,
// and the scratch disk's write and read bandwidths, measured separately.
//
// The production calibration (src/plan/hardware_profile.cpp,
// measure_work_units_per_second) times a 13-qubit, 2-layer H/CX circuit. At 13
// qubits a gate has at most 2^12 groups, below kParallelMinGroups (2^16,
// src/kernels/dense_kernels.h), so that measurement runs on one thread
// whatever the thread count. This probe repeats the same measurement -- the
// same circuit shape, the cost model's own work units, the same discarded
// pass and deadline-bounded repetition -- at 13, 16, 17, 18, 20 and 22 qubits,
// on one thread and on every hardware thread, and prints how long each
// measurement took (the calibration cost a threaded rate would add).
//
// Usage: menos_dense_rate_probe [scratch_directory]
// With a directory, the disk tier's bandwidth probe
// (plan::measure_scratch_bandwidth, at most 1 GiB) runs there too.
//
// CSV on stdout. A report, not a unit test.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>

#include "common/thread_count.h"
#include "engines/dense/dense_engine.h"
#include "menos/circuit.h"
#include "menos/simulation_engine.h"
#include "plan/hardware_profile.h"

namespace {

using Clock = std::chrono::steady_clock;
using namespace menos;

// The production calibration's own deadline and minimum repetitions
// (src/plan/hardware_profile.cpp, kThroughputDeadlineSeconds and
// kThroughputMinRepetitions).
constexpr double kDeadlineSeconds = 0.015;
constexpr std::uint64_t kMinRepetitions = 2;
constexpr std::uint32_t kGateLayers = 2;

Circuit calibration_circuit(std::uint32_t n) {
    Circuit circuit(n, 0);
    for (std::uint32_t layer = 0; layer < kGateLayers; ++layer) {
        for (std::uint32_t q = 0; q < n; ++q) {
            circuit.append(Opcode::H, {q});
        }
        for (std::uint32_t q = layer % 2; q + 1 < n; q += 2) {
            circuit.append(Opcode::CX, {q, q + 1});
        }
    }
    return circuit;
}

struct Rate {
    double work_units_per_second = 0.0;
    double measurement_seconds = 0.0;  // both passes, as calibration pays them
    std::uint64_t repetitions = 0;     // in the kept pass
};

Rate measure(std::uint32_t n, std::uint32_t threads) {
    const Circuit circuit = calibration_circuit(n);
    const engines::dense::DenseEngine engine;
    const CostEstimate estimate =
        engine.estimate(EngineEstimateRequest{.num_qubits = n,
                                              .operations = &circuit.operations(),
                                              .want_statevector = true,
                                              .want_counts = false,
                                              .want_expectation = false,
                                              .observable = nullptr});
    const EngineExecutionRequest request{.num_qubits = n,
                                         .operations = circuit.operations(),
                                         .want_statevector = true,
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
                                         .max_threads = threads};
    Rate rate;
    const auto pass = [&]() {
        std::uint64_t repetitions = 0;
        const Clock::time_point start = Clock::now();
        double seconds = 0.0;
        do {
            static_cast<void>(engine.execute(request));
            ++repetitions;
            seconds = std::chrono::duration<double>(Clock::now() - start).count();
        } while (seconds < kDeadlineSeconds || repetitions < kMinRepetitions);
        rate.repetitions = repetitions;
        return static_cast<double>(estimate.estimated_work_units) *
               static_cast<double>(repetitions) / seconds;
    };
    const Clock::time_point start = Clock::now();
    static_cast<void>(pass());  // discarded, as in calibration
    rate.work_units_per_second = pass();
    rate.measurement_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return rate;
}

}  // namespace

int main(int argc, char** argv) {
    const std::uint32_t all = common::default_thread_count();
    std::printf("kind,qubits,threads,work_units_per_s,measurement_s,repetitions\n");
    for (const std::uint32_t n : {13U, 16U, 17U, 18U, 20U, 22U}) {
        for (const std::uint32_t threads : {1U, all}) {
            const Rate r = measure(n, threads);
            std::printf("dense_rate,%u,%u,%.6e,%.6f,%llu\n",
                        n,
                        threads,
                        r.work_units_per_second,
                        r.measurement_seconds,
                        static_cast<unsigned long long>(r.repetitions));
            std::fflush(stdout);
        }
    }
    if (argc > 1) {
        const std::filesystem::path directory(argv[1]);
        const Clock::time_point start = Clock::now();
        const plan::ScratchBandwidth bw = plan::measure_scratch_bandwidth(directory);
        const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        std::printf("kind,write_B_per_s,read_B_per_s,measured_bytes,measurement_s\n");
        std::printf("scratch,%.6e,%.6e,%llu,%.6f\n",
                    bw.write_bytes_per_second,
                    bw.read_bytes_per_second,
                    static_cast<unsigned long long>(bw.measured_bytes),
                    seconds);
    }
    return 0;
}
