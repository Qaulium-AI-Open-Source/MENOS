#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "api/replan_controller.h"
#include "common/error.h"
#include "engines/dense/dense_engine.h"
#include "engines/dense_cuda/cuda_dense_engine.h"
#include "ir/fuse_gates.h"
#include "kernels/cuda/device_buffer.h"
#include "kernels/cuda/device_info.h"
#include "menos/circuit.h"
#include "menos/result.h"
#include "menos/simulation_engine.h"
#include "plan/hardware_profile.h"
#include "plan/job_spec.h"
#include "plan/planner.h"

// The ExecutionMonitor contract against the CUDA dense engine (Phase 11,
// docs/adr/0015).
//
// tests/api/execution_monitor_test.cpp proves the seam for the five CPU
// engines. The CUDA engine calls the same seam, but until this file nothing
// exercised it, and it is the one engine where abandoning has a resource
// consequence beyond host memory: a live cudaMalloc allocation that has to be
// released while the stack unwinds. That is asserted here by MEASURING free
// device memory, not by trusting that DeviceBuffer is RAII.
//
// Skipped loudly, never silently passed, when no device is present.

namespace menos {
namespace {

using engines::dense::DenseEngine;
using engines::dense_cuda::CudaDenseEngine;

// Counts boundaries and throws at the Nth. The same shape as the CPU suite's
// monitor; redefined here because the two suites are separate translation
// units and this one must not depend on the other.
class CountingMonitor final : public ExecutionMonitor {
public:
    struct Abandoned {};

    CountingMonitor() = default;
    explicit CountingMonitor(std::uint64_t throw_at) : throw_at_(throw_at) {}

    void at_safe_boundary(std::uint64_t boundaries_crossed) const override {
        ++calls_;
        last_ = boundaries_crossed;
        if (throw_at_ != 0 && boundaries_crossed >= throw_at_) {
            throw Abandoned{};
        }
    }

    [[nodiscard]] std::uint64_t calls() const {
        return calls_;
    }
    [[nodiscard]] std::uint64_t last() const {
        return last_;
    }

private:
    std::uint64_t throw_at_ = 0;
    mutable std::uint64_t calls_ = 0;
    mutable std::uint64_t last_ = 0;
};

// One aggregate rather than two adjacent std::uint32_t parameters, so a qubit
// count cannot be silently transposed with a layer count at a call site.
struct Shape {
    std::uint32_t num_qubits = 0;
    std::uint32_t layers = 0;
};

Circuit layered(Shape shape) {
    const std::uint32_t num_qubits = shape.num_qubits;
    const std::uint32_t layers = shape.layers;
    Circuit circuit(num_qubits, 0);
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        for (std::uint32_t q = 0; q < num_qubits; ++q) {
            circuit.append(Opcode::H, {static_cast<QubitId>(q)});
        }
        circuit.append(Opcode::T, {0});
        for (std::uint32_t q = layer % 2; q + 1 < num_qubits; q += 2) {
            circuit.append(Opcode::CX, {static_cast<QubitId>(q), static_cast<QubitId>(q + 1)});
        }
    }
    return circuit;
}

EngineExecutionRequest make_request(const Circuit& circuit,
                                    bool want_statevector,
                                    const ExecutionMonitor* monitor) {
    return EngineExecutionRequest{
        .num_qubits = circuit.num_qubits(),
        .operations = circuit.operations(),
        .want_statevector = want_statevector,
        .want_counts = false,
        .want_expectation = false,
        .shots = 0,
        .measure_targets = {},
        .bitstring_width = circuit.num_clbits(),
        .seed = 7,
        .global_phase = 0.0,
        .budget_bytes = std::uint64_t{1} << 34,
        .is_sole_region = true,
        .observable = nullptr,
        .accuracy_policy = {},
        .monitor = monitor,
    };
}

// Free device memory right now, straight from the driver. Deliberately NOT
// kernels::cuda::device_info().free_memory_bytes, which is cached at first
// query and would report the same number before and after.
// The device memory a job could use right now: the driver's free figure plus
// what MENOS's own pool holds for reuse. Since Phase 22 (docs/adr/0026
// section 3.4) a released DeviceBuffer goes back to the pool rather than to
// the driver, so the driver's free figure alone no longer rises on release;
// this sum does, and it is the figure the planner budgets against.
std::uint64_t available_device_bytes() {
    return kernels::cuda::device_available_bytes();
}

class CudaExecutionMonitorTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!engines::dense_cuda::cuda_device_available()) {
            GTEST_SKIP() << "no CUDA device available -- CUDA monitor coverage did NOT run";
        }
        // Force lazy context creation and kernel loading NOW. The first launch
        // in a process allocates driver-side memory that is never returned, and
        // if it happened inside a measured window it would look like a leak.
        const CudaDenseEngine engine;
        const Circuit warmup = layered({.num_qubits = 10, .layers = 1});
        static_cast<void>(engine.execute(make_request(warmup, false, nullptr)));
    }
};

// The seam is called at the same boundaries as on the CPU: once per fused
// block with gate fusion (the default since Phase 24 item 7, docs/adr/0028
// section 17), once per operation with fusion off.
TEST_F(CudaExecutionMonitorTest, TheCudaEngineCallsTheMonitorOncePerBlockOrOncePerOperation) {
    const CudaDenseEngine engine;
    const Circuit circuit = layered({.num_qubits = 8, .layers = 3});
    const std::size_t blocks = ir::plan_fusion(circuit.operations()).block_count();
    ASSERT_LT(blocks, circuit.operations().size()) << "the circuit must fuse something";
    {
        const CountingMonitor monitor;
        static_cast<void>(engine.execute(make_request(circuit, false, &monitor)));
        EXPECT_EQ(monitor.calls(), blocks);
        EXPECT_EQ(monitor.last(), blocks);
    }
    {
        const CountingMonitor monitor;
        EngineExecutionRequest request = make_request(circuit, false, &monitor);
        request.fuse_gates = false;
        static_cast<void>(engine.execute(request));
        EXPECT_EQ(monitor.calls(), circuit.operations().size());
        EXPECT_EQ(monitor.last(), circuit.operations().size());
    }
}

// A monitor that never abandons must not perturb a single bit of the result.
// Bitwise, not within a tolerance: the CUDA statevector is bitwise identical to
// the CPU's (docs/adr/0014), and monitoring must not be able to break that.
TEST_F(CudaExecutionMonitorTest, ANullMonitorLeavesTheCudaStatevectorBitwiseIdentical) {
    const CudaDenseEngine engine;
    const Circuit circuit = layered({.num_qubits = 10, .layers = 3});

    const EngineExecutionResult without = engine.execute(make_request(circuit, true, nullptr));
    const CountingMonitor monitor;
    const EngineExecutionResult with = engine.execute(make_request(circuit, true, &monitor));

    ASSERT_TRUE(without.statevector.has_value());
    EXPECT_EQ(without.statevector, with.statevector);
    EXPECT_GT(monitor.calls(), 0U);
}

// Abandoning stops the run AT the requested boundary. Boundary k is reached only
// after operation k's kernel has been launched and synchronized, so a throw
// there leaves no kernel in flight -- the property that makes it safe to free
// the buffer during unwinding.
TEST_F(CudaExecutionMonitorTest, AThrowingMonitorAbandonsTheCudaEngineAtTheRequestedBoundary) {
    const CudaDenseEngine engine;
    const Circuit circuit = layered({.num_qubits = 10, .layers = 3});
    ASSERT_GT(circuit.operations().size(), 5U);

    const CountingMonitor monitor(4);
    EXPECT_THROW(static_cast<void>(engine.execute(make_request(circuit, true, &monitor))),
                 CountingMonitor::Abandoned);

    EXPECT_EQ(monitor.calls(), 4U) << "the run must stop AT boundary 4, not run on past it";
    EXPECT_EQ(monitor.last(), 4U);
}

// An abandoned run must leave nothing behind that can corrupt the next one: the
// kernels keep matrices in constant memory, so a stale upload would show up
// here as a wrong amplitude. The CPU engine is the oracle, as everywhere else.
TEST_F(CudaExecutionMonitorTest, AnAbandonedCudaRunLeavesTheEngineProducingCorrectResults) {
    const CudaDenseEngine gpu;
    const DenseEngine cpu;
    const Circuit circuit = layered({.num_qubits = 10, .layers = 3});

    for (int repeat = 0; repeat < 5; ++repeat) {
        const CountingMonitor monitor(3);
        EXPECT_THROW(static_cast<void>(gpu.execute(make_request(circuit, true, &monitor))),
                     CountingMonitor::Abandoned);
    }

    const EngineExecutionResult after = gpu.execute(make_request(circuit, true, nullptr));
    const EngineExecutionResult reference = cpu.execute(make_request(circuit, true, nullptr));
    ASSERT_TRUE(after.statevector.has_value());
    EXPECT_EQ(after.statevector, reference.statevector)
        << "a run after repeated abandons must still match the CPU bitwise";
}

// CONTROL for the two tests below. Without it, "available memory did not
// drop" could pass because the probe cannot see allocations at all. Holding a
// live 256 MiB buffer must visibly reduce available memory, so a leak of that
// size would too.
TEST_F(CudaExecutionMonitorTest, TheMemoryProbeCanSeeALiveDeviceAllocation) {
    constexpr std::size_t kAmplitudes = std::size_t{1} << 24;  // 256 MiB
    const std::uint64_t before = available_device_bytes();
    {
        const kernels::cuda::DeviceBuffer buffer(kAmplitudes,
                                                 kernels::cuda::device_available_bytes());
        const std::uint64_t during = available_device_bytes();
        EXPECT_LT(during + (std::uint64_t{200} << 20), before)
            << "a live 256 MiB allocation must lower available memory by well over 200 MiB";
    }
    EXPECT_GE(available_device_bytes() + (std::uint64_t{32} << 20), before)
        << "and releasing it must give the memory back (to the pool, Phase 22)";
}

// The direct measurement: available device memory after many abandons equals
// available device memory before them. Each abandoned run holds a 128 MiB
// buffer at the moment it is abandoned, so a leak of one buffer per abandon
// would show as roughly 1 GiB here, against a 32 MiB tolerance for unrelated
// driver activity.
TEST_F(CudaExecutionMonitorTest, AbandoningReleasesTheDeviceAllocation) {
    const CudaDenseEngine engine;
    const Circuit circuit =
        layered({.num_qubits = 23, .layers = 2});  // 16 * 2^23 = 128 MiB of state

    const std::uint64_t before = available_device_bytes();
    for (int repeat = 0; repeat < 8; ++repeat) {
        const CountingMonitor monitor(3);
        EXPECT_THROW(static_cast<void>(engine.execute(make_request(circuit, false, &monitor))),
                     CountingMonitor::Abandoned);
    }
    const std::uint64_t after = available_device_bytes();

    EXPECT_GE(after + (std::uint64_t{32} << 20), before)
        << "available device memory fell from " << before << " to " << after
        << " bytes across 8 abandoned runs: the device allocation was not released";
}

// The behavioural form of the same claim, and independent of the probe above.
// Ten abandoned runs of a 512 MiB state would need 5 GiB if every one leaked,
// against roughly 3.6 GiB of usable device memory on the development GPU, so a
// leak surfaces as an out-of-memory failure long before the loop ends. Nothing
// but Abandoned may escape.
TEST_F(CudaExecutionMonitorTest, TenAbandonsOfAnOversubscribingTotalNeverExhaustTheDevice) {
    const CudaDenseEngine engine;
    const Circuit circuit =
        layered({.num_qubits = 25, .layers = 2});  // 16 * 2^25 = 512 MiB of state

    const std::uint64_t available = kernels::cuda::device_available_bytes();
    constexpr std::uint64_t kStateBytes = std::uint64_t{512} << 20;
    constexpr int kAbandons = 10;
    if (kStateBytes * kAbandons <= available) {
        GTEST_SKIP() << "this device has " << available
                     << " bytes free, so ten 512 MiB states would fit without any release -- the "
                        "test could not detect a leak here";
    }

    for (int repeat = 0; repeat < kAbandons; ++repeat) {
        const CountingMonitor monitor(3);
        EXPECT_THROW(static_cast<void>(engine.execute(make_request(circuit, false, &monitor))),
                     CountingMonitor::Abandoned)
            << "abandon " << repeat << " failed with something other than the abandon signal";
    }
}

// ---------------------------------------------------------------------------
// The whole loop on the real GPU: a CUDA overrun is abandoned on the device and
// the job is restarted on the CPU (docs/adr/0015).
//
// The synthetic profile is the same one cuda_backend_selection_test.cpp uses: it
// pins the planner's decision so it means the same on every machine. The
// EXECUTION is real -- the first attempt really runs on the GPU and really is
// abandoned, and the second really runs on the CPU.
// ---------------------------------------------------------------------------

plan::HardwareProfile fast_device_profile() {
    plan::HardwareProfile hw;
    hw.total_memory_bytes = std::uint64_t{1} << 36;
    hw.memory_bandwidth_bytes_per_second = 1.0e10;
    hw.work_units_per_second = 1.0e7;
    hw.cuda_present = true;
    hw.cuda_device_total_bytes = std::uint64_t{1} << 32;
    hw.cuda_device_available_bytes = std::uint64_t{1} << 32;
    hw.cuda_host_to_device_bytes_per_second = 1.0e10;
    hw.cuda_device_to_host_bytes_per_second = 1.0e10;
    hw.cuda_kernel_launch_seconds = 1.0e-9;
    hw.cuda_device_work_units_per_second = 1.0e10;
    return hw;
}

plan::JobSpec statevector_job(const Circuit& circuit, double budget_seconds) {
    return plan::JobSpec{
        .circuit = circuit,
        .shots = 0,
        .want_statevector = true,
        .seed = 3,
        .memory_budget_bytes = std::uint64_t{1} << 34,
        .expectation = std::nullopt,
        .accuracy_policy = {},
        .time_budget_seconds = budget_seconds,
    };
}

// A clock factory whose first `crossing_attempts` attempts read 2000 s after a
// few boundaries (over a 1000 s budget) and whose later attempts never cross.
// `attempts` counts how many were requested.
api::ElapsedSourceFactory crossing_clocks(int crossing_attempts,
                                          const std::shared_ptr<int>& attempts) {
    return [crossing_attempts, attempts]() -> api::ElapsedSource {
        const int attempt = (*attempts)++;
        if (attempt >= crossing_attempts) {
            return []() { return 0.0; };
        }
        auto reads = std::make_shared<int>(0);
        return [reads]() { return (*reads)++ < 3 ? 0.0 : 2000.0; };
    };
}

TEST_F(CudaExecutionMonitorTest, AGpuOverrunIsAbandonedOnTheDeviceAndRestartedOnTheCpu) {
    const Circuit circuit = layered({.num_qubits = 12, .layers = 4});
    const plan::HardwareProfile hw = fast_device_profile();
    const plan::JobSpec job = statevector_job(circuit, 1000.0);
    const Plan initial = plan::plan_job(job, hw);
    ASSERT_EQ(initial.regions[0].backend, Backend::Cuda) << "the scenario starts on the GPU";

    const auto attempts = std::make_shared<int>(0);
    const Result result = api::run_with_replanning(job, hw, initial, crossing_clocks(1, attempts));

    ASSERT_EQ(result.replans().size(), 1U);
    const ReplanEvent& event = result.replans()[0];
    EXPECT_EQ(event.engine, Engine::Dense);
    EXPECT_EQ(event.backend, Backend::Cuda) << "the abandoned attempt ran on the GPU";
    EXPECT_EQ(event.boundaries_crossed, 4U) << "abandoned at the first boundary over budget";
    EXPECT_EQ(event.superseded_plan.regions[0].backend, Backend::Cuda);

    // The replan landed on the CPU rather than being refused.
    EXPECT_EQ(result.plan().strategy, Strategy::DirectDense);
    EXPECT_EQ(result.plan().regions[0].engine, Engine::Dense);
    EXPECT_EQ(result.plan().regions[0].backend, Backend::Cpu);
    EXPECT_EQ(*attempts, 2) << "one abandoned GPU attempt and one CPU attempt that completed";

    // Restart correctness across BACKENDS: the CPU run that replaced the abandoned
    // GPU run must give the answer a cold CPU run gives, bitwise.
    plan::HardwareProfile no_device = hw;
    no_device.cuda_present = false;
    const plan::JobSpec cold_job = statevector_job(circuit, 1000.0);
    const Plan cold_plan = plan::plan_job(cold_job, no_device);
    ASSERT_EQ(cold_plan.regions[0].backend, Backend::Cpu);
    const Result cold = api::run_with_replanning(
        cold_job, no_device, cold_plan, [] { return api::steady_elapsed_source(); });
    ASSERT_TRUE(result.statevector().has_value());
    EXPECT_EQ(result.statevector(), cold.statevector());

    const std::string report = result.explain();
    EXPECT_NE(report.find("abandoned after 4 safe boundaries on engine=dense backend=cuda"),
              std::string::npos)
        << report;
    EXPECT_NE(report.find("engine=dense backend=cpu"), std::string::npos) << report;
}

// The converse through the real loop: the GPU overran, then the CPU overran too,
// so both backends carry an over-budget floor and the job is refused.
TEST_F(CudaExecutionMonitorTest, AnOverrunOnTheGpuThenTheCpuEndsInTheStrictRefusal) {
    const Circuit circuit = layered({.num_qubits = 12, .layers = 4});
    const plan::HardwareProfile hw = fast_device_profile();
    const plan::JobSpec job = statevector_job(circuit, 1000.0);
    const Plan initial = plan::plan_job(job, hw);
    ASSERT_EQ(initial.regions[0].backend, Backend::Cuda);

    const auto attempts = std::make_shared<int>(0);
    try {
        static_cast<void>(api::run_with_replanning(job, hw, initial, crossing_clocks(2, attempts)));
        FAIL() << "both backends overran, so the job must be refused";
    } catch (const common::ResourceExceeded& e) {
        EXPECT_NE(std::string(e.what()).find("no different plan meets that budget"),
                  std::string::npos)
            << e.what();
    }
    EXPECT_EQ(*attempts, 2) << "the GPU, then the CPU, each tried exactly once";
}

}  // namespace
}  // namespace menos
