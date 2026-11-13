// Benchmark group 2 (docs/BUILD_PLAN.md Phase 3, 3.9): factorized versus
// direct circuits with controlled component sizes. Both strategies are
// forced explicitly (bypassing the planner's own selection) so the same
// circuit can be measured both ways -- this is a performance comparison,
// not a test of which strategy the planner picks (that's
// tests/plan/planner_test.cpp's job).

#include <benchmark/benchmark.h>

#include <numeric>
#include <vector>

#include "analysis/interaction_graph.h"
#include "common/checked_math.h"
#include "engines/dense/dense_engine.h"
#include "menos/menos.h"
#include "plan/available_engines.h"
#include "plan/hardware_profile.h"
#include "plan/job_spec.h"
#include "plan/region_builder.h"
#include "runtime/runtime.h"

namespace menos {
namespace {

// num_components independent components, each of component_size qubits,
// connected internally by a CX ladder (same shape as
// tests/runtime/runtime_test.cpp's make_three_independent_five_qubit_components,
// generalized).
Circuit make_independent_components_circuit(std::uint32_t num_components,
                                            std::uint32_t component_size) {
    Circuit circuit(num_components * component_size, 0);
    for (std::uint32_t block = 0; block < num_components; ++block) {
        const QubitId base = block * component_size;
        for (QubitId i = 0; i + 1 < component_size; ++i) {
            circuit.append(Opcode::CX,
                           {static_cast<QubitId>(base + i), static_cast<QubitId>(base + i + 1)});
        }
    }
    return circuit;
}

void run_direct(benchmark::State& state, const Circuit& circuit) {
    const plan::JobSpec job_spec{.circuit = circuit,
                                 .shots = 0,
                                 .want_statevector = true,
                                 .seed = 1,
                                 .memory_budget_bytes = std::nullopt,
                                 .expectation = std::nullopt};
    const engines::dense::DenseEngine dense_engine;
    for (auto _ : state) {
        const EngineEstimateRequest estimate_request{
            .num_qubits = circuit.num_qubits(),
            .operations = &circuit.operations(),
            .want_statevector = true,
            .want_counts = false,
            .want_expectation = false,
            .observable = nullptr,
        };
        const CostEstimate estimate = dense_engine.estimate(estimate_request);
        Plan direct_plan;
        direct_plan.strategy = Strategy::DirectDense;
        direct_plan.exact = true;
        direct_plan.regions.push_back(
            plan::build_direct_region(circuit, Engine::Dense, Backend::Cpu, estimate, nullptr));
        // This benchmark forces a plan directly, bypassing plan_job (and
        // its light-cone reduction) entirely -- circuit is already what
        // Runtime::execute() needs as the "reduced" circuit, since no
        // reduction ever ran.
        direct_plan.reduced_circuit = circuit;
        direct_plan.estimated_state_bytes = direct_plan.regions[0].estimated_state_bytes;
        direct_plan.estimated_work_units = 0;

        const runtime::Runtime rt;
        Result result = rt.execute(direct_plan, job_spec);
        benchmark::DoNotOptimize(result);
    }
}

void run_factorized(benchmark::State& state, const Circuit& circuit) {
    const analysis::InteractionGraph graph = analysis::build_interaction_graph(circuit);
    const plan::JobSpec job_spec{.circuit = circuit,
                                 .shots = 0,
                                 .want_statevector = true,
                                 .seed = 1,
                                 .memory_budget_bytes = std::nullopt,
                                 .expectation = std::nullopt};
    const std::vector<const SimulationEngine*>& engines = plan::available_engines();
    for (auto _ : state) {
        Plan factorized_plan;
        factorized_plan.strategy = Strategy::Factorized;
        factorized_plan.exact = true;
        factorized_plan.regions = plan::build_factorized_regions(circuit,
                                                                 graph,
                                                                 engines,
                                                                 /*want_statevector=*/true,
                                                                 /*want_counts=*/false,
                                                                 /*want_expectation=*/false,
                                                                 /*observable=*/nullptr,
                                                                 /*budget_bytes=*/1'000'000'000ULL,
                                                                 /*accuracy_policy=*/{},
                                                                 /*shots=*/0);
        factorized_plan.reduced_circuit = circuit;
        std::uint64_t peak = 0;
        for (const Region& region : factorized_plan.regions) {
            peak = std::max(peak, region.estimated_state_bytes);
        }
        // want_statevector is always true here, so the runtime allocates
        // a global 16*2^n buffer in addition to the largest live region
        // (src/runtime/recombine_statevector.h) -- estimated_state_bytes
        // must account for both, matching cost::estimate_factorized's own
        // formula, or the runtime's own budget check on the global buffer
        // rejects it.
        const std::uint64_t global_bytes = common::checked_mul(
            common::checked_pow2(circuit.num_qubits()), sizeof(std::complex<double>));
        factorized_plan.estimated_state_bytes = common::checked_add(peak, global_bytes);
        factorized_plan.estimated_work_units = 0;

        const runtime::Runtime rt;
        Result result = rt.execute(factorized_plan, job_spec);
        benchmark::DoNotOptimize(result);
    }
}

// Several components of fixed size 4 -- direct's cost grows exponentially
// with the TOTAL qubit count while factorized's grows only with the
// LARGEST region, so the gap should widen as num_components increases.
void BM_DirectManySmallComponents(benchmark::State& state) {
    const auto num_components = static_cast<std::uint32_t>(state.range(0));
    const Circuit circuit = make_independent_components_circuit(num_components, 4);
    run_direct(state, circuit);
}

void BM_FactorizedManySmallComponents(benchmark::State& state) {
    const auto num_components = static_cast<std::uint32_t>(state.range(0));
    const Circuit circuit = make_independent_components_circuit(num_components, 4);
    run_factorized(state, circuit);
}

BENCHMARK(BM_DirectManySmallComponents)->Arg(2)->Arg(3)->Arg(4)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_FactorizedManySmallComponents)->Arg(2)->Arg(3)->Arg(4)->Unit(benchmark::kMillisecond);

}  // namespace
}  // namespace menos
