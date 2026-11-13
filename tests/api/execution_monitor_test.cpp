#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "engines/dense/dense_engine.h"
#include "engines/mps/mps_engine.h"
#include "engines/pauli/pauli_propagation_engine.h"
#include "engines/stabilizer/stabilizer_engine.h"
#include "engines/tensor_network/tensor_network_engine.h"
#include "ir/fuse_gates.h"
#include "menos/circuit.h"
#include "menos/observable.h"
#include "menos/simulation_engine.h"

// The ExecutionMonitor contract, checked against every engine that implements
// it (Phase 11, docs/adr/0015). These are deliberately engine-level rather than
// end-to-end: the point is that the SEAM behaves identically everywhere, not
// that any particular plan is chosen.
//
// Two properties are asserted for each engine:
//   1. a null monitor changes nothing -- results are bitwise identical to a run
//      with no monitor at all, which is what lets every pre-Phase-11 caller
//      keep its behavior;
//   2. a throwing monitor actually abandons the run, and the throw propagates
//      out of execute() rather than being swallowed. Nine pre-existing catch
//      sites exist in src/, every one catching common::ResourceExceeded
//      specifically; a signal type they could catch would be silently ignored,
//      so this pins it.

namespace menos {
namespace {

// Counts boundaries, and optionally throws at the Nth one.
class CountingMonitor final : public ExecutionMonitor {
public:
    CountingMonitor() = default;
    explicit CountingMonitor(std::uint64_t throw_at) : throw_at_(throw_at) {}

    void at_safe_boundary(std::uint64_t boundaries_crossed) const override {
        ++calls_;
        last_ = boundaries_crossed;
        if (throw_at_ != 0 && boundaries_crossed >= throw_at_) {
            throw Abandoned{};
        }
    }

    struct Abandoned {};

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

Circuit non_clifford_circuit(std::uint32_t num_qubits) {
    Circuit circuit(num_qubits, num_qubits);
    for (std::uint32_t layer = 0; layer < 3; ++layer) {
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

// Pauli propagation supports RZ/P/RX/RY/U but not T, so its circuit uses RZ
// for the non-Clifford rotation. execute() is called directly here, bypassing
// the estimate() that would normally have rejected an unsupported opcode.
Circuit pauli_circuit(std::uint32_t num_qubits) {
    Circuit circuit(num_qubits, num_qubits);
    for (std::uint32_t layer = 0; layer < 3; ++layer) {
        for (std::uint32_t q = 0; q < num_qubits; ++q) {
            circuit.append(Opcode::H, {static_cast<QubitId>(q)});
        }
        circuit.append(Opcode::RZ, {0}, {0.37});
        for (std::uint32_t q = layer % 2; q + 1 < num_qubits; q += 2) {
            circuit.append(Opcode::CX, {static_cast<QubitId>(q), static_cast<QubitId>(q + 1)});
        }
    }
    return circuit;
}

Circuit clifford_circuit(std::uint32_t num_qubits) {
    Circuit circuit(num_qubits, num_qubits);
    for (std::uint32_t layer = 0; layer < 3; ++layer) {
        for (std::uint32_t q = 0; q < num_qubits; ++q) {
            circuit.append(Opcode::H, {static_cast<QubitId>(q)});
        }
        for (std::uint32_t q = layer % 2; q + 1 < num_qubits; q += 2) {
            circuit.append(Opcode::CX, {static_cast<QubitId>(q), static_cast<QubitId>(q + 1)});
        }
    }
    return circuit;
}

// Single-qubit Z on qubit 0, as a one-term observable.
Observable single_z() {
    PauliString pauli;
    pauli.num_qubits = 1;
    pauli.x_words = {0};
    pauli.z_words = {1};
    Observable observable;
    observable.terms.emplace_back(pauli, std::complex<double>{1.0, 0.0});
    return observable;
}

EngineExecutionRequest make_request(const Circuit& circuit,
                                    bool want_statevector,
                                    bool want_expectation,
                                    const Observable* observable,
                                    const ExecutionMonitor* monitor) {
    return EngineExecutionRequest{
        .num_qubits = circuit.num_qubits(),
        .operations = circuit.operations(),
        .want_statevector = want_statevector,
        .want_counts = false,
        .want_expectation = want_expectation,
        .shots = 0,
        .measure_targets = {},
        .bitstring_width = circuit.num_clbits(),
        .seed = 7,
        .global_phase = 0.0,
        .budget_bytes = std::uint64_t{1} << 32,
        .is_sole_region = true,
        .observable = observable,
        .accuracy_policy = {},
        .monitor = monitor,
    };
}

// With gate fusion (the default since Phase 24 item 7, docs/adr/0028
// section 17) the dense engine's safe boundary is the gap between two fused
// blocks, so it reports once per block; with fusion off, once per operation
// as before.
TEST(ExecutionMonitorTest, DenseCallsTheMonitorOncePerBlockOrOncePerOperation) {
    const engines::dense::DenseEngine engine;
    const Circuit circuit = non_clifford_circuit(6);
    const std::size_t blocks = ir::plan_fusion(circuit.operations()).block_count();
    ASSERT_LT(blocks, circuit.operations().size()) << "the circuit must fuse something";
    {
        const CountingMonitor monitor;
        static_cast<void>(engine.execute(make_request(circuit, true, false, nullptr, &monitor)));
        EXPECT_EQ(monitor.calls(), blocks);
        EXPECT_EQ(monitor.last(), blocks);
    }
    {
        const CountingMonitor monitor;
        EngineExecutionRequest request = make_request(circuit, true, false, nullptr, &monitor);
        request.fuse_gates = false;
        static_cast<void>(engine.execute(request));
        EXPECT_EQ(monitor.calls(), circuit.operations().size());
        EXPECT_EQ(monitor.last(), circuit.operations().size());
    }
}

TEST(ExecutionMonitorTest, StabilizerCallsTheMonitorOncePerOperation) {
    const engines::stabilizer::StabilizerEngine engine;
    const Circuit circuit = clifford_circuit(6);
    const CountingMonitor monitor;
    static_cast<void>(engine.execute(make_request(circuit, false, false, nullptr, &monitor)));
    EXPECT_EQ(monitor.calls(), circuit.operations().size());
}

TEST(ExecutionMonitorTest, MpsCallsTheMonitorOncePerOperation) {
    const engines::mps::MPSEngine engine;
    const Circuit circuit = non_clifford_circuit(6);
    const CountingMonitor monitor;
    static_cast<void>(engine.execute(make_request(circuit, false, false, nullptr, &monitor)));
    EXPECT_EQ(monitor.calls(), circuit.operations().size());
}

// Pauli propagation walks the operations backwards, once per observable term,
// and the counter spans the whole execute() call rather than restarting per
// term -- so a one-term observable sees exactly one boundary per operation.
TEST(ExecutionMonitorTest, PauliCallsTheMonitorOncePerOperationPerTerm) {
    const engines::pauli::PauliPropagationEngine engine;
    const Circuit circuit = pauli_circuit(6);
    const Observable observable = single_z();
    const CountingMonitor monitor;
    static_cast<void>(engine.execute(make_request(circuit, false, true, &observable, &monitor)));
    EXPECT_EQ(monitor.calls(), circuit.operations().size());
}

// The tensor-network engine is the documented exception: its execute() is a
// contraction per observable term, not an operation sweep, so its only boundary
// is between terms. Asserted explicitly so the limitation is visible in the
// suite rather than only in a comment.
TEST(ExecutionMonitorTest, TensorNetworkCallsTheMonitorOncePerObservableTerm) {
    const engines::tensor_network::TensorNetworkEngine engine;
    const Circuit circuit = non_clifford_circuit(4);
    const Observable observable = single_z();
    const CountingMonitor monitor;
    static_cast<void>(engine.execute(make_request(circuit, false, true, &observable, &monitor)));
    EXPECT_EQ(monitor.calls(), observable.terms.size());
    EXPECT_LT(monitor.calls(), circuit.operations().size())
        << "the tensor-network engine has no per-operation boundary";
}

// A null monitor must change nothing at all. This is what keeps every
// pre-Phase-11 caller byte-for-byte unaffected.
TEST(ExecutionMonitorTest, ANullMonitorProducesAnIdenticalStatevector) {
    const engines::dense::DenseEngine engine;
    const Circuit circuit = non_clifford_circuit(6);

    const EngineExecutionResult without =
        engine.execute(make_request(circuit, true, false, nullptr, nullptr));
    const CountingMonitor monitor;
    const EngineExecutionResult with =
        engine.execute(make_request(circuit, true, false, nullptr, &monitor));

    // Comparing the optionals themselves, not their contents: this asserts
    // both that each is engaged and that the amplitudes are equal, in one
    // step. Bitwise, not within a tolerance -- the monitor must not perturb
    // arithmetic at all.
    ASSERT_TRUE(without.statevector.has_value());
    EXPECT_EQ(without.statevector, with.statevector);
    EXPECT_GT(monitor.calls(), 0U);
}

// Every engine must let the abandon signal escape execute(). src/ contains nine
// pre-existing catch sites and every one catches common::ResourceExceeded
// specifically; a signal type caught by one of those would be silently
// discarded somewhere between the engine and the replan loop.
TEST(ExecutionMonitorTest, AThrowingMonitorAbandonsEveryEngine) {
    const Circuit non_clifford = non_clifford_circuit(6);
    const Circuit clifford = clifford_circuit(6);
    const Circuit pauli_eligible = pauli_circuit(6);
    const Observable observable = single_z();

    const engines::dense::DenseEngine dense;
    EXPECT_THROW(
        {
            const CountingMonitor monitor(3);
            static_cast<void>(
                dense.execute(make_request(non_clifford, true, false, nullptr, &monitor)));
        },
        CountingMonitor::Abandoned);

    const engines::stabilizer::StabilizerEngine stabilizer;
    EXPECT_THROW(
        {
            const CountingMonitor monitor(3);
            static_cast<void>(
                stabilizer.execute(make_request(clifford, false, false, nullptr, &monitor)));
        },
        CountingMonitor::Abandoned);

    const engines::mps::MPSEngine mps;
    EXPECT_THROW(
        {
            const CountingMonitor monitor(3);
            static_cast<void>(
                mps.execute(make_request(non_clifford, false, false, nullptr, &monitor)));
        },
        CountingMonitor::Abandoned);

    const engines::pauli::PauliPropagationEngine pauli;
    EXPECT_THROW(
        {
            const CountingMonitor monitor(3);
            static_cast<void>(
                pauli.execute(make_request(pauli_eligible, false, true, &observable, &monitor)));
        },
        CountingMonitor::Abandoned);

    const engines::tensor_network::TensorNetworkEngine tensor_network;
    EXPECT_THROW(
        {
            const CountingMonitor monitor(1);
            static_cast<void>(tensor_network.execute(
                make_request(non_clifford_circuit(4), false, true, &observable, &monitor)));
        },
        CountingMonitor::Abandoned);
}

// Abandoning must not leak. Run under AddressSanitizer (the dev preset) this
// exercises unwinding out of the middle of each engine's own allocation.
TEST(ExecutionMonitorTest, AbandoningReleasesTheEnginesAllocation) {
    const engines::dense::DenseEngine engine;
    const Circuit circuit = non_clifford_circuit(12);
    for (int repeat = 0; repeat < 50; ++repeat) {
        const CountingMonitor monitor(2);
        EXPECT_THROW(static_cast<void>(
                         engine.execute(make_request(circuit, true, false, nullptr, &monitor))),
                     CountingMonitor::Abandoned);
    }
}

}  // namespace
}  // namespace menos
