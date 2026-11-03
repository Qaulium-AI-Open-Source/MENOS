#pragma once

#include <array>
#include <complex>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace menos {

using QubitId = std::uint32_t;
using ClbitId = std::uint32_t;

// Closed set for this phase (docs/BUILD_PLAN.md Phase 1). Unitary and
// Measure are members of this same enum rather than a separate mechanism.
enum class Opcode {
    X,
    Y,
    Z,
    H,
    S,
    Sdg,
    T,
    Tdg,
    SX,
    SXdg,
    RX,
    RY,
    RZ,
    P,
    U,
    CX,
    CZ,
    SWAP,
    CP,
    RZZ,
    CCX,
    Unitary,
    Measure,
};

// A 2^5 x 2^5 complex128 matrix is 16 KiB, small enough to copy and
// validate cheaply. Larger operands should be decomposed into named gates
// rather than carried as one opaque block.
inline constexpr int kMaxUnitaryQubits = 5;

// A named-gate operation: bit j of the gate matrix row/column index
// corresponds to operand qubits[j] (CLAUDE.md's operand-order rule).
struct GateOp {
    Opcode opcode;
    std::vector<QubitId> qubits;
    std::vector<double> params;
};

// An explicit 2^k x 2^k row-major unitary, k == qubits.size().
struct UnitaryOp {
    std::vector<QubitId> qubits;
    std::vector<std::complex<double>> matrix;
};

// A classical readout error on one measurement: the probability of recording
// each outcome given the qubit's true outcome, row-major by true outcome:
// assignment = {P(0|0), P(1|0), P(0|1), P(1|1)}. Each row sums to 1. It
// changes recorded counts only, never an expectation value (docs/adr/0016
// section 3.2).
struct ReadoutError {
    std::array<double, 4> assignment;
};

// Writes the outcome of qubit to classical bit clbit, recorded through
// readout_error when one is set.
struct MeasureOp {
    QubitId qubit;
    ClbitId clbit;
    std::optional<ReadoutError> readout_error = std::nullopt;
};

// A completely positive, trace-preserving noise channel given by its Kraus
// operators: rho -> sum_i K_i rho K_i^dagger. Each K_i is a 2^k x 2^k
// row-major matrix, k == qubits.size(), with the same operand-order rule as
// UnitaryOp: bit j of the row/column index corresponds to operand qubits[j]
// (docs/adr/0016 section 3.1). Only an engine that simulates mixed states can
// run it; every other engine reports a job containing one as infeasible.
struct ChannelOp {
    std::vector<QubitId> qubits;
    std::vector<std::vector<std::complex<double>>> kraus;
};

using Operation = std::variant<GateOp, UnitaryOp, MeasureOp, ChannelOp>;

// A validated-on-construction-boundary circuit representation (KIR).
// Preconditions/exceptions: see menos::ir::validate() (invoked by run()),
// not by the constructor or append methods themselves -- appending never
// throws for structural reasons; validation happens once, at the runtime
// entry point.
class Circuit {
public:
    // Precondition: num_qubits >= 1. Not checked here; checked by
    // menos::ir::validate() at the runtime entry point.
    Circuit(std::uint32_t num_qubits, std::uint32_t num_clbits);

    void append(Opcode opcode, std::vector<QubitId> qubits, std::vector<double> params = {});
    void append_unitary(std::vector<QubitId> qubits, std::vector<std::complex<double>> matrix);
    void append_measure(QubitId qubit,
                        ClbitId clbit,
                        std::optional<ReadoutError> readout_error = std::nullopt);
    // Appends a ChannelOp. Validated, like every operation, at the runtime
    // entry point (menos::ir::validate): trace preservation within
    // common::kDefaultAbsTol, matrix shapes, finite entries, at most
    // kMaxUnitaryQubits operands.
    void append_channel(std::vector<QubitId> qubits,
                        std::vector<std::vector<std::complex<double>>> kraus);

    // CSR-style batched ingestion of named-gate operations (GateOp only --
    // append_unitary/append_measure stay individual calls; they're rare,
    // O(num_qubits) per circuit, not O(num_operations)). Operation i's
    // qubits are qubits[qubit_offsets[i] .. qubit_offsets[i+1]), its
    // params are params[param_offsets[i] .. param_offsets[i+1]).
    // Equivalent, operation-for-operation and in the same order, to
    // calling append(opcodes[i], ...) for i in [0, opcodes.size()).
    // Exists to collapse O(num_operations) call-boundary crossings (e.g.
    // from the Python binding) into one call; see docs/adr/0004.
    //
    // Precondition, checked via common::ensure -> InternalError (a
    // contract violation of this entry point's own shape, not user
    // circuit content -- ordinary per-operation validation still happens
    // only at menos::ir::validate() time, exactly like append()):
    // qubit_offsets.size() == opcodes.size() + 1, strictly non-decreasing,
    // qubit_offsets.back() == qubits.size(); the same shape holds for
    // param_offsets/params.
    void append_batch(std::vector<Opcode> opcodes,
                      std::vector<QubitId> qubits,
                      std::vector<std::uint32_t> qubit_offsets,
                      std::vector<double> params,
                      std::vector<std::uint32_t> param_offsets);

    [[nodiscard]] std::uint32_t num_qubits() const;
    [[nodiscard]] std::uint32_t num_clbits() const;
    [[nodiscard]] const std::vector<Operation>& operations() const;

    // A circuit carries global_phase; exp(i * global_phase) multiplies the
    // initial |0...0> state (CLAUDE.md's global-phase domain invariant).
    double global_phase = 0.0;

private:
    std::uint32_t num_qubits_;
    std::uint32_t num_clbits_;
    std::vector<Operation> operations_;
};

}  // namespace menos
