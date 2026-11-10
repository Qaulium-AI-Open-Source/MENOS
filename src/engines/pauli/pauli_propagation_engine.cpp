#include "engines/pauli/pauli_propagation_engine.h"

#include <cmath>
#include <complex>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "common/checked_math.h"
#include "common/error.h"
#include "common/memory_bounds.h"
#include "common/memory_ledger.h"
#include "common/pauli_row.h"
#include "ir/gate_table.h"
#include "ir/noise.h"

namespace menos::engines::pauli {
namespace {

using common::checked_add;
using common::checked_mul;
using common::ResourceExceeded;

// One propagated term: a Pauli row (docs/adr/0006 convention, x/z bits +
// sign r) together with the complex coefficient accumulated so far --
// the original observable term's own coefficient is folded in only once,
// after propagation finishes (see propagate_term below).
struct WeightedTerm {
    common::PauliRow row;
    std::complex<double> weight;
};

// docs/adr/0007: backward propagation through a gate G replaces the
// tracked term with G^dagger * P * G (Heisenberg picture), the opposite
// direction from Tableau's forward G * P * G^dagger state evolution.
// Every self-adjoint gate (X Y Z H CX CZ SWAP) is unaffected by this
// direction flip. S/Sdg and SX/SXdg are adjoints of each other, not of
// themselves, so backward through a circuit S gate calls apply_sdg (and
// vice versa): apply_sdg already computes Sdg*P*Sdg^dagger = S^dagger *
// P * (S^dagger)^dagger = S^dagger * P * S, which is exactly the
// backward rule for a circuit S gate. Verified directly against
// docs/adr/0006's ten gate rules, not re-derived here.
void apply_clifford_backward(common::PauliRow& row, const GateOp& gate) {
    switch (gate.opcode) {
        case Opcode::X:
            common::apply_x(row, gate.qubits[0]);
            return;
        case Opcode::Y:
            common::apply_y(row, gate.qubits[0]);
            return;
        case Opcode::Z:
            common::apply_z(row, gate.qubits[0]);
            return;
        case Opcode::H:
            common::apply_h(row, gate.qubits[0]);
            return;
        case Opcode::S:
            common::apply_sdg(row, gate.qubits[0]);
            return;
        case Opcode::Sdg:
            common::apply_s(row, gate.qubits[0]);
            return;
        case Opcode::SX:
            common::apply_sxdg(row, gate.qubits[0]);
            return;
        case Opcode::SXdg:
            common::apply_sx(row, gate.qubits[0]);
            return;
        case Opcode::CX:
            common::apply_cx(row, gate.qubits[0], gate.qubits[1]);
            return;
        case Opcode::CZ:
            common::apply_cz(row, gate.qubits[0], gate.qubits[1]);
            return;
        case Opcode::SWAP:
            common::apply_swap(row, gate.qubits[0], gate.qubits[1]);
            return;
        default:
            throw common::InternalError(
                "pauli propagation apply_clifford_backward: opcode is not one of the ten "
                "supported Clifford gates");
    }
}

// RZ(theta)^dagger * P * RZ(theta) and P(lambda)^dagger * P' * P(lambda),
// derived directly from src/ir/gate_table.cpp's matrix_rz/matrix_p (P
// and RZ differ only by a global phase that cancels under conjugation,
// so one formula serves both -- docs/adr/0007). Z (and I) commute with
// RZ/P and are returned unchanged (one term, no split); X and Y mix into
// each other (two terms).
// qubit (an index) and theta (an angle) are not realistically confusable
// at any call site.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
std::vector<WeightedTerm> split_rz_like(const WeightedTerm& term,
                                        std::uint32_t qubit,
                                        double theta) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    const bool x_bit = common::get_bit(term.row.x, qubit);
    const bool z_bit = common::get_bit(term.row.z, qubit);
    if (!x_bit) {
        return {term};
    }
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    WeightedTerm x_part = term;
    common::set_bit(x_part.row.x, qubit, true);
    common::set_bit(x_part.row.z, qubit, false);
    WeightedTerm y_part = term;
    common::set_bit(y_part.row.x, qubit, true);
    common::set_bit(y_part.row.z, qubit, true);
    if (!z_bit) {
        // input X: X -> cos(theta)*X - sin(theta)*Y
        x_part.weight *= c;
        y_part.weight *= -s;
    } else {
        // input Y: Y -> sin(theta)*X + cos(theta)*Y
        x_part.weight *= s;
        y_part.weight *= c;
    }
    return {std::move(x_part), std::move(y_part)};
}

// RX(theta)^dagger * P * RX(theta), derived directly from matrix_rx
// (docs/adr/0007). X (and I) are returned unchanged; Y and Z mix.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
std::vector<WeightedTerm> split_rx(const WeightedTerm& term, std::uint32_t qubit, double theta) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    const bool x_bit = common::get_bit(term.row.x, qubit);
    const bool z_bit = common::get_bit(term.row.z, qubit);
    if (!z_bit) {
        return {term};  // I (!x,!z) or X (x,!z): both unaffected by RX
    }
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    WeightedTerm y_part = term;
    common::set_bit(y_part.row.x, qubit, true);
    common::set_bit(y_part.row.z, qubit, true);
    WeightedTerm z_part = term;
    common::set_bit(z_part.row.x, qubit, false);
    common::set_bit(z_part.row.z, qubit, true);
    if (!x_bit) {
        // input Z: Z -> sin(theta)*Y + cos(theta)*Z
        y_part.weight *= s;
        z_part.weight *= c;
    } else {
        // input Y: Y -> cos(theta)*Y - sin(theta)*Z
        y_part.weight *= c;
        z_part.weight *= -s;
    }
    return {std::move(y_part), std::move(z_part)};
}

// RY(theta)^dagger * P * RY(theta), derived directly from matrix_ry
// (docs/adr/0007). Y (and I) are returned unchanged; X and Z mix.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
std::vector<WeightedTerm> split_ry(const WeightedTerm& term, std::uint32_t qubit, double theta) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    const bool x_bit = common::get_bit(term.row.x, qubit);
    const bool z_bit = common::get_bit(term.row.z, qubit);
    if (x_bit == z_bit) {
        return {term};  // I (!x,!z) or Y (x,z): both unaffected by RY
    }
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    WeightedTerm x_part = term;
    common::set_bit(x_part.row.x, qubit, true);
    common::set_bit(x_part.row.z, qubit, false);
    WeightedTerm z_part = term;
    common::set_bit(z_part.row.x, qubit, false);
    common::set_bit(z_part.row.z, qubit, true);
    if (x_bit) {
        // input X: X -> cos(theta)*X + sin(theta)*Z
        x_part.weight *= c;
        z_part.weight *= s;
    } else {
        // input Z: Z -> -sin(theta)*X + cos(theta)*Z
        x_part.weight *= -s;
        z_part.weight *= c;
    }
    return {std::move(x_part), std::move(z_part)};
}

// U(theta,phi,lambda)^dagger * P * U(theta,phi,lambda), derived directly
// from matrix_u's four entries by explicit matrix multiplication and
// cross-checked against U(theta,0,0) == RY(theta) and U(0,phi,lambda) ~
// RZ(phi+lambda) (docs/adr/0007). Unlike RZ/RX/RY/P, no input Pauli is
// fixed except I: a general U can send X, Y or Z to a combination of all
// three output axes, so this returns up to 3 terms (never 2, matching
// the corrected term-growth bound -- see estimate() below).
// theta/phi/lambda are always passed positionally from gate->params[0/1/2]
// at the one call site below, in that fixed order -- not realistically
// swappable in practice, and a swap would be caught immediately by the
// direct-matrix-conjugation and differential tests.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
std::vector<WeightedTerm> split_u(
    const WeightedTerm& term, std::uint32_t qubit, double theta, double phi, double lambda) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    const bool x_bit = common::get_bit(term.row.x, qubit);
    const bool z_bit = common::get_bit(term.row.z, qubit);
    if (!x_bit && !z_bit) {
        return {term};  // I is unaffected by any conjugation
    }
    const double c_t = std::cos(theta);
    const double s_t = std::sin(theta);
    const double c_p = std::cos(phi);
    const double s_p = std::sin(phi);
    const double c_l = std::cos(lambda);
    const double s_l = std::sin(lambda);

    double a = 0.0;   // output X coefficient
    double b = 0.0;   // output Y coefficient
    double cz = 0.0;  // output Z coefficient
    if (x_bit && !z_bit) {
        // input X
        a = c_t * c_p * c_l - s_p * s_l;
        b = -s_p * c_l - c_t * c_p * s_l;
        cz = s_t * c_p;
    } else if (!x_bit && z_bit) {
        // input Z
        a = -s_t * c_l;
        b = s_t * s_l;
        cz = c_t;
    } else {
        // input Y
        a = c_p * s_l + c_t * s_p * c_l;
        b = c_p * c_l - c_t * s_p * s_l;
        cz = s_t * s_p;
    }

    std::vector<WeightedTerm> result;
    result.reserve(3);
    if (a != 0.0) {
        WeightedTerm t = term;
        common::set_bit(t.row.x, qubit, true);
        common::set_bit(t.row.z, qubit, false);
        t.weight *= a;
        result.push_back(std::move(t));
    }
    if (b != 0.0) {
        WeightedTerm t = term;
        common::set_bit(t.row.x, qubit, true);
        common::set_bit(t.row.z, qubit, true);
        t.weight *= b;
        result.push_back(std::move(t));
    }
    if (cz != 0.0) {
        WeightedTerm t = term;
        common::set_bit(t.row.x, qubit, false);
        common::set_bit(t.row.z, qubit, true);
        t.weight *= cz;
        result.push_back(std::move(t));
    }
    return result;
}

// Applies split_one to every existing term and flattens the results --
// the shared shape behind every RZ/RX/RY/P/U step below.
template <typename SplitFn>
std::vector<WeightedTerm> propagate_split(const std::vector<WeightedTerm>& terms,
                                          SplitFn&& split_one) {
    std::vector<WeightedTerm> next;
    for (const WeightedTerm& t : terms) {
        for (WeightedTerm& s : split_one(t)) {
            next.push_back(std::move(s));
        }
    }
    return next;
}

// Non-Clifford, non-RZ/RX/RY/P/U operations (T, Tdg, CP, CCX, Unitary)
// are not supported this phase: T's branching factor under conjugation
// is not a finite fixed bound the way RZ/RX/RY/P/U's is (it is not even
// a rotation about a fixed real axis), and CP/CCX are multi-qubit
// non-Clifford, needing their own derivation not yet done. MeasureOp is
// always allowed -- it has no effect on the expectation of the
// pre-measurement state, exactly like it has none on Tableau/StateVector
// (docs/adr/0006, 0007).
//
// Exhaustive over Operation, like every dispatch in this file: a new
// alternative fails the static_assert until it is given its own rule, so it
// can never be silently treated as supported (docs/adr/0016 section 7).
std::optional<std::string> first_unsupported_reason(const std::vector<Operation>& operations) {
    for (std::size_t i = 0; i < operations.size(); ++i) {
        std::optional<std::string> reason = std::visit(
            [i](const auto& o) -> std::optional<std::string> {
                using T = std::decay_t<decltype(o)>;
                if constexpr (std::is_same_v<T, GateOp>) {
                    const ir::GateInfo& info = ir::gate_info(o.opcode);
                    const bool supported = info.is_clifford || o.opcode == Opcode::RZ ||
                                           o.opcode == Opcode::RX || o.opcode == Opcode::RY ||
                                           o.opcode == Opcode::P || o.opcode == Opcode::U;
                    if (!supported) {
                        return "operation " + std::to_string(i) + " (" + std::string(info.name) +
                               ") is not supported by Pauli propagation (Clifford gates and "
                               "RZ/RX/RY/P/U only)";
                    }
                    return std::nullopt;
                } else if constexpr (std::is_same_v<T, UnitaryOp>) {
                    return "operation " + std::to_string(i) +
                           " (unitary) is not supported by Pauli propagation";
                } else if constexpr (std::is_same_v<T, ChannelOp>) {
                    // estimate() already refused it through ir::first_noise_reason.
                    return "operation " + std::to_string(i) +
                           " (noise channel) is not supported by Pauli propagation";
                } else {
                    static_assert(std::is_same_v<T, MeasureOp>);
                    return std::nullopt;
                }
            },
            operations[i]);
        if (reason.has_value()) {
            return reason;
        }
    }
    return std::nullopt;
}

// The GateOp an operation holds, or nullptr for an operation propagation
// passes through: a MeasureOp has no effect on the pre-measurement state, and
// a UnitaryOp never reaches execute() because first_unsupported_reason makes
// estimate() infeasible for it.
const GateOp* gate_or_null(const Operation& op) {
    return std::visit(
        [](const auto& o) -> const GateOp* {
            using T = std::decay_t<decltype(o)>;
            if constexpr (std::is_same_v<T, GateOp>) {
                return &o;
            } else if constexpr (std::is_same_v<T, ChannelOp>) {
                // Unreachable: estimate() refuses a noisy region.
                throw common::InternalError(
                    "pauli propagation: a noise channel reached a pure-state engine");
            } else {
                static_assert(std::is_same_v<T, UnitaryOp> || std::is_same_v<T, MeasureOp>);
                return nullptr;
            }
        },
        op);
}

// True iff every GateOp in operations is a named Clifford gate (vacuously
// true for zero GateOps -- an idle/measure-only region is "fully
// Clifford" too). A fully Clifford region is not routed to Pauli
// propagation even when it is the cheaper of the two by work units: the
// owner's explicit routing rule (docs/BUILD_PLAN.md Phase 5's planner
// test requirement, "a Clifford region routes to DirectStabilizer not
// DirectPauli or dense") is a hard eligibility constraint, not a cost
// preference -- Pauli propagation's per-term cost genuinely undercuts
// the stabilizer tableau's per-term elimination cost for a small number
// of observable terms (no cross-term caching yet, docs/adr/0007), so
// leaving this to cost comparison alone would let Pauli win Clifford
// regions the spec reserves for the stabilizer engine.
bool is_fully_clifford(const std::vector<Operation>& operations) {
    for (const Operation& op : operations) {
        const GateOp* gate = gate_or_null(op);
        if (gate != nullptr && !ir::gate_info(gate->opcode).is_clifford) {
            return false;
        }
    }
    return true;
}

// Propagates one starting Pauli string backward through operations
// (reverse program order) and evaluates every resulting term against
// |0...0>: a term with any X-bit set is orthogonal to |0...0> and
// contributes 0; a pure Z/I term contributes its own sign (r).
// monitor/completed thread through so the boundary is per OPERATION rather
// than per observable term (Phase 11, docs/adr/0015). Between two operations
// the working term list is a complete backward-propagated decomposition of the
// original Pauli string, so it is safe to abandon there; `completed` is shared
// across every term of one execute() call so it counts that whole invocation.
std::complex<double> propagate_term(const PauliString& start,
                                    std::span<const Operation> operations,
                                    std::uint64_t& peak_terms_seen,
                                    const ExecutionMonitor* monitor,
                                    std::uint64_t& completed) {
    std::vector<WeightedTerm> terms;
    terms.push_back(
        WeightedTerm{common::PauliRow{start.x_words, start.z_words, false}, {1.0, 0.0}});
    // The term list and each term's x and z words, updated after every gate.
    const std::uint64_t term_words_bytes = 2 * start.x_words.size() * sizeof(std::uint64_t);
    common::MemoryLedger::Charge terms_charge;
    const auto charge_terms = [&] {
        terms_charge.set(common::buffer_bytes(terms) + terms.size() * term_words_bytes);
    };
    charge_terms();

    for (std::size_t i = operations.size(); i-- > 0;) {
        const GateOp* gate = gate_or_null(operations[i]);
        if (gate == nullptr) {
            // MeasureOp: no effect. UnitaryOp: rejected by estimate() already.
            // Still a boundary, so the count matches the operation count.
            ++completed;
            if (monitor != nullptr) {
                monitor->at_safe_boundary(completed);
            }
            continue;
        }
        switch (gate->opcode) {
            case Opcode::X:
            case Opcode::Y:
            case Opcode::Z:
            case Opcode::H:
            case Opcode::S:
            case Opcode::Sdg:
            case Opcode::SX:
            case Opcode::SXdg:
            case Opcode::CX:
            case Opcode::CZ:
            case Opcode::SWAP: {
                for (WeightedTerm& t : terms) {
                    apply_clifford_backward(t.row, *gate);
                }
                break;
            }
            case Opcode::RZ:
            case Opcode::P: {
                const double angle = gate->params[0];
                terms = propagate_split(terms, [&](const WeightedTerm& t) {
                    return split_rz_like(t, gate->qubits[0], angle);
                });
                break;
            }
            case Opcode::RX: {
                const double angle = gate->params[0];
                terms = propagate_split(terms, [&](const WeightedTerm& t) {
                    return split_rx(t, gate->qubits[0], angle);
                });
                break;
            }
            case Opcode::RY: {
                const double angle = gate->params[0];
                terms = propagate_split(terms, [&](const WeightedTerm& t) {
                    return split_ry(t, gate->qubits[0], angle);
                });
                break;
            }
            case Opcode::U: {
                const double theta = gate->params[0];
                const double phi = gate->params[1];
                const double lambda = gate->params[2];
                terms = propagate_split(terms, [&](const WeightedTerm& t) {
                    return split_u(t, gate->qubits[0], theta, phi, lambda);
                });
                break;
            }
            default:
                throw common::InternalError(
                    "pauli propagation propagate_term: opcode should have been rejected by "
                    "estimate()");
        }
        peak_terms_seen = std::max(peak_terms_seen, static_cast<std::uint64_t>(terms.size()));
        charge_terms();
        ++completed;
        if (monitor != nullptr) {
            monitor->at_safe_boundary(completed);
        }
    }

    std::complex<double> total{0.0, 0.0};
    for (const WeightedTerm& t : terms) {
        bool nonzero_x = false;
        for (std::uint64_t word : t.row.x) {
            if (word != 0) {
                nonzero_x = true;
                break;
            }
        }
        if (!nonzero_x) {
            total += t.weight * (t.row.r ? -1.0 : 1.0);
        }
    }
    return total;
}

// Everything execute() allocates, at the most it holds at once (Phase 21,
// docs/adr/0025 section 3.4), for a term list of at most max_terms terms of
// n qubits: one complex per observable term, plus one rotation's split, when
// the old list (its capacity at most twice its size, from push_back growth)
// and the new one (at its last reallocation) are both live, with the few
// terms a split builds and returns in flight. max_terms is the estimate's own
// term-growth bound, number of terms times the branching product, as before
// Phase 21 (docs/adr/0025 section 7: its overcount is left as it is).
// Term counts and a qubit count: distinct roles, named at the one call site.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::uint64_t peak_bytes_of(std::uint64_t max_terms,
                            std::uint32_t num_qubits,
                            std::uint64_t num_observable_terms) {
    const std::uint64_t words = (std::uint64_t{num_qubits} + 63) / 64;
    // A term's x and z words are two allocations.
    const std::uint64_t term_words =
        2 * common::heap_block_bytes(checked_mul(words, sizeof(std::uint64_t)));
    const std::uint64_t term = checked_add(sizeof(WeightedTerm), term_words);
    // A split's locals, its initializer list and the vector built from it.
    constexpr std::uint64_t kTermsInFlight = 6;
    std::uint64_t bytes = common::grown_vector_bytes(max_terms, sizeof(WeightedTerm));
    bytes = checked_add(bytes,
                        checked_add(checked_mul(2 * max_terms, sizeof(WeightedTerm)),
                                    common::kHeapBlockOverheadBytes));
    bytes = checked_add(bytes, checked_mul(2 * max_terms, term_words));
    // The terms in flight, and the vector a split returns them in.
    bytes = checked_add(
        bytes, checked_add(checked_mul(kTermsInFlight, term), common::kHeapBlockOverheadBytes));
    return checked_add(
        bytes,
        common::heap_block_bytes(checked_mul(num_observable_terms, sizeof(std::complex<double>))));
}

}  // namespace

std::string_view PauliPropagationEngine::name() const {
    return "pauli";
}

Engine PauliPropagationEngine::engine_id() const {
    return Engine::Pauli;
}

Backend PauliPropagationEngine::backend() const {
    return Backend::Cpu;
}

EngineCapabilities PauliPropagationEngine::capabilities() const {
    return EngineCapabilities{/*counts=*/false, /*statevector=*/false, /*expectation=*/true};
}

CostEstimate PauliPropagationEngine::estimate(const EngineEstimateRequest& request) const {
    if (request.want_statevector) {
        return CostEstimate{/*feasible=*/false,
                            0,
                            0,
                            "the Pauli propagation engine never tracks a statevector",
                            {}};
    }
    if (request.want_counts) {
        return CostEstimate{/*feasible=*/false,
                            0,
                            0,
                            "the Pauli propagation engine produces an expectation value only, "
                            "never counts",
                            {}};
    }
    if (!request.want_expectation) {
        return CostEstimate{/*feasible=*/false,
                            0,
                            0,
                            "the Pauli propagation engine has nothing to compute without an "
                            "expectation request",
                            {}};
    }
    if (const std::optional<std::string> reason = ir::first_noise_reason(*request.operations);
        reason.has_value()) {
        return CostEstimate{/*feasible=*/false, 0, 0, *reason, {}};
    }
    if (const std::optional<std::string> reason = first_unsupported_reason(*request.operations);
        reason.has_value()) {
        return CostEstimate{/*feasible=*/false, 0, 0, *reason, {}};
    }
    if (is_fully_clifford(*request.operations)) {
        return CostEstimate{/*feasible=*/false,
                            0,
                            0,
                            "the region is fully Clifford -- the stabilizer engine is used for "
                            "a Clifford region's expectation, not Pauli propagation",
                            {}};
    }

    try {
        // Corrected term-growth bound (docs/adr/0007): RZ/RX/RY/P each
        // branch a term into at most 2, U into at most 3 -- never a
        // uniform factor of 2 for every non-Clifford gate.
        std::uint64_t single_axis_count = 0;
        std::uint64_t u_count = 0;
        std::uint64_t num_gates = 0;
        for (const Operation& op : *request.operations) {
            if (const auto* gate = std::get_if<GateOp>(&op)) {
                ++num_gates;
                switch (gate->opcode) {
                    case Opcode::RZ:
                    case Opcode::RX:
                    case Opcode::RY:
                    case Opcode::P:
                        ++single_axis_count;
                        break;
                    case Opcode::U:
                        ++u_count;
                        break;
                    default:
                        break;
                }
            }
        }

        const std::uint64_t t0 = static_cast<std::uint64_t>(request.observable->terms.size());
        std::uint64_t branching = 1;
        for (std::uint64_t i = 0; i < single_axis_count; ++i) {
            branching = checked_mul(branching, 2);
        }
        for (std::uint64_t i = 0; i < u_count; ++i) {
            branching = checked_mul(branching, 3);
        }
        const std::uint64_t max_terms = checked_mul(t0, branching);

        const std::uint64_t peak_bytes = peak_bytes_of(max_terms, request.num_qubits, t0);
        const std::uint64_t work = checked_mul(max_terms, num_gates);
        CostEstimate estimate{/*feasible=*/true, peak_bytes, work, /*infeasible_reason=*/"", {}};
        // Time terms (docs/adr/0028 section 20). The worst-case branching
        // above is the right bound for memory and the wrong figure for time:
        // a term splits only when a rotation anticommutes with it, and on the
        // measured workloads the runtime stayed flat while the bound grew 4x
        // per rotation (section 16.10.1). The time model therefore charges
        // the observable's own terms through every gate, the cost of a run in
        // which no term branches: component 0 the gates; component 1 the term
        // updates, t0 per gate, each with a fixed cost; component 2 their
        // words, n/64 per term update. It is a lower bound on a run that
        // branches heavily; such a run overruns its prediction and is caught
        // by the Phase 11 monitor like any other. For the same reason the
        // bandwidth floor is taken from the term list a run allocates when
        // nothing branches, not from the worst-case peak above.
        const double words = static_cast<double>((std::uint64_t{request.num_qubits} + 63) / 64);
        const double term_updates = static_cast<double>(t0) * static_cast<double>(num_gates);
        auto& pauli = estimate.time.counts[static_cast<std::size_t>(Engine::Pauli)];
        pauli[0] = static_cast<double>(num_gates);
        pauli[1] = term_updates;
        pauli[2] = term_updates * words;
        estimate.time.floor_bytes = static_cast<double>(peak_bytes_of(t0, request.num_qubits, t0));
        return estimate;
    } catch (const ResourceExceeded& e) {
        return CostEstimate{/*feasible=*/false, 0, 0, e.what(), {}};
    }
}

EngineExecutionResult PauliPropagationEngine::execute(const EngineExecutionRequest& request) const {
    // Read in place: until Phase 21 every operation was copied first
    // (docs/adr/0025 section 3.3).
    const std::span<const Operation> operations = request.operations;

    std::vector<std::complex<double>> terms;
    terms.reserve(request.observable->terms.size());
    std::uint64_t peak_terms = 0;
    std::uint64_t completed = 0;
    for (const auto& term : request.observable->terms) {
        terms.push_back(
            propagate_term(term.first, operations, peak_terms, request.monitor, completed));
    }

    EngineExecutionResult result;
    result.expectation_terms = std::move(terms);
    const std::uint64_t words = (std::uint64_t{request.num_qubits} + 63) / 64;
    const std::uint64_t row_bytes = checked_add(
        checked_mul(std::uint64_t{2}, checked_mul(words, std::uint64_t{8})), std::uint64_t{16});
    result.actual_bytes = checked_mul(std::max<std::uint64_t>(peak_terms, 1), row_bytes);
    return result;
}

}  // namespace menos::engines::pauli
