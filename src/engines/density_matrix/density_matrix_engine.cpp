#include "engines/density_matrix/density_matrix_engine.h"

#include <algorithm>
#include <array>
#include <bit>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "common/checked_math.h"
#include "common/error.h"
#include "common/memory_bounds.h"
#include "common/rng.h"
#include "common/thread_count.h"
#include "engines/dense/apply_matrix.h"
#include "engines/dense/cumulative_distribution.h"
#include "engines/dense/sample.h"
#include "engines/dense/state_vector.h"
#include "engines/noise/readout.h"
#include "ir/gate_table.h"

namespace menos::engines::density_matrix {
namespace {

using common::checked_add;
using common::checked_mul;
using common::checked_pow2;
using common::ResourceExceeded;
using Complex = std::complex<double>;

// 2^exponent for an exponent computed in 64 bits, so 2n or 2n + 2k can never
// wrap before checked_pow2 rejects it as unrepresentable.
std::uint64_t pow2(std::uint64_t exponent) {
    if (exponent >= 64) {
        throw ResourceExceeded("density matrix: 2^" + std::to_string(exponent) +
                               " is not representable");
    }
    return checked_pow2(static_cast<unsigned>(exponent));
}

// The column qubits of rho for row qubits `qubits` of an n-qubit region.
std::vector<std::uint32_t> column_qubits(std::span<const QubitId> qubits, std::uint32_t n) {
    std::vector<std::uint32_t> columns(qubits.size());
    for (std::size_t i = 0; i < qubits.size(); ++i) {
        columns[i] = qubits[i] + n;
    }
    return columns;
}

std::vector<Complex> conjugated(std::span<const Complex> matrix) {
    std::vector<Complex> out(matrix.size());
    for (std::size_t i = 0; i < matrix.size(); ++i) {
        out[i] = std::conj(matrix[i]);
    }
    return out;
}

// rho -> U rho U^dagger: U on the row qubits, conj(U) on the column qubits.
void apply_unitary(dense::StateVector& rho,
                   std::span<const Complex> u,
                   std::span<const QubitId> qubits,
                   std::uint32_t n,
                   std::uint32_t max_threads) {
    dense::apply_matrix(rho.amplitudes(), u, qubits, max_threads);
    dense::apply_matrix(rho.amplitudes(), conjugated(u), column_qubits(qubits, n), max_threads);
}

// The superoperator S = sum_i K_i (x) conj(K_i) of a k-qubit channel, as a
// 4^k x 4^k row-major matrix whose index is r | (c << k) for a row part r and a
// column part c. With operands [q_0..q_{k-1}, q_0+n..q_{k-1}+n] that is exactly
// apply_matrix's operand-order rule: bit j < k is row qubit q_j, bit k + j is
// column qubit q_j + n.
std::vector<Complex> superoperator(const std::vector<std::vector<Complex>>& kraus, std::size_t k) {
    const std::size_t dim = std::size_t{1} << k;
    const std::size_t sdim = dim * dim;
    std::vector<Complex> s(sdim * sdim, Complex{0.0, 0.0});
    for (const std::vector<Complex>& op : kraus) {
        for (std::size_t r = 0; r < dim; ++r) {
            for (std::size_t c = 0; c < dim; ++c) {
                const std::size_t row = r | (c << k);
                for (std::size_t r2 = 0; r2 < dim; ++r2) {
                    for (std::size_t c2 = 0; c2 < dim; ++c2) {
                        s[row * sdim + (r2 | (c2 << k))] +=
                            op[r * dim + r2] * std::conj(op[c * dim + c2]);
                    }
                }
            }
        }
    }
    return s;
}

// n is the density matrix's qubit count and max_threads the team cap, passed
// together by the one call site.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
void apply_channel(dense::StateVector& rho,
                   const ChannelOp& channel,
                   std::uint32_t n,
                   std::uint32_t max_threads) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    std::vector<std::uint32_t> operands(channel.qubits.begin(), channel.qubits.end());
    const std::vector<std::uint32_t> columns = column_qubits(channel.qubits, n);
    operands.insert(operands.end(), columns.begin(), columns.end());
    dense::apply_matrix(rho.amplitudes(),
                        superoperator(channel.kraus, channel.qubits.size()),
                        operands,
                        max_threads);
}

// The gap between two operations is this engine's safe boundary (Phase 11,
// docs/adr/0015): rho is a complete density matrix there and nothing has been
// handed to the caller. Exhaustive over Operation.
void apply_operations(dense::StateVector& rho,
                      std::span<const Operation> operations,
                      std::uint32_t n,
                      const ExecutionMonitor* monitor,
                      std::uint32_t max_threads) {
    std::uint64_t completed = 0;
    for (const Operation& operation : operations) {
        std::visit(
            [&](const auto& op) {
                using T = std::decay_t<decltype(op)>;
                if constexpr (std::is_same_v<T, GateOp>) {
                    apply_unitary(
                        rho, ir::generate_matrix(op.opcode, op.params), op.qubits, n, max_threads);
                } else if constexpr (std::is_same_v<T, UnitaryOp>) {
                    apply_unitary(rho, op.matrix, op.qubits, n, max_threads);
                } else if constexpr (std::is_same_v<T, ChannelOp>) {
                    apply_channel(rho, op, n, max_threads);
                } else {
                    static_assert(std::is_same_v<T, MeasureOp>);
                    // Outcomes come from the diagonal after the sweep.
                }
            },
            operation);
        ++completed;
        if (monitor != nullptr) {
            monitor->at_safe_boundary(completed);
        }
    }
}

using noise::apply_readout;
using noise::readout_targets;
using noise::ReadoutTarget;

// A region's outcome distribution with its readout errors, for the Factorized
// counts path and for a sole region whose measurements carry readout errors.
// Each draw takes one word for the true outcome, then one per readout target.
class DensityMatrixCountsSampler : public RegionCountsSampler {
public:
    // Takes the probabilities over: the distribution accumulates them in place.
    DensityMatrixCountsSampler(std::vector<double>&& probabilities,
                               std::span<const std::pair<QubitId, ClbitId>> measure_targets,
                               std::uint32_t bitstring_width,
                               std::vector<ReadoutTarget> readouts)
        : distribution_(std::move(probabilities), measure_targets, bitstring_width),
          readouts_(std::move(readouts)) {}

    [[nodiscard]] BitString sample_one(common::Rng& rng) const override {
        BitString outcome = distribution_.sample_one(rng);
        apply_readout(outcome, readouts_, rng);
        return outcome;
    }

private:
    dense::CumulativeDistribution distribution_;
    std::vector<ReadoutTarget> readouts_;
};

// Tr(rho P) for the Pauli string with masks x, z, in the convention
// P = i^{|x & z|} X^x Z^z (docs/adr/0007): P|c> = i^{|x&z|} (-1)^{|c&z|} |c ^ x>,
// so Tr(rho P) = i^{|x&z|} sum_c (-1)^{|c&z|} rho[c, c ^ x].
Complex trace_with_pauli(std::span<const Complex> rho,
                         std::uint32_t n,
                         std::uint64_t x,
                         std::uint64_t z) {
    const std::uint64_t dim = std::uint64_t{1} << n;
    Complex sum{0.0, 0.0};
    for (std::uint64_t c = 0; c < dim; ++c) {
        const Complex entry = rho[c | ((c ^ x) << n)];
        sum += (std::popcount(c & z) % 2 == 0) ? entry : -entry;
    }
    constexpr std::array<Complex, 4> kPowersOfI = {
        Complex{1.0, 0.0}, Complex{0.0, 1.0}, Complex{-1.0, 0.0}, Complex{0.0, -1.0}};
    return kPowersOfI[static_cast<std::size_t>(std::popcount(x & z) % 4)] * sum;
}

// Everything execute() allocates, at the most it holds at once (Phase 21,
// docs/adr/0025 section 3.4): rho, plus the larger of one operation's
// transient and the outputs.
//   * A k-qubit gate or unitary: its generated matrix (a GateOp), the
//     conjugated copy, the column-qubit list and the kernel's scratch.
//   * A k-qubit channel: the 2k-operand list (grown by one insert), the
//     column-qubit list, the 4^k x 4^k superoperator and the kernel's scratch
//     for 2k operands.
//   * Counts: the diagonal (8 bytes per basis state), which the sampler takes
//     over, the readout targets, the sampler object, and a sole region's
//     sampling buffers and histogram. One complex per observable term.
std::uint64_t peak_bytes_of(const EngineEstimateRequest& request) {
    using common::heap_block_bytes;
    const std::uint64_t n = request.num_qubits;
    const std::uint64_t rho = heap_block_bytes(checked_mul(pow2(2 * n), sizeof(Complex)));
    const std::uint32_t threads = common::default_thread_count();

    std::uint64_t operation = 0;
    for (const Operation& op : *request.operations) {
        std::visit(
            [&](const auto& o) {
                using T = std::decay_t<decltype(o)>;
                if constexpr (std::is_same_v<T, GateOp> || std::is_same_v<T, UnitaryOp>) {
                    // The conjugated copy and column list, beside a GateOp's
                    // generated matrix.
                    const std::uint64_t k = o.qubits.size();
                    const std::uint64_t matrix =
                        heap_block_bytes(checked_mul(pow2(2 * k), sizeof(Complex)));
                    std::uint64_t bytes = checked_add(
                        checked_add(matrix,
                                    heap_block_bytes(checked_mul(k, sizeof(std::uint32_t)))),
                        dense::apply_matrix_scratch_bytes(k, threads));
                    if constexpr (std::is_same_v<T, GateOp>) {
                        bytes = checked_add(bytes, matrix);
                    }
                    operation = std::max(operation, bytes);
                } else if constexpr (std::is_same_v<T, ChannelOp>) {
                    const std::uint64_t k = o.qubits.size();
                    const std::uint64_t bytes = checked_add(
                        checked_add(common::grown_vector_bytes(2 * k, sizeof(std::uint32_t)),
                                    heap_block_bytes(checked_mul(k, sizeof(std::uint32_t)))),
                        checked_add(heap_block_bytes(checked_mul(pow2(4 * k), sizeof(Complex))),
                                    dense::apply_matrix_scratch_bytes(2 * k, threads)));
                    operation = std::max(operation, bytes);
                } else {
                    static_assert(std::is_same_v<T, MeasureOp>);
                }
            },
            op);
    }

    std::uint64_t outputs = 0;
    if (request.want_counts) {
        const std::uint64_t width = request.bitstring_width;
        // The diagonal is the sampler's cumulative array: the sampler takes it over.
        const std::uint64_t sampler =
            checked_add(heap_block_bytes(sizeof(DensityMatrixCountsSampler)),
                        dense::CumulativeDistribution::heap_bytes(pow2(n), width));
        outputs = checked_add(sampler, noise::readout_targets_bytes(width));
        if (request.is_sole_region) {
            outputs =
                checked_add(outputs,
                            dense::sample_bytes(request.shots,
                                                static_cast<std::uint32_t>(std::min(n, width)),
                                                request.bitstring_width));
        }
    }
    if (request.want_expectation) {
        outputs = checked_add(
            outputs,
            heap_block_bytes(checked_mul(
                static_cast<std::uint64_t>(request.observable->terms.size()), sizeof(Complex))));
    }
    return checked_add(rho, std::max(operation, outputs));
}

}  // namespace

std::string_view DensityMatrixEngine::name() const {
    return "density_matrix";
}

Engine DensityMatrixEngine::engine_id() const {
    return Engine::DensityMatrix;
}

Backend DensityMatrixEngine::backend() const {
    return Backend::Cpu;
}

EngineCapabilities DensityMatrixEngine::capabilities() const {
    return EngineCapabilities{/*counts=*/true, /*statevector=*/false, /*expectation=*/true};
}

// Work units (docs/adr/0016 section 5), all with checked arithmetic, for an
// n-qubit region: 4^n to initialize; per gate or unitary on k qubits,
// 2 * 2^(2n + k) for its two kernel passes; per channel on k qubits,
// 2^(2n + 2k) for one pass with 2k operands; 2^n for the diagonal pass when
// counts are requested; 2^n per observable term. Peak bytes: peak_bytes_of
// above.
CostEstimate DensityMatrixEngine::estimate(const EngineEstimateRequest& request) const {
    if (request.want_statevector) {
        return CostEstimate{/*feasible=*/false,
                            0,
                            0,
                            "the density-matrix engine does not produce a statevector -- a "
                            "mixed state has none",
                            {}};
    }
    try {
        const std::uint64_t n = request.num_qubits;
        const std::uint64_t entries = pow2(2 * n);
        std::uint64_t work = entries;
        for (const Operation& op : *request.operations) {
            std::visit(
                [&](const auto& o) {
                    using T = std::decay_t<decltype(o)>;
                    if constexpr (std::is_same_v<T, GateOp> || std::is_same_v<T, UnitaryOp>) {
                        const std::uint64_t k = o.qubits.size();
                        work = checked_add(work, checked_mul(std::uint64_t{2}, pow2(2 * n + k)));
                    } else if constexpr (std::is_same_v<T, ChannelOp>) {
                        const std::uint64_t k = o.qubits.size();
                        work = checked_add(work, pow2(2 * n + 2 * k));
                    } else {
                        static_assert(std::is_same_v<T, MeasureOp>);
                    }
                },
                op);
        }
        if (request.want_counts) {
            work = checked_add(work, pow2(n));
        }
        if (request.want_expectation) {
            const auto num_terms = static_cast<std::uint64_t>(request.observable->terms.size());
            work = checked_add(work, checked_mul(num_terms, pow2(n)));
        }
        return CostEstimate{
            /*feasible=*/true, peak_bytes_of(request), work, /*infeasible_reason=*/"", {}};
    } catch (const ResourceExceeded& e) {
        return CostEstimate{/*feasible=*/false, 0, 0, e.what(), {}};
    }
}

EngineExecutionResult DensityMatrixEngine::execute(const EngineExecutionRequest& request) const {
    common::ensure(!request.want_statevector,
                   "density matrix: a statevector was requested; estimate() refuses that");
    const std::uint32_t n = request.num_qubits;
    // Global phase cancels in rho = |psi><psi|, so request.global_phase is unused.
    dense::StateVector rho(static_cast<int>(2 * n), request.budget_bytes);
    rho.amplitudes()[0] = Complex{1.0, 0.0};  // |0...0><0...0|
    apply_operations(rho, request.operations, n, request.monitor, request.max_threads);

    EngineExecutionResult result;
    result.actual_bytes = rho.size_bytes();

    if (request.want_counts) {
        const std::uint64_t dim = std::uint64_t{1} << n;
        std::vector<double> probabilities(dim);
        for (std::uint64_t x = 0; x < dim; ++x) {
            probabilities[x] = rho.amplitudes()[x | (x << n)].real();
        }
        result.actual_bytes = checked_add(result.actual_bytes, dim * sizeof(double));

        const std::span<const std::pair<QubitId, ClbitId>> targets = request.measure_targets;
        std::vector<ReadoutTarget> readouts = readout_targets(request.operations, targets);
        if (request.is_sole_region && readouts.empty()) {
            result.counts = dense::sample(
                probabilities, request.shots, request.seed, targets, request.bitstring_width);
        } else if (request.is_sole_region) {
            // Per shot: one word for the true outcome, then one per readout target.
            const DensityMatrixCountsSampler sampler(
                std::move(probabilities), targets, request.bitstring_width, std::move(readouts));
            common::Rng rng(request.seed);
            std::map<BitString, std::uint64_t> counts;
            for (std::uint64_t shot = 0; shot < request.shots; ++shot) {
                counts[sampler.sample_one(rng)] += 1;
            }
            result.counts = std::move(counts);
        } else {
            result.counts_sampler = std::make_unique<DensityMatrixCountsSampler>(
                std::move(probabilities), targets, request.bitstring_width, std::move(readouts));
        }
    }

    if (request.want_expectation) {
        std::vector<Complex> terms;
        terms.reserve(request.observable->terms.size());
        for (const auto& term : request.observable->terms) {
            // One word per mask: estimate() limits n to well under 64 (4^n entries).
            terms.push_back(trace_with_pauli(
                rho.amplitudes(), n, term.first.x_words[0], term.first.z_words[0]));
        }
        result.expectation_terms = std::move(terms);
    }

    return result;
}

}  // namespace menos::engines::density_matrix
