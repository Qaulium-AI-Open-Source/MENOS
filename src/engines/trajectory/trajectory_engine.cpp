#include "engines/trajectory/trajectory_engine.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "common/checked_math.h"
#include "common/error.h"
#include "common/rng.h"
#include "common/splitmix64.h"
#include "engines/dense/apply_matrix.h"
#include "engines/dense/cumulative_distribution.h"
#include "engines/dense/state_vector.h"
#include "engines/noise/readout.h"
#include "ir/gate_table.h"
#include "kernels/dense_kernels.h"

namespace menos::engines::trajectory {
namespace {

using common::checked_add;
using common::checked_mul;
using common::checked_pow2;
using common::ResourceExceeded;
using Complex = std::complex<double>;
using Matrix = std::vector<Complex>;  // row-major, dim x dim

// SplitMix64's state advances by this constant per output
// (src/common/splitmix64.h), which is what lets trajectory_seed() jump
// straight to output t + 1.
constexpr std::uint64_t kSplitMix64Gamma = 0x9E3779B97F4A7C15ULL;

std::size_t operand_count(const Operation& op) {
    return std::visit(
        [](const auto& o) -> std::size_t {
            using T = std::decay_t<decltype(o)>;
            if constexpr (std::is_same_v<T, MeasureOp>) {
                return 0;
            } else {
                return o.qubits.size();
            }
        },
        op);
}

// M_i = K_i^dagger K_i for each Kraus operator of a k-qubit channel, so that
// p_i = Re Tr(M_i rho_q) (docs/adr/0017 section 2). Computed once per channel
// per execute(), never per trajectory.
std::vector<Matrix> kraus_products(const ChannelOp& channel) {
    const std::size_t dim = std::size_t{1} << channel.qubits.size();
    std::vector<Matrix> products;
    products.reserve(channel.kraus.size());
    for (const Matrix& kraus : channel.kraus) {
        Matrix m(dim * dim, Complex{0.0, 0.0});
        for (std::size_t a = 0; a < dim; ++a) {
            for (std::size_t b = 0; b < dim; ++b) {
                Complex sum{0.0, 0.0};
                for (std::size_t r = 0; r < dim; ++r) {
                    sum += std::conj(kraus[r * dim + a]) * kraus[r * dim + b];
                }
                m[a * dim + b] = sum;
            }
        }
        products.push_back(std::move(m));
    }
    return products;
}

// rho_q[r][c] = sum over the other qubits of psi[r] * conj(psi[c]), the
// reduced density matrix of psi on `qubits`, in one pass over the 2^n
// amplitudes. Local index bit j is qubit qubits[j], the same operand-order
// rule dense::apply_matrix uses, so a Kraus operator and rho_q index alike.
Matrix reduced_density_matrix(std::span<const Complex> psi, std::span<const QubitId> qubits) {
    const std::size_t k = qubits.size();
    const std::size_t dim = std::size_t{1} << k;
    std::vector<std::uint64_t> offsets(dim, 0);
    std::uint64_t mask = 0;
    for (std::size_t local = 0; local < dim; ++local) {
        for (std::size_t j = 0; j < k; ++j) {
            if (((local >> j) & 1U) != 0) {
                offsets[local] |= std::uint64_t{1} << qubits[j];
            }
        }
    }
    for (const QubitId q : qubits) {
        mask |= std::uint64_t{1} << q;
    }
    Matrix rho(dim * dim, Complex{0.0, 0.0});
    std::vector<Complex> slice(dim);
    const std::uint64_t size = psi.size();
    for (std::uint64_t base = 0; base < size; ++base) {
        if ((base & mask) != 0) {
            continue;
        }
        for (std::size_t a = 0; a < dim; ++a) {
            slice[a] = psi[base | offsets[a]];
        }
        for (std::size_t r = 0; r < dim; ++r) {
            for (std::size_t c = 0; c < dim; ++c) {
                rho[r * dim + c] += slice[r] * std::conj(slice[c]);
            }
        }
    }
    return rho;
}

// Picks a Kraus branch with probability p_i = Re Tr(M_i rho_q): the first i
// whose cumulative probability exceeds u. If rounding leaves u above the total,
// the last branch with p_i > 0 is taken (docs/adr/0017 section 2).
// dim is 2^k for the channel's k qubits.
std::size_t choose_branch(std::size_t dim,
                          const std::vector<Matrix>& products,
                          const Matrix& rho,
                          double u) {
    double cumulative = 0.0;
    std::optional<std::size_t> last_positive;
    for (std::size_t i = 0; i < products.size(); ++i) {
        double p = 0.0;
        for (std::size_t a = 0; a < dim; ++a) {
            for (std::size_t b = 0; b < dim; ++b) {
                // Tr(M rho) = sum_{a,b} M[a][b] rho[b][a].
                p += (products[i][a * dim + b] * rho[b * dim + a]).real();
            }
        }
        if (p > 0.0) {
            last_positive = i;
            cumulative += p;
            if (u < cumulative) {
                return i;
            }
        }
    }
    if (!last_positive.has_value()) {
        throw common::InternalError(
            "trajectory: every Kraus branch has probability 0 -- the channel is not trace "
            "preserving on this state");
    }
    return *last_positive;
}

// psi /= ||psi||, using the norm measured from psi itself rather than
// sqrt(p_i), so rounding cannot accumulate across channels.
void renormalize(std::span<Complex> psi) {
    double norm_squared = 0.0;
    for (const Complex& a : psi) {
        norm_squared += std::norm(a);
    }
    common::ensure(norm_squared > 0.0, "trajectory: a chosen Kraus branch annihilated the state");
    const double scale = 1.0 / std::sqrt(norm_squared);
    for (Complex& a : psi) {
        a *= scale;
    }
}

// The trajectory engine's per-operation boundary counter. The gap between two
// operations, and between two trajectories, is a safe boundary: psi is a
// complete normalized state there and nothing has been handed to the caller.
struct Boundaries {
    const ExecutionMonitor* monitor;
    std::uint64_t crossed = 0;

    void cross() {
        ++crossed;
        if (monitor != nullptr) {
            monitor->at_safe_boundary(crossed);
        }
    }
};

// Runs one trajectory over `operations` into psi, which the caller has reset.
// Draws exactly one uniform per channel, in operation order.
void run_trajectory(dense::StateVector& psi,
                    std::span<const Operation> operations,
                    const std::vector<std::vector<Matrix>>& products_by_op,
                    common::Rng& rng,
                    Boundaries& boundaries,
                    std::uint32_t max_threads) {
    for (std::size_t index = 0; index < operations.size(); ++index) {
        std::visit(
            [&](const auto& op) {
                using T = std::decay_t<decltype(op)>;
                if constexpr (std::is_same_v<T, GateOp>) {
                    dense::apply_matrix(psi.amplitudes(),
                                        ir::generate_matrix(op.opcode, op.params),
                                        op.qubits,
                                        max_threads);
                } else if constexpr (std::is_same_v<T, UnitaryOp>) {
                    dense::apply_matrix(psi.amplitudes(), op.matrix, op.qubits, max_threads);
                } else if constexpr (std::is_same_v<T, ChannelOp>) {
                    const Matrix rho = reduced_density_matrix(psi.amplitudes(), op.qubits);
                    const std::size_t branch = choose_branch(std::size_t{1} << op.qubits.size(),
                                                             products_by_op[index],
                                                             rho,
                                                             rng.next_uniform());
                    dense::apply_matrix(psi.amplitudes(), op.kraus[branch], op.qubits, max_threads);
                    renormalize(psi.amplitudes());
                } else {
                    static_assert(std::is_same_v<T, MeasureOp>);
                    // Outcomes are drawn from the final state.
                }
            },
            operations[index]);
        boundaries.cross();
    }
}

void reset(dense::StateVector& psi) {
    std::ranges::fill(psi.amplitudes(), Complex{0.0, 0.0});
    // Global phase cancels in every output this engine produces (probabilities
    // and <psi|P|psi>), so it is not applied.
    psi.amplitudes()[0] = Complex{1.0, 0.0};
}

// Welford's running mean and sum of squared deviations of the per-trajectory
// expectation X_t, for the standard error s / sqrt(N), with
// s^2 = sum |X_t - mean|^2 / (N - 1) (docs/adr/0017 section 3).
struct RunningMoments {
    std::uint64_t count = 0;
    Complex mean{0.0, 0.0};
    double sum_squared_deviation = 0.0;

    void add(Complex x) {
        ++count;
        const Complex delta = x - mean;
        mean += delta / static_cast<double>(count);
        sum_squared_deviation += (std::conj(delta) * (x - mean)).real();
    }

    [[nodiscard]] double standard_error() const {
        common::ensure(count >= 2, "trajectory: a standard error needs at least 2 trajectories");
        const auto n = static_cast<double>(count);
        return std::sqrt(sum_squared_deviation / (n - 1.0) / n);
    }
};

}  // namespace

std::uint64_t trajectory_seed(std::uint64_t base_seed, std::uint64_t t) {
    // Unsigned arithmetic wraps exactly as SplitMix64's own state does.
    common::SplitMix64 mixer(base_seed + t * kSplitMix64Gamma);
    return mixer.next();
}

std::string_view TrajectoryEngine::name() const {
    return "trajectory";
}

Engine TrajectoryEngine::engine_id() const {
    return Engine::Trajectory;
}

Backend TrajectoryEngine::backend() const {
    return Backend::Cpu;
}

EngineCapabilities TrajectoryEngine::capabilities() const {
    return EngineCapabilities{/*counts=*/true, /*statevector=*/false, /*expectation=*/true};
}

// Work units (docs/adr/0017 section 5), all with checked arithmetic, for an
// n-qubit circuit and N trajectories. Per trajectory: 2^n for the reset; each
// gate or unitary on k qubits at the dense engine's 2^(n+k); each channel on k
// qubits at 2^(n+k) for the reduced density matrix, plus one 2^(n+k)
// apply_matrix pass, plus 2^n to renormalize; and 2^n per observable term when
// an expectation is requested. Each of the first `shots` trajectories also
// builds its 2^n outcome distribution. Peak bytes: 16 * 2^n for psi, the
// 8 * 2^n distribution when counts are requested, and every channel's
// precomputed K^dagger K products (16 * 4^k each), which live for the whole
// run.
CostEstimate TrajectoryEngine::estimate(const EngineEstimateRequest& request) const {
    if (request.want_statevector) {
        return CostEstimate{/*feasible=*/false,
                            0,
                            0,
                            "the trajectory engine does not produce a statevector -- a noisy "
                            "job has no single final state",
                            {}};
    }
    if (request.trajectories == 0) {
        return CostEstimate{
            /*feasible=*/false, 0, 0, "the trajectory engine was given no trajectory count", {}};
    }
    try {
        const std::uint32_t n = request.num_qubits;
        const std::uint64_t states = checked_pow2(n);
        std::uint64_t per_trajectory = states;  // reset
        std::uint64_t product_bytes = 0;
        for (const Operation& op : *request.operations) {
            const std::size_t k = operand_count(op);
            if (k == 0) {
                continue;  // MeasureOp
            }
            const std::uint64_t pass = checked_pow2(n + static_cast<std::uint32_t>(k));
            per_trajectory = checked_add(per_trajectory, pass);
            if (const auto* channel = std::get_if<ChannelOp>(&op)) {
                per_trajectory = checked_add(per_trajectory, checked_add(pass, states));
                const std::uint64_t matrix_bytes =
                    checked_mul(checked_pow2(2 * static_cast<std::uint32_t>(k)), sizeof(Complex));
                product_bytes =
                    checked_add(product_bytes, checked_mul(channel->kraus.size(), matrix_bytes));
            }
        }
        if (request.want_expectation) {
            const auto num_terms = static_cast<std::uint64_t>(request.observable->terms.size());
            per_trajectory = checked_add(per_trajectory, checked_mul(num_terms, states));
        }
        std::uint64_t work = checked_mul(request.trajectories, per_trajectory);
        std::uint64_t peak_bytes = checked_add(checked_mul(states, sizeof(Complex)), product_bytes);
        if (request.want_counts) {
            // One distribution per shot-producing trajectory; shots <= N.
            work = checked_add(work, checked_mul(request.trajectories, states));
            peak_bytes = checked_add(peak_bytes, checked_mul(states, sizeof(double)));
        }
        return CostEstimate{/*feasible=*/true, peak_bytes, work, /*infeasible_reason=*/"", {}};
    } catch (const ResourceExceeded& e) {
        return CostEstimate{/*feasible=*/false, 0, 0, e.what(), {}};
    }
}

EngineExecutionResult TrajectoryEngine::execute(const EngineExecutionRequest& request) const {
    common::ensure(!request.want_statevector,
                   "trajectory: a statevector was requested; estimate() refuses that");
    common::ensure(request.is_sole_region,
                   "trajectory: the engine runs only as a sole region (docs/adr/0017 section 5)");
    const std::uint64_t num_trajectories = request.trajectories;
    common::ensure(num_trajectories >= 1, "trajectory: no trajectory count was given");
    common::ensure(!request.want_counts || request.shots <= num_trajectories,
                   "trajectory: more shots than trajectories");
    common::ensure(!request.want_expectation || num_trajectories >= 2,
                   "trajectory: an expectation needs at least 2 trajectories");

    const std::span<const Operation> operations = request.operations;
    std::vector<std::vector<Matrix>> products_by_op(operations.size());
    std::uint64_t product_bytes = 0;
    for (std::size_t i = 0; i < operations.size(); ++i) {
        if (const auto* channel = std::get_if<ChannelOp>(&operations[i])) {
            products_by_op[i] = kraus_products(*channel);
            for (const Matrix& m : products_by_op[i]) {
                product_bytes = checked_add(product_bytes, m.size() * sizeof(Complex));
            }
        }
    }

    dense::StateVector psi(static_cast<int>(request.num_qubits), request.budget_bytes);
    const std::vector<noise::ReadoutTarget> readouts =
        noise::readout_targets(operations, request.measure_targets);

    std::vector<Complex> term_sums;
    RunningMoments moments;
    if (request.want_expectation) {
        term_sums.assign(request.observable->terms.size(), Complex{0.0, 0.0});
    }
    std::map<BitString, std::uint64_t> counts;

    Boundaries boundaries{request.monitor};
    for (std::uint64_t t = 0; t < num_trajectories; ++t) {
        common::Rng rng(trajectory_seed(request.seed, t));
        reset(psi);
        run_trajectory(psi, operations, products_by_op, rng, boundaries, request.max_threads);

        if (request.want_expectation) {
            Complex x{0.0, 0.0};
            for (std::size_t i = 0; i < term_sums.size(); ++i) {
                const auto& [pauli, coefficient] = request.observable->terms[i];
                // One word per mask: the state fits in memory, so n is far
                // below 64.
                const Complex value = kernels::cpu::expectation_of_term(
                    psi.amplitudes(), pauli.x_words[0], pauli.z_words[0]);
                term_sums[i] += value;
                x += coefficient * value;
            }
            moments.add(x);
        }
        if (request.want_counts && t < request.shots) {
            // After the channel draws: one word for the shot, then one uniform
            // per readout-errored target, in measure_targets order.
            // One cumulative pass, with no per-shot table (docs/adr/0025).
            BitString outcome = dense::CumulativeDistribution::sample_once(
                psi.amplitudes(), request.measure_targets, request.bitstring_width, rng);
            noise::apply_readout(outcome, readouts, rng);
            counts[outcome] += 1;
        }
        boundaries.cross();
    }

    EngineExecutionResult result;
    result.statistical = true;
    result.actual_bytes = checked_add(psi.size_bytes(), product_bytes);
    if (request.want_counts) {
        result.counts = std::move(counts);
    }
    if (request.want_expectation) {
        const auto n = static_cast<double>(num_trajectories);
        for (Complex& sum : term_sums) {
            sum /= n;
        }
        result.expectation_terms = std::move(term_sums);
        result.expectation_standard_error = moments.standard_error();
    }
    return result;
}

}  // namespace menos::engines::trajectory
