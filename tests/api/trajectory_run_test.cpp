// End to end: trajectory simulation of noisy circuits (Phase 13,
// docs/adr/0017). The opt-ins are validated, a job that does not opt in keeps
// Phase 12's exact plan, and trajectory results are statistical, flagged, and
// agree with the exact density-matrix values within their reported error.
//
// Every statistical test uses fixed seeds, so its outcome is deterministic. The
// thresholds are the owner-approved ones (docs/adr/0017 section 6, Q4), and the
// quantiles they need are computed here from their definitions rather than
// hard-coded. A failure is investigated, never reseeded away (section 8).

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "api/replan_controller.h"
#include "common/error.h"
#include "menos/menos.h"
#include "plan/hardware_profile.h"
#include "plan/job_spec.h"
#include "plan/planner.h"

namespace menos {
namespace {

using Complex = std::complex<double>;
using Matrix = std::vector<Complex>;
using Kraus = std::vector<Matrix>;

// ---------------------------------------------------------------------------
// Channels
// ---------------------------------------------------------------------------

const std::array<Matrix, 4> kPaulis = {
    Matrix{1.0, 0.0, 0.0, 1.0},
    Matrix{0.0, 1.0, 1.0, 0.0},
    Matrix{0.0, Complex{0.0, -1.0}, Complex{0.0, 1.0}, 0.0},
    Matrix{1.0, 0.0, 0.0, -1.0},
};

// The Kronecker product for apply_matrix's operand order: local bit 0 is the
// first operand, so `low` acts on bit 0 and `high` on the bits above it.
Matrix kron(const Matrix& high, const Matrix& low) {
    const auto dim_low = static_cast<std::size_t>(std::sqrt(static_cast<double>(low.size())));
    const auto dim_high = static_cast<std::size_t>(std::sqrt(static_cast<double>(high.size())));
    const std::size_t dim = dim_low * dim_high;
    Matrix out(dim * dim);
    for (std::size_t r = 0; r < dim; ++r) {
        for (std::size_t c = 0; c < dim; ++c) {
            out[r * dim + c] = high[(r / dim_low) * dim_high + (c / dim_low)] *
                               low[(r % dim_low) * dim_low + (c % dim_low)];
        }
    }
    return out;
}

Matrix scaled(const Matrix& m, double s) {
    Matrix out = m;
    for (Complex& v : out) {
        v *= s;
    }
    return out;
}

// rho -> (1 - p) rho + p I/2.
Kraus depolarizing_1q(double p) {
    Kraus out{scaled(kPaulis[0], std::sqrt(1.0 - 3.0 * p / 4.0))};
    for (std::size_t i = 1; i < 4; ++i) {
        out.push_back(scaled(kPaulis[i], std::sqrt(p / 4.0)));
    }
    return out;
}

// rho -> (1 - p) rho + p I/4: 16 Pauli-product Kraus operators.
Kraus depolarizing_2q(double p) {
    Kraus out;
    for (std::size_t a = 0; a < 4; ++a) {
        for (std::size_t b = 0; b < 4; ++b) {
            const double w = (a == 0 && b == 0) ? 1.0 - 15.0 * p / 16.0 : p / 16.0;
            out.push_back(scaled(kron(kPaulis[a], kPaulis[b]), std::sqrt(w)));
        }
    }
    return out;
}

// A correlated three-qubit bit flip: XXX with probability p. Its Kraus
// operators are 8x8, the widest channel the fixtures use.
Kraus correlated_bit_flip_3q(double p) {
    const Matrix x3 = kron(kPaulis[1], kron(kPaulis[1], kPaulis[1]));
    const Matrix i3 = kron(kPaulis[0], kron(kPaulis[0], kPaulis[0]));
    return {scaled(i3, std::sqrt(1.0 - p)), scaled(x3, std::sqrt(p))};
}

// |1> decays to |0> with probability gamma: state-dependent branch weights.
Kraus amplitude_damping(double gamma) {
    return {{1.0, 0.0, 0.0, std::sqrt(1.0 - gamma)}, {0.0, std::sqrt(gamma), 0.0, 0.0}};
}

Observable observable_of(
    std::uint32_t n, const std::vector<std::tuple<std::uint64_t, std::uint64_t, double>>& terms) {
    Observable observable;
    for (const auto& [x, z, c] : terms) {
        observable.terms.emplace_back(PauliString{n, {x}, {z}}, Complex{c, 0.0});
    }
    return observable;
}

struct Fixture {
    std::string name;
    Circuit circuit;
    Observable observable;
};

void cx_chain(Circuit& circuit) {
    for (QubitId q = 0; q + 1 < circuit.num_qubits(); ++q) {
        circuit.append(Opcode::CX, {q, q + 1});
    }
}

// The expectation fixtures of docs/adr/0017 section 6: depolarizing, amplitude
// damping and their composition; 1-, 2- and 3-qubit channels; n = 2..10. Every
// circuit is connected, so no Factorized plan competes.
std::vector<Fixture> expectation_fixtures() {
    std::vector<Fixture> out;
    {
        Circuit c(2, 0);
        c.append(Opcode::H, {0});
        c.append(Opcode::CX, {0, 1});
        c.append_channel({1}, depolarizing_1q(0.2));
        out.push_back({"bell_depolarizing", c, observable_of(2, {{0, 3, 1.0}})});
    }
    {
        Circuit c(2, 0);
        c.append(Opcode::X, {0});
        c.append(Opcode::RY, {1}, {0.7});
        c.append(Opcode::CX, {0, 1});
        c.append_channel({0}, amplitude_damping(0.3));
        out.push_back({"amplitude_damping", c, observable_of(2, {{0, 1, 0.6}, {2, 0, 0.4}})});
    }
    {
        Circuit c(3, 0);
        c.append(Opcode::H, {0});
        cx_chain(c);
        for (QubitId q = 0; q < 3; ++q) {
            c.append_channel({q}, amplitude_damping(0.25));
        }
        out.push_back({"ghz_damped", c, observable_of(3, {{7, 0, 1.0}})});
    }
    {
        Circuit c(3, 0);
        c.append(Opcode::RX, {0}, {0.4});
        c.append(Opcode::RY, {1}, {1.1});
        c.append(Opcode::H, {2});
        cx_chain(c);
        c.append_channel({0, 1}, depolarizing_2q(0.15));
        out.push_back({"two_qubit_depolarizing", c, observable_of(3, {{0, 3, 1.0}, {4, 4, 0.5}})});
    }
    {
        // Rotations rather than H on every qubit: |+...+> is invariant under the
        // CX chain and under the bit flip, which would make every trajectory's
        // value identical and the fixture uninformative.
        Circuit c(5, 0);
        for (QubitId q = 0; q < 5; ++q) {
            c.append(Opcode::RY, {q}, {0.35 * static_cast<double>(q + 1)});
        }
        cx_chain(c);
        c.append_channel({1, 2, 3}, correlated_bit_flip_3q(0.2));
        c.append_channel({4}, depolarizing_1q(0.1));
        out.push_back(
            {"three_qubit_channel", c, observable_of(5, {{0, 6, 1.0}, {0, 24, 1.0}, {1, 0, 0.3}})});
    }
    {
        Circuit c(6, 0);
        for (QubitId q = 0; q < 6; ++q) {
            c.append(Opcode::RY, {q}, {0.3 * static_cast<double>(q + 1)});
        }
        cx_chain(c);
        c.append_channel({2}, amplitude_damping(0.2));
        c.append_channel({2}, depolarizing_1q(0.15));
        out.push_back({"composition", c, observable_of(6, {{0, 33, 1.0}, {4, 0, 0.3}})});
    }
    {
        Circuit c(8, 0);
        c.append(Opcode::H, {0});
        cx_chain(c);
        c.append(Opcode::RZ, {3}, {0.5});
        for (const QubitId q : {0U, 4U, 7U}) {
            c.append_channel({q}, depolarizing_1q(0.1));
        }
        out.push_back({"eight_qubits", c, observable_of(8, {{0, 129, 1.0}, {0, 8, 0.5}})});
    }
    {
        Circuit c(10, 0);
        for (QubitId q = 0; q < 10; ++q) {
            c.append(Opcode::RY, {q}, {0.2 * static_cast<double>(q + 1)});
        }
        cx_chain(c);
        for (const QubitId q : {0U, 5U, 9U}) {
            c.append_channel({q}, amplitude_damping(0.15));
        }
        c.append_channel({4, 5}, depolarizing_2q(0.1));
        out.push_back(
            {"ten_qubits", c, observable_of(10, {{0, 1, 1.0}, {0, 512, 1.0}, {48, 0, 0.5}})});
    }
    return out;
}

// ---------------------------------------------------------------------------
// Statistics, from their definitions
// ---------------------------------------------------------------------------

// z with P(Z > z) = upper_tail for a standard normal Z, by bisection on
// P(Z > z) = erfc(z / sqrt 2) / 2.
double normal_upper_quantile(double upper_tail) {
    double lo = 0.0;
    double hi = 40.0;
    for (int i = 0; i < 200; ++i) {
        const double mid = 0.5 * (lo + hi);
        if (0.5 * std::erfc(mid / std::sqrt(2.0)) > upper_tail) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return 0.5 * (lo + hi);
}

// The regularized lower incomplete gamma function P(a, x): the series for
// x < a + 1, the continued fraction otherwise (Numerical Recipes, gser/gcf).
double regularized_lower_gamma(double a, double x) {
    if (x <= 0.0) {
        return 0.0;
    }
    const double log_prefix = -x + a * std::log(x) - std::lgamma(a);
    if (x < a + 1.0) {
        double term = 1.0 / a;
        double sum = term;
        for (int n = 1; n < 10000; ++n) {
            term *= x / (a + n);
            sum += term;
            if (std::abs(term) < std::abs(sum) * 1e-16) {
                break;
            }
        }
        return sum * std::exp(log_prefix);
    }
    constexpr double kTiny = 1e-300;
    double b = x + 1.0 - a;
    double c = 1.0 / kTiny;
    double d = 1.0 / b;
    double h = d;
    for (int i = 1; i < 10000; ++i) {
        const double an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        d = std::abs(d) < kTiny ? kTiny : d;
        c = b + an / c;
        c = std::abs(c) < kTiny ? kTiny : c;
        d = 1.0 / d;
        const double delta = d * c;
        h *= delta;
        if (std::abs(delta - 1.0) < 1e-16) {
            break;
        }
    }
    return 1.0 - std::exp(log_prefix) * h;
}

// x with P(chi^2_k <= x) = p.
double chi_square_quantile(double k, double p) {
    double lo = 0.0;
    double hi = 20.0 * k;
    for (int i = 0; i < 300; ++i) {
        const double mid = 0.5 * (lo + hi);
        if (regularized_lower_gamma(0.5 * k, 0.5 * mid) < p) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return 0.5 * (lo + hi);
}

// The exact two-sided binomial p-value of observing k successes in n trials
// with success probability p: the total probability of every outcome no more
// likely than k. k, n and p have different roles and types at every call site.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
double binomial_two_sided_p(std::uint64_t k, std::uint64_t n, double p) {
    const auto log_pmf = [&](std::uint64_t i) {
        const auto di = static_cast<double>(i);
        const auto dn = static_cast<double>(n);
        return std::lgamma(dn + 1.0) - std::lgamma(di + 1.0) - std::lgamma(dn - di + 1.0) +
               di * std::log(p) + (dn - di) * std::log1p(-p);
    };
    const double observed = log_pmf(k);
    double total = 0.0;
    for (std::uint64_t i = 0; i <= n; ++i) {
        const double l = log_pmf(i);
        if (l <= observed + 1e-9) {
            total += std::exp(l);
        }
    }
    return std::min(1.0, total);
}

// ---------------------------------------------------------------------------
// Running jobs
// ---------------------------------------------------------------------------

const CandidateSummary* candidate(const Plan& plan, Strategy strategy) {
    for (const CandidateSummary& c : plan.candidates) {
        if (c.strategy == strategy) {
            return &c;
        }
    }
    return nullptr;
}

// The exact Phase 12 value: no opt-in, so the density-matrix engine runs it.
Complex exact_value(const Fixture& f) {
    const Result result =
        run(f.circuit, RunOptions{.shots = 0, .seed = 1, .expectation = f.observable});
    EXPECT_TRUE(result.exact()) << f.name;
    return result.expectation_value().value_or(Complex{});
}

// Runs `options` on trajectories regardless of which plan is cheaper, by
// setting the memory budget to the trajectory candidate's own peak: the density
// matrix needs strictly more (16 * 4^n against 16 * 2^n), so only trajectories
// fit. This is the "memory budget below 16 * 4^n" route of docs/adr/0017
// section 5, used to validate the engine on circuits where it would lose on
// cost.
// Runs the job on trajectories, with the density matrix excluded by an observed
// runtime floor above a generous time budget -- the planner's own mechanism,
// as after a replan -- and otherwise exactly as menos::run would: the same job,
// planned by plan_job and run by run_with_replanning, with a clock that never
// crosses. Until Phase 21 this excluded the density matrix with a memory budget
// at the trajectory plan's figure, which only the old figures allowed: with
// the real ones, a small density matrix is below the trajectory engine's own
// buffers (docs/adr/0025 section 3.4).
Result run_on_trajectories(const Circuit& circuit, const RunOptions& options) {
    const plan::JobSpec job{
        .circuit = circuit,
        .shots = options.shots,
        .want_statevector = options.want_statevector,
        .seed = options.seed,
        .memory_budget_bytes = std::uint64_t{1} << 34,
        .expectation = options.expectation,
        .accuracy_policy = options.accuracy_policy,
        .time_budget_seconds = 1.0e18,
        .noise_method = options.noise_method,
        .expectation_precision = options.expectation_precision,
    };
    const plan::HardwareProfile& profile = plan::calibrated_hardware_profile();
    plan::RuntimeObservations observations;
    observations.observe(Engine::DensityMatrix, Backend::Cpu, 2.0e18);
    const Plan trajectory_plan = plan::plan_job(job, profile, observations);
    EXPECT_EQ(trajectory_plan.strategy, Strategy::DirectTrajectory) << trajectory_plan.explain();
    return api::run_with_replanning(
        job, profile, trajectory_plan, []() -> api::ElapsedSource { return []() { return 0.0; }; });
}

Circuit noisy_one_qubit() {
    Circuit circuit(1, 1);
    circuit.append(Opcode::H, {0});
    circuit.append_channel({0}, depolarizing_1q(0.2));
    return circuit;
}

// ---------------------------------------------------------------------------
// Validation of the opt-ins
// ---------------------------------------------------------------------------

TEST(TrajectoryRunTest, ANonPositivePrecisionIsRefused) {
    for (const double bad : {0.0,
                             -0.1,
                             std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()}) {
        const RunOptions options{.shots = 0,
                                 .seed = 1,
                                 .expectation = observable_of(1, {{1, 0, 1.0}}),
                                 .expectation_precision = bad};
        EXPECT_THROW((void)run(noisy_one_qubit(), options), common::InvalidCircuit) << bad;
    }
}

TEST(TrajectoryRunTest, APrecisionWithoutAnExpectationIsRefused) {
    Circuit circuit = noisy_one_qubit();
    circuit.append_measure(0, 0);
    const RunOptions options{.shots = 100, .seed = 1, .expectation_precision = 0.01};
    EXPECT_THROW((void)run(circuit, options), common::InvalidCircuit);
}

// ---------------------------------------------------------------------------
// Gate criterion 5: Phase 12 behavior is unchanged without an opt-in.
// ---------------------------------------------------------------------------

TEST(TrajectoryRunTest, ANoisyJobWithoutAnOptInNeverSeesTheTrajectoryCandidate) {
    const Fixture f = expectation_fixtures()[0];
    const Plan plan =
        preview(f.circuit, RunOptions{.shots = 0, .seed = 1, .expectation = f.observable});
    EXPECT_EQ(candidate(plan, Strategy::DirectTrajectory), nullptr);
    EXPECT_EQ(plan.strategy, Strategy::DirectDensityMatrix);
    EXPECT_TRUE(plan.exact);
    EXPECT_FALSE(plan.trajectory_summary.has_value());
    EXPECT_EQ(plan.explain().find("trajectory"), std::string::npos);
}

TEST(TrajectoryRunTest, ANoiselessJobNeverSeesTheTrajectoryCandidateEvenWhenOptedIn) {
    Circuit circuit(2, 2);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::CX, {0, 1});
    circuit.append_measure(0, 0);
    circuit.append_measure(1, 1);
    const RunOptions plain{.shots = 100, .seed = 1};
    RunOptions opted = plain;
    opted.noise_method = NoiseMethod::Auto;
    EXPECT_EQ(candidate(preview(circuit, opted), Strategy::DirectTrajectory), nullptr);
    EXPECT_EQ(preview(circuit, opted).explain(), preview(circuit, plain).explain());
}

// Precision opts in only the expectation; counts still need Auto (owner
// decision, 2026-10-02), so the job stays exact on the density matrix.
TEST(TrajectoryRunTest, PrecisionWithCountsButWithoutAutoStaysOnTheDensityMatrix) {
    Circuit circuit(2, 2);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::CX, {0, 1});
    circuit.append_channel({1}, depolarizing_1q(0.2));
    circuit.append_measure(0, 0);
    circuit.append_measure(1, 1);
    const RunOptions options{.shots = 100,
                             .seed = 1,
                             .expectation = observable_of(2, {{0, 3, 1.0}}),
                             .expectation_precision = 0.5};
    const Plan plan = preview(circuit, options);
    const CandidateSummary* trajectory = candidate(plan, Strategy::DirectTrajectory);
    ASSERT_NE(trajectory, nullptr);
    EXPECT_FALSE(trajectory->feasible);
    EXPECT_NE(trajectory->infeasible_reason.find("noise_method=Auto"), std::string::npos);
    EXPECT_EQ(plan.strategy, Strategy::DirectDensityMatrix);
}

TEST(TrajectoryRunTest, AutoWithAnExpectationButNoPrecisionIsInfeasibleWithAReason) {
    const Fixture f = expectation_fixtures()[0];
    RunOptions options{.shots = 0, .seed = 1, .expectation = f.observable};
    options.noise_method = NoiseMethod::Auto;
    const Plan plan = preview(f.circuit, options);
    const CandidateSummary* trajectory = candidate(plan, Strategy::DirectTrajectory);
    ASSERT_NE(trajectory, nullptr);
    EXPECT_FALSE(trajectory->feasible);
    EXPECT_NE(trajectory->infeasible_reason.find("expectation_precision"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Gate criterion 1: agreement with the density matrix within the reported
// error, on the documented fixture set.
// ---------------------------------------------------------------------------

TEST(TrajectoryRunTest, ExpectationsAgreeWithTheDensityMatrixWithinTheirStandardError) {
    const std::vector<Fixture> fixtures = expectation_fixtures();
    const auto k = static_cast<double>(fixtures.size());
    // Bonferroni: a family-wise false-alarm rate of 0.001 over K fixtures.
    const double z_star = normal_upper_quantile(0.001 / (2.0 * k));
    for (const Fixture& f : fixtures) {
        const Complex exact = exact_value(f);
        // eps = C / 40 gives every fixture N = 1600 trajectories: enough power
        // for the z-test at any n, without a 10-qubit fixture dominating the
        // suite's run time. The z-test is valid at any eps.
        double c = 0.0;
        for (const auto& [pauli, coefficient] : f.observable.terms) {
            c += std::abs(coefficient);  // no fixture has an identity term
        }
        const Result result = run_on_trajectories(f.circuit,
                                                  RunOptions{.shots = 0,
                                                             .seed = 2026,
                                                             .expectation = f.observable,
                                                             .expectation_precision = c / 40.0});
        ASSERT_TRUE(result.expectation_value().has_value()) << f.name;
        ASSERT_TRUE(result.expectation_standard_error().has_value()) << f.name;
        // Guarded by the ASSERT_TRUEs above; clang-tidy cannot see through them.
        // NOLINTBEGIN(bugprone-unchecked-optional-access)
        const double estimate = result.expectation_value()->real();
        const double se = *result.expectation_standard_error();
        // NOLINTEND(bugprone-unchecked-optional-access)
        ASSERT_GT(se, 0.0) << f.name;
        // |z| <= z*, i.e. |estimate - exact| <= z* se. kRounding covers only the
        // two engines' floating-point rounding, which is all that is left when
        // an observable's per-trajectory value barely varies and se is tiny.
        constexpr double kRounding = 1e-12;
        // Printed so the gate evidence (docs/adr/0017 section 10) can quote it.
        std::printf("[trajectory-z] %-24s n=%u exact=%.6f estimate=%.6f se=%.3g z=%+.3f z*=%.3f\n",
                    f.name.c_str(),
                    f.circuit.num_qubits(),
                    exact.real(),
                    estimate,
                    se,
                    (estimate - exact.real()) / se,
                    z_star);
        EXPECT_LE(std::abs(estimate - exact.real()), z_star * se + kRounding)
            << f.name << ": trajectory " << estimate << " +/- " << se << ", exact " << exact.real()
            << ", z = " << (estimate - exact.real()) / se << " vs z* = " << z_star;
        EXPECT_FALSE(result.exact()) << f.name;
    }
}

TEST(TrajectoryRunTest, CountsJobsRunOneShotPerTrajectoryAndAreNotExact) {
    Circuit circuit(2, 2);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::CX, {0, 1});
    circuit.append_channel({1}, amplitude_damping(0.3));
    circuit.append_measure(0, 0, ReadoutError{{0.9, 0.1, 0.2, 0.8}});
    circuit.append_measure(1, 1);
    RunOptions options{.shots = 500, .seed = 9};
    options.noise_method = NoiseMethod::Auto;
    const Result result = run_on_trajectories(circuit, options);
    std::uint64_t total = 0;
    for (const auto& [outcome, count] : result.counts()) {
        total += count;
    }
    EXPECT_EQ(total, 500U);
    EXPECT_FALSE(result.exact());
    EXPECT_FALSE(result.expectation_standard_error().has_value());
    ASSERT_TRUE(result.plan().trajectory_summary.has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access): asserted above.
    EXPECT_EQ(result.plan().trajectory_summary->trajectories, 500U);
}

// Counts against the exact density matrix (docs/adr/0017 section 6.2): the
// exact distribution comes from Phase 12 Z-string expectations through the
// Walsh transform P(x) = 2^-n sum_z (-1)^{|x & z|} <Z^z>, then each readout
// error's assignment matrix; the trajectory counts must pass a chi-square
// goodness-of-fit test at 0.001.
TEST(TrajectoryRunTest, CountsPassAChiSquareTestAgainstTheDensityMatrix) {
    constexpr std::uint32_t kN = 3;
    constexpr std::uint64_t kDim = std::uint64_t{1} << kN;
    constexpr std::uint64_t kShots = 20000;
    const ReadoutError readout{{0.9, 0.1, 0.2, 0.8}};  // on qubit 1 only

    Circuit unmeasured(kN, kN);
    for (QubitId q = 0; q < kN; ++q) {
        unmeasured.append(Opcode::RY, {q}, {0.5 + 0.4 * static_cast<double>(q)});
    }
    unmeasured.append_channel({0}, amplitude_damping(0.3));
    cx_chain(unmeasured);
    unmeasured.append_channel({2}, depolarizing_1q(0.2));

    // Exact true-outcome probabilities from the density matrix.
    std::vector<double> z_expectation(kDim, 1.0);
    for (std::uint64_t z = 1; z < kDim; ++z) {
        const Result exact =
            run(unmeasured,
                RunOptions{.shots = 0, .seed = 1, .expectation = observable_of(kN, {{0, z, 1.0}})});
        z_expectation[z] = exact.expectation_value().value_or(Complex{}).real();
    }
    std::vector<double> truth(kDim, 0.0);
    for (std::uint64_t x = 0; x < kDim; ++x) {
        for (std::uint64_t z = 0; z < kDim; ++z) {
            const double sign = (std::popcount(x & z) % 2 == 0) ? 1.0 : -1.0;
            truth[x] += sign * z_expectation[z] / static_cast<double>(kDim);
        }
    }
    // Pushed through qubit 1's readout: assignment[t * 2 + s] = P(s | t).
    std::vector<double> expected(kDim, 0.0);
    for (std::uint64_t x = 0; x < kDim; ++x) {
        const std::uint64_t t = (x >> 1) & 1U;
        for (std::uint64_t s = 0; s < 2; ++s) {
            const std::uint64_t y = (x & ~std::uint64_t{2}) | (s << 1);
            expected[y] += truth[x] * readout.assignment[t * 2 + s] * static_cast<double>(kShots);
        }
    }

    Circuit measured = unmeasured;
    measured.append_measure(0, 0);
    measured.append_measure(1, 1, readout);
    measured.append_measure(2, 2);
    RunOptions options{.shots = kShots, .seed = 31};
    options.noise_method = NoiseMethod::Auto;
    const Result result = run_on_trajectories(measured, options);

    double statistic = 0.0;
    std::uint64_t total = 0;
    for (std::uint64_t y = 0; y < kDim; ++y) {
        ASSERT_GT(expected[y], 5.0) << "every outcome must be likely enough for chi-square";
        BitString key(kN);
        for (std::uint32_t b = 0; b < kN; ++b) {
            key.set(b, ((y >> b) & 1U) != 0);
        }
        const auto it = result.counts().find(key);
        const double observed = it == result.counts().end() ? 0.0 : static_cast<double>(it->second);
        total += static_cast<std::uint64_t>(observed);
        statistic += (observed - expected[y]) * (observed - expected[y]) / expected[y];
    }
    EXPECT_EQ(total, kShots);
    const double critical = chi_square_quantile(static_cast<double>(kDim - 1), 0.999);
    std::printf("[trajectory-chi2] counts n=%u shots=%llu chi2=%.3f critical=%.3f\n",
                kN,
                static_cast<unsigned long long>(kShots),
                statistic,
                critical);
    EXPECT_LT(statistic, critical);
}

// ---------------------------------------------------------------------------
// Gate criterion 2: the reported errors are genuine standard errors, and
// non-exact results are flagged.
// ---------------------------------------------------------------------------

// 400 independent seeds on two fixtures (docs/adr/0017 section 6.3): (a) the
// fraction of |z| <= z_0.975 is consistent with 0.95 (two-sided exact binomial
// test at 0.001); (b) the spread of the estimates across seeds matches the
// reported standard error (chi-square interval at 0.001, 399 degrees of
// freedom). (a) catches a biased mean, (b) a wrongly scaled error.
TEST(TrajectoryRunTest, ReportedStandardErrorsAreGenuineAcrossFourHundredSeeds) {
    constexpr std::uint64_t kSeeds = 400;
    constexpr double kEps = 0.05;  // C = 1 for both fixtures, so N = 400
    const double z_975 = normal_upper_quantile(0.025);
    const double dof = static_cast<double>(kSeeds - 1);
    const double chi_lo = chi_square_quantile(dof, 0.0005);
    const double chi_hi = chi_square_quantile(dof, 0.9995);
    const std::vector<Fixture> all = expectation_fixtures();
    for (const Fixture& f : {all[0], all[2]}) {
        const double exact = exact_value(f).real();
        std::uint64_t covered = 0;
        double sum = 0.0;
        double sum_squares = 0.0;
        double se_sum = 0.0;
        for (std::uint64_t seed = 0; seed < kSeeds; ++seed) {
            const Result result = run_on_trajectories(f.circuit,
                                                      RunOptions{.shots = 0,
                                                                 .seed = 10000 + seed,
                                                                 .expectation = f.observable,
                                                                 .expectation_precision = kEps});
            ASSERT_TRUE(result.expectation_standard_error().has_value());
            // NOLINTBEGIN(bugprone-unchecked-optional-access): asserted above.
            const double estimate = result.expectation_value()->real();
            const double se = *result.expectation_standard_error();
            const std::uint64_t n = result.plan().trajectory_summary->trajectories;
            // NOLINTEND(bugprone-unchecked-optional-access)
            ASSERT_EQ(n, 400U) << "N = ceil(C^2 / eps^2) with C = 1, eps = 0.05";
            // The bound holds up to the sampling spread of s itself.
            EXPECT_LE(se, kEps * (1.0 + 5.0 / std::sqrt(2.0 * static_cast<double>(n - 1))));
            covered += std::abs(estimate - exact) <= z_975 * se ? 1U : 0U;
            sum += estimate;
            sum_squares += estimate * estimate;
            se_sum += se;
        }
        const double p_value = binomial_two_sided_p(covered, kSeeds, 0.95);
        EXPECT_GE(p_value, 0.001) << f.name << ": " << covered << "/" << kSeeds << " covered";
        const double mean = sum / static_cast<double>(kSeeds);
        const double sample_variance =
            (sum_squares - static_cast<double>(kSeeds) * mean * mean) / dof;
        const double mean_se = se_sum / static_cast<double>(kSeeds);
        const double statistic = dof * sample_variance / (mean_se * mean_se);
        std::printf("[trajectory-coverage] %s covered=%llu/%llu p=%.4g chi2=%.2f in [%.2f, %.2f]\n",
                    f.name.c_str(),
                    static_cast<unsigned long long>(covered),
                    static_cast<unsigned long long>(kSeeds),
                    p_value,
                    statistic,
                    chi_lo,
                    chi_hi);
        EXPECT_GE(statistic, chi_lo) << f.name << ": errors overstated";
        EXPECT_LE(statistic, chi_hi) << f.name << ": errors understated";
    }
}

TEST(TrajectoryRunTest, AResultIsFlaggedAndExplainedAsStatistical) {
    const Fixture f = expectation_fixtures()[1];
    const Result result = run_on_trajectories(
        f.circuit,
        RunOptions{
            .shots = 0, .seed = 3, .expectation = f.observable, .expectation_precision = 0.05});
    EXPECT_FALSE(result.exact());
    EXPECT_FALSE(result.plan().exact);
    const std::string report = result.explain();
    EXPECT_NE(report.find("direct_trajectory (approximate)"), std::string::npos) << report;
    // C = 0.6 + 0.4 = 1, eps = 0.05: N = 400.
    EXPECT_NE(report.find("n=400 from max(shots, max(2, ceil(C^2/eps^2))) C=1 eps=0.05"),
              std::string::npos)
        << report;
    EXPECT_NE(report.find("std_error="), std::string::npos) << report;
    EXPECT_NE(report.find("statistical"), std::string::npos) << report;
}

// ---------------------------------------------------------------------------
// Gate criterion 3: the planner chooses by cost.
// ---------------------------------------------------------------------------

// The channel comes before the CX chain and the observables below act on the
// LAST qubit, so the light cone keeps every qubit and the noise. (Z on qubit 0
// commutes with every CX in the chain, so an observable there would reduce the
// job to one noisy qubit and no longer test what these tests are about.)
Circuit noisy_chain(std::uint32_t n) {
    Circuit c(n, 0);
    for (QubitId q = 0; q < n; ++q) {
        c.append(Opcode::RY, {q}, {0.1 * static_cast<double>(q + 1)});
    }
    c.append_channel({0}, depolarizing_1q(0.1));
    cx_chain(c);
    return c;
}

Observable z_on_last(std::uint32_t n) {
    return observable_of(n, {{0, std::uint64_t{1} << (n - 1), 1.0}});
}

TEST(TrajectoryRunTest, ATightPrecisionOnASmallCircuitLeavesTheDensityMatrixCheaper) {
    const Circuit circuit = noisy_chain(2);
    const RunOptions options{.shots = 0,
                             .seed = 1,
                             .memory_budget_bytes = std::uint64_t{1} << 30,
                             .expectation = z_on_last(2),
                             .expectation_precision = 0.01};
    const Plan plan = preview(circuit, options);
    const CandidateSummary* trajectory = candidate(plan, Strategy::DirectTrajectory);
    const CandidateSummary* density = candidate(plan, Strategy::DirectDensityMatrix);
    ASSERT_NE(trajectory, nullptr);
    ASSERT_NE(density, nullptr);
    ASSERT_TRUE(trajectory->feasible);
    ASSERT_TRUE(density->feasible);
    EXPECT_LT(density->estimated_work_units, trajectory->estimated_work_units);
    EXPECT_EQ(plan.strategy, Strategy::DirectDensityMatrix);
}

TEST(TrajectoryRunTest, ALooserPrecisionOnALargerCircuitMakesTrajectoriesCheaper) {
    const Circuit circuit = noisy_chain(12);
    const RunOptions options{.shots = 0,
                             .seed = 1,
                             .memory_budget_bytes = std::uint64_t{1} << 30,
                             .expectation = z_on_last(12),
                             .expectation_precision = 0.1};
    const Plan plan = preview(circuit, options);
    const CandidateSummary* trajectory = candidate(plan, Strategy::DirectTrajectory);
    const CandidateSummary* density = candidate(plan, Strategy::DirectDensityMatrix);
    ASSERT_NE(trajectory, nullptr);
    ASSERT_NE(density, nullptr);
    ASSERT_TRUE(trajectory->feasible);
    ASSERT_TRUE(density->feasible);
    EXPECT_LT(trajectory->estimated_work_units, density->estimated_work_units);
    EXPECT_EQ(plan.strategy, Strategy::DirectTrajectory);
    // And it runs: n = 12 on trajectories, N = 100.
    const Result result = run(circuit, options);
    EXPECT_EQ(result.plan().strategy, Strategy::DirectTrajectory);
    EXPECT_TRUE(result.expectation_standard_error().has_value());
}

TEST(TrajectoryRunTest, AMemoryBudgetBelowTheDensityMatrixLeavesOnlyTrajectories) {
    constexpr std::uint32_t kN = 8;
    const Circuit circuit = noisy_chain(kN);
    RunOptions options{.shots = 0,
                       .seed = 1,
                       .memory_budget_bytes = std::uint64_t{1} << 30,
                       .expectation = z_on_last(kN),
                       .expectation_precision = 0.005};
    // With room for both, the tight precision makes the density matrix cheaper.
    EXPECT_EQ(preview(circuit, options).strategy, Strategy::DirectDensityMatrix);
    // Below 16 * 4^n bytes it cannot run at all, and trajectories still fit.
    options.memory_budget_bytes = 16 * (std::uint64_t{1} << (2 * kN)) - 1;
    const Plan plan = preview(circuit, options);
    const CandidateSummary* density = candidate(plan, Strategy::DirectDensityMatrix);
    ASSERT_NE(density, nullptr);
    EXPECT_FALSE(density->feasible);
    EXPECT_EQ(plan.strategy, Strategy::DirectTrajectory);
}

// ---------------------------------------------------------------------------
// Gate criterion 4: deterministic under a fixed seed, including after a replan.
// ---------------------------------------------------------------------------

TEST(TrajectoryRunTest, TheSameSeedGivesTheSameResult) {
    const Fixture f = expectation_fixtures()[3];
    const RunOptions options{
        .shots = 0, .seed = 77, .expectation = f.observable, .expectation_precision = 0.05};
    const Result a = run_on_trajectories(f.circuit, options);
    const Result b = run_on_trajectories(f.circuit, options);
    EXPECT_EQ(a.expectation_value(), b.expectation_value());
    EXPECT_EQ(a.expectation_standard_error(), b.expectation_standard_error());
    RunOptions other = options;
    other.seed = 78;
    EXPECT_NE(run_on_trajectories(f.circuit, other).expectation_value(), a.expectation_value());
}

// One work unit and one byte per second, so seconds are computable by hand (the
// convention of tests/api/replan_test.cpp).
constexpr plan::HardwareProfile kUnitRateProfile{
    .total_memory_bytes = 1ULL << 40,
    .memory_bandwidth_bytes_per_second = 1.0,
    .work_units_per_second = 1.0,
};

// A density-matrix run that overruns its time budget replans through the
// unchanged planner; with the caller opted in, trajectories take over, and the
// result is identical to running the trajectory plan directly with that seed,
// because a replan restarts from scratch (docs/adr/0017 section 4).
TEST(TrajectoryRunTest, AReplanFromTheDensityMatrixToTrajectoriesIsDeterministic) {
    const Circuit circuit = noisy_chain(2);
    const plan::JobSpec job{
        .circuit = circuit,
        .shots = 0,
        .want_statevector = false,
        .seed = 4242,
        .memory_budget_bytes = std::uint64_t{1} << 30,
        .expectation = z_on_last(2),
        .accuracy_policy = {},
        .time_budget_seconds = 1.0e9,
        .noise_method = NoiseMethod::Exact,
        .expectation_precision = 0.01,
    };
    const Plan initial = plan::plan_job(job, kUnitRateProfile);
    ASSERT_EQ(initial.strategy, Strategy::DirectDensityMatrix);

    // The first attempt's clock crosses the budget after two boundaries; the
    // replacement attempt's never does.
    int attempts = 0;
    const api::ElapsedSourceFactory crossing_once = [&attempts]() -> api::ElapsedSource {
        if (attempts++ > 0) {
            return []() { return 0.0; };
        }
        auto reads = std::make_shared<int>(0);
        return [reads]() { return (*reads)++ < 2 ? 0.0 : 2.0e9; };
    };
    const Result replanned =
        api::run_with_replanning(job, kUnitRateProfile, initial, crossing_once);
    ASSERT_EQ(replanned.replans().size(), 1U) << replanned.explain();
    EXPECT_EQ(replanned.replans()[0].superseded_plan.strategy, Strategy::DirectDensityMatrix);
    ASSERT_EQ(replanned.plan().strategy, Strategy::DirectTrajectory) << replanned.explain();

    // The same job planned straight onto trajectories -- with the density
    // matrix excluded by the same observed floor the replan recorded -- gives
    // the identical result. (Until Phase 21 the density matrix was excluded
    // with a memory budget at the trajectory plan's figure; with the real
    // figures, a 2-qubit density matrix is smaller than the trajectory
    // engine's own buffers, docs/adr/0025 section 3.4.)
    plan::RuntimeObservations observations;
    observations.observe(Engine::DensityMatrix, Backend::Cpu, 2.0e9);
    const plan::JobSpec& direct_job = job;
    const Plan direct_plan = plan::plan_job(direct_job, kUnitRateProfile, observations);
    ASSERT_EQ(direct_plan.strategy, Strategy::DirectTrajectory);
    const Result direct = api::run_with_replanning(
        direct_job, kUnitRateProfile, direct_plan, []() -> api::ElapsedSource {
            return []() { return 0.0; };
        });
    EXPECT_EQ(replanned.expectation_value(), direct.expectation_value());
    EXPECT_EQ(replanned.expectation_standard_error(), direct.expectation_standard_error());
}

}  // namespace
}  // namespace menos
