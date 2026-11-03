#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "api/replan_controller.h"
#include "common/error.h"
#include "menos/menos.h"
#include "plan/hardware_profile.h"
#include "plan/planner.h"

// Runtime monitoring and replanning end to end (Phase 11, docs/adr/0015).
//
// The threshold is wall-clock, which is not reproducible, so the tests that
// assert on the DECISION drive an injected clock through
// api::run_with_replanning and a synthetic HardwareProfile. That is the same
// discipline Phase 10 used for backend selection: with the inputs pinned, the
// planner's and the monitor's behaviour is a pure function of the numbers and
// means the same thing on every machine.
//
// One test at the end uses the real clock and the real calibration, to show the
// mechanism fires on genuine hardware rather than only against a fake.

namespace menos::api {
namespace {

// One work unit per second, one byte per second: seconds then equal
// max(work_units, peak_bytes), so every figure below is computable by hand.
constexpr plan::HardwareProfile kUnitRateProfile{
    .total_memory_bytes = 1ULL << 40,
    .memory_bandwidth_bytes_per_second = 1.0,
    .work_units_per_second = 1.0,
};

// Clifford with counts, so both the stabilizer and the dense candidates are
// feasible and an observation can move the winner between them.
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
    for (std::uint32_t q = 0; q < num_qubits; ++q) {
        circuit.append_measure(static_cast<QubitId>(q), static_cast<ClbitId>(q));
    }
    return circuit;
}

plan::JobSpec make_job(const Circuit& circuit, std::optional<double> budget) {
    return plan::JobSpec{
        .circuit = circuit,
        .shots = 32,
        .want_statevector = false,
        .seed = 11,
        .memory_budget_bytes = std::uint64_t{1} << 30,
        .expectation = std::nullopt,
        .accuracy_policy = {},
        .time_budget_seconds = budget,
    };
}

// A clock that reports `before_seconds` until it has been read
// `cross_after_reads` times, then reports `after_seconds` for ever. A fresh one
// is handed out per attempt, and `attempts` counts how many were requested, so
// a test can assert that a restart really happened.
//
// The script is one aggregate rather than three scalar parameters, so every
// construction below names its fields and no two can be transposed silently.
class ScriptedClock {
public:
    struct Script {
        double before_seconds = 0.0;
        double after_seconds = 0.0;
        int cross_after_reads = 0;
        // How many of the first attempts cross. Later attempts get a clock that
        // never does, which is what a genuinely faster plan looks like. One is
        // the common case; more than one produces a CHAIN of replans.
        int crossing_attempts = 1;
    };

    explicit ScriptedClock(Script script) : script_(script) {}

    // The first `crossing_attempts` attempts' clocks cross; later ones never do.
    ElapsedSourceFactory factory() {
        return [this]() -> ElapsedSource {
            const int attempt = attempts_++;
            if (attempt >= script_.crossing_attempts) {
                return []() { return 0.0; };
            }
            auto reads = std::make_shared<int>(0);
            return [this, reads]() {
                const int n = (*reads)++;
                return n < script_.cross_after_reads ? script_.before_seconds
                                                     : script_.after_seconds;
            };
        };
    }

    [[nodiscard]] int attempts() const {
        return attempts_;
    }

private:
    Script script_;
    int attempts_ = 0;
};

const CandidateSummary& winner_of(const Plan& plan) {
    for (const CandidateSummary& candidate : plan.candidates) {
        if (candidate.is_winner) {
            return candidate;
        }
    }
    throw std::logic_error("no winner");
}

// ---------------------------------------------------------------------------
// Gate criterion 2: a documented threshold deterministically triggers a replan.
// ---------------------------------------------------------------------------

TEST(ReplanTest, AnInjectedClockCrossesTheBudgetAtAKnownBoundaryAndTriggersAReplan) {
    const Circuit circuit = clifford_circuit(4);
    const plan::JobSpec job = make_job(circuit, /*budget=*/1.0e9);
    const Plan initial = plan::plan_job(job, kUnitRateProfile);
    ASSERT_EQ(initial.strategy, Strategy::DirectStabilizer);

    // Cross on the 6th boundary: under budget for five operations, then over.
    ScriptedClock clock({.before_seconds = 0.0, .after_seconds = 2.0e9, .cross_after_reads = 5});
    const Result result = run_with_replanning(job, kUnitRateProfile, initial, clock.factory());

    ASSERT_EQ(result.replans().size(), 1U) << "exactly one replan was scripted";
    const ReplanEvent& event = result.replans()[0];
    EXPECT_EQ(event.sequence, 0U);
    EXPECT_DOUBLE_EQ(event.threshold_seconds, 1.0e9);
    EXPECT_DOUBLE_EQ(event.observed_seconds, 2.0e9);
    EXPECT_EQ(event.boundaries_crossed, 6U) << "the 6th boundary is the first over budget";
    EXPECT_EQ(event.engine, Engine::Stabilizer);
    EXPECT_EQ(event.backend, Backend::Cpu);
    EXPECT_EQ(event.superseded_plan.strategy, Strategy::DirectStabilizer);

    // Gate criterion 4: the new plan differs, because of what was observed.
    EXPECT_NE(result.plan().strategy, Strategy::DirectStabilizer);
    EXPECT_EQ(clock.attempts(), 2) << "one abandoned attempt and one that completed";
}

// The threshold is the budget itself. A run that stays under it must not
// replan, and must produce a Result with an empty history.
TEST(ReplanTest, ARunThatStaysUnderTheBudgetNeverReplans) {
    const Circuit circuit = clifford_circuit(4);
    const plan::JobSpec job = make_job(circuit, /*budget=*/1.0e9);
    const Plan initial = plan::plan_job(job, kUnitRateProfile);

    ScriptedClock clock({.before_seconds = 0.0, .after_seconds = 0.0, .cross_after_reads = 0});
    const Result result = run_with_replanning(job, kUnitRateProfile, initial, clock.factory());

    EXPECT_TRUE(result.replans().empty());
    EXPECT_EQ(result.plan().strategy, initial.strategy);
    EXPECT_EQ(clock.attempts(), 1);
}

// ---------------------------------------------------------------------------
// Gate criterion 5: restart/re-execution correctness.
// ---------------------------------------------------------------------------

// The correctness claim for a restart is that abandoning leaves NO residue: the
// run that follows must produce exactly what running that same plan from a cold
// start produces. So the reference is the REPLANNED plan run directly, not the
// original one -- comparing two different engines' 32-shot histograms would
// compare two samples of two different RNG streams and prove nothing.
TEST(ReplanTest, ARestartedRunIsIdenticalToRunningTheNewPlanFromCold) {
    const Circuit circuit = clifford_circuit(4);
    const plan::JobSpec monitored = make_job(circuit, /*budget=*/1.0e9);
    const Plan initial = plan::plan_job(monitored, kUnitRateProfile);
    ASSERT_EQ(initial.strategy, Strategy::DirectStabilizer);

    // Abandon the first attempt and restart.
    ScriptedClock clock({.before_seconds = 0.0, .after_seconds = 2.0e9, .cross_after_reads = 3});
    const Result restarted =
        run_with_replanning(monitored, kUnitRateProfile, initial, clock.factory());
    ASSERT_EQ(restarted.replans().size(), 1U);
    EXPECT_NE(restarted.plan().strategy, Strategy::DirectStabilizer);

    // The cold reference: hand the planner the same observation up front so it
    // produces the same plan, then run that with nothing monitoring it at all.
    plan::RuntimeObservations observations;
    observations.observe(initial.regions[0].engine,
                         initial.regions[0].backend,
                         restarted.replans()[0].observed_seconds);
    // The same budget, because it is the budget that turns an observed floor
    // into an exclusion -- without one the floor raises stabilizer's estimate
    // but stabilizer still wins on work units, and the reference would be the
    // wrong plan. The real clock cannot cross a 1e9-second budget on a 4-qubit
    // circuit, so this attempt runs to completion unmonitored in effect.
    const plan::JobSpec cold_job = make_job(circuit, /*budget=*/1.0e9);
    const Plan cold_plan = plan::plan_job(cold_job, kUnitRateProfile, observations);
    ASSERT_EQ(cold_plan.strategy, restarted.plan().strategy)
        << "the reference must be the same plan the restart chose";
    const Result cold =
        run_with_replanning(cold_job, kUnitRateProfile, cold_plan, &steady_elapsed_source);
    ASSERT_TRUE(cold.replans().empty());

    // Bitwise: same plan, same seed, same sampler. Anything but equality would
    // mean the abandoned attempt perturbed the one that followed.
    EXPECT_EQ(restarted.counts(), cold.counts());
    EXPECT_EQ(restarted.exact(), cold.exact());
    EXPECT_EQ(restarted.actual_state_bytes(), cold.actual_state_bytes());

    std::uint64_t shots = 0;
    for (const auto& [key, count] : restarted.counts()) {
        static_cast<void>(key);
        shots += count;
    }
    EXPECT_EQ(shots, 32U) << "every shot must still be accounted for";
}

// Re-running the same abandoned job twice must give the same answer twice --
// CLAUDE.md's determinism invariant has to survive a replan.
TEST(ReplanTest, ReplanningIsDeterministic) {
    const Circuit circuit = clifford_circuit(4);
    const plan::JobSpec job = make_job(circuit, /*budget=*/1.0e9);
    const Plan initial = plan::plan_job(job, kUnitRateProfile);

    ScriptedClock first_clock(
        {.before_seconds = 0.0, .after_seconds = 2.0e9, .cross_after_reads = 4});
    const Result first = run_with_replanning(job, kUnitRateProfile, initial, first_clock.factory());
    ScriptedClock second_clock(
        {.before_seconds = 0.0, .after_seconds = 2.0e9, .cross_after_reads = 4});
    const Result second =
        run_with_replanning(job, kUnitRateProfile, initial, second_clock.factory());

    EXPECT_EQ(first.counts(), second.counts());
    EXPECT_EQ(first.plan().strategy, second.plan().strategy);
    ASSERT_EQ(first.replans().size(), second.replans().size());
    EXPECT_EQ(first.replans()[0].engine, second.replans()[0].engine);
    EXPECT_EQ(first.replans()[0].boundaries_crossed, second.replans()[0].boundaries_crossed);
}

// ---------------------------------------------------------------------------
// Same-plan behaviour (gate criterion resolution, docs/adr/0015).
//
// A replan that reproduced the plan which just overran would be a livelock, and
// the original design carried a branch for it. That branch is unreachable -- the
// proof is in run_with_replanning's own comment -- so it was removed rather than
// left as code no test can execute. What is asserted instead is the conclusion
// itself, through the real loop, over a chain long enough to give it a chance to
// fail: every replan runs a DIFFERENT engine from every earlier attempt.
// ---------------------------------------------------------------------------

// Two crossings in a row, then a third attempt that completes. The Clifford job
// has exactly three feasible single-engine candidates (stabilizer, MPS, dense),
// so this walks the whole enumeration.
TEST(ReplanTest, EveryReplanInAChainOfCrossingsRunsADifferentEngine) {
    const Circuit circuit = clifford_circuit(4);
    const plan::JobSpec job = make_job(circuit, /*budget=*/1.0e9);
    const Plan initial = plan::plan_job(job, kUnitRateProfile);
    ASSERT_EQ(initial.strategy, Strategy::DirectStabilizer);

    ScriptedClock clock({.before_seconds = 0.0,
                         .after_seconds = 2.0e9,
                         .cross_after_reads = 2,
                         .crossing_attempts = 2});
    const Result result = run_with_replanning(job, kUnitRateProfile, initial, clock.factory());

    ASSERT_EQ(result.replans().size(), 2U);
    EXPECT_EQ(clock.attempts(), 3) << "two abandoned attempts and one that completed";

    // The sequence is numbered from zero and in order.
    EXPECT_EQ(result.replans()[0].sequence, 0U);
    EXPECT_EQ(result.replans()[1].sequence, 1U);

    // Every attempt ran a different engine from every attempt before it.
    std::vector<Engine> ran;
    ran.push_back(result.replans()[0].engine);
    ran.push_back(result.replans()[1].engine);
    ran.push_back(result.plan().regions[0].engine);
    EXPECT_EQ(ran[0], Engine::Stabilizer);
    EXPECT_NE(ran[1], ran[0]);
    EXPECT_NE(ran[2], ran[0]);
    EXPECT_NE(ran[2], ran[1]);

    // Each event's superseded plan is the plan that really ran that attempt, and
    // the second event's superseded plan is not the first's replacement in name
    // only: it carries its own strategy.
    EXPECT_NE(result.replans()[0].superseded_plan.strategy,
              result.replans()[1].superseded_plan.strategy);
    EXPECT_NE(result.replans()[1].superseded_plan.strategy, result.plan().strategy);

    // The final report shows the whole chain.
    const std::string report = result.explain();
    EXPECT_NE(report.find("replan 0"), std::string::npos) << report;
    EXPECT_NE(report.find("replan 1"), std::string::npos) << report;
}

// The same chain when EVERY attempt crosses: it must end in the strict refusal
// after visiting each feasible engine exactly once. Terminating at all is the
// property; there is no attempt count to tune because the count is the number of
// feasible candidates.
TEST(ReplanTest, ARunWhoseEveryAttemptCrossesEndsInRefusalAfterEachEngineIsTriedOnce) {
    const Circuit circuit = clifford_circuit(4);
    const plan::JobSpec job = make_job(circuit, /*budget=*/1.0e9);
    const Plan initial = plan::plan_job(job, kUnitRateProfile);

    ScriptedClock clock({.before_seconds = 0.0,
                         .after_seconds = 2.0e9,
                         .cross_after_reads = 2,
                         .crossing_attempts = 1000});  // every attempt crosses
    EXPECT_THROW(
        static_cast<void>(run_with_replanning(job, kUnitRateProfile, initial, clock.factory())),
        common::ResourceExceeded);
    EXPECT_EQ(clock.attempts(), 3)
        << "stabilizer, MPS and dense are the three feasible candidates; each must run once and "
           "none twice";
}

// ---------------------------------------------------------------------------
// The strict refusal.
// ---------------------------------------------------------------------------

// When the budget is crossed and nothing else meets it, the job is refused.
// This is the strict reading of time_budget_seconds carried into run time: a
// run is never quietly allowed to overrun.
TEST(ReplanTest, ACrossingWithNoAlternativePlanThrowsResourceExceeded) {
    const Circuit circuit = clifford_circuit(4);
    // A budget so small that, once the running engine is excluded by its
    // observed floor, no other candidate's own estimate fits either.
    const plan::JobSpec job = make_job(circuit, /*budget=*/1.0e-6);
    // The initial plan has to be built without the budget, because with it
    // nothing would be feasible at plan time at all -- which is the Phase 10
    // behaviour and is already covered there.
    const Plan initial = plan::plan_job(make_job(circuit, std::nullopt), kUnitRateProfile);

    ScriptedClock clock({.before_seconds = 0.0, .after_seconds = 1.0, .cross_after_reads = 2});
    EXPECT_THROW(
        static_cast<void>(run_with_replanning(job, kUnitRateProfile, initial, clock.factory())),
        common::ResourceExceeded);
}

TEST(ReplanTest, TheRefusalMessageNamesTheObservationAndTheBudget) {
    const Circuit circuit = clifford_circuit(4);
    const plan::JobSpec job = make_job(circuit, /*budget=*/1.0e-6);
    const Plan initial = plan::plan_job(make_job(circuit, std::nullopt), kUnitRateProfile);
    ScriptedClock clock({.before_seconds = 0.0, .after_seconds = 0.25, .cross_after_reads = 2});
    try {
        static_cast<void>(run_with_replanning(job, kUnitRateProfile, initial, clock.factory()));
        FAIL() << "a crossing with no alternative must throw";
    } catch (const common::ResourceExceeded& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("observed runtime"), std::string::npos) << message;
        EXPECT_NE(message.find("time budget"), std::string::npos) << message;
        EXPECT_NE(message.find("stabilizer"), std::string::npos)
            << "the message must name the engine that overran: " << message;
    }
}

// Phase 12 (docs/adr/0016): a noisy job that cannot be split has exactly one
// candidate able to run it, the density-matrix engine. An overrun observed on it
// makes that candidate infeasible, and no pure-state engine can take a noisy job
// over, so the replan ends in a refusal naming the engine -- the same refusal,
// with the same message, on every run.
TEST(ReplanTest, ANoisyJobWhoseDensityMatrixRunOverrunsIsRefusedDeterministically) {
    const double damped = std::sqrt(0.5);
    Circuit circuit(3, 3);
    for (QubitId q = 0; q < 3; ++q) {
        circuit.append(Opcode::H, {q});
    }
    circuit.append(Opcode::CX, {0, 1});
    circuit.append(Opcode::CX, {1, 2});
    circuit.append_channel({1}, {{1.0, 0.0, 0.0, damped}, {0.0, damped, 0.0, 0.0}});
    for (QubitId q = 0; q < 3; ++q) {
        circuit.append_measure(q, q);
    }
    const plan::JobSpec job = make_job(circuit, /*budget=*/1.0e9);
    const Plan initial = plan::plan_job(job, kUnitRateProfile);
    ASSERT_EQ(initial.strategy, Strategy::DirectDensityMatrix);

    std::vector<std::string> messages;
    for (int run = 0; run < 2; ++run) {
        ScriptedClock clock(
            {.before_seconds = 0.0, .after_seconds = 2.0e9, .cross_after_reads = 2});
        try {
            static_cast<void>(run_with_replanning(job, kUnitRateProfile, initial, clock.factory()));
            FAIL() << "only the density-matrix engine can run a noisy job, so an overrun on it "
                      "must be refused";
        } catch (const common::ResourceExceeded& e) {
            messages.emplace_back(e.what());
        }
        EXPECT_EQ(clock.attempts(), 1);
    }
    ASSERT_EQ(messages.size(), 2U);
    EXPECT_NE(messages[0].find("density_matrix"), std::string::npos) << messages[0];
    EXPECT_EQ(messages[0], messages[1]);
}

// A multi-region plan divides its elapsed time between several engines, and
// this phase does not attribute it per region. The documented consequence is
// refusal rather than a guessed attribution, and it is asserted here so the
// limitation is visible in the suite.
TEST(ReplanTest, AMultiRegionPlanIsRefusedRatherThanAttributedPerRegion) {
    Circuit circuit(4, 4);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::CX, {0, 1});
    circuit.append(Opcode::H, {2});
    circuit.append(Opcode::CX, {2, 3});
    for (std::uint32_t q = 0; q < 4; ++q) {
        circuit.append_measure(static_cast<QubitId>(q), static_cast<ClbitId>(q));
    }
    const plan::JobSpec unbudgeted = make_job(circuit, std::nullopt);
    const Plan initial = plan::plan_job(unbudgeted, kUnitRateProfile);
    ASSERT_GT(initial.regions.size(), 1U) << "this test needs a multi-region plan";

    const plan::JobSpec job = make_job(circuit, /*budget=*/1.0e9);
    ScriptedClock clock({.before_seconds = 0.0, .after_seconds = 2.0e9, .cross_after_reads = 1});
    try {
        static_cast<void>(run_with_replanning(job, kUnitRateProfile, initial, clock.factory()));
        FAIL() << "a multi-region overrun must be refused, not guessed at";
    } catch (const common::ResourceExceeded& e) {
        EXPECT_NE(std::string(e.what()).find("not attributed per region"), std::string::npos)
            << e.what();
    }
}

// ---------------------------------------------------------------------------
// Gate criteria 7 and 8: explain().
// ---------------------------------------------------------------------------

TEST(ReplanTest, ExplainReportsThePredictionObservationThresholdReasonAndSequence) {
    const Circuit circuit = clifford_circuit(4);
    const plan::JobSpec job = make_job(circuit, /*budget=*/1.0e9);
    const Plan initial = plan::plan_job(job, kUnitRateProfile);
    ScriptedClock clock({.before_seconds = 0.0, .after_seconds = 2.0e9, .cross_after_reads = 5});
    const Result result = run_with_replanning(job, kUnitRateProfile, initial, clock.factory());
    const std::string report = result.explain();

    EXPECT_NE(report.find("replan 0"), std::string::npos) << report;
    EXPECT_NE(report.find("threshold="), std::string::npos) << report;
    EXPECT_NE(report.find("crossed at elapsed="), std::string::npos) << report;
    EXPECT_NE(report.find("abandoned after 6 safe boundaries"), std::string::npos)
        << "the report must state how far the abandoned attempt got: " << report;
    EXPECT_NE(report.find("engine=stabilizer"), std::string::npos) << report;
    EXPECT_NE(report.find("backend=cpu"), std::string::npos) << report;
    EXPECT_NE(report.find("was  strategy=direct_stabilizer"), std::string::npos) << report;
    EXPECT_NE(report.find("predicted="), std::string::npos) << report;
    EXPECT_NE(report.find("now  strategy="), std::string::npos) << report;
    EXPECT_NE(report.find("reason=\"observed runtime exceeded the time budget"), std::string::npos)
        << report;
    // The superseded plan is reachable in full, not merely summarized.
    EXPECT_FALSE(result.replans()[0].superseded_plan.explain().empty());
}

// Gate criterion 8, the strongest form: with no replan, the report is
// byte-identical to the one the same Result produced before Phase 11 existed --
// which is to say, it contains no replan text whatsoever.
TEST(ReplanTest, ARunWithNoReplanEmitsNoReplanTextAtAll) {
    const Circuit circuit = clifford_circuit(4);
    const plan::JobSpec job = make_job(circuit, std::nullopt);
    const Plan initial = plan::plan_job(job, kUnitRateProfile);
    const Result result =
        run_with_replanning(job, kUnitRateProfile, initial, &steady_elapsed_source);

    const std::string report = result.explain();
    EXPECT_EQ(report.find("replan"), std::string::npos) << report;
    // The plan body and the actual line, in the order they always were.
    EXPECT_LT(report.find("strategy "), report.find("actual"));
}

// ---------------------------------------------------------------------------
// Gate criterion 1: monitored on REAL execution, real clock, real calibration.
// ---------------------------------------------------------------------------

// This is the same 16-qubit shape the Phase 11 measurements used: the planner
// routes it to MPS on a predicted figure that measurement showed to be ~550x
// below the truth. With a budget between the two, the monitor must fire.
//
// The budget is deliberately far from both figures rather than tuned, so the
// assertion does not depend on this machine's exact speed -- only on the
// prediction being wrong by orders of magnitude, which is the finding this
// phase is built on.
TEST(ReplanTest, TheMonitorFiresOnRealHardwareWhenThePredictionIsOrdersOfMagnitudeLow) {
    Circuit circuit(16, 16);
    for (std::uint32_t layer = 0; layer < 8; ++layer) {
        for (std::uint32_t q = 0; q < 16; ++q) {
            circuit.append(Opcode::H, {static_cast<QubitId>(q)});
        }
        circuit.append(Opcode::T, {0});
        for (std::uint32_t q = layer % 2; q + 1 < 16; q += 2) {
            circuit.append(Opcode::CX, {static_cast<QubitId>(q), static_cast<QubitId>(q + 1)});
        }
    }
    for (std::uint32_t q = 0; q < 16; ++q) {
        circuit.append_measure(static_cast<QubitId>(q), static_cast<ClbitId>(q));
    }

    const plan::HardwareProfile& hardware = plan::calibrated_hardware_profile();
    if (hardware.work_units_per_second <= 0.0) {
        GTEST_SKIP() << "no calibration available -- real-hardware monitoring did NOT run";
    }

    const RunOptions unbudgeted{.shots = 64, .want_statevector = false, .seed = 5};
    // The model's prediction, read from a plan that carries a (generous)
    // budget: only a budgeted job is planned against the calibrated profile
    // (plan::hardware_profile_for), so only its candidates carry seconds.
    RunOptions generously_budgeted = unbudgeted;
    generously_budgeted.time_budget_seconds = 1e9;
    const Plan preview_plan = preview(circuit, generously_budgeted);
    const double predicted = winner_of(preview_plan).estimated_seconds;
    ASSERT_GT(predicted, 0.0);

    // A budget 100x the prediction. If the model were right this could never
    // fire; it fires because the model is wrong by far more than 100x.
    RunOptions budgeted = unbudgeted;
    budgeted.time_budget_seconds = predicted * 100.0;

    try {
        const Result result = run(circuit, budgeted);
        // Either a replan happened, or the run finished inside the budget
        // after all. Both are legitimate outcomes on unknown hardware; what is
        // NOT legitimate is finishing while having blown the budget silently.
        if (result.replans().empty()) {
            EXPECT_LE(result.execute_seconds() + result.sample_seconds(),
                      *budgeted.time_budget_seconds)
                << "a run that did not replan must have respected its budget";
        } else {
            const ReplanEvent& event = result.replans()[0];
            EXPECT_GT(event.observed_seconds, *budgeted.time_budget_seconds);
            EXPECT_EQ(event.engine, preview_plan.regions[0].engine);
            EXPECT_NE(result.explain().find("replan 0"), std::string::npos);
        }
    } catch (const common::ResourceExceeded&) {
        // Also legitimate and also a pass: the threshold fired and no plan met
        // the budget. The mechanism ran either way, which is what is asserted.
        SUCCEED() << "the threshold fired and the job was strictly refused";
    }
}

// Phase 17 (docs/adr/0021 section 3). Before Phase 17 nothing validated the
// budget: zero and negative values were refused only because no candidate fit,
// and NaN was silently ignored (no comparison with NaN is true, so neither the
// planner nor the monitor applied it). Each is now refused up front.
TEST(ReplanTest, ANonFiniteOrNonPositiveBudgetIsRefused) {
    Circuit circuit(1, 1);
    circuit.append(Opcode::H, {0});
    circuit.append_measure(0, 0);
    for (const double budget : {0.0, -1.0, std::nan(""), std::numeric_limits<double>::infinity()}) {
        RunOptions options{.shots = 10, .seed = 1};
        options.time_budget_seconds = budget;
        EXPECT_THROW(static_cast<void>(run(circuit, options)), common::InvalidCircuit)
            << "budget " << budget;
    }
}

}  // namespace
}  // namespace menos::api
