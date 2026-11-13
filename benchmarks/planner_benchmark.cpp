// Planner latency benchmark (Phase 24 step 1, docs/adr/0028 section 5).
//
// One Google Benchmark per job of the fixed corpus (tests/plan/planner_corpus.h),
// each timing plan::plan_job alone against the job's hand-built HardwareProfile,
// so neither calibration nor the plan cache is in the figure. A few
// "cache_hit/" benchmarks time plan_job_cached when the cache holds the plan.
//
// Each benchmark also reports, from one extra untimed call, the heap traffic of
// a single plan (benchmarks/counting_allocator.cpp):
//   allocs         operator new calls
//   alloc_bytes    bytes requested
//   peak_bytes     peak live heap above what was live before the call
//
// Run (release build with -DMENOS_BUILD_BENCHMARKS=ON):
//   ./menos_planner_benchmark --benchmark_repetitions=5
//       --benchmark_report_aggregates_only=true --benchmark_min_time=0.2s

#include <benchmark/benchmark.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "analysis/planner_counters.h"
#include "common/error.h"
#include "counting_allocator.h"
#include "plan/plan_cache.h"
#include "plan/planner.h"
#include "planner_corpus.h"

namespace {

using menos::planner_corpus::CorpusJob;

// A corpus job may have no feasible plan (the large random-pairs jobs do):
// refusing it is planning work too, so it is timed the same way.
std::optional<menos::Plan> plan_or_refuse(const CorpusJob& job) {
    try {
        return menos::plan::plan_job(job.spec, job.hardware);
    } catch (const menos::common::ResourceExceeded&) {
        return std::nullopt;
    }
}

void report_allocations(benchmark::State& state, const CorpusJob& job) {
    menos_benchmark::reset_peak_to_current();
#ifdef MENOS_PLANNER_COUNTERS
    menos::analysis::detail::planner_counters = {};
#endif
    const menos_benchmark::AllocationSnapshot before = menos_benchmark::allocation_snapshot();
    {
        std::optional<menos::Plan> plan = plan_or_refuse(job);
        benchmark::DoNotOptimize(plan);
        state.counters["refused"] = plan.has_value() ? 0.0 : 1.0;
    }
    const menos_benchmark::AllocationSnapshot after = menos_benchmark::allocation_snapshot();
    state.counters["allocs"] = static_cast<double>(after.allocations - before.allocations);
    state.counters["alloc_bytes"] =
        static_cast<double>(after.bytes_requested - before.bytes_requested);
    state.counters["peak_bytes"] = static_cast<double>(after.peak_live_bytes - before.live_bytes);
#ifdef MENOS_PLANNER_COUNTERS
    const menos::analysis::detail::PlannerCounters& c = menos::analysis::detail::planner_counters;
    state.counters["rank_hits"] = static_cast<double>(c.rank_hits);
    state.counters["rank_misses"] = static_cast<double>(c.rank_misses);
    state.counters["rank_evictions"] = static_cast<double>(c.rank_evictions);
    state.counters["rank_uncached"] = static_cast<double>(c.rank_uncached);
    state.counters["rank_peak_bytes"] = static_cast<double>(c.rank_peak_bytes);
    state.counters["minfill_searches"] = static_cast<double>(c.minfill_searches);
    state.counters["minfill_steps"] = static_cast<double>(c.minfill_steps);
    state.counters["minfill_nodes"] = static_cast<double>(c.minfill_nodes);
    state.counters["minfill_stopped"] = static_cast<double>(c.minfill_stopped);
#endif
}

void plan_job_benchmark(benchmark::State& state, const CorpusJob& job) {
    for (auto _ : state) {
        std::optional<menos::Plan> plan = plan_or_refuse(job);
        benchmark::DoNotOptimize(plan);
    }
    report_allocations(state, job);
}

void cache_hit_benchmark(benchmark::State& state, const CorpusJob& job) {
    menos::plan::plan_cache_clear();
    {
        menos::Plan warm = menos::plan::plan_job_cached(job.spec, job.hardware);
        benchmark::DoNotOptimize(warm);
    }
    for (auto _ : state) {
        menos::Plan plan = menos::plan::plan_job_cached(job.spec, job.hardware);
        benchmark::DoNotOptimize(plan);
    }
    const menos::plan::PlanCacheStats stats = menos::plan::plan_cache_stats();
    state.counters["hits"] = static_cast<double>(stats.hits);
    state.counters["misses"] = static_cast<double>(stats.misses);
}

// Built once, kept alive for the whole run: the benchmarks hold references.
const std::vector<CorpusJob>& corpus() {
    static const std::vector<CorpusJob> kJobs = menos::planner_corpus::planner_corpus();
    return kJobs;
}

// Benchmark-only jobs, outside the golden corpus (Phase 24 step 1,
// docs/adr/0028 section 16.9). "expectation_50_terms": the case measured on
// 2026-11-02 (docs/BUILD_PLAN.md, Phase 24, "Planning cost"), a 20-qubit,
// depth-10 circuit with a 50-term Pauli expectation, which took 4.4 s to plan
// before branch-and-bound was merged. Planned under the same hand-built
// profile as the corpus, and again with a non-binding time budget (the
// calibrated-profile path, where every candidate is converted to seconds).
const std::vector<CorpusJob>& extra_jobs() {
    static const std::vector<CorpusJob> kJobs = [] {
        namespace pc = menos::planner_corpus;
        std::vector<CorpusJob> jobs;
        jobs.push_back(CorpusJob{
            "expectation_50_terms/n20_d10",
            pc::expectation_job(pc::brickwork(20, 10, 301, false), pc::random_paulis(20, 50, 302)),
            pc::uncalibrated_profile()});
        menos::plan::JobSpec budgeted =
            pc::expectation_job(pc::brickwork(20, 10, 301, false), pc::random_paulis(20, 50, 302));
        budgeted.time_budget_seconds = 1.0e6;
        jobs.push_back(CorpusJob{"expectation_50_terms_budgeted/n20_d10",
                                 std::move(budgeted),
                                 pc::fixed_calibrated_profile()});
        return jobs;
    }();
    return kJobs;
}

const bool kRegistered = [] {
    for (const CorpusJob& job : corpus()) {
        benchmark::RegisterBenchmark(("plan_job/" + job.name).c_str(), [&job](benchmark::State& s) {
            plan_job_benchmark(s, job);
        })->Unit(benchmark::kMicrosecond);
    }
    for (const CorpusJob& job : extra_jobs()) {
        benchmark::RegisterBenchmark(("plan_job/" + job.name).c_str(), [&job](benchmark::State& s) {
            plan_job_benchmark(s, job);
        })->Unit(benchmark::kMicrosecond);
    }
    for (const CorpusJob& job : corpus()) {
        if (job.name == "statevector_brickwork/medium" || job.name == "counts_brickwork/large" ||
            job.name == "expectation_ising/medium") {
            benchmark::RegisterBenchmark(
                ("cache_hit/" + job.name).c_str(),
                [&job](benchmark::State& s) { cache_hit_benchmark(s, job); })
                ->Unit(benchmark::kMicrosecond);
        }
    }
    return true;
}();

}  // namespace
