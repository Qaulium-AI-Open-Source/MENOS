// Does calibration's run-to-run variation change which plan is chosen?
// (Phase 21, owner question 4 of 2026-11-03.)
//
// Reads calibration samples, one per line, in the CSV format of
// calibration_probe (bench/raw/calibration_cpu.csv, calibration_cuda.csv).
// Plans every fixed-corpus job once per sample, with the sample's derived
// constants in the HardwareProfile and nothing else changed, under several
// time budgets (none, and 1 ms to 1000 s), and reports every (job, budget)
// whose outcome -- winning strategy, region backends, or refusal -- is not the
// same for every sample.
//
// The planner reads the calibration in two places only: estimated seconds,
// which matter only against a time budget, and the CPU-versus-CUDA
// comparisons (the dense backend and Factorized's per-region backends), which
// need a CUDA build and a sample with cuda_present = 1. Device memory is not
// in the CSV; samples with a device are given this machine's figures
// (3770 MiB total, 3681 MiB free; plan/hardware_profile.h).
//
// usage: menos_calibration_sensitivity SAMPLES.csv

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "common/error.h"
#include "plan/enum_names.h"
#include "plan/hardware_profile.h"
#include "plan/planner.h"
#include "planner_corpus.h"

namespace {

using menos::planner_corpus::CorpusJob;

std::vector<menos::plan::HardwareProfile> read_samples(const char* path) {
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);  // header
    std::vector<menos::plan::HardwareProfile> samples;
    while (std::getline(in, line)) {
        std::vector<double> f;
        std::stringstream fields(line);
        std::string field;
        while (std::getline(fields, field, ',')) {
            f.push_back(std::stod(field));
        }
        if (f.size() != 9) {
            continue;
        }
        menos::plan::HardwareProfile hw{.total_memory_bytes =
                                            menos::planner_corpus::kTotalMemoryBytes};
        hw.work_units_per_second = f[2];
        hw.memory_bandwidth_bytes_per_second = f[3];
        hw.cuda_present = f[4] != 0.0;
        if (hw.cuda_present) {
            hw.cuda_device_total_bytes = std::uint64_t{3770} << 20;
            hw.cuda_device_available_bytes = std::uint64_t{3681} << 20;
            hw.cuda_host_to_device_bytes_per_second = f[5];
            hw.cuda_device_to_host_bytes_per_second = f[6];
            hw.cuda_kernel_launch_seconds = f[7];
            hw.cuda_device_work_units_per_second = f[8];
        }
        samples.push_back(hw);
    }
    return samples;
}

std::string outcome(const CorpusJob& job, const menos::plan::HardwareProfile& hw) {
    try {
        const menos::Plan plan = menos::plan::plan_job(job.spec, hw);
        std::string out = std::string(menos::plan::strategy_name(plan.strategy));
        for (const menos::Region& region : plan.regions) {
            out += region.backend == menos::Backend::Cuda ? " cuda" : " cpu";
        }
        return out;
    } catch (const menos::common::ResourceExceeded&) {
        return "refused (ResourceExceeded)";
    } catch (const menos::common::UnsupportedFeature&) {
        return "refused (UnsupportedFeature)";
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s SAMPLES.csv\n", argv[0]);
        return 2;
    }
    const std::vector<menos::plan::HardwareProfile> samples = read_samples(argv[1]);
    const std::vector<std::optional<double>> budgets{
        std::nullopt, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0};
    std::size_t cases = 0;
    std::size_t varying = 0;
    for (CorpusJob job : menos::planner_corpus::planner_corpus()) {
        for (const std::optional<double>& budget : budgets) {
            job.spec.time_budget_seconds = budget;
            std::map<std::string, std::size_t> outcomes;
            for (const menos::plan::HardwareProfile& hw : samples) {
                ++outcomes[outcome(job, hw)];
            }
            ++cases;
            if (outcomes.size() > 1) {
                ++varying;
                const std::string budget_text =
                    budget.has_value() ? std::to_string(budget.value_or(0.0)) + "s" : "none";
                std::cout << job.name << " budget=" << budget_text << ":";
                for (const auto& [o, n] : outcomes) {
                    std::cout << "  [" << o << "] x" << n;
                }
                std::cout << "\n";
            }
        }
    }
    std::cout << samples.size() << " samples; " << varying << " of " << cases
              << " (job, budget) cases change outcome with the sample\n";
    return 0;
}
