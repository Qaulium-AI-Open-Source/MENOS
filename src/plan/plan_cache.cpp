#include "plan/plan_cache.h"

#include <cmath>
#include <cstring>
#include <list>
#include <mutex>
#include <numbers>
#include <type_traits>
#include <unordered_map>
#include <variant>

#include "common/error.h"
#include "ir/gate_table.h"
#include "ir/noise.h"
#include "plan/planner.h"

namespace menos::plan {
namespace {

// --- key serialisation: raw little-endian bytes of every structural field,
// in a fixed order, with a tag before every variable-length run so that two
// different structures can never serialise to the same bytes.

template <typename T>
void put(std::string& key, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    const char* bytes = reinterpret_cast<const char*>(&value);
    key.append(bytes, sizeof(T));
}

template <typename T>
void put_span(std::string& key, const std::vector<T>& values) {
    put(key, static_cast<std::uint64_t>(values.size()));
    for (const T& v : values) {
        put(key, v);
    }
}

// True when a multi-qubit parametrised gate is within kSpecialAngleMargin
// of an angle at which its operator Schmidt rank drops below its generic
// value -- the one planner input that depends on a parameter's value. Only
// gates of arity >= 2 matter: analysis::compute_schmidt_rank_bound never
// looks at a single-qubit gate (no cut passes through its one operand).
//   RZZ(t) = cos(t/2) I.I - i sin(t/2) Z.Z: rank 2 unless sin or cos vanishes.
//   CP(l)  = |0><0|.I + |1><1|.diag(1, e^{il}): rank 2 unless e^{il} == 1.
// Any other multi-qubit parametrised gate added later is treated as "may be
// special" (never cached) until it is given a rule here.
bool near_special_angle(const GateOp& gate) {
    if (gate.qubits.size() < 2 || gate.params.empty()) {
        return false;
    }
    switch (gate.opcode) {
        case Opcode::RZZ: {
            const double half = gate.params[0] / 2.0;
            return std::fabs(std::sin(half)) <= kSpecialAngleMargin ||
                   std::fabs(std::cos(half)) <= kSpecialAngleMargin;
        }
        case Opcode::CP: {
            // |e^{il} - 1| = 2 |sin(l/2)|.
            return 2.0 * std::fabs(std::sin(gate.params[0] / 2.0)) <= kSpecialAngleMargin;
        }
        default:
            return true;
    }
}

enum class OpTag : std::uint8_t { Gate = 1, Unitary = 2, Measure = 3 };

struct LruEntry {
    std::string key;
    Plan plan;
};

struct Cache {
    std::mutex mutex;
    std::list<LruEntry> entries;  // front = most recently used
    std::unordered_map<std::string, std::list<LruEntry>::iterator> index;
    PlanCacheStats stats;
};

Cache& cache() {
    static Cache instance;
    return instance;
}

bool is_direct(Strategy strategy) {
    switch (strategy) {
        case Strategy::DirectDense:
        case Strategy::DirectStabilizer:
        case Strategy::DirectPauli:
        case Strategy::DirectMPS:
        case Strategy::DirectTensorNetwork:
        case Strategy::DirectDensityMatrix:
        case Strategy::DirectTrajectory:
            return true;
        case Strategy::Factorized:
        case Strategy::CutAndRecombine:
            return false;
    }
    throw common::InternalError("plan_cache: unknown Strategy");
}

}  // namespace

std::optional<std::string> plan_cache_key(const JobSpec& job_spec) {
    // A budgeted job is replanned from observations (docs/adr/0015); keep it
    // out. A noisy job is out of scope (its channels carry numeric
    // parameters of their own); so is a readout error.
    if (job_spec.time_budget_seconds.has_value()) {
        return std::nullopt;
    }
    if (ir::first_noise_reason(job_spec.circuit.operations()).has_value()) {
        return std::nullopt;
    }

    std::string key;
    key.reserve(64 + 16 * job_spec.circuit.operations().size());
    put(key, std::uint32_t{1});  // key format version
    put(key, job_spec.circuit.num_qubits());
    put(key, job_spec.circuit.num_clbits());
    put(key, static_cast<std::uint64_t>(job_spec.circuit.operations().size()));
    bool cacheable = true;
    for (const Operation& op : job_spec.circuit.operations()) {
        std::visit(
            [&](const auto& o) {
                using T = std::decay_t<decltype(o)>;
                if constexpr (std::is_same_v<T, GateOp>) {
                    if (near_special_angle(o)) {
                        cacheable = false;
                    }
                    put(key, OpTag::Gate);
                    put(key, o.opcode);
                    put_span(key, o.qubits);
                    put(key, static_cast<std::uint32_t>(o.params.size()));
                } else if constexpr (std::is_same_v<T, UnitaryOp>) {
                    // An explicit matrix has no parameters to vary: it is
                    // part of the structure, byte for byte.
                    put(key, OpTag::Unitary);
                    put_span(key, o.qubits);
                    put_span(key, o.matrix);
                } else if constexpr (std::is_same_v<T, MeasureOp>) {
                    put(key, OpTag::Measure);
                    put(key, o.qubit);
                    put(key, o.clbit);
                } else {
                    static_assert(std::is_same_v<T, ChannelOp>);
                    cacheable = false;  // first_noise_reason already refused it
                }
            },
            op);
    }
    if (!cacheable) {
        return std::nullopt;
    }

    put(key, job_spec.shots);
    put(key, job_spec.want_statevector);
    put(key, job_spec.memory_budget_bytes.has_value());
    put(key, job_spec.memory_budget_bytes.value_or(0));
    put(key, job_spec.expectation.has_value());
    if (job_spec.expectation.has_value()) {
        put(key, static_cast<std::uint64_t>(job_spec.expectation->terms.size()));
        for (const auto& [pauli, coefficient] : job_spec.expectation->terms) {
            put(key, pauli.num_qubits);
            put_span(key, pauli.x_words);
            put_span(key, pauli.z_words);
            put(key, coefficient);
        }
    }
    put(key, job_spec.accuracy_policy.kind);
    put(key, job_spec.accuracy_policy.tolerance);
    put(key, job_spec.accuracy_policy.max_bond_dimension.has_value());
    put(key, job_spec.accuracy_policy.max_bond_dimension.value_or(0));
    put(key, job_spec.noise_method);
    put(key, job_spec.expectation_precision.has_value());
    put(key, job_spec.expectation_precision.value_or(0.0));
    put(key, job_spec.max_threads);
    put(key, job_spec.backend);
    // Not in the key: seed (sampling only, never read by plan_job) and the
    // circuit's global_phase (dropped by the light-cone pass, applied by the
    // runtime from the job's own circuit).
    return key;
}

Plan plan_job_cached(const JobSpec& job_spec, const HardwareProfile& hardware_profile) {
    const std::optional<std::string> key = plan_cache_key(job_spec);
    Cache& c = cache();
    if (!key.has_value()) {
        std::lock_guard<std::mutex> lock(c.mutex);
        ++c.stats.bypasses;
        return plan_job(job_spec, hardware_profile);
    }
    {
        std::lock_guard<std::mutex> lock(c.mutex);
        const auto found = c.index.find(*key);
        if (found != c.index.end()) {
            c.entries.splice(c.entries.begin(), c.entries, found->second);
            ++c.stats.hits;
            return reparameterize_direct_plan(found->second->plan, job_spec);
        }
        ++c.stats.misses;
    }
    // Planned outside the lock: plan_job can take seconds, and another thread
    // planning the same job concurrently only costs a duplicate insert.
    Plan plan = plan_job(job_spec, hardware_profile);
    if (!is_direct(plan.strategy)) {
        std::lock_guard<std::mutex> lock(c.mutex);
        ++c.stats.bypasses;
        --c.stats.misses;
        return plan;
    }
    std::lock_guard<std::mutex> lock(c.mutex);
    if (const auto found = c.index.find(*key); found != c.index.end()) {
        c.entries.erase(found->second);
        c.index.erase(found);
    }
    c.entries.push_front(LruEntry{*key, plan});
    c.index[*key] = c.entries.begin();
    while (c.entries.size() > kPlanCacheCapacity) {
        c.index.erase(c.entries.back().key);
        c.entries.pop_back();
    }
    return plan;
}

PlanCacheStats plan_cache_stats() {
    Cache& c = cache();
    std::lock_guard<std::mutex> lock(c.mutex);
    return c.stats;
}

void plan_cache_clear() {
    Cache& c = cache();
    std::lock_guard<std::mutex> lock(c.mutex);
    c.entries.clear();
    c.index.clear();
    c.stats = PlanCacheStats{};
}

}  // namespace menos::plan
