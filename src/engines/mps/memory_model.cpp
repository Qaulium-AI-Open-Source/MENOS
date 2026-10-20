#include "engines/mps/memory_model.h"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <type_traits>
#include <variant>

#include "common/checked_math.h"
#include "common/error.h"
#include "common/memory_bounds.h"
#include "common/svd.h"
#include "engines/mps/mps_state.h"
#include "engines/mps/sampling.h"

namespace menos::engines::mps {
namespace {

using common::checked_add;
using common::checked_mul;
using common::checked_pow2;
using common::heap_block_bytes;

constexpr std::uint64_t kComplex = sizeof(std::complex<double>);

// Site indices and byte counts are both integers here, and every call names
// what it passes.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)

// Replays execute() on dimensions. Each method mirrors the function it is
// named after and calls note() at every point where that function's live set
// is largest, with `held` the bytes its callers keep live around it.
class Replay {
public:
    // base is held for the whole run, beside everything the replay notes.
    Replay(std::uint32_t num_qubits, const AccuracyPolicy& policy, std::uint64_t base)
        : n_(num_qubits),
          policy_(policy),
          base_(base),
          bonds_(num_qubits > 0 ? num_qubits - 1 : 0, 1),
          capacity_(num_qubits, common::heap_block_bytes(2 * kComplex)) {
        // MPSState's own site array, and each trivial site's {1, 0}. A site's
        // figure is its data's allocation, overhead included: an n-site chain
        // makes n of them.
        sites_ = checked_add(common::heap_block_bytes(checked_mul(num_qubits, sizeof(SiteTensor))),
                             checked_mul(num_qubits, common::heap_block_bytes(2 * kComplex)));
        note(0);
    }

    [[nodiscard]] std::uint64_t left(std::size_t i) const {
        return i == 0 ? 1 : bonds_[i - 1];
    }
    [[nodiscard]] std::uint64_t right(std::size_t i) const {
        return i + 1 == n_ ? 1 : bonds_[i];
    }
    [[nodiscard]] std::uint64_t sites() const {
        return sites_;
    }
    [[nodiscard]] std::size_t center() const {
        return center_;
    }
    [[nodiscard]] std::uint64_t peak() const {
        return peak_;
    }
    [[nodiscard]] std::uint64_t max_bond() const {
        return max_bond_;
    }
    [[nodiscard]] std::uint64_t splits() const {
        return splits_;
    }
    [[nodiscard]] const std::vector<std::uint64_t>& bonds() const {
        return bonds_;
    }

    void note(std::uint64_t extra) {
        peak_ = std::max(peak_, checked_add(checked_add(sites_, base_), extra));
    }

    // MPSState::move_orthogonality_center_to.
    void move_center_to(std::size_t target, std::uint64_t held) {
        while (center_ < target) {
            shift_right(center_, held);
            ++center_;
        }
        while (center_ > target) {
            shift_left(center_, held);
            --center_;
        }
    }

    // apply_single_qubit_gate: a new site beside the old one, then moved in.
    void single(std::size_t q, std::uint64_t held) {
        const std::uint64_t site = site_bytes(left(q), right(q));
        note(checked_add(held, heap_block_bytes(site)));
        replace_site(q, site);
    }

    // apply_local_block_gate on sites [s, s + k).
    void block(std::size_t s, std::size_t k, std::uint64_t held) {
        move_center_to(s, held);
        const std::uint64_t chi_left = left(s);
        // theta, grown one site at a time; each step's new tensor is built
        // beside the previous one.
        std::uint64_t theta =
            heap_block_bytes(checked_mul(checked_mul(chi_left, 2 * right(s)), kComplex));
        note(checked_add(held, theta));
        for (std::size_t step = 1; step < k; ++step) {
            const std::uint64_t grown = heap_block_bytes(
                checked_mul(checked_mul(chi_left, checked_pow2(static_cast<unsigned>(step + 1))),
                            checked_mul(right(s + step), kComplex)));
            note(checked_add(held, checked_add(theta, grown)));
            theta = grown;
        }
        const std::uint64_t chi_right_last = right(s + k - 1);
        // theta stays live until the function returns; theta2, of the same
        // size, becomes the running remainder.
        std::uint64_t remainder = theta;
        std::uint64_t remainder_size =
            checked_mul(checked_mul(chi_left, checked_pow2(static_cast<unsigned>(k))),
                        checked_mul(chi_right_last, kComplex));
        const std::uint64_t base_held = checked_add(held, theta);
        note(checked_add(base_held, remainder));

        std::uint64_t cur_left = chi_left;
        std::size_t remaining_bits = k;
        for (std::size_t j = 0; j + 1 < k; ++j) {
            const std::uint64_t m = 2 * cur_left;
            const std::uint64_t cols = checked_mul(
                checked_pow2(static_cast<unsigned>(remaining_bits - 1)), chi_right_last);
            const std::uint64_t kept = kept_count(std::min(m, cols));
            const std::uint64_t live = checked_add(base_held, remainder);
            note(checked_add(live, common::svd_peak_bytes(m, cols)));
            const std::uint64_t result = common::svd_result_bytes(m, cols);
            const std::uint64_t new_site = checked_mul(checked_mul(m, kept), kComplex);
            note(checked_add(live, checked_add(result, heap_block_bytes(new_site))));
            replace_site(s + j, new_site);
            remainder_size = checked_mul(checked_mul(kept, cols), kComplex);
            const std::uint64_t next_remainder = heap_block_bytes(remainder_size);
            note(checked_add(live, checked_add(result, next_remainder)));
            remainder = next_remainder;
            set_bond(s + j, kept);
            cur_left = kept;
            --remaining_bits;
            ++splits_;
        }
        // The last remainder becomes the last site, moved in.
        replace_site(s + k - 1, remainder_size);
        center_ = s + k - 1;
    }

private:
    [[nodiscard]] std::uint64_t site_bytes(std::uint64_t chi_left, std::uint64_t chi_right) const {
        return checked_mul(checked_mul(chi_left, 2 * chi_right), kComplex);
    }

    // choose_kept_count: Exact keeps every value; Controlled keeps at most
    // that many, capped at max_bond_dimension, and at least one.
    [[nodiscard]] std::uint64_t kept_count(std::uint64_t full) const {
        std::uint64_t kept = full;
        if (policy_.kind == AccuracyPolicyKind::Controlled &&
            policy_.max_bond_dimension.has_value()) {
            kept = std::min<std::uint64_t>(kept, *policy_.max_bond_dimension);
        }
        return std::max<std::uint64_t>(kept, 1);
    }

    void replace_site(std::size_t i, std::uint64_t bytes) {
        const std::uint64_t allocated = common::heap_block_bytes(bytes);
        sites_ = checked_add(sites_ - capacity_[i], allocated);
        capacity_[i] = allocated;
    }

    void set_bond(std::size_t i, std::uint64_t bond) {
        bonds_[i] = bond;
        max_bond_ = std::max(max_bond_, bond);
    }

    // shift_center_right(p): SVD of site p as (2 chi_left) x chi_right. U is
    // copy-assigned into site p's own buffer, which is never smaller, so its
    // capacity is unchanged; S * Vt is contracted into a new site p + 1 built
    // beside the old one.
    void shift_right(std::size_t p, std::uint64_t held) {
        const std::uint64_t m = 2 * left(p);
        const std::uint64_t cols = right(p);
        const std::uint64_t k = std::min(m, cols);
        const std::uint64_t new_next = site_bytes(k, right(p + 1));
        note(checked_add(
            held,
            std::max(common::svd_peak_bytes(m, cols),
                     checked_add(common::svd_result_bytes(m, cols), heap_block_bytes(new_next)))));
        set_bond(p, k);
        replace_site(p + 1, new_next);
    }

    // shift_center_left(p): SVD of site p as chi_left x (2 chi_right); Vt is
    // copy-assigned into site p's buffer, U * S contracted into a new site
    // p - 1.
    void shift_left(std::size_t p, std::uint64_t held) {
        const std::uint64_t m = left(p);
        const std::uint64_t cols = 2 * right(p);
        const std::uint64_t k = std::min(m, cols);
        const std::uint64_t new_prev = site_bytes(left(p - 1), k);
        note(checked_add(
            held,
            std::max(common::svd_peak_bytes(m, cols),
                     checked_add(common::svd_result_bytes(m, cols), heap_block_bytes(new_prev)))));
        set_bond(p - 1, k);
        replace_site(p - 1, new_prev);
    }

    std::uint32_t n_;
    AccuracyPolicy policy_;
    std::uint64_t base_;
    std::vector<std::uint64_t> bonds_;
    std::vector<std::uint64_t> capacity_;
    std::uint64_t sites_ = 0;
    std::size_t center_ = 0;
    std::uint64_t peak_ = 0;
    std::uint64_t max_bond_ = 1;
    std::uint64_t splits_ = 0;
};

// NOLINTEND(bugprone-easily-swappable-parameters)

// apply_gate(): the SWAP network that gathers the operands, the gate's own
// block, and the network undone, with the buffers apply_gate keeps live
// throughout (the sorted operands and their ranks, the localized gate matrix,
// the SWAP matrix in both forms, the swap log and the occupancy list).
void replay_gate(Replay& replay,
                 const std::vector<QubitId>& qubits,
                 std::uint64_t generated_matrix_bytes) {
    const std::size_t k = qubits.size();
    if (k == 1) {
        replay.single(qubits[0], generated_matrix_bytes);
        return;
    }
    std::vector<QubitId> sorted(qubits.begin(), qubits.end());
    std::sort(sorted.begin(), sorted.end());
    const std::size_t span_start = sorted.front();
    const std::size_t span_len = sorted.back() - span_start + 1;

    // Each its own allocation: the sorted operands, their ranks and a
    // localization's bit map; the localized gate matrix; the SWAP matrix, its
    // localized copy, and that localization's rank list and bit map.
    std::uint64_t held = generated_matrix_bytes;
    held = checked_add(held, heap_block_bytes(k * sizeof(QubitId)));
    held = checked_add(held, 2 * heap_block_bytes(k * sizeof(std::size_t)));
    held = checked_add(
        held, heap_block_bytes(checked_mul(checked_pow2(static_cast<unsigned>(2 * k)), kComplex)));
    held = checked_add(held, 2 * heap_block_bytes(16 * kComplex));
    held = checked_add(held, 2 * heap_block_bytes(2 * sizeof(std::size_t)));
    held = checked_add(held, common::grown_vector_bytes(k * span_len, sizeof(std::size_t)));
    const std::uint64_t gathering = checked_add(held, heap_block_bytes(span_len * sizeof(QubitId)));

    // gather_qubits: walk each operand left to the next open slot.
    std::vector<std::size_t> swap_log;
    std::vector<std::size_t> occupant(span_len);
    for (std::size_t p = 0; p < span_len; ++p) {
        occupant[p] = span_start + p;
    }
    std::size_t insert_at = 0;
    for (std::size_t p = 0; p < span_len; ++p) {
        if (!std::binary_search(sorted.begin(), sorted.end(), occupant[p])) {
            continue;
        }
        for (std::size_t pos = p; pos > insert_at; --pos) {
            const std::size_t site_a = span_start + pos - 1;
            replay.block(site_a, 2, gathering);
            swap_log.push_back(site_a);
            std::swap(occupant[pos - 1], occupant[pos]);
        }
        ++insert_at;
    }

    replay.block(span_start, k, held);

    for (auto it = swap_log.rbegin(); it != swap_log.rend(); ++it) {
        replay.block(*it, 2, held);
    }
}

void replay_operations(Replay& replay, const std::vector<Operation>& operations) {
    for (const Operation& op : operations) {
        std::visit(
            [&](const auto& o) {
                using T = std::decay_t<decltype(o)>;
                if constexpr (std::is_same_v<T, GateOp>) {
                    replay_gate(
                        replay,
                        o.qubits,
                        heap_block_bytes(checked_mul(
                            checked_pow2(static_cast<unsigned>(2 * o.qubits.size())), kComplex)));
                } else if constexpr (std::is_same_v<T, UnitaryOp>) {
                    replay_gate(replay, o.qubits, 0);
                } else if constexpr (std::is_same_v<T, ChannelOp>) {
                    throw common::InternalError("mps: a noise channel reached a pure-state engine");
                } else {
                    static_assert(std::is_same_v<T, MeasureOp>);
                }
            },
            op);
    }
}

}  // namespace

MemoryModel model_execution(const EngineEstimateRequest& request) {
    const std::uint32_t n = request.num_qubits;
    common::ensure(n >= 1, "mps model_execution: num_qubits must be at least 1");

    // Under Controlled the truncation record grows by one double per
    // truncating split, alongside everything else, so a first replay counts
    // the splits and the second carries the record's bound throughout. Under
    // Exact nothing is ever truncated.
    std::uint64_t record = 0;
    if (request.accuracy_policy.kind == AccuracyPolicyKind::Controlled) {
        Replay counting(n, request.accuracy_policy, 0);
        replay_operations(counting, *request.operations);
        record = common::grown_vector_bytes(counting.splits(), sizeof(double));
    }
    Replay replay(n, request.accuracy_policy, record);
    replay_operations(replay, *request.operations);

    MemoryModel model;
    model.final_bonds = replay.bonds();

    std::uint64_t held = 0;

    // The expectation, computed before sampling: per term, the running
    // left environment beside the next one, at every site.
    if (request.want_expectation) {
        held = checked_add(
            held,
            heap_block_bytes(checked_mul(
                static_cast<std::uint64_t>(request.observable->terms.size()), kComplex)));
        std::uint64_t environment = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint64_t l = replay.left(i);
            const std::uint64_t r = replay.right(i);
            environment = std::max(environment,
                                   checked_add(heap_block_bytes(checked_mul(l * l, kComplex)),
                                               heap_block_bytes(checked_mul(r * r, kComplex))));
        }
        replay.note(checked_add(held, environment));
    }

    // Counts: the sampler takes the state, copies the measure targets and
    // moves the center to site 0. A sole region then draws every shot into a
    // histogram; a Factorized region returns the sampler, heap-allocated.
    if (request.want_counts) {
        const std::uint32_t width = request.bitstring_width;
        held = checked_add(
            held, heap_block_bytes(checked_mul(width, sizeof(std::pair<QubitId, ClbitId>))));
        if (!request.is_sole_region) {
            held = checked_add(held, common::heap_block_bytes(sizeof(MPSCountsSampler)));
        }
        replay.move_center_to(0, held);
        if (request.is_sole_region) {
            std::uint64_t widest = 1;
            for (std::size_t i = 0; i < n; ++i) {
                widest = std::max(widest, replay.right(i));
            }
            // Per shot: the running left vector (with its old buffer while it
            // grows), each site's two candidate vectors, the outcome bits and
            // the drawn BitString.
            const std::uint64_t vector = heap_block_bytes(checked_mul(widest, kComplex));
            const std::uint64_t shot = checked_add(
                checked_mul(4, vector),
                checked_add(heap_block_bytes((std::uint64_t{n} + 63) / 64 * sizeof(std::uint64_t)),
                            common::bitstring_heap_bytes(width)));
            const std::uint64_t histogram = common::counts_map_bytes(
                common::distinct_outcomes_bound(request.shots, std::min(n, width)), width);
            replay.note(checked_add(held, checked_add(histogram, shot)));
        }
    }

    model.peak_bytes = replay.peak();
    model.max_bond = replay.max_bond();
    return model;
}

}  // namespace menos::engines::mps
