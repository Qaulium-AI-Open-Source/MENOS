#include "engines/dense_tiered/chunked_executor.h"

#include <bit>
#include <cstdint>
#include <functional>
#include <new>
#include <string>
#include <type_traits>
#include <variant>

#include "common/checked_math.h"
#include "common/error.h"
#include "common/memory_bounds.h"
#include "engines/dense/apply_matrix.h"
#include "engines/dense/fused_block.h"
#include "engines/dense/sample.h"
#include "ir/gate_table.h"
#include "kernels/dense_kernels.h"

namespace menos::engines::dense_tiered {

using Complex = std::complex<double>;

// ---------------------------------------------------------------------------
// Window
// ---------------------------------------------------------------------------

namespace {

constexpr std::uint64_t kAlignmentAmplitudes = FileBackingStore::kAlignment / sizeof(Complex);

}  // namespace

std::uint64_t Window::heap_bytes(std::uint32_t window_qubits) {
    // The vector holds the window plus one alignment's worth of slack, so a
    // sector-aligned span of the full size fits wherever the heap puts it.
    return common::heap_block_bytes(common::checked_mul(
        common::checked_add(common::checked_pow2(window_qubits), kAlignmentAmplitudes),
        sizeof(Complex)));
}

Window::Window(std::uint32_t window_qubits) : window_qubits_(window_qubits) {
    const std::uint64_t amplitudes = common::checked_pow2(window_qubits);
    try {
        storage_.resize(amplitudes + kAlignmentAmplitudes);
    } catch (const std::bad_alloc&) {
        throw common::ResourceExceeded("the disk tier's window of " +
                                       std::to_string(Window::heap_bytes(window_qubits)) +
                                       " bytes could not be allocated");
    }
    const auto address = reinterpret_cast<std::uintptr_t>(storage_.data());  // NOLINT
    const std::uintptr_t misalignment = address % FileBackingStore::kAlignment;
    const std::uintptr_t skip =
        misalignment == 0 ? 0 : (FileBackingStore::kAlignment - misalignment) / sizeof(Complex);
    aligned_ = std::span<Complex>(storage_.data() + skip, amplitudes);
    charge_ = common::MemoryLedger::Charge(common::buffer_bytes(storage_));
}

std::uint32_t Window::window_qubits() const {
    return window_qubits_;
}

std::uint64_t Window::size_bytes() const {
    return common::buffer_bytes(storage_);
}

std::span<Complex> Window::amplitudes() {
    return aligned_;
}

std::span<Complex> Window::slot(std::uint64_t index, std::uint64_t chunk_amplitudes) {
    return aligned_.subspan(index * chunk_amplitudes, chunk_amplitudes);
}

// ---------------------------------------------------------------------------
// The passes
// ---------------------------------------------------------------------------

// A value and a mask: distinct roles, named at every call site.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::uint64_t deposit_bits(std::uint64_t value, std::uint64_t mask) {
    std::uint64_t result = 0;
    for (std::uint64_t rest = mask, bit = 0; rest != 0; rest &= rest - 1, ++bit) {
        if (((value >> bit) & std::uint64_t{1}) != 0) {
            result |= rest & (~rest + 1);  // the lowest set bit of rest
        }
    }
    return result;
}

void initialize_store(const FileBackingStore& store, Window& window, double global_phase) {
    const std::span<Complex> slot = window.slot(0, store.chunk_amplitudes());
    // The first chunk holds the one non-zero amplitude, from the same kernel
    // the in-memory engine initialises with.
    kernels::cpu::initialize(slot, global_phase);
    store.write(0, slot);
    slot[0] = Complex{0.0, 0.0};
    for (std::uint64_t chunk = 1; chunk < store.chunk_count(); ++chunk) {
        store.write(chunk, slot);
    }
}

namespace {

// The window bit a region-local qubit takes inside a pass whose global set
// is `global_mask`: its own bit for a local qubit, chunk_qubits plus its
// rank among the set's qubits for a global one.
std::uint32_t window_bit(QubitId qubit, std::uint32_t chunk_qubits, std::uint64_t global_mask) {
    if (qubit < chunk_qubits) {
        return qubit;
    }
    const std::uint64_t below = (std::uint64_t{1} << (qubit - chunk_qubits)) - 1;
    return chunk_qubits + static_cast<std::uint32_t>(std::popcount(global_mask & below));
}

}  // namespace

void run_passes(const FileBackingStore& store,
                Window& window,
                const PassPlan& plan,
                std::span<const Operation> operations,
                const ExecutionMonitor* monitor,
                std::uint32_t max_threads,
                const ir::FusionPlan* fusion) {
    const std::uint32_t chunk_qubits = plan.chunk_qubits;
    const std::uint64_t chunk = store.chunk_amplitudes();
    common::ensure(
        chunk == (std::uint64_t{1} << chunk_qubits) && window.window_qubits() >= plan.window_qubits,
        "run_passes: the store, the window and the plan disagree");
    const std::uint32_t global_bits = plan.num_qubits - chunk_qubits;
    const std::uint64_t all_globals = (std::uint64_t{1} << global_bits) - 1;

    std::vector<std::uint32_t> operands;
    for (const Pass& pass : plan.passes) {
        const std::uint64_t set = pass.global_mask;
        const auto in_set = static_cast<std::uint32_t>(std::popcount(set));
        const std::uint64_t rest = all_globals & ~set;
        const std::uint64_t groups = std::uint64_t{1} << (global_bits - in_set);
        const std::uint64_t slots = std::uint64_t{1} << in_set;
        const std::span<Complex> used = window.amplitudes().subspan(0, chunk << in_set);

        for (std::uint64_t group = 0; group < groups; ++group) {
            const std::uint64_t base = deposit_bits(group, rest);
            for (std::uint64_t s = 0; s < slots; ++s) {
                store.read(base | deposit_bits(s, set), window.slot(s, chunk));
            }
            // A gate's operands remapped into the window, then the kernel.
            const auto apply_in_window = [&](std::span<const Complex> matrix,
                                             std::span<const QubitId> qubits) {
                operands.clear();
                for (const QubitId q : qubits) {
                    operands.push_back(window_bit(q, chunk_qubits, set));
                }
                dense::apply_matrix(used, matrix, operands, max_threads);
            };
            if (fusion != nullptr) {
                // The pass's units are blocks; apply each as the in-memory
                // engines do, with its own matrix or its gate's.
                for (std::size_t b = pass.first_operation; b < pass.end_operation; ++b) {
                    const std::span<const std::uint32_t> members = fusion->block_operations(b);
                    if (members.empty() || fusion->block_operands(b).empty()) {
                        continue;  // a barrier block applies nothing
                    }
                    if (members.size() == 1) {
                        std::visit(
                            [&](const auto& op) {
                                using T = std::decay_t<decltype(op)>;
                                if constexpr (std::is_same_v<T, GateOp>) {
                                    const std::vector<Complex> matrix =
                                        ir::generate_matrix(op.opcode, op.params);
                                    apply_in_window(matrix, op.qubits);
                                } else if constexpr (std::is_same_v<T, UnitaryOp>) {
                                    apply_in_window(op.matrix, op.qubits);
                                } else {
                                    throw common::InternalError(
                                        "disk tier: a block member is not a gate");
                                }
                            },
                            operations[members[0]]);
                    } else {
                        const std::vector<Complex> matrix =
                            dense::fused_block_matrix(*fusion, b, operations);
                        apply_in_window(matrix, fusion->block_operands(b));
                    }
                }
            }
            for (std::size_t i = fusion != nullptr ? pass.end_operation : pass.first_operation;
                 i < pass.end_operation;
                 ++i) {
                std::visit(
                    [&](const auto& op) {
                        using T = std::decay_t<decltype(op)>;
                        if constexpr (std::is_same_v<T, GateOp> || std::is_same_v<T, UnitaryOp>) {
                            operands.clear();
                            for (const QubitId q : op.qubits) {
                                operands.push_back(window_bit(q, chunk_qubits, set));
                            }
                            if constexpr (std::is_same_v<T, GateOp>) {
                                const std::vector<Complex> matrix =
                                    ir::generate_matrix(op.opcode, op.params);
                                dense::apply_matrix(used, matrix, operands, max_threads);
                            } else {
                                dense::apply_matrix(used, op.matrix, operands, max_threads);
                            }
                        } else if constexpr (std::is_same_v<T, ChannelOp>) {
                            // Unreachable: estimate() refuses a noisy region.
                            throw common::InternalError(
                                "disk tier: a noise channel reached a pure-state engine");
                        } else {
                            static_assert(std::is_same_v<T, MeasureOp>);
                            // Outcomes are resolved by the sampling pass.
                        }
                    },
                    operations[i]);
            }
            for (std::uint64_t s = 0; s < slots; ++s) {
                store.write(base | deposit_bits(s, set), window.slot(s, chunk));
            }
        }
        // The pass's safe boundary: every chunk holds the state after its
        // last operation, and nothing has been handed to the caller.
        if (monitor != nullptr) {
            monitor->at_safe_boundary(pass.end_operation);
        }
    }
}

// ---------------------------------------------------------------------------
// The outputs
// ---------------------------------------------------------------------------

std::map<BitString, std::uint64_t> sample_store(
    const FileBackingStore& store,
    Window& window,
    std::uint64_t shots,
    std::uint64_t seed,
    std::span<const std::pair<std::uint32_t, std::uint32_t>> measure_targets,
    std::uint32_t bitstring_width) {
    const std::uint64_t chunk = store.chunk_amplitudes();
    const std::function<std::span<const Complex>(std::uint64_t)> read_chunk =
        [&store, &window, chunk](std::uint64_t index) {
            const std::span<Complex> slot = window.slot(0, chunk);
            store.read(index, slot);
            return std::span<const Complex>(slot);
        };
    return dense::sample_streamed(store.chunk_count() * chunk,
                                  chunk,
                                  read_chunk,
                                  shots,
                                  seed,
                                  measure_targets,
                                  bitstring_width);
}

std::complex<double> expectation_of_term(const FileBackingStore& store,
                                         Window& window,
                                         std::uint64_t x_mask,
                                         std::uint64_t z_mask) {
    const std::uint64_t chunk = store.chunk_amplitudes();
    const auto chunk_qubits = static_cast<std::uint32_t>(std::countr_zero(chunk));
    const std::uint64_t local_mask = chunk - 1;
    const std::uint64_t x_high = x_mask >> chunk_qubits;
    const std::uint64_t z_high = z_mask >> chunk_qubits;
    common::ensure(x_high == 0 || window.window_qubits() > chunk_qubits,
                   "expectation_of_term: a paired term needs a window of two chunks");
    const std::span<Complex> own = window.slot(0, chunk);
    Complex acc{0.0, 0.0};
    for (std::uint64_t index = 0; index < store.chunk_count(); ++index) {
        store.read(index, own);
        const std::uint64_t partner_index = index ^ x_high;
        std::span<const Complex> partner = own;
        if (partner_index != index) {
            const std::span<Complex> other = window.slot(1, chunk);
            store.read(partner_index, other);
            partner = other;
        }
        // The sign the masks' high bits give every amplitude of this chunk:
        // (-1)^popcount(z & (m ^ x)) over the chunk-number bits.
        const bool global_negative = (std::popcount(z_high & partner_index) & 1) != 0;
        kernels::cpu::expectation_of_term_accumulate(
            own, partner, x_mask & local_mask, z_mask & local_mask, global_negative, acc);
    }
    return kernels::cpu::expectation_term_phase(x_mask, z_mask) * acc;
}

}  // namespace menos::engines::dense_tiered
