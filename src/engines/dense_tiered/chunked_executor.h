#pragma once

#include <complex>
#include <cstdint>
#include <map>
#include <span>
#include <utility>
#include <vector>

#include "common/memory_ledger.h"
#include "engines/dense_tiered/file_backing_store.h"
#include "engines/dense_tiered/pass_plan.h"
#include "ir/fuse_gates.h"
#include "menos/bitstring.h"
#include "menos/circuit.h"
#include "menos/simulation_engine.h"

// The chunked pass executor (Phase 23, docs/adr/0027 section 3.2): the dense
// algorithm over a state that lives in a FileBackingStore, run through a
// window of 2^W amplitudes in host memory. Every output is bitwise the
// in-memory DenseEngine's for the same request, because each gate multiplies
// the same amplitudes by the same matrix in the same order whichever window
// slot they sit in, the expectation accumulates the same products in the same
// order, and the counts come from the same sort-then-walk over the same
// probabilities (sections 3.2 and 3.4).
namespace menos::engines::dense_tiered {

// The window: 2^window_qubits amplitudes, sector-aligned for the store's
// unbuffered I/O, charged to the installed ledger's host space. Slot s of
// chunk size C is amplitudes [s * C, (s + 1) * C).
class Window {
public:
    // Throws common::ResourceExceeded naming the bytes if the allocation
    // fails.
    explicit Window(std::uint32_t window_qubits);

    [[nodiscard]] std::uint32_t window_qubits() const;
    // The bytes the window holds, as charged: what the estimate counts for it.
    [[nodiscard]] std::uint64_t size_bytes() const;
    [[nodiscard]] static std::uint64_t heap_bytes(std::uint32_t window_qubits);
    [[nodiscard]] std::span<std::complex<double>> amplitudes();
    [[nodiscard]] std::span<std::complex<double>> slot(std::uint64_t index,
                                                       std::uint64_t chunk_amplitudes);

private:
    std::uint32_t window_qubits_;
    std::vector<std::complex<double>> storage_;
    std::span<std::complex<double>> aligned_;
    common::MemoryLedger::Charge charge_;
};

// Writes exp(i * global_phase) |0...0> to the store, chunk by chunk, through
// the window's first slot: the same amplitude kernels::cpu::initialize puts
// in an in-memory state.
void initialize_store(const FileBackingStore& store, Window& window, double global_phase);

// Runs every pass of `plan` over `operations`, or with `fusion` over its
// blocks (Phase 24 item 7, docs/adr/0028 section 17: a fused block's matrix
// is engines/dense/fused_block.h's, the same bytes the in-memory engines
// apply). For each pass and each group
// of chunks its global set selects, the group's chunks are read into the
// window's slots, the pass's gates are applied to the window with their
// operands remapped (a local qubit keeps its bit; global qubit g of the set
// becomes window bit chunk_qubits + its rank in the set), and the chunks are
// written back. The monitor, if any, is called once at the end of each pass,
// the executor's safe boundary, with the number of operations applied so far;
// a throw there leaves the store to its own destructor.
void run_passes(const FileBackingStore& store,
                Window& window,
                const PassPlan& plan,
                std::span<const Operation> operations,
                const ExecutionMonitor* monitor,
                std::uint32_t max_threads,
                const ir::FusionPlan* fusion = nullptr);

// The counts of `shots` draws from the stored state, read chunk by chunk
// through the window's first slot into engines::dense::sample_streamed.
[[nodiscard]] std::map<BitString, std::uint64_t> sample_store(
    const FileBackingStore& store,
    Window& window,
    std::uint64_t shots,
    std::uint64_t seed,
    std::span<const std::pair<std::uint32_t, std::uint32_t>> measure_targets,
    std::uint32_t bitstring_width);

// <psi|P|psi> for the term (x_mask, z_mask) over the stored state, chunk by
// chunk through kernels::cpu::expectation_of_term_accumulate: each chunk in
// the first slot, and the chunk its X mask's high bits pair it with in the
// second when they differ (which needs a window of at least two chunks).
[[nodiscard]] std::complex<double> expectation_of_term(const FileBackingStore& store,
                                                       Window& window,
                                                       std::uint64_t x_mask,
                                                       std::uint64_t z_mask);

// The bits of `value`, lowest first, placed at the set bits of `mask`, lowest
// first (a portable PDEP). Public for the tests of the chunk addressing.
[[nodiscard]] std::uint64_t deposit_bits(std::uint64_t value, std::uint64_t mask);

}  // namespace menos::engines::dense_tiered
