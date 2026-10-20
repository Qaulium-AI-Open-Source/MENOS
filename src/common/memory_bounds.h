#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>

#include "common/checked_math.h"
#include "menos/bitstring.h"

// Upper bounds on the heap bytes of the containers every engine's estimate()
// charges (Phase 21, docs/adr/0025 section 3.4). Each is derived from the
// container's own layout (sizeof and the allocation the standard library
// makes), never from a tuned constant, and each is checked against the heap
// tracker and the process's peak RSS by tests/memory on libstdc++ and the
// MSVC STL. Arithmetic is checked: an overflow throws common::ResourceExceeded,
// which every estimate() already reports as infeasible.
namespace menos::common {

// What one allocation can cost beyond the bytes a container asks for, charged
// on every allocation an estimate counts (Phase 21, docs/adr/0025 section
// 3.4), as the sum of two measured, documented parts:
//   * the standard library's own request beyond the size: none for libstdc++;
//     the MSVC STL asks for size + sizeof(void*) + 31 for any allocation of
//     4096 bytes or more, to align it to 32 bytes (<xmemory>,
//     _Big_allocation_threshold), plus one more pointer in a Debug build --
//     at most 47 (measured: 39 on a 4 MiB statevector, Release);
//   * the heap's own cost: its header (8 bytes for glibc's malloc, at most 16
//     for the Windows heap) and the padding to its 16-byte granularity (at
//     most 15) -- at most 31.
// 47 + 31 = 78, rounded up to the heap's 16-byte granularity.
inline constexpr std::uint64_t kHeapBlockOverheadBytes = 80;

// One allocation of `requested` bytes, with its overhead; none for 0 bytes.
[[nodiscard]] inline std::uint64_t heap_block_bytes(std::uint64_t requested) {
    return requested == 0 ? 0 : checked_add(requested, kHeapBlockOverheadBytes);
}

// One node of a std::map or std::set holding `value_bytes`: the value, plus
// the links and colour that both libstdc++ and the MSVC STL pack into four
// pointer-sized words, as one allocation.
[[nodiscard]] inline std::uint64_t tree_node_bytes(std::uint64_t value_bytes) {
    return heap_block_bytes(checked_add(value_bytes, 4 * sizeof(void*)));
}

// The heap words of one BitString of bitstring_width bits, one allocation (its
// inline part is counted wherever the BitString itself lives).
[[nodiscard]] inline std::uint64_t bitstring_heap_bytes(std::uint32_t bitstring_width) {
    return heap_block_bytes(
        checked_mul((std::uint64_t{bitstring_width} + 63) / 64, sizeof(std::uint64_t)));
}

// One std::map<BitString, std::uint64_t> node, and its key's heap words.
[[nodiscard]] inline std::uint64_t counts_node_bytes(std::uint32_t bitstring_width) {
    return checked_add(tree_node_bytes(sizeof(std::pair<const BitString, std::uint64_t>)),
                       bitstring_heap_bytes(bitstring_width));
}

// The largest number of distinct outcomes `shots` draws over `measured_qubits`
// qubits can produce: min(shots, 2^measured_qubits).
[[nodiscard]] inline std::uint64_t distinct_outcomes_bound(std::uint64_t shots,
                                                           std::uint32_t measured_qubits) {
    if (measured_qubits >= 63) {
        return shots;
    }
    return std::min(shots, std::uint64_t{1} << measured_qubits);
}

// A counts histogram of up to `distinct` outcomes, plus one more node: the MSVC
// STL allocates a sentinel node when the map is constructed, and an insertion
// builds its key before the node exists (one BitString's words in flight).
[[nodiscard]] inline std::uint64_t counts_map_bytes(std::uint64_t distinct,
                                                    std::uint32_t bitstring_width) {
    return checked_mul(checked_add(distinct, 1), counts_node_bytes(bitstring_width));
}

// A std::vector grown one push_back at a time to `count` elements, at the
// moment of its last reallocation: the old buffer and the new one are both
// live, two allocations. Both libstdc++ (x2) and the MSVC STL (x1.5) grow by at
// most a factor of two, so the new capacity is below 2 * count and the old one
// below count.
[[nodiscard]] inline std::uint64_t grown_vector_bytes(std::uint64_t count,
                                                      std::uint64_t element_bytes) {
    if (count == 0) {
        return 0;
    }
    return checked_add(checked_mul(checked_mul(count, 3), element_bytes),
                       2 * kHeapBlockOverheadBytes);
}

}  // namespace menos::common
