#pragma once

#include <cstdint>

// A test-only replacement of the global operator new/delete that counts the
// bytes every C++ allocation in this process requests (docs/adr/0025 section
// 3.1). Linked only into menos_memory_tests and menos_memory_probe: replacing
// operator new is per executable, and no production target may do it.
//
// Counts requested sizes, not the allocator's own per-block overhead, and sees
// nothing allocated with malloc (LAPACKE's workspace, OpenBLAS's buffers) or on
// a device. The peak-RSS probe covers those.
//
// Under AddressSanitizer (the dev and windows-dev presets) the sanitizer owns
// operator new, so no replacement is compiled and available() is false.
namespace menos::test::heap_tracker {

[[nodiscard]] bool available();

// Bytes currently allocated through operator new and not yet freed.
[[nodiscard]] std::uint64_t live_bytes();

// The largest live_bytes() since the last reset_peak().
[[nodiscard]] std::uint64_t peak_bytes();

// Sets the peak to the current live figure.
void reset_peak();

}  // namespace menos::test::heap_tracker
