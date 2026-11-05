// A counting global operator new/delete, linked only into the planner
// benchmark (Phase 24 step 1, docs/adr/0028 section 5). It counts heap
// allocations, bytes requested and the peak of live bytes, and otherwise
// forwards to malloc/free, so it measures the planner's own allocation
// behaviour without valgrind or heaptrack. Never linked into the library.
//
// Live bytes are tracked with malloc_usable_size (glibc) or _msize (the MSVC
// CRT), the size malloc actually reserved, so the peak is the allocator-level
// peak, not the sum of requested sizes. Over-aligned new/delete are not replaced; nothing on the
// planning path uses them.

#include "counting_allocator.h"

#include <malloc.h>

#include <atomic>
#include <cstdlib>
#include <new>

namespace {

std::size_t usable_size(void* p) {
#if defined(_WIN32)
    return _msize(p);
#else
    return malloc_usable_size(p);
#endif
}

std::atomic<std::uint64_t> g_allocations{0};
std::atomic<std::uint64_t> g_bytes_requested{0};
std::atomic<std::int64_t> g_live_bytes{0};
std::atomic<std::int64_t> g_peak_live_bytes{0};

void* counted_alloc(std::size_t size) {
    void* p = std::malloc(size == 0 ? 1 : size);
    if (p == nullptr) {
        throw std::bad_alloc();
    }
    g_allocations.fetch_add(1, std::memory_order_relaxed);
    g_bytes_requested.fetch_add(size, std::memory_order_relaxed);
    const auto usable = static_cast<std::int64_t>(usable_size(p));
    const std::int64_t live = g_live_bytes.fetch_add(usable, std::memory_order_relaxed) + usable;
    std::int64_t peak = g_peak_live_bytes.load(std::memory_order_relaxed);
    while (live > peak &&
           !g_peak_live_bytes.compare_exchange_weak(peak, live, std::memory_order_relaxed)) {
    }
    return p;
}

void counted_free(void* p) noexcept {
    if (p == nullptr) {
        return;
    }
    g_live_bytes.fetch_sub(static_cast<std::int64_t>(usable_size(p)), std::memory_order_relaxed);
    std::free(p);
}

}  // namespace

namespace menos_benchmark {

AllocationSnapshot allocation_snapshot() {
    return AllocationSnapshot{
        g_allocations.load(std::memory_order_relaxed),
        g_bytes_requested.load(std::memory_order_relaxed),
        g_live_bytes.load(std::memory_order_relaxed),
        g_peak_live_bytes.load(std::memory_order_relaxed),
    };
}

void reset_peak_to_current() {
    g_peak_live_bytes.store(g_live_bytes.load(std::memory_order_relaxed),
                            std::memory_order_relaxed);
}

}  // namespace menos_benchmark

// NOLINTBEGIN(misc-new-delete-overloads,cert-dcl54-cpp,hicpp-new-delete-operators)
void* operator new(std::size_t size) {
    return counted_alloc(size);
}
void* operator new[](std::size_t size) {
    return counted_alloc(size);
}
void* operator new(std::size_t size, const std::nothrow_t& /*tag*/) noexcept {
    try {
        return counted_alloc(size);
    } catch (...) {
        return nullptr;
    }
}
void* operator new[](std::size_t size, const std::nothrow_t& /*tag*/) noexcept {
    try {
        return counted_alloc(size);
    } catch (...) {
        return nullptr;
    }
}
void operator delete(void* p) noexcept {
    counted_free(p);
}
void operator delete[](void* p) noexcept {
    counted_free(p);
}
void operator delete(void* p, std::size_t /*size*/) noexcept {
    counted_free(p);
}
void operator delete[](void* p, std::size_t /*size*/) noexcept {
    counted_free(p);
}
void operator delete(void* p, const std::nothrow_t& /*tag*/) noexcept {
    counted_free(p);
}
void operator delete[](void* p, const std::nothrow_t& /*tag*/) noexcept {
    counted_free(p);
}
// NOLINTEND(misc-new-delete-overloads,cert-dcl54-cpp,hicpp-new-delete-operators)
