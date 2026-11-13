#include "heap_tracker.h"

// MENOS_HEAP_TRACKER_DISABLED: menos_memory_probe measures the real allocator,
// so it links this file without the replacement.
#if defined(MENOS_HEAP_TRACKER_DISABLED) || defined(__SANITIZE_ADDRESS__)
#define MENOS_HEAP_TRACKER_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define MENOS_HEAP_TRACKER_ASAN 1
#endif
#endif

#ifndef MENOS_HEAP_TRACKER_ASAN

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>

// NOLINTBEGIN(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory,cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr,misc-new-delete-overloads)

namespace menos::test::heap_tracker {
namespace {

std::atomic<std::uint64_t> g_live{0};
std::atomic<std::uint64_t> g_peak{0};

// Stored immediately before every returned pointer, so every delete form finds
// the size and the block malloc returned without knowing the alignment it was
// allocated with.
struct Header {
    void* raw;
    std::size_t size;
};

constexpr std::size_t kMinAlignment = 16;

void record_allocation(std::size_t size) {
    const std::uint64_t live = g_live.fetch_add(size, std::memory_order_relaxed) + size;
    std::uint64_t peak = g_peak.load(std::memory_order_relaxed);
    while (live > peak && !g_peak.compare_exchange_weak(peak, live, std::memory_order_relaxed)) {
    }
}

void* allocate(std::size_t size, std::size_t alignment) noexcept {
    if (alignment < kMinAlignment) {
        alignment = kMinAlignment;
    }
    const std::size_t total = size + sizeof(Header) + alignment;
    void* raw = std::malloc(total);
    if (raw == nullptr) {
        return nullptr;
    }
    const auto base = reinterpret_cast<std::uintptr_t>(raw) + sizeof(Header);
    const std::uintptr_t aligned = (base + alignment - 1) & ~(std::uintptr_t{alignment} - 1);
    auto* header = reinterpret_cast<Header*>(aligned - sizeof(Header));
    header->raw = raw;
    header->size = size;
    record_allocation(size);
    return reinterpret_cast<void*>(aligned);
}

void* allocate_or_throw(std::size_t size, std::size_t alignment) {
    void* p = allocate(size, alignment);
    if (p == nullptr) {
        throw std::bad_alloc();
    }
    return p;
}

void release(void* p) noexcept {
    if (p == nullptr) {
        return;
    }
    const auto* header =
        reinterpret_cast<const Header*>(reinterpret_cast<std::uintptr_t>(p) - sizeof(Header));
    g_live.fetch_sub(header->size, std::memory_order_relaxed);
    std::free(header->raw);
}

}  // namespace

bool available() {
    return true;
}

std::uint64_t live_bytes() {
    return g_live.load(std::memory_order_relaxed);
}

std::uint64_t peak_bytes() {
    return g_peak.load(std::memory_order_relaxed);
}

void reset_peak() {
    g_peak.store(g_live.load(std::memory_order_relaxed), std::memory_order_relaxed);
}

}  // namespace menos::test::heap_tracker

namespace tracker = menos::test::heap_tracker;

void* operator new(std::size_t size) {
    return tracker::allocate_or_throw(size, __STDCPP_DEFAULT_NEW_ALIGNMENT__);
}
void* operator new[](std::size_t size) {
    return tracker::allocate_or_throw(size, __STDCPP_DEFAULT_NEW_ALIGNMENT__);
}
void* operator new(std::size_t size, const std::nothrow_t& /*tag*/) noexcept {
    return tracker::allocate(size, __STDCPP_DEFAULT_NEW_ALIGNMENT__);
}
void* operator new[](std::size_t size, const std::nothrow_t& /*tag*/) noexcept {
    return tracker::allocate(size, __STDCPP_DEFAULT_NEW_ALIGNMENT__);
}
void* operator new(std::size_t size, std::align_val_t alignment) {
    return tracker::allocate_or_throw(size, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
    return tracker::allocate_or_throw(size, static_cast<std::size_t>(alignment));
}
void* operator new(std::size_t size,
                   std::align_val_t alignment,
                   const std::nothrow_t& /*tag*/) noexcept {
    return tracker::allocate(size, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t size,
                     std::align_val_t alignment,
                     const std::nothrow_t& /*tag*/) noexcept {
    return tracker::allocate(size, static_cast<std::size_t>(alignment));
}

void operator delete(void* p) noexcept {
    tracker::release(p);
}
void operator delete[](void* p) noexcept {
    tracker::release(p);
}
void operator delete(void* p, const std::nothrow_t& /*tag*/) noexcept {
    tracker::release(p);
}
void operator delete[](void* p, const std::nothrow_t& /*tag*/) noexcept {
    tracker::release(p);
}
void operator delete(void* p, std::size_t /*size*/) noexcept {
    tracker::release(p);
}
void operator delete[](void* p, std::size_t /*size*/) noexcept {
    tracker::release(p);
}
void operator delete(void* p, std::align_val_t /*alignment*/) noexcept {
    tracker::release(p);
}
void operator delete[](void* p, std::align_val_t /*alignment*/) noexcept {
    tracker::release(p);
}
void operator delete(void* p, std::size_t /*size*/, std::align_val_t /*alignment*/) noexcept {
    tracker::release(p);
}
void operator delete[](void* p, std::size_t /*size*/, std::align_val_t /*alignment*/) noexcept {
    tracker::release(p);
}
void operator delete(void* p,
                     std::align_val_t /*alignment*/,
                     const std::nothrow_t& /*tag*/) noexcept {
    tracker::release(p);
}
void operator delete[](void* p,
                       std::align_val_t /*alignment*/,
                       const std::nothrow_t& /*tag*/) noexcept {
    tracker::release(p);
}

// NOLINTEND(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory,cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr,misc-new-delete-overloads)

#else  // MENOS_HEAP_TRACKER_ASAN

namespace menos::test::heap_tracker {

bool available() {
    return false;
}
std::uint64_t live_bytes() {
    return 0;
}
std::uint64_t peak_bytes() {
    return 0;
}
void reset_peak() {}

}  // namespace menos::test::heap_tracker

#endif  // MENOS_HEAP_TRACKER_ASAN
