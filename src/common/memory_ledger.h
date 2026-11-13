#pragma once

#include <cstdint>
#include <utility>

#include "menos/bitstring.h"

// The memory a job's execution actually held, as Result::peak_host_bytes(),
// Result::peak_device_bytes() and Result::peak_disk_bytes() report it (Phase 21,
// docs/adr/0025 section 3.2; the disk space is Phase 23, docs/adr/0027).
//
// The runtime installs one MemoryLedger for each execution attempt, on the
// thread that runs it. Every buffer whose size grows with the problem holds a
// Charge for the bytes it asked for, from its allocation until it is freed or
// handed over to the result: the engines' states, scratch and SVD workspace,
// their samplers and histograms, the tensor network's live tensors, the
// stabilizer tableaux, the Pauli term lists, the device buffers, and the
// runtime's own combination buffers, and the disk tier's scratch file. The
// ledger keeps their running total and its highest point, for host memory,
// device memory and disk space separately.
//
// Charges are made explicitly where the buffers are allocated, rather than
// through an allocator as section 3.2 first proposed: the statevector is moved
// into Result as a plain std::vector, whose type a ledger allocator would
// change, and the kernels' worker threads never allocate, so one thread-local
// installation reaches every charge.
//
// What the ledger leaves out is what does not grow with the problem -- small
// matrices, per-gate bookkeeping, a container's own header -- and the
// allocator's overhead, so its peak is at most what the process allocated
// (the heap tracker in tests/memory checks this on every fixture) and at most
// the plan's estimate. A Charge made with no ledger installed -- in
// calibration, benchmarks, or an engine called directly -- costs one
// thread-local read and records nothing.
namespace menos::common {

class MemoryLedger {
public:
    // Host memory, CUDA device memory, or the scratch file of the disk tier
    // (Phase 23): three separate totals, because a byte in one is not a byte
    // in another and each has its own budget.
    enum class Space { Host, Device, Disk };

    MemoryLedger() = default;
    MemoryLedger(const MemoryLedger&) = delete;
    MemoryLedger& operator=(const MemoryLedger&) = delete;
    MemoryLedger(MemoryLedger&&) = delete;
    MemoryLedger& operator=(MemoryLedger&&) = delete;
    ~MemoryLedger() = default;

    // The highest total of live charges since construction.
    [[nodiscard]] std::uint64_t peak_host_bytes() const;
    [[nodiscard]] std::uint64_t peak_device_bytes() const;
    [[nodiscard]] std::uint64_t peak_disk_bytes() const;
    // The total of the charges live now.
    [[nodiscard]] std::uint64_t live_host_bytes() const;
    [[nodiscard]] std::uint64_t live_device_bytes() const;
    [[nodiscard]] std::uint64_t live_disk_bytes() const;

    // The ledger installed on this thread, or null.
    [[nodiscard]] static MemoryLedger* installed();

    // Installs a ledger on the calling thread for its own lifetime, and puts
    // back whichever was installed before.
    class Scope {
    public:
        explicit Scope(MemoryLedger& ledger);
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        Scope(Scope&&) = delete;
        Scope& operator=(Scope&&) = delete;
        ~Scope();

    private:
        MemoryLedger* previous_;
    };

    // `bytes` of one buffer, charged to the ledger installed on this thread
    // when the Charge is made, until it is destroyed or set again. It lives
    // inside the object that owns the buffer: moving the object moves the
    // charge with it, and copying the object, which copies the buffer,
    // charges the copy's bytes again, to the ledger installed at the time.
    class Charge {
    public:
        // Nothing yet, on the ledger installed now; set() charges it.
        Charge();
        explicit Charge(std::uint64_t bytes, Space space = Space::Host);
        Charge(const Charge& other);
        Charge& operator=(const Charge& other);
        Charge(Charge&& other) noexcept;
        Charge& operator=(Charge&& other) noexcept;
        ~Charge();

        // Replaces the charged figure, for a buffer that grows or shrinks.
        void set(std::uint64_t bytes);
        // Ends the charge early, for a buffer freed before its owner ends.
        void release();

    private:
        MemoryLedger* ledger_ = nullptr;
        Space space_ = Space::Host;
        std::uint64_t bytes_ = 0;
    };

private:
    void add(Space space, std::uint64_t bytes);
    void remove(Space space, std::uint64_t bytes);
    [[nodiscard]] std::uint64_t& live_of(Space space);
    [[nodiscard]] std::uint64_t& peak_of(Space space);

    std::uint64_t live_host_ = 0;
    std::uint64_t peak_host_ = 0;
    std::uint64_t live_device_ = 0;
    std::uint64_t peak_device_ = 0;
    std::uint64_t live_disk_ = 0;
    std::uint64_t peak_disk_ = 0;
};

// The bytes a std::vector's buffer holds.
template <typename Vector>
[[nodiscard]] std::uint64_t buffer_bytes(const Vector& v) {
    return static_cast<std::uint64_t>(v.capacity()) * sizeof(typename Vector::value_type);
}

// The bytes one BitString of bitstring_width bits asks for: its words.
[[nodiscard]] inline std::uint64_t bitstring_words_bytes(std::uint32_t bitstring_width) {
    return (std::uint64_t{bitstring_width} + 63) / 64 * sizeof(std::uint64_t);
}

// The bytes a counts histogram of `entries` outcomes of bitstring_width bits
// asked for: each node -- its value, and the links and colour that libstdc++
// and the MSVC STL both pack into four pointer-sized words ahead of it -- and
// its key's words. The sentinel node is left out.
[[nodiscard]] inline std::uint64_t histogram_bytes(std::uint64_t entries,
                                                   std::uint32_t bitstring_width) {
    return entries * (sizeof(std::pair<const BitString, std::uint64_t>) + 4 * sizeof(void*) +
                      bitstring_words_bytes(bitstring_width));
}

}  // namespace menos::common
