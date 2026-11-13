#include "common/memory_ledger.h"

#include <algorithm>
#include <utility>

namespace menos::common {
namespace {

// One per thread: an execution attempt runs on one thread, and only its
// thread allocates the buffers it charges.
thread_local MemoryLedger* installed_ledger = nullptr;

}  // namespace

std::uint64_t MemoryLedger::peak_host_bytes() const {
    return peak_host_;
}

std::uint64_t MemoryLedger::peak_device_bytes() const {
    return peak_device_;
}

std::uint64_t MemoryLedger::live_host_bytes() const {
    return live_host_;
}

std::uint64_t MemoryLedger::live_device_bytes() const {
    return live_device_;
}

std::uint64_t MemoryLedger::peak_disk_bytes() const {
    return peak_disk_;
}

std::uint64_t MemoryLedger::live_disk_bytes() const {
    return live_disk_;
}

MemoryLedger* MemoryLedger::installed() {
    return installed_ledger;
}

std::uint64_t& MemoryLedger::live_of(Space space) {
    switch (space) {
        case Space::Host:
            return live_host_;
        case Space::Device:
            return live_device_;
        case Space::Disk:
            return live_disk_;
    }
    return live_host_;  // unreachable: every Space is a case above
}

std::uint64_t& MemoryLedger::peak_of(Space space) {
    switch (space) {
        case Space::Host:
            return peak_host_;
        case Space::Device:
            return peak_device_;
        case Space::Disk:
            return peak_disk_;
    }
    return peak_host_;  // unreachable: every Space is a case above
}

void MemoryLedger::add(Space space, std::uint64_t bytes) {
    std::uint64_t& live = live_of(space);
    live += bytes;
    std::uint64_t& peak = peak_of(space);
    peak = std::max(peak, live);
}

void MemoryLedger::remove(Space space, std::uint64_t bytes) {
    live_of(space) -= bytes;
}

MemoryLedger::Scope::Scope(MemoryLedger& ledger) : previous_(installed_ledger) {
    installed_ledger = &ledger;
}

MemoryLedger::Scope::~Scope() {
    installed_ledger = previous_;
}

MemoryLedger::Charge::Charge() : ledger_(installed_ledger) {}

MemoryLedger::Charge::Charge(std::uint64_t bytes, Space space)
    : ledger_(installed_ledger), space_(space) {
    set(bytes);
}

MemoryLedger::Charge::Charge(const Charge& other) : Charge(other.bytes_, other.space_) {}

MemoryLedger::Charge& MemoryLedger::Charge::operator=(const Charge& other) {
    if (this != &other) {
        // The owner's old buffer is replaced by a copy of other's: charged as
        // a replacement, never as both at once.
        release();
        if (ledger_ == nullptr) {
            ledger_ = installed_ledger;
        }
        space_ = other.space_;
        set(other.bytes_);
    }
    return *this;
}

MemoryLedger::Charge::Charge(Charge&& other) noexcept
    : ledger_(std::exchange(other.ledger_, nullptr)),
      space_(other.space_),
      bytes_(std::exchange(other.bytes_, 0)) {}

MemoryLedger::Charge& MemoryLedger::Charge::operator=(Charge&& other) noexcept {
    if (this != &other) {
        release();
        ledger_ = std::exchange(other.ledger_, nullptr);
        space_ = other.space_;
        bytes_ = std::exchange(other.bytes_, 0);
    }
    return *this;
}

MemoryLedger::Charge::~Charge() {
    release();
}

void MemoryLedger::Charge::set(std::uint64_t bytes) {
    if (ledger_ == nullptr) {
        return;
    }
    // Removed before added, so a buffer replaced by a larger one is not
    // counted twice -- the caller frees the old one first or charges it
    // separately.
    ledger_->remove(space_, bytes_);
    bytes_ = bytes;
    ledger_->add(space_, bytes_);
}

void MemoryLedger::Charge::release() {
    if (ledger_ != nullptr) {
        ledger_->remove(space_, bytes_);
        bytes_ = 0;
    }
}

}  // namespace menos::common
