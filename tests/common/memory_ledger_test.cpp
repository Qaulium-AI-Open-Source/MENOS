#include "common/memory_ledger.h"

#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <utility>
#include <vector>

// Phase 21 (docs/adr/0025 section 3.2): the ledger behind
// Result::peak_host_bytes() and peak_device_bytes(), and from Phase 23
// (docs/adr/0027) peak_disk_bytes().

namespace menos::common {
namespace {

using Charge = MemoryLedger::Charge;
using Space = MemoryLedger::Space;

TEST(MemoryLedgerTest, LiveChargesAddUpAndThePeakKeepsTheirHighestTotal) {
    MemoryLedger ledger;
    const MemoryLedger::Scope scope(ledger);
    {
        const Charge a(100);
        {
            const Charge b(50);
            EXPECT_EQ(ledger.live_host_bytes(), 150U);
        }
        EXPECT_EQ(ledger.live_host_bytes(), 100U);
        const Charge c(30);
        EXPECT_EQ(ledger.live_host_bytes(), 130U);
    }
    EXPECT_EQ(ledger.live_host_bytes(), 0U);
    EXPECT_EQ(ledger.peak_host_bytes(), 150U);
}

TEST(MemoryLedgerTest, HostAndDeviceAreSeparate) {
    MemoryLedger ledger;
    const MemoryLedger::Scope scope(ledger);
    {
        const Charge host(64);
        const Charge device(4096, Space::Device);
        EXPECT_EQ(ledger.live_host_bytes(), 64U);
        EXPECT_EQ(ledger.live_device_bytes(), 4096U);
    }
    EXPECT_EQ(ledger.peak_host_bytes(), 64U);
    EXPECT_EQ(ledger.peak_device_bytes(), 4096U);
}

TEST(MemoryLedgerTest, DiskIsAThirdSeparateSpace) {
    MemoryLedger ledger;
    const MemoryLedger::Scope scope(ledger);
    {
        const Charge host(64);
        const Charge device(4096, Space::Device);
        const Charge disk(1U << 20, Space::Disk);
        EXPECT_EQ(ledger.live_host_bytes(), 64U);
        EXPECT_EQ(ledger.live_device_bytes(), 4096U);
        EXPECT_EQ(ledger.live_disk_bytes(), 1U << 20);
        {
            const Charge window(2U << 20, Space::Disk);
            EXPECT_EQ(ledger.live_disk_bytes(), 3U << 20);
        }
        EXPECT_EQ(ledger.live_disk_bytes(), 1U << 20);
    }
    EXPECT_EQ(ledger.live_disk_bytes(), 0U);
    EXPECT_EQ(ledger.peak_host_bytes(), 64U);
    EXPECT_EQ(ledger.peak_device_bytes(), 4096U);
    EXPECT_EQ(ledger.peak_disk_bytes(), 3U << 20);
}

TEST(MemoryLedgerTest, SetReplacesTheFigureAndReleaseEndsIt) {
    MemoryLedger ledger;
    const MemoryLedger::Scope scope(ledger);
    Charge growing;
    growing.set(10);
    growing.set(40);
    EXPECT_EQ(ledger.live_host_bytes(), 40U);
    growing.set(20);
    EXPECT_EQ(ledger.live_host_bytes(), 20U);
    growing.release();
    EXPECT_EQ(ledger.live_host_bytes(), 0U);
    growing.release();  // a second release is a no-op
    EXPECT_EQ(ledger.live_host_bytes(), 0U);
    EXPECT_EQ(ledger.peak_host_bytes(), 40U);
}

TEST(MemoryLedgerTest, AMovedChargeMovesWithItsOwner) {
    MemoryLedger ledger;
    const MemoryLedger::Scope scope(ledger);
    Charge first(70);
    Charge second(std::move(first));
    EXPECT_EQ(ledger.live_host_bytes(), 70U);
    Charge third;
    third = std::move(second);
    EXPECT_EQ(ledger.live_host_bytes(), 70U);
    third.release();
    EXPECT_EQ(ledger.live_host_bytes(), 0U);
}

TEST(MemoryLedgerTest, ACopiedChargeChargesTheCopiedBufferAgain) {
    MemoryLedger ledger;
    const MemoryLedger::Scope scope(ledger);
    const Charge original(25);
    {
        const Charge copy(original);  // NOLINT(performance-unnecessary-copy-initialization)
        EXPECT_EQ(ledger.live_host_bytes(), 50U);
        Charge assigned(5);
        assigned = original;
        EXPECT_EQ(ledger.live_host_bytes(), 75U);
    }
    EXPECT_EQ(ledger.live_host_bytes(), 25U);
    EXPECT_EQ(ledger.peak_host_bytes(), 75U);
}

TEST(MemoryLedgerTest, WithNoLedgerInstalledAChargeRecordsNothing) {
    ASSERT_EQ(MemoryLedger::installed(), nullptr);
    Charge charge(1000);
    charge.set(2000);
    MemoryLedger ledger;
    {
        // Charged before this ledger was installed, so not on it.
        const MemoryLedger::Scope scope(ledger);
        charge.set(3000);
    }
    EXPECT_EQ(ledger.peak_host_bytes(), 0U);
}

TEST(MemoryLedgerTest, ScopesNestAndRestoreTheOuterLedger) {
    MemoryLedger outer;
    MemoryLedger inner;
    const MemoryLedger::Scope outer_scope(outer);
    {
        const MemoryLedger::Scope inner_scope(inner);
        EXPECT_EQ(MemoryLedger::installed(), &inner);
        const Charge charge(8);
    }
    EXPECT_EQ(MemoryLedger::installed(), &outer);
    EXPECT_EQ(inner.peak_host_bytes(), 8U);
    EXPECT_EQ(outer.peak_host_bytes(), 0U);
}

TEST(MemoryLedgerTest, BufferAndHistogramFiguresAreTheBytesAskedFor) {
    std::vector<std::complex<double>> amplitudes(16);
    EXPECT_EQ(buffer_bytes(amplitudes), 16 * sizeof(std::complex<double>));
    EXPECT_EQ(bitstring_words_bytes(0), 0U);
    EXPECT_EQ(bitstring_words_bytes(64), 8U);
    EXPECT_EQ(bitstring_words_bytes(65), 16U);
    EXPECT_EQ(histogram_bytes(3, 65),
              3 * (sizeof(std::pair<const BitString, std::uint64_t>) + 4 * sizeof(void*) + 16));
}

}  // namespace
}  // namespace menos::common
