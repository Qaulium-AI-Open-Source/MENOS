// Phase 23 (docs/adr/0027 section 3.3): the disk tier's scratch file. Whole
// chunks round-trip exactly, the file is charged to the ledger's disk space
// and gone the moment the store is, and a short directory is refused with
// the figures named. The scratch directory is the test's working directory:
// the build tree, a real disk on every platform (WSL's /tmp is a tmpfs that
// refuses O_DIRECT, so temp_directory_path() would test the buffered path
// only).

#include "engines/dense_tiered/file_backing_store.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include "common/error.h"
#include "common/memory_ledger.h"

namespace menos::engines::dense_tiered {
namespace {

using Complex = std::complex<double>;

// A buffer whose start is a multiple of FileBackingStore::kAlignment, as the
// executor's window provides.
class AlignedChunk {
public:
    explicit AlignedChunk(std::uint64_t amplitudes)
        : storage_(amplitudes + FileBackingStore::kAlignment / sizeof(Complex)) {
        auto address = reinterpret_cast<std::uintptr_t>(storage_.data());  // NOLINT
        const std::uintptr_t aligned = (address + FileBackingStore::kAlignment - 1) &
                                       ~std::uintptr_t{FileBackingStore::kAlignment - 1};
        const std::size_t skip = static_cast<std::size_t>(aligned - address) / sizeof(Complex);
        span_ = std::span<Complex>(storage_.data() + skip, amplitudes);
    }
    [[nodiscard]] std::span<Complex> span() {
        return span_;
    }

private:
    std::vector<Complex> storage_;
    std::span<Complex> span_;
};

std::filesystem::path scratch_directory() {
    return std::filesystem::current_path();
}

std::vector<Complex> random_amplitudes(std::uint64_t count, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<Complex> v(count);
    for (Complex& c : v) {
        c = Complex{dist(rng), dist(rng)};
    }
    return v;
}

void expect_round_trip(std::uint64_t num_amplitudes, std::uint64_t chunk_amplitudes) {
    std::mt19937_64 rng(2305 + chunk_amplitudes);
    const std::vector<Complex> state = random_amplitudes(num_amplitudes, rng);
    FileBackingStore store(scratch_directory(), num_amplitudes, chunk_amplitudes);
    EXPECT_EQ(store.chunk_count(), num_amplitudes / chunk_amplitudes);
    EXPECT_EQ(store.chunk_bytes(), chunk_amplitudes * sizeof(Complex));
    EXPECT_EQ(store.file_bytes(), num_amplitudes * sizeof(Complex));
    // Written in a shuffled order, read back in another.
    std::vector<std::uint64_t> order(store.chunk_count());
    for (std::uint64_t c = 0; c < order.size(); ++c) {
        order[c] = c;
    }
    std::shuffle(order.begin(), order.end(), rng);
    AlignedChunk buffer(chunk_amplitudes);
    for (const std::uint64_t c : order) {
        std::copy_n(state.data() + c * chunk_amplitudes, chunk_amplitudes, buffer.span().data());
        store.write(c, buffer.span());
    }
    std::shuffle(order.begin(), order.end(), rng);
    for (const std::uint64_t c : order) {
        std::fill(buffer.span().begin(), buffer.span().end(), Complex{0.0, 0.0});
        store.read(c, buffer.span());
        EXPECT_EQ(std::memcmp(buffer.span().data(),
                              state.data() + c * chunk_amplitudes,
                              chunk_amplitudes * sizeof(Complex)),
                  0)
            << "chunk " << c << " of " << chunk_amplitudes;
    }
    // Overwritten chunks read back as their last write.
    std::fill(buffer.span().begin(), buffer.span().end(), Complex{0.5, -0.25});
    store.write(0, buffer.span());
    std::fill(buffer.span().begin(), buffer.span().end(), Complex{0.0, 0.0});
    store.read(0, buffer.span());
    for (const Complex& c : buffer.span()) {
        EXPECT_EQ(c, (Complex{0.5, -0.25}));
    }
}

TEST(FileBackingStoreTest, WholeChunksRoundTripExactly) {
    expect_round_trip(1U << 12, 1U << 8);  // 4096-byte chunks: unbuffered where supported
    expect_round_trip(1U << 12, 1U << 10);
    expect_round_trip(1U << 12, 1U << 12);  // one chunk
    expect_round_trip(1U << 8, 1U << 4);    // 256-byte chunks: the buffered path
}

TEST(FileBackingStoreTest, ChunksOfWholeSectorsAreUnbufferedOnARealDisk) {
    const FileBackingStore sectors(scratch_directory(), 1U << 12, 1U << 8);
    const FileBackingStore small(scratch_directory(), 1U << 8, 1U << 4);
    EXPECT_FALSE(small.unbuffered());
#if defined(_WIN32)
    EXPECT_TRUE(sectors.unbuffered());
#else
    // ext4, xfs and the like take O_DIRECT; a tmpfs build tree would not.
    if (!sectors.unbuffered()) {
        GTEST_SKIP() << "the build tree's file system refused O_DIRECT; buffered I/O was tested";
    }
#endif
}

TEST(FileBackingStoreTest, TheFileIsChargedToDiskSpaceAndGoneWithTheStore) {
    common::MemoryLedger ledger;
    const common::MemoryLedger::Scope scope(ledger);
    std::filesystem::path path;
    {
        const FileBackingStore store(scratch_directory(), 1U << 10, 1U << 8);
        path = store.path();
        EXPECT_EQ(ledger.live_disk_bytes(), (1U << 10) * sizeof(Complex));
        EXPECT_EQ(ledger.live_host_bytes(), 0U);
#if defined(_WIN32)
        // Deleted on close: present, and nobody else's, while the store lives.
        EXPECT_TRUE(std::filesystem::exists(path));
#else
        // Unlinked the moment it was opened: the descriptor is its only link.
        EXPECT_FALSE(std::filesystem::exists(path));
#endif
    }
    EXPECT_FALSE(std::filesystem::exists(path)) << path;
    EXPECT_EQ(ledger.live_disk_bytes(), 0U);
    EXPECT_EQ(ledger.peak_disk_bytes(), (1U << 10) * sizeof(Complex));
}

TEST(FileBackingStoreTest, AnExceptionMidUseStillRemovesTheFile) {
    std::filesystem::path path;
    const auto use = [&] {
        const FileBackingStore store(scratch_directory(), 1U << 10, 1U << 8);
        path = store.path();
        AlignedChunk buffer(1U << 8);
        store.write(3, buffer.span());
        throw std::runtime_error("abandoned");
    };
    EXPECT_THROW(use(), std::runtime_error);
    EXPECT_FALSE(std::filesystem::exists(path)) << path;
    // The next store works as before.
    const FileBackingStore again(scratch_directory(), 1U << 10, 1U << 8);
    EXPECT_NE(again.path(), path);
}

TEST(FileBackingStoreTest, AShortDirectoryIsRefusedNamingTheFigures) {
    try {
        const FileBackingStore store(
            scratch_directory(), 1U << 20, 1U << 8, /*available_bytes=*/1000);
        FAIL() << "expected ResourceExceeded";
    } catch (const common::ResourceExceeded& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("16777216 bytes of scratch space"), std::string::npos) << what;
        EXPECT_NE(what.find("1000 bytes available"), std::string::npos) << what;
        EXPECT_NE(what.find(scratch_directory().string()), std::string::npos) << what;
    }
    // Exactly enough is enough.
    EXPECT_NO_THROW((FileBackingStore(scratch_directory(), 1U << 10, 1U << 8, (1U << 10) * 16)));
}

TEST(FileBackingStoreTest, AMissingDirectoryIsRefusedNamingIt) {
    const std::filesystem::path missing = scratch_directory() / "no-such-directory-2305";
    try {
        const FileBackingStore store(missing, 1U << 10, 1U << 8);
        FAIL() << "expected ResourceExceeded";
    } catch (const common::ResourceExceeded& e) {
        EXPECT_NE(std::string(e.what()).find(missing.string()), std::string::npos) << e.what();
    }
    // With the space figure injected, the creation itself is what fails.
    try {
        const FileBackingStore store(missing, 1U << 10, 1U << 8, /*available_bytes=*/1U << 30);
        FAIL() << "expected ResourceExceeded";
    } catch (const common::ResourceExceeded& e) {
        EXPECT_NE(std::string(e.what()).find("cannot create a scratch file"), std::string::npos)
            << e.what();
    }
}

TEST(FileBackingStoreTest, TwoStoresInOneProcessHaveDistinctFiles) {
    const FileBackingStore a(scratch_directory(), 1U << 10, 1U << 8);
    const FileBackingStore b(scratch_directory(), 1U << 10, 1U << 8);
    EXPECT_NE(a.path(), b.path());
    AlignedChunk buffer(1U << 8);
    std::fill(buffer.span().begin(), buffer.span().end(), Complex{1.0, 0.0});
    a.write(0, buffer.span());
    std::fill(buffer.span().begin(), buffer.span().end(), Complex{2.0, 0.0});
    b.write(0, buffer.span());
    a.read(0, buffer.span());
    EXPECT_EQ(buffer.span()[0], (Complex{1.0, 0.0}));
    b.read(0, buffer.span());
    EXPECT_EQ(buffer.span()[0], (Complex{2.0, 0.0}));
}

}  // namespace
}  // namespace menos::engines::dense_tiered
