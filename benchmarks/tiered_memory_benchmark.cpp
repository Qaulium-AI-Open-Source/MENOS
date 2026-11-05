// The Phase 23 step-1 tiered-memory benchmark, CPU side (docs/adr/0027
// sections 4 and 6, step 1): what the disk tier would pay and what it would
// compete with, measured before any engine code exists.
//
//   (0) facts: the scratch directory, its capacity and available space, the
//       host budget and the thread count;
//   (1) disk: one chunk of 256 MiB, 1 GiB and 2 GiB written and read back
//       through a scratch file, as docs/adr/0027 section 3.3 would do it --
//       explicit positioned I/O (pread/pwrite on POSIX, ReadFile/WriteFile with
//       an offset on Windows), buffered and unbuffered (O_DIRECT /
//       FILE_FLAG_NO_BUFFERING), and a memory-mapped view (mmap /
//       CreateFileMapping) at the smallest chunk; the flush is timed on its
//       own, and on POSIX the page cache is dropped before a buffered read so
//       that the read is the disk's, not the cache's;
//   (2) passes: the pass count of docs/adr/0027 section 3.2's greedy rule for
//       four circuit families at n = 28-34 and L = 24-28 (analysis only, no
//       execution), with the number of gates that touch a global qubit;
//   (3) in memory: DenseEngine::execute on the crossover circuit at n = 26-30,
//       and on an empty circuit of the same width, so the per-gate cost of the
//       competitor the tier is priced against is a measured figure.
//
// Every timed point is the median of repetitions after one untimed warm-up,
// with the spread beside it (benchmark_spread.h). The disk rows are what the
// cost model of section 3.6 will be calibrated from; the pass counts are what
// it multiplies them by.
//
// This is a benchmark, not a test (CLAUDE.md): it asserts nothing. Setting
// MENOS_TIERED_MEMORY_JSON=<path> also writes every table as JSON, for
// docs/benchmarks/phase-23-baseline/. MENOS_TIERED_SCRATCH_DIR=<dir> names the
// scratch directory (default: the platform temporary directory). Both are
// environment variables because the shared main() hands argv to Google
// Benchmark.
//
// The scratch files are created with exclusive names and deleted when their
// owner goes out of scope, also when a measurement throws.

#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
// std::getenv is the portable call for the two environment variables above;
// MSVC's C4996 on it recommends a non-standard replacement, and /WX would
// otherwise make the warning fatal.
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <new>
#include <nlohmann/json.hpp>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
// clang-format off
#include <windows.h>
#include <malloc.h>
// clang-format on
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "benchmark_spread.h"
#include "common/error.h"
#include "crossover_circuit.h"
#include "engines/dense/dense_engine.h"
#include "menos/circuit.h"
#include "menos/simulation_engine.h"
#include "menos_benchmark_environment.h"
#include "tiered_pass_count.h"

namespace {

using Clock = std::chrono::steady_clock;
using menos::benchmarks::spread;
using menos::benchmarks::Spread;

// Repetitions per point, as in Phases 15, 16 and 22 (owner, docs/adr/0019 Q1).
constexpr int kRepetitions = 10;
// The disk rows and the largest in-memory rows move gigabytes per repetition;
// five keep a run under half an hour and the SSD's written bytes in the low
// hundreds of gigabytes per run.
constexpr int kDiskRepetitions = 5;
constexpr int kLargeRepetitions = 3;

constexpr std::uint64_t kMiB = std::uint64_t{1} << 20;
constexpr std::uint64_t kGiB = std::uint64_t{1} << 30;
constexpr std::uint64_t kChunkSizes[] = {256 * kMiB, 1 * kGiB, 2 * kGiB};
// Every chunk size is measured on a file of this size, so the chunk count per
// pass differs (8, 2, 1) but the bytes moved per pass do not.
constexpr std::uint64_t kFileBytes = 2 * kGiB;
// Space the benchmark leaves untouched on the volume, the reserve of
// docs/adr/0027 Q2 (5% of the volume or 8 GiB, whichever is larger).
constexpr std::uint64_t kReserveBytes = 8 * kGiB;
constexpr std::size_t kAlignment = 4096;  // a sector, for unbuffered I/O

constexpr std::uint32_t kMinInMemoryQubits = 26;
constexpr std::uint32_t kMaxInMemoryQubits = 30;
constexpr std::uint32_t kLargeFromQubits = 28;

constexpr double kGiga = 1e9;
constexpr double kMilli = 1e3;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

nlohmann::json spread_json(const Spread& s) {
    return {{"median", s.median}, {"min", s.min}, {"max", s.max}, {"q1", s.q1}, {"q3", s.q3}};
}

double gb_per_second(std::uint64_t bytes, double seconds) {
    return seconds > 0.0 ? static_cast<double>(bytes) / kGiga / seconds : 0.0;
}

// ---------------------------------------------------------------------------
// Platform pieces: an aligned buffer, a scratch file, a mapped view
// ---------------------------------------------------------------------------

// Sector-aligned, as unbuffered I/O requires on both platforms.
class AlignedBuffer {
public:
    explicit AlignedBuffer(std::size_t bytes) : bytes_(bytes) {
#if defined(_WIN32)
        data_ = static_cast<unsigned char*>(_aligned_malloc(bytes, kAlignment));
#else
        data_ = static_cast<unsigned char*>(std::aligned_alloc(kAlignment, bytes));
#endif
        if (data_ == nullptr) {
            throw std::bad_alloc();
        }
        // Resident and non-zero before any timing, so a write moves real bytes
        // and a read lands in mapped pages.
        for (std::size_t i = 0; i < bytes; i += kAlignment) {
            data_[i] = static_cast<unsigned char>(i >> 12);
        }
    }
    ~AlignedBuffer() {
#if defined(_WIN32)
        _aligned_free(data_);
#else
        std::free(data_);
#endif
    }
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    AlignedBuffer(AlignedBuffer&&) = delete;
    AlignedBuffer& operator=(AlignedBuffer&&) = delete;

    [[nodiscard]] std::span<unsigned char> bytes() {
        return {data_, bytes_};
    }

private:
    unsigned char* data_ = nullptr;
    std::size_t bytes_;
};

std::filesystem::path unique_scratch_name(const std::filesystem::path& directory) {
    static int counter = 0;
    ++counter;
#if defined(_WIN32)
    const auto pid = static_cast<unsigned long>(GetCurrentProcessId());
#else
    const auto pid = static_cast<unsigned long>(getpid());
#endif
    return directory /
           ("menos-tiered-" + std::to_string(pid) + "-" + std::to_string(counter) + ".bin");
}

// A scratch file of a fixed size with positioned reads and writes of whole
// chunks, buffered or unbuffered, deleted on destruction. The shape
// docs/adr/0027 section 3.3 gives FileBackingStore.
class ScratchFile {
public:
    ScratchFile(const std::filesystem::path& directory, std::uint64_t bytes, bool unbuffered)
        : path_(unique_scratch_name(directory)), bytes_(bytes), unbuffered_(unbuffered) {
#if defined(_WIN32)
        DWORD flags = FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_SEQUENTIAL_SCAN;
        if (unbuffered) {
            flags |= FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH;
        }
        handle_ = CreateFileW(
            path_.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_NEW, flags, nullptr);
        menos::common::ensure(handle_ != INVALID_HANDLE_VALUE, "CreateFileW failed");
        LARGE_INTEGER size;
        size.QuadPart = static_cast<LONGLONG>(bytes);
        menos::common::ensure(SetFilePointerEx(handle_, size, nullptr, FILE_BEGIN) != 0,
                              "SetFilePointerEx failed");
        menos::common::ensure(SetEndOfFile(handle_) != 0, "SetEndOfFile failed");
#else
        int flags = O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC;
#ifdef O_DIRECT
        if (unbuffered) {
            flags |= O_DIRECT;
        }
#else
        menos::common::ensure(!unbuffered, "O_DIRECT is not available on this platform");
#endif
        fd_ = ::open(path_.c_str(), flags, 0600);
        menos::common::ensure(fd_ >= 0, "open failed");
        menos::common::ensure(::ftruncate(fd_, static_cast<off_t>(bytes)) == 0, "ftruncate failed");
#endif
    }
    ~ScratchFile() {
#if defined(_WIN32)
        if (handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(handle_);
        }
        DeleteFileW(path_.c_str());
#else
        if (fd_ >= 0) {
            ::close(fd_);
        }
        ::unlink(path_.c_str());
#endif
    }
    ScratchFile(const ScratchFile&) = delete;
    ScratchFile& operator=(const ScratchFile&) = delete;
    ScratchFile(ScratchFile&&) = delete;
    ScratchFile& operator=(ScratchFile&&) = delete;

    [[nodiscard]] std::uint64_t size_bytes() const {
        return bytes_;
    }
    [[nodiscard]] bool unbuffered() const {
        return unbuffered_;
    }

    void write_at(std::uint64_t offset, std::span<const unsigned char> source) const {
        std::size_t done = 0;
        while (done < source.size()) {
#if defined(_WIN32)
            const DWORD piece = piece_of(source.size() - done);
            OVERLAPPED overlapped = overlapped_at(offset + done);
            DWORD moved = 0;
            menos::common::ensure(
                WriteFile(handle_, source.data() + done, piece, &moved, &overlapped) != 0 &&
                    moved > 0,
                "WriteFile failed");
            done += moved;
#else
            const ssize_t moved = ::pwrite(
                fd_, source.data() + done, source.size() - done, static_cast<off_t>(offset + done));
            menos::common::ensure(moved > 0, "pwrite failed");
            done += static_cast<std::size_t>(moved);
#endif
        }
    }
    void read_at(std::uint64_t offset, std::span<unsigned char> destination) const {
        std::size_t done = 0;
        while (done < destination.size()) {
#if defined(_WIN32)
            const DWORD piece = piece_of(destination.size() - done);
            OVERLAPPED overlapped = overlapped_at(offset + done);
            DWORD moved = 0;
            menos::common::ensure(
                ReadFile(handle_, destination.data() + done, piece, &moved, &overlapped) != 0 &&
                    moved > 0,
                "ReadFile failed");
            done += moved;
#else
            const ssize_t moved = ::pread(fd_,
                                          destination.data() + done,
                                          destination.size() - done,
                                          static_cast<off_t>(offset + done));
            menos::common::ensure(moved > 0, "pread failed");
            done += static_cast<std::size_t>(moved);
#endif
        }
    }

    // Everything written reaches the device.
    void flush() const {
#if defined(_WIN32)
        menos::common::ensure(FlushFileBuffers(handle_) != 0, "FlushFileBuffers failed");
#else
        menos::common::ensure(::fdatasync(fd_) == 0, "fdatasync failed");
#endif
    }

    // Asks the OS to forget the file's cached pages, so a buffered read that
    // follows is served by the disk. Linux honours it; Windows has no such
    // call, which the row's label states.
    [[nodiscard]] bool drop_cache() const {
#if defined(_WIN32)
        return false;
#elif defined(POSIX_FADV_DONTNEED)
        return ::posix_fadvise(fd_, 0, 0, POSIX_FADV_DONTNEED) == 0;
#else
        return false;
#endif
    }

#if defined(_WIN32)
    [[nodiscard]] HANDLE handle() const {
        return handle_;
    }
#else
    [[nodiscard]] int descriptor() const {
        return fd_;
    }
#endif

private:
#if defined(_WIN32)
    // ReadFile/WriteFile take a DWORD, so a transfer above 4 GiB is split into
    // pieces of at most 1 GiB; every chunk here is at most 2 GiB.
    static DWORD piece_of(std::size_t remaining) {
        return static_cast<DWORD>(std::min<std::size_t>(remaining, static_cast<std::size_t>(kGiB)));
    }
    static OVERLAPPED overlapped_at(std::uint64_t offset) {
        OVERLAPPED overlapped{};
        overlapped.Offset = static_cast<DWORD>(offset & 0xFFFFFFFFULL);
        overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);
        return overlapped;
    }
#endif

    std::filesystem::path path_;
    std::uint64_t bytes_;
    bool unbuffered_;
#if defined(_WIN32)
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
};

// The whole file mapped read-write; what the proposal's "memory-mapped file"
// alternative would use, measured so the choice of section 3.3 is on evidence.
class MappedView {
public:
    explicit MappedView(const ScratchFile& file) : bytes_(file.size_bytes()) {
#if defined(_WIN32)
        mapping_ = CreateFileMappingW(file.handle(), nullptr, PAGE_READWRITE, 0, 0, nullptr);
        menos::common::ensure(mapping_ != nullptr, "CreateFileMappingW failed");
        data_ = static_cast<unsigned char*>(MapViewOfFile(mapping_, FILE_MAP_ALL_ACCESS, 0, 0, 0));
        menos::common::ensure(data_ != nullptr, "MapViewOfFile failed");
#else
        void* mapped =
            ::mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, file.descriptor(), 0);
        // MAP_FAILED is (void*)-1 behind a C-style cast in the system header;
        // compared as an integer so that -Wold-style-cast has nothing to say.
        menos::common::ensure(
            reinterpret_cast<std::uintptr_t>(mapped) != static_cast<std::uintptr_t>(-1),
            "mmap failed");
        data_ = static_cast<unsigned char*>(mapped);
#endif
    }
    ~MappedView() {
#if defined(_WIN32)
        UnmapViewOfFile(data_);
        CloseHandle(mapping_);
#else
        ::munmap(data_, bytes_);
#endif
    }
    MappedView(const MappedView&) = delete;
    MappedView& operator=(const MappedView&) = delete;
    MappedView(MappedView&&) = delete;
    MappedView& operator=(MappedView&&) = delete;

    [[nodiscard]] unsigned char* data() const {
        return data_;
    }

    // Dirty pages written out: the mapped analogue of flush().
    void sync() const {
#if defined(_WIN32)
        menos::common::ensure(FlushViewOfFile(data_, 0) != 0, "FlushViewOfFile failed");
#else
        menos::common::ensure(::msync(data_, bytes_, MS_SYNC) == 0, "msync failed");
#endif
    }

private:
    std::uint64_t bytes_;
    unsigned char* data_ = nullptr;
#if defined(_WIN32)
    HANDLE mapping_ = nullptr;
#endif
};

// ---------------------------------------------------------------------------
// (0) facts
// ---------------------------------------------------------------------------

struct Facts {
    std::filesystem::path scratch_directory;
    std::uint64_t capacity_bytes = 0;
    std::uint64_t available_bytes = 0;
    std::uint64_t host_budget_bytes = 0;
    unsigned hardware_threads = 0;
};

Facts query_facts() {
    Facts facts;
    const char* const dir = std::getenv("MENOS_TIERED_SCRATCH_DIR");
    facts.scratch_directory = (dir != nullptr && *dir != '\0')
                                  ? std::filesystem::path(dir)
                                  : std::filesystem::temp_directory_path();
    const std::filesystem::space_info space = std::filesystem::space(facts.scratch_directory);
    facts.capacity_bytes = space.capacity;
    facts.available_bytes = space.available;
    facts.host_budget_bytes = menos_benchmark::safe_allocation_budget_bytes();
    facts.hardware_threads = std::thread::hardware_concurrency();
    return facts;
}

nlohmann::json facts_section(const Facts& facts) {
    std::printf("\n== (0) facts ==\n");
    std::printf("scratch directory        : %s\n", facts.scratch_directory.string().c_str());
    std::printf("volume capacity          : %llu bytes\n",
                static_cast<unsigned long long>(facts.capacity_bytes));
    std::printf("volume available         : %llu bytes\n",
                static_cast<unsigned long long>(facts.available_bytes));
    std::printf("host budget              : %llu bytes\n",
                static_cast<unsigned long long>(facts.host_budget_bytes));
    std::printf("hardware threads         : %u\n", facts.hardware_threads);
    return {
        {"scratch_directory", facts.scratch_directory.string()},
        {"volume_capacity_bytes", facts.capacity_bytes},
        {"volume_available_bytes", facts.available_bytes},
        {"host_budget_bytes", facts.host_budget_bytes},
        {"hardware_threads", facts.hardware_threads},
    };
}

// ---------------------------------------------------------------------------
// (1) disk
// ---------------------------------------------------------------------------

struct DiskPath {
    std::string name;
    Spread write{};
    Spread flush{};
    Spread read{};
    bool read_after_drop = false;  // the read followed a successful cache drop
    bool measured = false;
};

// One pass writes the whole file in chunks, flushes, (drops the cache,) and
// reads the whole file back in chunks; each part timed on its own.
DiskPath time_explicit(const Facts& facts,
                       std::uint64_t chunk_bytes,
                       bool unbuffered,
                       AlignedBuffer& buffer) {
    DiskPath row;
    row.name = unbuffered ? "explicit unbuffered" : "explicit buffered";
    std::vector<double> writes;
    std::vector<double> flushes;
    std::vector<double> reads;
    const std::uint64_t chunks = kFileBytes / chunk_bytes;
    const std::span<unsigned char> chunk =
        buffer.bytes().subspan(0, static_cast<std::size_t>(chunk_bytes));
    for (int r = -1; r < kDiskRepetitions; ++r) {
        const ScratchFile file(facts.scratch_directory, kFileBytes, unbuffered);
        const Clock::time_point t0 = Clock::now();
        for (std::uint64_t c = 0; c < chunks; ++c) {
            file.write_at(c * chunk_bytes, chunk);
        }
        const Clock::time_point t1 = Clock::now();
        file.flush();
        const Clock::time_point t2 = Clock::now();
        row.read_after_drop = file.drop_cache();
        const Clock::time_point t3 = Clock::now();
        for (std::uint64_t c = 0; c < chunks; ++c) {
            file.read_at(c * chunk_bytes, chunk);
        }
        const Clock::time_point t4 = Clock::now();
        benchmark::DoNotOptimize(chunk.data());
        if (r >= 0) {
            writes.push_back(std::chrono::duration<double>(t1 - t0).count());
            flushes.push_back(std::chrono::duration<double>(t2 - t1).count());
            reads.push_back(std::chrono::duration<double>(t4 - t3).count());
        }
    }
    row.write = spread(std::move(writes));
    row.flush = spread(std::move(flushes));
    row.read = spread(std::move(reads));
    row.measured = true;
    return row;
}

DiskPath time_mapped(const Facts& facts, std::uint64_t chunk_bytes, AlignedBuffer& buffer) {
    DiskPath row;
    row.name = "mapped";
    std::vector<double> writes;
    std::vector<double> flushes;
    std::vector<double> reads;
    const std::uint64_t chunks = kFileBytes / chunk_bytes;
    const std::span<unsigned char> chunk =
        buffer.bytes().subspan(0, static_cast<std::size_t>(chunk_bytes));
    for (int r = -1; r < kDiskRepetitions; ++r) {
        const ScratchFile file(facts.scratch_directory, kFileBytes, false);
        const MappedView view(file);
        const Clock::time_point t0 = Clock::now();
        for (std::uint64_t c = 0; c < chunks; ++c) {
            std::memcpy(view.data() + c * chunk_bytes, chunk.data(), chunk.size());
        }
        const Clock::time_point t1 = Clock::now();
        view.sync();
        const Clock::time_point t2 = Clock::now();
        row.read_after_drop = file.drop_cache();
        const Clock::time_point t3 = Clock::now();
        for (std::uint64_t c = 0; c < chunks; ++c) {
            std::memcpy(chunk.data(), view.data() + c * chunk_bytes, chunk.size());
        }
        const Clock::time_point t4 = Clock::now();
        benchmark::DoNotOptimize(chunk.data());
        if (r >= 0) {
            writes.push_back(std::chrono::duration<double>(t1 - t0).count());
            flushes.push_back(std::chrono::duration<double>(t2 - t1).count());
            reads.push_back(std::chrono::duration<double>(t4 - t3).count());
        }
    }
    row.write = spread(std::move(writes));
    row.flush = spread(std::move(flushes));
    row.read = spread(std::move(reads));
    row.measured = true;
    return row;
}

void print_disk_row(std::uint64_t chunk_bytes, const DiskPath& row) {
    std::printf("%6llu %-20s %9.1f %7.2f %9.1f %9.1f %7.2f %7.2f %s\n",
                static_cast<unsigned long long>(chunk_bytes / kMiB),
                row.name.c_str(),
                row.write.median * kMilli,
                gb_per_second(kFileBytes, row.write.median),
                row.flush.median * kMilli,
                row.read.median * kMilli,
                gb_per_second(kFileBytes, row.read.median),
                gb_per_second(kFileBytes, row.write.median + row.flush.median),
                row.read_after_drop ? "dropped" : "possibly cached");
    std::fflush(stdout);
}

nlohmann::json disk_row_json(std::uint64_t chunk_bytes, const DiskPath& row) {
    return {
        {"chunk_bytes", chunk_bytes},
        {"file_bytes", kFileBytes},
        {"path", row.name},
        {"measured", row.measured},
        {"write_seconds", spread_json(row.write)},
        {"flush_seconds", spread_json(row.flush)},
        {"read_seconds", spread_json(row.read)},
        {"read_after_cache_drop", row.read_after_drop},
        {"write_gb_per_second", gb_per_second(kFileBytes, row.write.median)},
        {"write_and_flush_gb_per_second",
         gb_per_second(kFileBytes, row.write.median + row.flush.median)},
        {"read_gb_per_second", gb_per_second(kFileBytes, row.read.median)},
    };
}

nlohmann::json disk_section(const Facts& facts) {
    std::printf(
        "\n== (1) disk: a %llu MiB scratch file written and read in chunks (milliseconds, "
        "medians over %d; GB/s of the median) ==\n",
        static_cast<unsigned long long>(kFileBytes / kMiB),
        kDiskRepetitions);
    std::printf("%6s %-20s %9s %7s %9s %9s %7s %7s %s\n",
                "chunk",
                "path",
                "write",
                "GB/s",
                "flush",
                "read",
                "GB/s",
                "w+f",
                "read cache");
    nlohmann::json rows = nlohmann::json::array();
    if (facts.available_bytes < kFileBytes + kReserveBytes) {
        std::printf("(skipped: %llu bytes available, below the file plus the reserve)\n",
                    static_cast<unsigned long long>(facts.available_bytes));
        return rows;
    }
    for (const std::uint64_t chunk_bytes : kChunkSizes) {
        if (chunk_bytes > facts.host_budget_bytes / 4) {
            std::printf(
                "%6llu (skipped: the chunk buffer does not fit a quarter of the host budget)\n",
                static_cast<unsigned long long>(chunk_bytes / kMiB));
            rows.push_back({{"chunk_bytes", chunk_bytes}, {"measured", false}});
            continue;
        }
        AlignedBuffer buffer(static_cast<std::size_t>(chunk_bytes));
        const DiskPath buffered = time_explicit(facts, chunk_bytes, false, buffer);
        print_disk_row(chunk_bytes, buffered);
        rows.push_back(disk_row_json(chunk_bytes, buffered));
        DiskPath unbuffered;
        unbuffered.name = "explicit unbuffered";
        try {
            unbuffered = time_explicit(facts, chunk_bytes, true, buffer);
            print_disk_row(chunk_bytes, unbuffered);
        } catch (const menos::common::Error& error) {
            std::printf("%6llu %-20s (not measured: %s)\n",
                        static_cast<unsigned long long>(chunk_bytes / kMiB),
                        unbuffered.name.c_str(),
                        error.what());
        }
        rows.push_back(disk_row_json(chunk_bytes, unbuffered));
        if (chunk_bytes == kChunkSizes[0]) {
            const DiskPath mapped = time_mapped(facts, chunk_bytes, buffer);
            print_disk_row(chunk_bytes, mapped);
            rows.push_back(disk_row_json(chunk_bytes, mapped));
        }
    }
    const std::filesystem::space_info after = std::filesystem::space(facts.scratch_directory);
    std::printf("volume available after the sweep: %llu bytes\n",
                static_cast<unsigned long long>(after.available));
    return rows;
}

// ---------------------------------------------------------------------------
// (2) passes
// ---------------------------------------------------------------------------

// Four families with different global-qubit habits: the crossover family
// (every layer touches every qubit), a nearest-neighbour ladder (a CX chain
// that reaches the top qubits once per layer), random pairs (no locality at
// all) and a QFT-like all-to-all pattern (every pair once).
menos::Circuit ladder(std::uint32_t n) {
    menos::Circuit circuit(n, 0);
    for (int layer = 0; layer < menos::benchmarks::kCrossoverLayers; ++layer) {
        for (std::uint32_t q = 0; q < n; ++q) {
            circuit.append(menos::Opcode::H, {static_cast<menos::QubitId>(q)});
        }
        for (std::uint32_t q = 0; q + 1 < n; ++q) {
            circuit.append(menos::Opcode::CX,
                           {static_cast<menos::QubitId>(q), static_cast<menos::QubitId>(q + 1)});
        }
    }
    return circuit;
}

menos::Circuit random_pairs(std::uint32_t n) {
    constexpr int kGates = 200;
    std::mt19937_64 rng(1);
    std::uniform_int_distribution<std::uint32_t> pick(0, n - 1);
    menos::Circuit circuit(n, 0);
    for (int g = 0; g < kGates; ++g) {
        const std::uint32_t a = pick(rng);
        std::uint32_t b = pick(rng);
        while (b == a) {
            b = pick(rng);
        }
        if (g % 2 == 0) {
            circuit.append(menos::Opcode::RX, {static_cast<menos::QubitId>(a)}, {0.3});
        } else {
            circuit.append(menos::Opcode::CX,
                           {static_cast<menos::QubitId>(a), static_cast<menos::QubitId>(b)});
        }
    }
    return circuit;
}

menos::Circuit qft_like(std::uint32_t n) {
    menos::Circuit circuit(n, 0);
    for (std::uint32_t i = 0; i < n; ++i) {
        circuit.append(menos::Opcode::H, {static_cast<menos::QubitId>(i)});
        for (std::uint32_t j = i + 1; j < n; ++j) {
            circuit.append(menos::Opcode::CP,
                           {static_cast<menos::QubitId>(j), static_cast<menos::QubitId>(i)},
                           {3.141592653589793 / static_cast<double>(std::uint64_t{1} << (j - i))});
        }
    }
    return circuit;
}

nlohmann::json passes_section() {
    std::printf(
        "\n== (2) passes of the greedy rule (docs/adr/0027 section 3.2); w = the window's "
        "global-qubit capacity ==\n");
    std::printf("%-12s %3s %5s %3s %6s %6s %6s %6s %6s\n",
                "family",
                "n",
                "gates",
                "L",
                "global",
                "gmax",
                "w=gmax",
                "w+1",
                "w+2");
    nlohmann::json rows = nlohmann::json::array();
    const std::pair<const char*, std::function<menos::Circuit(std::uint32_t)>> families[] = {
        {"crossover", menos::benchmarks::make_crossover_circuit},
        {"ladder", ladder},
        {"random_pairs", random_pairs},
        {"qft_like", qft_like},
    };
    for (const auto& [name, make] : families) {
        for (std::uint32_t n = 28; n <= 34; n += 2) {
            const menos::Circuit circuit = make(n);
            for (std::uint32_t local = 24; local <= 28 && local < n; local += 2) {
                const menos::benchmarks::PassCount probe = menos::benchmarks::count_passes(
                    circuit.operations(), {.local_qubits = local, .window_globals = 5});
                const std::uint32_t gmax = probe.max_global_operands;
                std::uint64_t counts[3] = {0, 0, 0};
                for (std::uint32_t extra = 0; extra < 3; ++extra) {
                    counts[extra] = menos::benchmarks::count_passes(
                                        circuit.operations(),
                                        {.local_qubits = local, .window_globals = gmax + extra})
                                        .passes;
                }
                std::printf("%-12s %3u %5zu %3u %6llu %6u %6llu %6llu %6llu\n",
                            name,
                            n,
                            circuit.operations().size(),
                            local,
                            static_cast<unsigned long long>(probe.global_gates),
                            gmax,
                            static_cast<unsigned long long>(counts[0]),
                            static_cast<unsigned long long>(counts[1]),
                            static_cast<unsigned long long>(counts[2]));
                rows.push_back({
                    {"family", name},
                    {"n", n},
                    {"gates", circuit.operations().size()},
                    {"local_qubits", local},
                    {"global_gates", probe.global_gates},
                    {"max_global_operands", gmax},
                    {"passes_window_gmax", counts[0]},
                    {"passes_window_gmax_plus_1", counts[1]},
                    {"passes_window_gmax_plus_2", counts[2]},
                });
            }
        }
    }
    std::fflush(stdout);
    return rows;
}

// ---------------------------------------------------------------------------
// (3) in memory
// ---------------------------------------------------------------------------

Spread time_execute(const menos::engines::dense::DenseEngine& engine,
                    const menos::EngineExecutionRequest& request,
                    int repetitions) {
    std::vector<double> samples;
    for (int r = -1; r < repetitions; ++r) {
        const Clock::time_point start = Clock::now();
        const menos::EngineExecutionResult result = engine.execute(request);
        const double seconds = seconds_since(start);
        std::uint64_t actual_bytes = result.actual_bytes;
        benchmark::DoNotOptimize(actual_bytes);
        if (r >= 0) {
            samples.push_back(seconds);
        }
    }
    return spread(std::move(samples));
}

nlohmann::json in_memory_section(const Facts& facts) {
    std::printf(
        "\n== (3) DenseEngine::execute in memory, crossover circuit and an empty circuit "
        "(seconds, medians) ==\n");
    std::printf("%3s %5s %4s %11s %11s %11s %9s\n",
                "n",
                "gates",
                "reps",
                "circuit",
                "empty",
                "per gate",
                "GB/s eq");
    nlohmann::json rows = nlohmann::json::array();
    const menos::engines::dense::DenseEngine engine;
    for (std::uint32_t n = kMinInMemoryQubits; n <= kMaxInMemoryQubits; ++n) {
        const std::uint64_t bytes = std::uint64_t{16} << n;
        if (bytes > facts.host_budget_bytes) {
            std::printf("%3u (skipped: does not fit the host budget)\n", n);
            rows.push_back({{"n", n}, {"measured", false}});
            continue;
        }
        const int repetitions = n >= kLargeFromQubits ? kLargeRepetitions : kRepetitions;
        const menos::Circuit circuit = menos::benchmarks::make_crossover_circuit(n);
        const menos::Circuit empty(n, 0);
        const auto request = [&](const menos::Circuit& c) {
            return menos::EngineExecutionRequest{
                .num_qubits = n,
                .operations = c.operations(),
                .want_statevector = true,
                .want_counts = false,
                .want_expectation = false,
                .shots = 0,
                .measure_targets = {},
                .bitstring_width = 0,
                .seed = 0,
                .global_phase = 0.0,
                .budget_bytes = facts.host_budget_bytes,
                .is_sole_region = true,
                .observable = nullptr,
            };
        };
        const Spread full = time_execute(engine, request(circuit), repetitions);
        const Spread init_only = time_execute(engine, request(empty), repetitions);
        const double per_gate =
            (full.median - init_only.median) / static_cast<double>(circuit.operations().size());
        // Each gate streams the state once: read and write, 2 * 16 * 2^n bytes.
        const double traffic_gb_per_second = gb_per_second(2 * bytes, per_gate);
        std::printf("%3u %5zu %4d %11.4f %11.4f %11.6f %9.2f\n",
                    n,
                    circuit.operations().size(),
                    repetitions,
                    full.median,
                    init_only.median,
                    per_gate,
                    traffic_gb_per_second);
        std::fflush(stdout);
        rows.push_back({
            {"n", n},
            {"measured", true},
            {"gates", circuit.operations().size()},
            {"repetitions", repetitions},
            {"circuit_seconds", spread_json(full)},
            {"empty_circuit_seconds", spread_json(init_only)},
            {"per_gate_seconds", per_gate},
            {"per_gate_traffic_gb_per_second", traffic_gb_per_second},
        });
    }
    return rows;
}

void write_json(const nlohmann::json& document) {
    const char* const path = std::getenv("MENOS_TIERED_MEMORY_JSON");
    if (path == nullptr || *path == '\0') {
        return;
    }
    std::ofstream out(path);
    out << document.dump(2) << "\n";
    out.close();
    std::printf("JSON written     : %s%s\n", path, out ? "" : " (WRITE FAILED)");
}

void report() {
    std::printf("\n=== MENOS tiered memory benchmark, CPU side (Phase 23 step 1) ===\n");
    std::printf("environment      : %s\n", menos_benchmark::describe_environment().c_str());
    std::printf(
        "repetitions      : %d per point (%d for the disk rows, %d from n = %u in memory) "
        "after a warm-up; medians unless stated\n",
        kRepetitions,
        kDiskRepetitions,
        kLargeRepetitions,
        kLargeFromQubits);
    const Facts facts = query_facts();
    nlohmann::json document = {{"schema_version", 1}};
    document["environment"] = menos_benchmark::describe_environment();
    document["repetitions"] = kRepetitions;
    document["disk_repetitions"] = kDiskRepetitions;
    document["large_repetitions"] = kLargeRepetitions;
    document["facts"] = facts_section(facts);
    document["disk"] = disk_section(facts);
    document["passes"] = passes_section();
    document["in_memory"] = in_memory_section(facts);
    std::printf("\n");
    write_json(document);
}

}  // namespace

// Report-only, like BM_CudaMemoryReport.
static void BM_TieredMemoryReport(benchmark::State& state) {
    for (auto _ : state) {
        report();
        state.SkipWithError("report-only benchmark: see the printed tables");
        break;
    }
}
BENCHMARK(BM_TieredMemoryReport)->Iterations(1)->Unit(benchmark::kSecond);
