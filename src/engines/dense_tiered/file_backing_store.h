#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>

#include "common/memory_ledger.h"

// The disk tier's state: a scratch file of 16 * 2^n bytes read and written
// in whole chunks (Phase 23, docs/adr/0027 section 3.3).
//
// The file is created in the caller's scratch directory under a name of the
// form menos-scratch-<pid>-<k>.state and exists only while this object does:
// on POSIX it is unlinked the moment it is open, so it vanishes with the file
// descriptor on any exit, a crash included; on Windows it is opened with
// FILE_FLAG_DELETE_ON_CLOSE, which the kernel honours when the last handle
// closes, however the process ends. Nothing else can open it meanwhile.
//
// I/O is explicit and positioned (pread/pwrite; ReadFile/WriteFile with an
// offset), unbuffered when the chunk is a whole number of sectors (O_DIRECT;
// FILE_FLAG_NO_BUFFERING with FILE_FLAG_WRITE_THROUGH): step 1 measured
// buffered writes 3x slower on NTFS, where the cache manager copies every
// byte, and half as fast to durable on ext4 (docs/adr/0027 section 10.1,
// owner decision 2). A file system that refuses O_DIRECT, such as a tmpfs,
// gets buffered I/O and unbuffered() says so. Unbuffered I/O needs
// sector-aligned buffers: the executor's window provides them, and read()
// and write() state the precondition.
namespace menos::engines::dense_tiered {

class FileBackingStore {
public:
    // The alignment unbuffered I/O needs of every buffer, offset and length:
    // one sector of the largest common size.
    static constexpr std::size_t kAlignment = 4096;

    // Creates the file for num_amplitudes complex128 amplitudes in chunks of
    // chunk_amplitudes (both powers of two, chunk_amplitudes dividing
    // num_amplitudes), charged to the installed ledger's disk space.
    // Refuses with ResourceExceeded, naming the bytes and the figure it was
    // checked against, when the directory's available space (or
    // available_bytes, which a test injects in its place) is below the file's
    // size, and with ResourceExceeded naming the directory when the file
    // cannot be created there.
    FileBackingStore(const std::filesystem::path& directory,
                     std::uint64_t num_amplitudes,
                     std::uint64_t chunk_amplitudes,
                     std::optional<std::uint64_t> available_bytes = std::nullopt);
    ~FileBackingStore();
    FileBackingStore(const FileBackingStore&) = delete;
    FileBackingStore& operator=(const FileBackingStore&) = delete;
    FileBackingStore(FileBackingStore&&) = delete;
    FileBackingStore& operator=(FileBackingStore&&) = delete;

    // Chunk `chunk` into `into`, or from `from`, which hold exactly
    // chunk_amplitudes() amplitudes. Precondition when unbuffered(): the
    // buffer's address is a multiple of kAlignment. A failed transfer throws
    // InternalError naming the call.
    void read(std::uint64_t chunk, std::span<std::complex<double>> into) const;
    void write(std::uint64_t chunk, std::span<const std::complex<double>> from) const;

    [[nodiscard]] std::uint64_t chunk_count() const;
    [[nodiscard]] std::uint64_t chunk_amplitudes() const;
    [[nodiscard]] std::uint64_t chunk_bytes() const;
    [[nodiscard]] std::uint64_t file_bytes() const;
    [[nodiscard]] bool unbuffered() const;
    [[nodiscard]] const std::filesystem::path& path() const;

private:
    void transfer(std::uint64_t offset, void* data, std::size_t bytes, bool writing) const;
    void close_file();

    std::filesystem::path path_;
    std::uint64_t num_amplitudes_;
    std::uint64_t chunk_amplitudes_;
    bool unbuffered_ = false;
#if defined(_WIN32)
    void* handle_ = nullptr;  // a HANDLE, kept as void* so <windows.h> stays out of the header
#else
    int fd_ = -1;
#endif
    common::MemoryLedger::Charge charge_;
};

}  // namespace menos::engines::dense_tiered
