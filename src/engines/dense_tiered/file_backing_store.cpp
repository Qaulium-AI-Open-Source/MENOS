#include "engines/dense_tiered/file_backing_store.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#endif

#include <algorithm>
#include <atomic>
#include <string>
#include <system_error>

#include "common/checked_math.h"
#include "common/error.h"

namespace menos::engines::dense_tiered {
namespace {

// One ReadFile/WriteFile moves at most a DWORD of bytes; a chunk of 2^28
// amplitudes is 4 GiB, so transfers go in pieces of this size.
constexpr std::size_t kMaxPieceBytes = std::size_t{1} << 30;

std::string last_error_text() {
#if defined(_WIN32)
    return "error " + std::to_string(GetLastError());
#else
    return std::strerror(errno);
#endif
}

unsigned long process_id() {
#if defined(_WIN32)
    return static_cast<unsigned long>(GetCurrentProcessId());
#else
    return static_cast<unsigned long>(getpid());
#endif
}

std::filesystem::path scratch_name(const std::filesystem::path& directory, unsigned long serial) {
    return directory / ("menos-scratch-" + std::to_string(process_id()) + "-" +
                        std::to_string(serial) + ".state");
}

// The next serial for a scratch name in this process.
unsigned long next_serial() {
    static std::atomic<unsigned long> counter{0};
    return ++counter;
}

}  // namespace

FileBackingStore::FileBackingStore(const std::filesystem::path& directory,
                                   std::uint64_t num_amplitudes,
                                   std::uint64_t chunk_amplitudes,
                                   std::optional<std::uint64_t> available_bytes)
    : num_amplitudes_(num_amplitudes), chunk_amplitudes_(chunk_amplitudes) {
    common::ensure(
        num_amplitudes > 0 && chunk_amplitudes > 0 && num_amplitudes % chunk_amplitudes == 0,
        "FileBackingStore: the chunk must divide the state");
    const std::uint64_t bytes = common::checked_mul(num_amplitudes, sizeof(std::complex<double>));

    // The space check, against the directory as it is now, or the figure a
    // test injects.
    std::uint64_t available = 0;
    if (available_bytes.has_value()) {
        available = *available_bytes;
    } else {
        std::error_code error;
        const std::filesystem::space_info info = std::filesystem::space(directory, error);
        if (error) {
            throw common::ResourceExceeded("the scratch directory " + directory.string() +
                                           " is not usable: " + error.message());
        }
        available = info.available;
    }
    if (bytes > available) {
        throw common::ResourceExceeded(
            "the disk tier needs " + std::to_string(bytes) + " bytes of scratch space in " +
            directory.string() + ", which has " + std::to_string(available) + " bytes available");
    }

    // Unbuffered I/O needs whole sectors; a chunk below one sector (reached
    // only through the internal chunk-size seam) is buffered.
    const bool want_unbuffered = (chunk_bytes() % kAlignment) == 0;

    // A fresh name each time; an existing file of the same name (another
    // process with the same pid in another namespace, say) is never reused.
    for (int attempt = 0; attempt < 16; ++attempt) {
        path_ = scratch_name(directory, next_serial());
#if defined(_WIN32)
        DWORD flags =
            FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_SEQUENTIAL_SCAN | FILE_FLAG_DELETE_ON_CLOSE;
        if (want_unbuffered) {
            flags |= FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH;
        }
        HANDLE handle = CreateFileW(
            path_.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_NEW, flags, nullptr);
        if (handle == INVALID_HANDLE_VALUE) {
            if (GetLastError() == ERROR_FILE_EXISTS) {
                continue;
            }
            throw common::ResourceExceeded("cannot create a scratch file in " + directory.string() +
                                           ": " + last_error_text());
        }
        handle_ = handle;
        unbuffered_ = want_unbuffered;
        LARGE_INTEGER size;
        size.QuadPart = static_cast<LONGLONG>(bytes);
        if (SetFilePointerEx(handle, size, nullptr, FILE_BEGIN) == 0 || SetEndOfFile(handle) == 0) {
            const std::string text = last_error_text();
            close_file();
            throw common::ResourceExceeded("cannot size the scratch file " + path_.string() +
                                           " to " + std::to_string(bytes) + " bytes: " + text);
        }
#else
        int flags = O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC;
#ifdef O_DIRECT
        if (want_unbuffered) {
            flags |= O_DIRECT;
        }
#endif
        int fd = ::open(path_.c_str(), flags, 0600);
#ifdef O_DIRECT
        if (fd < 0 && want_unbuffered && errno == EINVAL) {
            // A file system without O_DIRECT (a tmpfs): buffered, and said so.
            flags &= ~O_DIRECT;
            fd = ::open(path_.c_str(), flags, 0600);
        }
#endif
        if (fd < 0) {
            if (errno == EEXIST) {
                continue;
            }
            throw common::ResourceExceeded("cannot create a scratch file in " + directory.string() +
                                           ": " + last_error_text());
        }
        fd_ = fd;
        unbuffered_ = want_unbuffered && (flags & O_DIRECT) != 0;
#if defined(__APPLE__)
        if (want_unbuffered) {
            unbuffered_ = ::fcntl(fd, F_NOCACHE, 1) == 0;
        }
#endif
        // Gone from the directory already: the descriptor is the only link.
        ::unlink(path_.c_str());
        if (::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
            const std::string text = last_error_text();
            close_file();
            throw common::ResourceExceeded("cannot size the scratch file " + path_.string() +
                                           " to " + std::to_string(bytes) + " bytes: " + text);
        }
#endif
        charge_ = common::MemoryLedger::Charge(bytes, common::MemoryLedger::Space::Disk);
        return;
    }
    throw common::ResourceExceeded("cannot create a scratch file in " + directory.string() +
                                   ": every name tried exists");
}

FileBackingStore::~FileBackingStore() {
    close_file();
}

void FileBackingStore::close_file() {
#if defined(_WIN32)
    if (handle_ != nullptr) {
        CloseHandle(static_cast<HANDLE>(handle_));
        handle_ = nullptr;
    }
#else
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
#endif
}

void FileBackingStore::read(std::uint64_t chunk, std::span<std::complex<double>> into) const {
    common::ensure(chunk < chunk_count() && into.size() == chunk_amplitudes_,
                   "FileBackingStore::read: not a whole chunk");
    transfer(chunk * chunk_bytes(), into.data(), chunk_bytes(), /*writing=*/false);
}

void FileBackingStore::write(std::uint64_t chunk,
                             std::span<const std::complex<double>> from) const {
    common::ensure(chunk < chunk_count() && from.size() == chunk_amplitudes_,
                   "FileBackingStore::write: not a whole chunk");
    transfer(chunk * chunk_bytes(), from.data(), chunk_bytes(), /*writing=*/true);
}

void FileBackingStore::transfer(std::uint64_t offset,
                                const void* data,
                                std::size_t bytes,
                                bool writing) const {
    // The read path writes into the caller's buffer, which read() passed as
    // the non-const span it holds; the write path only reads. One pointer
    // type serves both because the platform calls take void pointers.
    auto* cursor = static_cast<unsigned char*>(
        const_cast<void*>(data));  // NOLINT(cppcoreguidelines-pro-type-const-cast)
    std::size_t done = 0;
    while (done < bytes) {
        const std::size_t piece = std::min(bytes - done, kMaxPieceBytes);
#if defined(_WIN32)
        OVERLAPPED overlapped{};
        const std::uint64_t at = offset + done;
        overlapped.Offset = static_cast<DWORD>(at & 0xFFFFFFFFU);
        overlapped.OffsetHigh = static_cast<DWORD>(at >> 32);
        DWORD moved = 0;
        const BOOL ok = writing ? WriteFile(static_cast<HANDLE>(handle_),
                                            cursor + done,
                                            static_cast<DWORD>(piece),
                                            &moved,
                                            &overlapped)
                                : ReadFile(static_cast<HANDLE>(handle_),
                                           cursor + done,
                                           static_cast<DWORD>(piece),
                                           &moved,
                                           &overlapped);
        if (ok == 0 || moved == 0) {
            throw common::InternalError(std::string(writing ? "WriteFile" : "ReadFile") +
                                        " on the scratch file " + path_.string() +
                                        " failed: " + last_error_text());
        }
        done += moved;
#else
        const ssize_t moved =
            writing ? ::pwrite(fd_, cursor + done, piece, static_cast<off_t>(offset + done))
                    : ::pread(fd_, cursor + done, piece, static_cast<off_t>(offset + done));
        if (moved <= 0) {
            throw common::InternalError(std::string(writing ? "pwrite" : "pread") +
                                        " on the scratch file " + path_.string() +
                                        " failed: " + last_error_text());
        }
        done += static_cast<std::size_t>(moved);
#endif
    }
}

std::uint64_t FileBackingStore::chunk_count() const {
    return num_amplitudes_ / chunk_amplitudes_;
}

std::uint64_t FileBackingStore::chunk_amplitudes() const {
    return chunk_amplitudes_;
}

std::uint64_t FileBackingStore::chunk_bytes() const {
    return chunk_amplitudes_ * sizeof(std::complex<double>);
}

std::uint64_t FileBackingStore::file_bytes() const {
    return num_amplitudes_ * sizeof(std::complex<double>);
}

bool FileBackingStore::unbuffered() const {
    return unbuffered_;
}

const std::filesystem::path& FileBackingStore::path() const {
    return path_;
}

}  // namespace menos::engines::dense_tiered
