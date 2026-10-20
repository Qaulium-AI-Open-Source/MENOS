#pragma once

#include <cstdint>
#include <string>

namespace menos::kernels::cuda {

// Facts about the CUDA device this process would use, queried from the CUDA
// runtime rather than assumed (docs/adr/0014).
//
// This lives in the kernels layer, below both engines and plan, because both
// need it and the dependency direction is one way: the CUDA dense engine
// budgets against it, and the planner's HardwareProfile reports it.
struct DeviceInfo {
    bool present = false;
    std::string name;
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    int multiprocessor_count = 0;
    // Total and currently-free device memory, from cudaMemGetInfo.
    //
    // These are the RUNTIME figures and they are materially smaller than the
    // marketing capacity: the development GPU is sold as 4096 MiB and reports
    // 3770 MiB total, 3681 MiB free. Budgeting against the nameplate would
    // over-commit the device by 326 MiB, so nothing here is ever derived from
    // a datasheet.
    std::uint64_t total_memory_bytes = 0;
    std::uint64_t free_memory_bytes = 0;
};

// Queried once per process and cached; a device's identity and capacity do not
// change while the process runs. Returns a DeviceInfo with present == false,
// and every other field zeroed, when the binary has no CUDA support or no
// usable device is present -- never throws, so a caller can simply ask.
[[nodiscard]] const DeviceInfo& device_info();

// The device's free memory now, from a fresh cudaMemGetInfo (Phase 21,
// docs/adr/0025 section 9, Q3), unlike DeviceInfo::free_memory_bytes, which
// was read once per process. 0 when no usable device is present or the query
// fails.
[[nodiscard]] std::uint64_t device_free_bytes();

// Device memory a region may use: device_free_bytes(). This is the free
// figure, not the total: the display server and any other process already
// hold some, and budgeting against total would refuse nothing and then fail at
// cudaMalloc.
[[nodiscard]] std::uint64_t device_available_bytes();

}  // namespace menos::kernels::cuda
