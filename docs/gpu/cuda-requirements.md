# CUDA Requirements

| Requirement | Detail | Verified |
|---|---|---|
| Operating system | Linux, x86-64; Windows x64 with MSVC (from version 1.3.1) | Yes |
| GPU | NVIDIA, with a known compute capability | RTX 3050 Laptop GPU, compute capability **8.6**; RTX 4060 Laptop GPU, compute capability **8.9** (built for 8.6 under WSL, for 8.9 on Windows) |
| Driver | Supporting your toolkit | 595.91.07 (advertises CUDA 13.2) with toolkit 13.4; 566.14 (advertises CUDA 12.7) with toolkit 12.4 under WSL and toolkit 12.9 on Windows |
| CUDA toolkit | **12.0 or newer**: `nvcc`, CUDA runtime headers and library, all from the same toolkit | **13.4.92** and **12.4.131** on Linux; **12.9.86** on Windows |
| Host compiler | One your `nvcc` accepts | GCC 15.2.0 with nvcc 13.4; GCC 13.4 with nvcc 12.4; MSVC 14.44 with nvcc 12.9 on Windows |
| CMake | 3.25 or newer | 4.2.3 |
| Build setting | `CMAKE_CUDA_ARCHITECTURES` set to your GPU | `86` on Linux, `89` on Windows |

## GPUs

- Code built for **compute capability 8.6** has been tested on 8.6 and 8.9 GPUs, and code built for **8.9** on the 8.9 GPU (Windows). No build for another architecture has been tested on its own GPU. The code uses no architecture-specific feature, but an untested GPU is an untested GPU.
- Only **one** device is used.
- Older GPUs than the architecture you compile for cannot run the kernels; compile for your own GPU's capability.

## Toolkit and compiler

CUDA **12.0** is the minimum because MENOS compiles its device code as C++20, which `nvcc` accepts only from 12.0 on. Configuring with an older `nvcc` stops with an error. CMake also refuses a build whose `nvcc`, CUDA headers and CUDA runtime library come from different major versions; that combination would otherwise build without complaint and misread device properties.

Your **driver** decides which toolkit you can run: a program built with a toolkit newer than the driver supports cannot use the GPU, and every CUDA test skips. For example, driver 566.14 runs CUDA 12.x builds but not CUDA 13 builds. Check `nvidia-smi` and pick a toolkit no newer than the CUDA version it reports, or one within the same major version (minor-version compatibility).

`nvcc` accepts only host compilers up to a certain version. CUDA 13.4's `nvcc` accepted GCC 15.2; the 12.4 toolkit accepts only GCC 13 or earlier. With an older toolkit, compile the **whole** build with a GCC it accepts (set both `CMAKE_CXX_COMPILER` and `CMAKE_CUDA_HOST_COMPILER`), as the `cuda12-dev` and `cuda12-release` presets do with GCC 13. GCC 12 is the oldest compiler MENOS supports. Pass the toolkit explicitly ([CUDA Installation](/gpu/cuda-installation)).

On Windows the rule is different. The CUDA 12 `nvcc` accepts Visual Studio 2017 to 2022 only, and MENOS's C++ is built with the Visual Studio 2026 toolset (14.50). So the device code alone is compiled with the MSVC 14.44 toolset as `nvcc`'s host compiler, and everything else stays on 14.50; `-allow-unsupported-compiler` is not used. See [CUDA Installation](/gpu/cuda-installation#on-windows).

## Warning flags

MENOS compiles its device code with the same warning discipline as the host code, with two flags dropped for CUDA sources because they fire inside `nvcc`'s own generated files and CUDA's headers rather than MENOS code. This matters only if you develop MENOS.

## Device memory

The device must have room for the state: `16 · 2ⁿ` bytes for an `n`-qubit dense region, taken from the **free** memory the driver reports. See [GPU Memory](/gpu/gpu-memory).

## Numerical behaviour

Device kernels are compiled with `-fmad=false` so results match the CPU bitwise. This is part of the build; you do not configure it.

## Platforms without CUDA support

CUDA is not supported on Linux aarch64 or macOS, and AMD (ROCm) and Apple (Metal) GPUs are not supported at all.

## Related

[CUDA Installation](/gpu/cuda-installation) · [Compatibility](/reference/compatibility) · [Troubleshooting CUDA](/gpu/troubleshooting-cuda)
