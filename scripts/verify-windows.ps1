# Native Windows verification (docs/adr/0022, docs/adr/0030). The Windows
# counterpart of scripts/verify.sh, which stays the Linux gate and keeps
# clang-format, clang-tidy, UBSan/LSan, the Linux CUDA build and the
# documentation build.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify-windows.ps1 [-Wheels] [-Jobs N]
#
# -Jobs is how many CPU-preset tests ctest runs at once (default: every
# hardware thread). The AddressSanitizer build needs fewer on a machine where
# that many calibrating tests at once exhaust its allocator or starve each
# other's timings (docs/adr/0028 section 23.5 used 4); the NUMBER_OF_PROCESSORS
# environment variable can no longer be overridden for this on current Windows
# builds (docs/adr/0030 section 10.6).
#
# Stages: enter a Visual Studio x64 developer environment; fetch OpenBLAS;
# configure, build and ctest windows-dev (MSVC + AddressSanitizer) and
# windows-release; build the benchmarks; fetch the CUDA toolkit and do the same
# for windows-cuda-dev and windows-cuda-release, whose CUDA tests run on the
# GPU when the machine has one and skip themselves when it has none; editable
# install + pytest on CPython 3.14; build a wheel and run the smoke test from a
# clean venv. With -Wheels, also cibuildwheel for CPython 3.10-3.14 (CUDA
# compiled in), each wheel running the backend check and the Python suite.
# Exits non-zero at the first failing stage.
param(
    [switch]$Wheels,
    [ValidateRange(1, 1024)][int]$Jobs = [int]$env:NUMBER_OF_PROCESSORS
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

function Stage([string]$Name) { Write-Output "== $Name ==" }
function Fail([string]$Name) { Write-Output "FAILED: $Name"; exit 1 }
function Run([string]$Name, [scriptblock]$Block) {
    & $Block
    if ($LASTEXITCODE -ne 0) { Fail $Name }
}

Stage "visual studio developer environment"
$Installer = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer"
$env:PATH = "$Installer;$env:PATH"
$VsPath = & vswhere.exe -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
# vswhere hides pre-release installs (for example Visual Studio 2026 Insiders)
# unless asked; look for one only when no release install has the tools.
if (-not $VsPath) {
    $VsPath = & vswhere.exe -latest -prerelease -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}
if (-not $VsPath) { Fail "no Visual Studio installation with the x64 C++ tools" }
Import-Module (Join-Path $VsPath "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
# The 14.50 toolset is pinned: 14.51 miscompiles MENOS and CMake refuses it
# (docs/adr/0022 section 3.1).
if (-not (Test-Path (Join-Path $VsPath "VC\Tools\MSVC\14.50.*"))) {
    Fail "the MSVC 14.50 toolset is not installed (Visual Studio Installer component Microsoft.VisualStudio.Component.VC.14.50.18.0.x86.x64)"
}
Enter-VsDevShell -VsInstallPath $VsPath -SkipAutomaticLocation `
    -DevCmdArguments "-arch=x64 -host_arch=x64 -vcvars_ver=14.50" | Out-Null
# Prefer the CMake and Ninja that ship with Visual Studio over any others on PATH.
$VsCMake = Join-Path $VsPath "Common7\IDE\CommonExtensions\Microsoft\CMake"
if (Test-Path $VsCMake) { $env:PATH = "$VsCMake\CMake\bin;$VsCMake\Ninja;$env:PATH" }
Set-Location $RepoRoot
Write-Output "cl: $((Get-Command cl).Source)"
Run "tool versions" { cmake --version | Select-Object -First 1; ninja --version; uv --version }

Stage "openblas"
$OpenBlasRoot = & (Join-Path $PSScriptRoot "fetch-openblas-windows.ps1")
$env:MENOS_OPENBLAS_ROOT = $OpenBlasRoot
Write-Output $OpenBlasRoot

# FetchContent's git clones can be very slow here (docs/adr/0002). As in
# verify.sh, reuse a repo-local copy of the sources when one exists; the first
# successful configure fills it.
$DepsCache = Join-Path $RepoRoot "build\_deps-cache"
function Get-CacheDefines {
    if ((Test-Path "$DepsCache\googletest-src") -and (Test-Path "$DepsCache\nlohmann_json-src")) {
        return @("-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=$DepsCache/googletest-src",
                 "-DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=$DepsCache/nlohmann_json-src")
    }
    return @()
}

foreach ($Preset in @("windows-dev", "windows-release")) {
    Stage "configure $Preset"
    $CacheDefines = Get-CacheDefines
    Run "configure $Preset" { cmake --preset $Preset @CacheDefines }
    if ((Get-CacheDefines).Count -eq 0) {
        New-Item -ItemType Directory -Force $DepsCache | Out-Null
        foreach ($Dep in @("googletest-src", "nlohmann_json-src")) {
            Copy-Item -Recurse -Force (Join-Path $RepoRoot "build\$Preset\_deps\$Dep") (Join-Path $DepsCache $Dep)
        }
    }
    Stage "build $Preset"
    Run "build $Preset" { cmake --build --preset $Preset }
    Stage "ctest $Preset"
    Run "ctest $Preset" { ctest --preset $Preset -j $Jobs }
}

Stage "benchmarks build (windows-release)"
$CacheDefines = Get-CacheDefines
Run "benchmarks configure" { cmake --preset windows-release -DMENOS_BUILD_BENCHMARKS=ON @CacheDefines }
Run "benchmarks build" { cmake --build --preset windows-release }
Run "benchmarks reconfigure" { cmake --preset windows-release -DMENOS_BUILD_BENCHMARKS=OFF }

# CUDA (docs/adr/0030). nvcc 12.x refuses the MSVC 14.50 toolset, so the .cu
# files are compiled with a Visual Studio 2022 toolset (14.4x) as nvcc's host
# compiler; CMake takes it from CUDAHOSTCXX and the toolkit from CUDA_PATH.
Stage "cuda toolkit"
$HostCl = Get-ChildItem (Join-Path $VsPath "VC\Tools\MSVC\14.4*\bin\Hostx64\x64\cl.exe") -ErrorAction SilentlyContinue |
    Sort-Object FullName | Select-Object -Last 1
if (-not $HostCl) {
    Fail "no Visual Studio 2022 toolset for nvcc (Visual Studio Installer component Microsoft.VisualStudio.Component.VC.14.44.17.14.x86.x64)"
}
# Forward slashes: both values are substituted into CMake presets and into
# cibuildwheel's environment, where a backslash is an escape character.
$env:CUDAHOSTCXX = $HostCl.FullName -replace "\\", "/"
$env:CUDA_PATH = (& (Join-Path $PSScriptRoot "fetch-cuda-windows.ps1")) -replace "\\", "/"
Write-Output "CUDA_PATH: $env:CUDA_PATH"
Write-Output "CUDAHOSTCXX: $env:CUDAHOSTCXX"
Run "nvcc version" { & "$env:CUDA_PATH/bin/nvcc.exe" --version | Select-Object -Last 2 }
$Gpu = $null
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $Gpu = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null
    if ($LASTEXITCODE -ne 0) { $Gpu = $null }
}
if ($Gpu) { Write-Output "GPU: $Gpu" } else { Write-Output "no GPU visible; CUDA tests will skip themselves" }

foreach ($Preset in @("windows-cuda-dev", "windows-cuda-release")) {
    Stage "configure $Preset"
    $CacheDefines = Get-CacheDefines
    Run "configure $Preset" { cmake --preset $Preset @CacheDefines }
    # When CMake finds CMAKE_CUDA_COMPILER changed (even only in the case of
    # the repository's path) it deletes the cache and configures again without
    # the preset's variables: the result is a build with no CUDA and no
    # sanitizer that would pass this stage untested (docs/adr/0030 section 10.6).
    $Cache = Get-Content (Join-Path $RepoRoot "build\$Preset\CMakeCache.txt")
    if (-not ($Cache -match '^MENOS_ENABLE_CUDA:BOOL=ON$')) {
        Fail "$Preset was configured without CUDA; delete build\$Preset\CMakeCache.txt and run again"
    }
    Stage "build $Preset"
    Run "build $Preset" { cmake --build --preset $Preset }
    Stage "ctest $Preset"
    # Not in parallel: the CUDA tests share one device, and several of them
    # assert on its free memory.
    Run "ctest $Preset" { ctest --preset $Preset }
}

Stage "benchmarks build (windows-cuda-release)"
$CacheDefines = Get-CacheDefines
Run "cuda benchmarks configure" { cmake --preset windows-cuda-release -DMENOS_BUILD_BENCHMARKS=ON @CacheDefines }
Run "cuda benchmarks build" { cmake --build --preset windows-cuda-release }
Run "cuda benchmarks reconfigure" { cmake --preset windows-cuda-release -DMENOS_BUILD_BENCHMARKS=OFF }

Stage "python: venv"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { Run "venv" { uv venv .venv --python 3.14 } }

Stage "python: editable install"
# These pins must match pyproject.toml's [build-system] requires.
Run "build backend install" { uv pip install --python $Python "scikit-build-core==1.0.3" "nanobind==3.1.0" }
$PyCacheDefines = @()
foreach ($Define in Get-CacheDefines) { $PyCacheDefines += @("-C", "cmake.define.$($Define.Substring(2))") }
Run "editable install" { uv pip install --python $Python -e ".[test]" --no-build-isolation @PyCacheDefines }

Stage "python: pytest (editable install)"
Run "pytest" { & $Python -m pytest python/tests --deselect python/tests/test_wheel_smoke.py -v }

Stage "python: build wheel"
if (Test-Path dist) { Remove-Item -Recurse -Force dist }
Run "wheel build" { uv build --wheel --python $Python --no-build-isolation @PyCacheDefines }

Stage "python: wheel smoke test in a clean venv"
$WheelVenv = Join-Path $env:TEMP "menos-wheel-venv"
if (Test-Path $WheelVenv) { Remove-Item -Recurse -Force $WheelVenv }
Run "clean venv" { uv venv $WheelVenv --python 3.14 }
$WheelPython = Join-Path $WheelVenv "Scripts\python.exe"
$Wheel = (Get-ChildItem dist\*.whl | Select-Object -First 1).FullName
Run "wheel install" { uv pip install --python $WheelPython $Wheel "qiskit==2.5.2" "pennylane==0.45.1" pytest }
Run "wheel smoke test" { & $WheelPython -m pytest python/tests/test_wheel_smoke.py -v }

if ($Wheels) {
    Stage "cibuildwheel: CPython 3.10-3.14 win_amd64"
    if (Test-Path wheelhouse) { Remove-Item -Recurse -Force wheelhouse }
    $CacheDefines = Get-CacheDefines
    if ($CacheDefines.Count -gt 0) {
        # scikit-build-core's form of the same FetchContent cache reuse.
        $env:SKBUILD_CMAKE_DEFINE = ($CacheDefines | ForEach-Object { $_.Substring(2) }) -join ";"
        $env:CIBW_ENVIRONMENT_PASS_WINDOWS = "SKBUILD_CMAKE_DEFINE"
    }
    # The wheels have CUDA compiled in (pyproject.toml). Each is first shown to
    # run on the CPU, and on this machine's GPU when it has one or to refuse a
    # forced CUDA run when it has none (scripts/check_wheel_backends.py).
    $BackendCheck = "python {project}/scripts/check_wheel_backends.py cuda"
    if ($Gpu) { $BackendCheck += " --gpu" }
    $env:CIBW_TEST_COMMAND_WINDOWS = "$BackendCheck && pytest {project}/python/tests -q"
    Run "cibuildwheel" { uvx --python 3.14 cibuildwheel==4.2.1 --platform windows --output-dir wheelhouse }
    $Built = @(Get-ChildItem wheelhouse\*.whl | ForEach-Object { $_.Name })
    $Built | ForEach-Object { Write-Output $_ }
    foreach ($Tag in @("cp310", "cp311", "cp312", "cp313", "cp314")) {
        if (-not ($Built | Where-Object { $_ -like "*-$Tag-$Tag-win_amd64.whl" })) { Fail "no $Tag win_amd64 wheel" }
    }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    foreach ($WheelFile in Get-ChildItem wheelhouse\*.whl) {
        $Zip = [System.IO.Compression.ZipFile]::OpenRead($WheelFile.FullName)
        $Names = @($Zip.Entries | ForEach-Object { $_.FullName })
        $Zip.Dispose()
        # Vendored by delvewheel under hashed names (docs/adr/0030 section 5).
        foreach ($Dll in @("libopenblas", "cudart64_12", "msvcp140")) {
            if (-not ($Names | Where-Object { $_ -like "menos.libs/$Dll-*.dll" })) {
                Fail "$($WheelFile.Name) does not carry menos.libs/$Dll-<hash>.dll"
            }
        }
    }
}

Write-Output "verify-windows: all stages passed"
exit 0
