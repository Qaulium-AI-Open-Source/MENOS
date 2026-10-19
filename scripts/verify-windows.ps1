# Native Windows verification (docs/adr/0022). The Windows counterpart of
# scripts/verify.sh, which stays the Linux gate and keeps clang-format,
# clang-tidy, UBSan/LSan, CUDA and the documentation build.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify-windows.ps1 [-Wheels]
#
# Stages: enter a Visual Studio x64 developer environment; fetch OpenBLAS;
# configure, build and ctest windows-dev (MSVC + AddressSanitizer) and
# windows-release; build the benchmarks; editable install + pytest on CPython
# 3.14; build a wheel and run the smoke test from a clean venv. With -Wheels,
# also cibuildwheel for CPython 3.10-3.14, each wheel running the Python suite.
# Exits non-zero at the first failing stage.
param(
    [switch]$Wheels
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
    Run "ctest $Preset" { ctest --preset $Preset -j $env:NUMBER_OF_PROCESSORS }
}

Stage "benchmarks build (windows-release)"
$CacheDefines = Get-CacheDefines
Run "benchmarks configure" { cmake --preset windows-release -DMENOS_BUILD_BENCHMARKS=ON @CacheDefines }
Run "benchmarks build" { cmake --build --preset windows-release }
Run "benchmarks reconfigure" { cmake --preset windows-release -DMENOS_BUILD_BENCHMARKS=OFF }

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
    Run "cibuildwheel" { uvx --python 3.14 cibuildwheel==4.2.1 --platform windows --output-dir wheelhouse }
    $Built = @(Get-ChildItem wheelhouse\*.whl | ForEach-Object { $_.Name })
    $Built | ForEach-Object { Write-Output $_ }
    foreach ($Tag in @("cp310", "cp311", "cp312", "cp313", "cp314")) {
        if (-not ($Built | Where-Object { $_ -like "*-$Tag-$Tag-win_amd64.whl" })) { Fail "no $Tag win_amd64 wheel" }
    }
}

Write-Output "verify-windows: all stages passed"
exit 0
