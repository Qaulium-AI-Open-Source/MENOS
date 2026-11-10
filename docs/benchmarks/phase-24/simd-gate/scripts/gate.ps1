# Phase 24 item 8 gate run on Windows (docs/adr/0028 section 19.7): the
# kernels and the fused brickwork sweep on the current commit against the
# unchanged kernels of the reference commit (default a326851), in the same
# session. The reference tree is a detached worktree at
# C:\Users\janak\menos-simd-ref built with the same flags (MSVC, Release,
# Ninja). The probes run in the order old, new, new, old.
#   kernels-{old,new}-{1,2}.csv   menos_dense_simd_probe (table A)
#   runs-{old,new}-{1,2}.csv      menos_dense_fusion_probe runs (table B)
# Refuses to start, and marks the run invalid, unless on AC at every checkpoint.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File docs\benchmarks\phase-24\simd-gate\scripts\gate.ps1 [max_qubits]
param([string]$MaxQubits = "24", [string]$Ref = "a326851")
$ErrorActionPreference = "Stop"
$RepoRoot = "C:\Users\janak\menos"
$Old = "C:\Users\janak\menos-simd-ref"
Set-Location $RepoRoot
$Out = Join-Path $RepoRoot "build\phase24-simd-gate\windows"
function Power() { $b = Get-CimInstance Win32_Battery; "status=$($b.BatteryStatus) charge=$($b.EstimatedChargeRemaining)" }
function Note([string]$Label) {
    $p = Power
    $load = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
    "$Label $((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')) cpu_load=$load% power: $p" | Out-File -Append -Encoding ascii -FilePath (Join-Path $Out "load.log")
    if ($p -notmatch "status=2") { Write-Output "NOT ON AC at ${Label}: the run is invalid"; exit 3 }
}
if (Test-Path $Out) { Remove-Item -Recurse -Force $Out }
New-Item -ItemType Directory -Force $Out | Out-Null
$commit = Join-Path $Out "commit.txt"
git log --oneline -1 | Out-File -Encoding ascii -FilePath $commit
git status --short --untracked-files=no | Out-File -Append -Encoding ascii -FilePath $commit
"reference $Ref" | Out-File -Append -Encoding ascii -FilePath $commit
$Installer = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer"
$env:PATH = "$Installer;$env:PATH"
$VsPath = & vswhere.exe -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
Import-Module (Join-Path $VsPath "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
Enter-VsDevShell -VsInstallPath $VsPath -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64 -vcvars_ver=14.50" | Out-Null
$VsCMake = Join-Path $VsPath "Common7\IDE\CommonExtensions\Microsoft\CMake"
if (Test-Path $VsCMake) { $env:PATH = "$VsCMake\CMake\bin;$VsCMake\Ninja;$env:PATH" }
Set-Location $RepoRoot
$env:MENOS_OPENBLAS_ROOT = & (Join-Path $RepoRoot "scripts\fetch-openblas-windows.ps1")
$Deps = Join-Path $RepoRoot "build\windows-bench\_deps"
if (-not (Test-Path $Old)) { cmd /c "git worktree add --detach `"$Old`" $Ref 2>&1" | Out-Null }
cmd /c "git -C `"$Old`" checkout -q --detach $Ref 2>&1" | Out-Null
cmd /c "cmake -S `"$Old`" -B `"$Old\build\windows-bench`" -G Ninja -DCMAKE_BUILD_TYPE=Release -DMENOS_BUILD_BENCHMARKS=ON -DMENOS_BUILD_PYTHON_BINDINGS=OFF -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=`"$Deps\googletest-src`" -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=`"$Deps\nlohmann_json-src`" -DFETCHCONTENT_SOURCE_DIR_GOOGLE_BENCHMARK=`"$Deps\google_benchmark-src`" > `"$Out\ref-configure.log`" 2>&1"
if ($LASTEXITCODE -ne 0) { Write-Output "reference configure failed"; exit 1 }
foreach ($dir in @("$Old\build\windows-bench", "build\windows-bench")) {
    cmd /c "cmake --build `"$dir`" --target menos_dense_simd_probe menos_dense_fusion_probe > `"$Out\build.log`" 2>&1"
    if ($LASTEXITCODE -ne 0) { Write-Output "build failed in $dir"; exit 1 }
}
function Kernels([string]$Dir, [string]$Tag) {
    cmd /c "`"$Dir\benchmarks\menos_dense_simd_probe.exe`" $MaxQubits > `"$Out\kernels-$Tag.csv`" 2> `"$Out\kernels-$Tag.err`""
    "exit $LASTEXITCODE" | Out-File -Append -Encoding ascii -FilePath (Join-Path $Out "kernels-$Tag.err")
}
function Runs([string]$Dir, [string]$Tag) {
    cmd /c "`"$Dir\benchmarks\menos_dense_fusion_probe.exe`" $MaxQubits runs > `"$Out\runs-$Tag.csv`" 2> `"$Out\runs-$Tag.err`""
    "exit $LASTEXITCODE" | Out-File -Append -Encoding ascii -FilePath (Join-Path $Out "runs-$Tag.err")
}
$OldBin = "$Old\build\windows-bench"
$NewBin = Join-Path $RepoRoot "build\windows-bench"
Note "start"
Kernels $OldBin "old-1"
Kernels $NewBin "new-1"
Kernels $NewBin "new-2"
Kernels $OldBin "old-2"
Note "kernels-done"
Runs $OldBin "old-1"
Runs $NewBin "new-1"
Runs $NewBin "new-2"
Runs $OldBin "old-2"
Note "end"
"== done $((Get-Date).ToUniversalTime().ToString('HH:mm:ss'))"
