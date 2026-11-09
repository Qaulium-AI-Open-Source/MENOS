# Phase 24 item 7, step 1 on Windows (docs/adr/0028 section 17.3): builds
# and runs benchmarks/dense_fusion_probe.cpp in build/windows-bench (MSVC,
# Release) and records its CSV, its environment line, the commit and the
# machine's load and power state. Refuses to start, and marks the run
# invalid, unless the machine is on AC before and after the probe.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File docs\benchmarks\phase-24\fusion-step1\scripts\step1.ps1 [max_qubits]
param([string]$MaxQubits = "24")
$ErrorActionPreference = "Stop"
$RepoRoot = "C:\Users\janak\menos"
Set-Location $RepoRoot
$Out = Join-Path $RepoRoot "build\phase24-fusion-step1\windows"
function Power() { $b = Get-CimInstance Win32_Battery; "status=$($b.BatteryStatus) charge=$($b.EstimatedChargeRemaining)" }
function Note([string]$Label) {
    $p = Power
    $load = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
    "$Label $((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')) cpu_load=$load% power: $p" | Tee-Object -Append -FilePath (Join-Path $Out "load.log")
    if ($p -notmatch "status=2") { Write-Output "NOT ON AC at ${Label}: the run is invalid"; exit 3 }
}
if (Test-Path $Out) { Remove-Item -Recurse -Force $Out }
New-Item -ItemType Directory -Force $Out | Out-Null
git log --oneline -1 | Tee-Object -FilePath (Join-Path $Out "commit.txt")
git status --short --untracked-files=no | Tee-Object -Append -FilePath (Join-Path $Out "commit.txt")
$Installer = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer"
$env:PATH = "$Installer;$env:PATH"
$VsPath = & vswhere.exe -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
Import-Module (Join-Path $VsPath "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
Enter-VsDevShell -VsInstallPath $VsPath -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64 -vcvars_ver=14.50" | Out-Null
$VsCMake = Join-Path $VsPath "Common7\IDE\CommonExtensions\Microsoft\CMake"
if (Test-Path $VsCMake) { $env:PATH = "$VsCMake\CMake\bin;$VsCMake\Ninja;$env:PATH" }
Set-Location $RepoRoot
$env:MENOS_OPENBLAS_ROOT = & (Join-Path $RepoRoot "scripts\fetch-openblas-windows.ps1")
cmake --build build/windows-bench --target menos_dense_fusion_probe | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Output "build failed"; exit 1 }
Note "start"
# Through cmd, so PowerShell 5.1 never sees the probe's stderr (under
# ErrorActionPreference=Stop that would end the run).
cmd /c "`"build\windows-bench\benchmarks\menos_dense_fusion_probe.exe`" $MaxQubits > `"$Out\probe.csv`" 2> `"$Out\probe.err`""
"probe exit $LASTEXITCODE" | Tee-Object -Append -FilePath (Join-Path $Out "probe.err")
Note "end"
"== done $((Get-Date).ToUniversalTime().ToString('HH:mm:ss'))"
