# Phase 23 step 1 run (a, b, c) on Windows: the CPU side of the benchmark with
# its scratch files on drive C:. Refuses to start, and marks the run invalid,
# unless the machine is on AC before and after every tool. Records the WSL
# virtual disk's size, which the WSL runs grow.
param([string]$Run)
$ErrorActionPreference = "Stop"
$RepoRoot = "C:\Users\janak\menos"
Set-Location $RepoRoot
$Out = Join-Path $RepoRoot "build\phase23-baseline\win-run-$Run"
function Power() { $b = Get-CimInstance Win32_Battery; "status=$($b.BatteryStatus) charge=$($b.EstimatedChargeRemaining)" }
function CheckAc([string]$Label) {
    $p = Power
    "$((Get-Date).ToUniversalTime().ToString('HH:mm:ss')) $Label power: $p" | Tee-Object -Append -FilePath (Join-Path $Out "power.log")
    if ($p -notmatch "status=2") { Write-Output "NOT ON AC at ${Label}: run $Run is invalid"; exit 3 }
}
if (Test-Path $Out) { Remove-Item -Recurse -Force $Out }
New-Item -ItemType Directory -Force $Out | Out-Null
git log --oneline -1 | Tee-Object -FilePath (Join-Path $Out "commit.txt")
git status --short --untracked-files=no | Tee-Object -Append -FilePath (Join-Path $Out "commit.txt")
$vhd = "C:\Users\janak\AppData\Local\wsl\{93b29be9-62d7-4a7c-abe6-e798d7a9cb5d}\ext4.vhdx"
"wsl vhd before: $((Get-Item $vhd).Length) bytes; C: free $((Get-PSDrive C).Free) bytes" | Tee-Object -FilePath (Join-Path $Out "disk.txt")
CheckAc "start"
$Installer = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer"
$env:PATH = "$Installer;$env:PATH"
$VsPath = & vswhere.exe -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
Import-Module (Join-Path $VsPath "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
Enter-VsDevShell -VsInstallPath $VsPath -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64 -vcvars_ver=14.50" | Out-Null
$VsCMake = Join-Path $VsPath "Common7\IDE\CommonExtensions\Microsoft\CMake"
if (Test-Path $VsCMake) { $env:PATH = "$VsCMake\CMake\bin;$VsCMake\Ninja;$env:PATH" }
Set-Location $RepoRoot
$env:MENOS_OPENBLAS_ROOT = & (Join-Path $RepoRoot "scripts\fetch-openblas-windows.ps1")
cmake --build build/windows-bench --target menos_tiered_memory_benchmark | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Output "build failed"; exit 1 }
CheckAc "after-build"
"== cpu side $((Get-Date).ToUniversalTime().ToString('HH:mm:ss'))"
$env:MENOS_TIERED_MEMORY_JSON = Join-Path $Out "cpu.json"
$env:MENOS_TIERED_SCRATCH_DIR = "C:\Users\janak\menos\build\phase23-scratch"
New-Item -ItemType Directory -Force $env:MENOS_TIERED_SCRATCH_DIR | Out-Null
# Through cmd, so PowerShell 5.1 never sees the benchmark's stderr (its banner
# goes there, and under ErrorActionPreference=Stop that would end the run).
$Log = Join-Path $Out "cpu.log"
cmd /c "`"build\windows-bench\benchmarks\menos_tiered_memory_benchmark.exe`" > `"$Log`" 2>&1"
"cpu benchmark exit $LASTEXITCODE"
CheckAc "end"
"wsl vhd after: $((Get-Item $vhd).Length) bytes; C: free $((Get-PSDrive C).Free) bytes" | Tee-Object -Append -FilePath (Join-Path $Out "disk.txt")
"== done $((Get-Date).ToUniversalTime().ToString('HH:mm:ss'))"
