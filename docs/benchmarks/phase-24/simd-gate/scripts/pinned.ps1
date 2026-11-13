# Phase 24 item 8 gate on Windows, the one-thread rows pinned to one core
# (docs/adr/0028 section 19.8): the reference and current
# menos_dense_simd_probe built by gate.ps1, alternated four times with
# `start /affinity` on logical CPU $Cpu (default 2), so a single thread cannot
# land on an efficiency core. Writes old-<r>.csv and new-<r>.csv to
# build\phase24-simd-gate\windows-pinned; bench\tools\simd_pinned_table.py
# prints the ratios.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File docs\benchmarks\phase-24\simd-gate\scripts\pinned.ps1
param([int]$Cpu = 2, [string]$MaxQubits = "20")
$ErrorActionPreference = "Stop"
$RepoRoot = "C:\Users\janak\menos"
$Out = Join-Path $RepoRoot "build\phase24-simd-gate\windows-pinned"
if (Test-Path $Out) { Remove-Item -Recurse -Force $Out }
New-Item -ItemType Directory -Force $Out | Out-Null
$Mask = "{0:X}" -f ([int64]1 -shl $Cpu)
$OldExe = "C:\Users\janak\menos-simd-ref\build\windows-bench\benchmarks\menos_dense_simd_probe.exe"
$NewExe = Join-Path $RepoRoot "build\windows-bench\benchmarks\menos_dense_simd_probe.exe"
$b = Get-CimInstance Win32_Battery
"start $((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')) power: status=$($b.BatteryStatus) affinity=0x$Mask" | Out-File -Encoding ascii (Join-Path $Out "load.log")
if ($b.BatteryStatus -ne 2) { Write-Output "NOT ON AC: the run is invalid"; exit 3 }
foreach ($r in 1..4) {
    cmd /c "start `"`" /wait /affinity $Mask cmd /c `"`"$OldExe`" $MaxQubits > `"$Out\old-$r.csv`" 2>nul`""
    cmd /c "start `"`" /wait /affinity $Mask cmd /c `"`"$NewExe`" $MaxQubits > `"$Out\new-$r.csv`" 2>nul`""
}
$b = Get-CimInstance Win32_Battery
"end $((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')) power: status=$($b.BatteryStatus)" | Out-File -Append -Encoding ascii (Join-Path $Out "load.log")
if ($b.BatteryStatus -ne 2) { Write-Output "NOT ON AC at the end: the run is invalid"; exit 3 }
"== done"
