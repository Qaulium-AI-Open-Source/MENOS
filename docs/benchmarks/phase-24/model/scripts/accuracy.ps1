# Phase 24 model fixes (docs/adr/0028 section 20): the cost-model accuracy
# benchmark on Windows, three separate processes, in build/windows-bench
# (MSVC, Release). Records accuracy_run{1,2,3}.{csv,err}, the commit and the
# load and power state into build\phase24-model\<tag>\windows. Refuses to
# start, and marks the run invalid, unless on AC at every checkpoint.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File docs\benchmarks\phase-24\model\scripts\accuracy.ps1 <tag> [-PinOneThread <hex mask> -PinThreaded <hex mask>]
#
# With the masks (docs/adr/0028 section 22) each run becomes one process per
# engine label, started under a CPU mask (cmd's start /affinity): the labels
# whose engines run on one thread and call no threaded library (stabilizer,
# pauli, tensor_network) under -PinOneThread, every other label (engines
# that start threads, or MPS, whose SVDs run on OpenBLAS's threads) under
# -PinThreaded. On the development machine 4000 is logical CPU 14, a
# performance core, and FFC03 the twelve performance-core threads (logical
# CPUs 0, 1 and 10-19). On 2026-11-11 Windows moved unpinned one-thread work
# onto efficiency cores partway through a process and ran it 3-10x slower
# than the same binary pinned, while the calibration, at the start of the
# process, ran on a performance core. The masks are recorded in commit.txt;
# each run's CSV is the labels' CSVs concatenated under one header, and its
# .err holds each label's calibration.
param([Parameter(Mandatory = $true)][string]$Tag, [string]$PinOneThread = "", [string]$PinThreaded = "")
$ErrorActionPreference = "Stop"
$RepoRoot = "C:\Users\janak\menos"
Set-Location $RepoRoot
$Out = Join-Path $RepoRoot "build\phase24-model\$Tag\windows"
function Note([string]$Label) {
    $b = Get-CimInstance Win32_Battery
    $load = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
    "$Label $((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')) cpu_load=$load% power: status=$($b.BatteryStatus) charge=$($b.EstimatedChargeRemaining)" | Out-File -Append -Encoding ascii -FilePath (Join-Path $Out "load.log")
    if ($b.BatteryStatus -ne 2) { Write-Output "NOT ON AC at ${Label}: the run is invalid"; exit 3 }
}
if (Test-Path $Out) { Remove-Item -Recurse -Force $Out }
New-Item -ItemType Directory -Force $Out | Out-Null
$commit = Join-Path $Out "commit.txt"
git log --oneline -1 | Out-File -Encoding ascii -FilePath $commit
git status --short --untracked-files=no | Out-File -Append -Encoding ascii -FilePath $commit
"pin_one_thread=$(if ($PinOneThread -eq '') { 'none' } else { $PinOneThread }) pin_threaded=$(if ($PinThreaded -eq '') { 'none' } else { $PinThreaded })" | Out-File -Append -Encoding ascii -FilePath $commit
$Installer = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer"
$env:PATH = "$Installer;$env:PATH"
$VsPath = & vswhere.exe -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
Import-Module (Join-Path $VsPath "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
Enter-VsDevShell -VsInstallPath $VsPath -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64 -vcvars_ver=14.50" | Out-Null
$VsCMake = Join-Path $VsPath "Common7\IDE\CommonExtensions\Microsoft\CMake"
if (Test-Path $VsCMake) { $env:PATH = "$VsCMake\CMake\bin;$VsCMake\Ninja;$env:PATH" }
Set-Location $RepoRoot
$env:MENOS_OPENBLAS_ROOT = & (Join-Path $RepoRoot "scripts\fetch-openblas-windows.ps1")
cmd /c "cmake --build build/windows-bench --target menos_cost_model_accuracy_benchmark > `"$Out\build.log`" 2>&1"
if ($LASTEXITCODE -ne 0) { Write-Output "build failed"; exit 1 }
$Exe = "`"build\windows-bench\benchmarks\menos_cost_model_accuracy_benchmark.exe`""
$OneThread = @("stabilizer", "pauli", "tensor_network")
$Labels = @("dense", "dense_cuda", "stabilizer", "pauli", "mps", "tensor_network", "cut", "factorized", "density_matrix", "trajectory")
foreach ($r in 1..3) {
    Note "accuracy_run$r"
    $Csv = "$Out\accuracy_run$r.csv"
    $Err = "$Out\accuracy_run$r.err"
    if ($PinOneThread -eq "" -and $PinThreaded -eq "") {
        cmd /c "$Exe > `"$Csv`" 2> `"$Err`""
        "exit $LASTEXITCODE" | Out-File -Append -Encoding ascii -FilePath $Err
        continue
    }
    $first = $true
    foreach ($label in $Labels) {
        $Mask = if ($OneThread -contains $label) { $PinOneThread } else { $PinThreaded }
        $Start = if ($Mask -eq "") { "" } else { "start `"`" /wait /b /affinity $Mask " }
        cmd /c "$Start$Exe $label > `"$Out\part.csv`" 2> `"$Out\part.err`""
        $code = $LASTEXITCODE
        $lines = Get-Content "$Out\part.csv"
        if (-not $first) { $lines = $lines | Select-Object -Skip 1 }
        $lines | Out-File -Append -Encoding ascii -FilePath $Csv
        "== $label affinity=$(if ($Mask -eq '') { 'none' } else { $Mask })" | Out-File -Append -Encoding ascii -FilePath $Err
        Get-Content "$Out\part.err" | Out-File -Append -Encoding ascii -FilePath $Err
        "exit $code" | Out-File -Append -Encoding ascii -FilePath $Err
        $first = $false
    }
    Remove-Item "$Out\part.csv", "$Out\part.err"
}
Note "end"
"== done $((Get-Date).ToUniversalTime().ToString('HH:mm:ss'))"
