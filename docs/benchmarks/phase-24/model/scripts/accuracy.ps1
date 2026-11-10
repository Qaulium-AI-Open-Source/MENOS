# Phase 24 model fixes (docs/adr/0028 section 20): the cost-model accuracy
# benchmark on Windows, three separate processes, in build/windows-bench
# (MSVC, Release). Records accuracy_run{1,2,3}.{csv,err}, the commit and the
# load and power state into build\phase24-model\<tag>\windows. Refuses to
# start, and marks the run invalid, unless on AC at every checkpoint.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File docs\benchmarks\phase-24\model\scripts\accuracy.ps1 <tag>
param([Parameter(Mandatory = $true)][string]$Tag)
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
foreach ($r in 1..3) {
    Note "accuracy_run$r"
    cmd /c "`"build\windows-bench\benchmarks\menos_cost_model_accuracy_benchmark.exe`" > `"$Out\accuracy_run$r.csv`" 2> `"$Out\accuracy_run$r.err`""
    "exit $LASTEXITCODE" | Out-File -Append -Encoding ascii -FilePath (Join-Path $Out "accuracy_run$r.err")
}
Note "end"
"== done $((Get-Date).ToUniversalTime().ToString('HH:mm:ss'))"
