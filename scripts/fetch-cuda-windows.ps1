# Downloads the pinned CUDA compiler (nvcc) and runtime (cudart) for the native
# Windows CUDA build and checks them against recorded SHA-256 hashes
# (docs/adr/0030 section 4). These are NVIDIA's redistributable archives for
# CUDA 12.9.1, the release the Linux wheels are built with; the hashes are the
# ones in NVIDIA's own manifest,
# https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.1.json
#
#   scripts\fetch-cuda-windows.ps1 [-Destination <dir>]
#
# Prints the unpacked directory, which is used as CUDA_PATH (the
# windows-cuda-* presets read it). Nothing is installed system-wide and no
# driver is touched. Does nothing if a verified copy is already there.
param(
    [string]$Destination = (Join-Path $PSScriptRoot "..\build\_deps")
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Version = "12.9.1"
$BaseUrl = "https://developer.download.nvidia.com/compute/cuda/redist"
$Archives = @(
    @{ Name = "cuda_nvcc-windows-x86_64-12.9.86-archive.zip"; Component = "cuda_nvcc"
       Sha256 = "227B109663B5E57D2718BCABB24A4BA0D9D4E52D958E327DC476F7C28691BE85" },
    @{ Name = "cuda_cudart-windows-x86_64-12.9.79-archive.zip"; Component = "cuda_cudart"
       Sha256 = "179E9C43B0735FFE67207B3DA556EB5A0C50F3047961882B7657D3B822D34EF8" }
)

function Get-Verified([string]$Url, [string]$Path, [string]$Sha256) {
    if (-not (Test-Path $Path) -or (Get-FileHash $Path -Algorithm SHA256).Hash -ne $Sha256) {
        Invoke-WebRequest -Uri $Url -OutFile $Path -UseBasicParsing
        $actual = (Get-FileHash $Path -Algorithm SHA256).Hash
        if ($actual -ne $Sha256) {
            Remove-Item $Path -Force
            throw "SHA-256 mismatch for ${Url}: expected $Sha256, got $actual"
        }
    }
}

New-Item -ItemType Directory -Force $Destination | Out-Null
$Destination = (Resolve-Path $Destination).Path
$Root = Join-Path $Destination "cuda-$Version"

$complete = (Test-Path (Join-Path $Root "bin\nvcc.exe")) -and
            (Test-Path (Join-Path $Root "bin\cudart64_12.dll")) -and
            (Test-Path (Join-Path $Root "include\cuda_runtime.h")) -and
            (Test-Path (Join-Path $Root "lib\x64\cudart.lib"))

# Windows' own tar (bsdtar) reads zip archives and can drop the archives'
# top-level directory, so both unpack into one toolkit root. It is named by
# path so that a GNU tar earlier on PATH (Git's, which cannot read zip) is not
# picked up.
$Tar = Join-Path $env:SystemRoot "System32\tar.exe"

foreach ($archive in $Archives) {
    $zip = Join-Path $Destination $archive.Name
    $url = "$BaseUrl/$($archive.Component)/windows-x86_64/$($archive.Name)"
    Get-Verified $url $zip $archive.Sha256
    if (-not $complete) {
        New-Item -ItemType Directory -Force $Root | Out-Null
        & $Tar -xf $zip -C $Root --strip-components 1
        if ($LASTEXITCODE -ne 0) { throw "could not unpack $zip" }
    }
}

foreach ($required in "bin\nvcc.exe", "bin\cudart64_12.dll", "include\cuda_runtime.h", "lib\x64\cudart.lib", "LICENSE") {
    if (-not (Test-Path (Join-Path $Root $required))) {
        throw "the CUDA $Version archives did not provide $required under $Root"
    }
}

Write-Output $Root
