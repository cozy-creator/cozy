# Install or upgrade cozy and this host's tool (cozy-runtime) —
# install.sh's Windows twin (#449).
#
#   scripts\install.ps1 -Asset <cozy-*.zip|.tar.gz> [-Sums <SHA256SUMS> | -Sha256 <hex>]
#                       [-Prefix <dir>]
#   scripts\install.ps1 -Binary <built cozy.exe> [-Prefix <dir>]
#
# THE CHECKSUM IS VERIFIED BEFORE ANYTHING IS REPLACED. A corrupted or substituted asset
# refuses with the two digests named and leaves the installed binary exactly as it was.
# The staged binary must run, and the host tools must install, before cozy is replaced.
# Keep $hostTools equal to hostruntime.InstallCommand.
#
# Replacement is SIDE-BY-SIDE, because Windows will not overwrite a running executable:
# the live cozy.exe is RENAMED aside (renaming a running exe is allowed), the new one
# takes its name, and the stray is deleted when nothing runs it any more — this install
# sweeps any strays a previous install left. A reader holding the old file keeps running
# it, exactly as the Unix rename promises.
#
# Windows has no release yet, so this installs a local build or asset (install.sh downloads).

[CmdletBinding()]
param(
    [string]$Asset = "",
    [string]$Binary = "",
    [string]$Sums = "",
    [string]$Sha256 = "",
    [string]$Prefix = "$(if ($env:COZY_PREFIX) { $env:COZY_PREFIX } else { Join-Path $env:LOCALAPPDATA 'cozy' })"
)

$ErrorActionPreference = "Stop"
$hostTools = @('tool', 'install', '--force', '--refresh-package', 'cozy-runtime', '--python', '3.12', 'cozy-runtime[media,model-execution]>=0.19.0')

if ([bool]$Asset -eq [bool]$Binary) { Write-Error "refusing: give exactly one of -Asset or -Binary"; exit 2 }
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "refusing: uv is required to install cozy-runtime — https://docs.astral.sh/uv/getting-started/installation/"; exit 6
}
if ($Binary -and -not (Test-Path $Binary)) { Write-Error "refusing: no such binary: $Binary"; exit 4 }
if ($Asset -and -not (Test-Path $Asset)) { Write-Error "refusing: no such asset: $Asset"; exit 4 }

if ($Asset -and -not $Sha256) {
    if (-not $Sums) { $Sums = Join-Path (Split-Path -Parent (Resolve-Path $Asset)) "SHA256SUMS" }
    if (-not (Test-Path $Sums)) {
        Write-Error "refusing: no checksum given and no SHA256SUMS beside the asset"; exit 4
    }
    $name = Split-Path -Leaf $Asset
    foreach ($line in Get-Content $Sums) {
        $parts = -split $line
        if ($parts.Count -ge 2 -and ($parts[1] -replace '^\./', '') -eq $name) { $Sha256 = $parts[0] }
    }
    if (-not $Sha256) { Write-Error "refusing: $Sums names no line for $name"; exit 4 }
}

if ($Asset) {
    $got = (Get-FileHash -Algorithm SHA256 $Asset).Hash.ToLowerInvariant()
    if ($got -ne $Sha256.ToLowerInvariant()) {
        Write-Error ("refusing: {0} does not match its declared checksum`n  declared sha256:{1}`n  actual   sha256:{2}`nnothing was replaced" -f (Split-Path -Leaf $Asset), $Sha256, $got)
        exit 13
    }
}

$bin = Join-Path $Prefix "bin"
New-Item -ItemType Directory -Force -Path $bin | Out-Null
$target = Join-Path $bin "cozy.exe"

$was = "none"
if (Test-Path $target) {
    $was = (& $target -v 2>$null).Trim()
    if (-not $was) { $was = "unreadable" }
}

# Stage INSIDE the target directory so the final move never crosses a volume.
$stage = Join-Path $bin (".cozy-install." + [System.IO.Path]::GetRandomFileName())
New-Item -ItemType Directory -Path $stage | Out-Null
try {
    if ($Binary) { Copy-Item -Path $Binary -Destination (Join-Path $stage "cozy.exe") }
    elseif ($Asset -match '\.zip$') { Expand-Archive -Path $Asset -DestinationPath $stage }
    else { tar -xzf $Asset -C $stage; if ($LASTEXITCODE -ne 0) { throw "tar failed" } }
    $new = @(Get-ChildItem -Path $stage -Recurse -Include cozy.exe, cozy | Select-Object -First 1)
    if (-not $new) { Write-Error "refusing: the asset carries no cozy binary"; exit 13 }
    & $new[0].FullName -v | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Error "refusing: the new cozy binary does not run; nothing was replaced"; exit 13 }

    & uv @hostTools
    if ($LASTEXITCODE -ne 0) { Write-Error "refusing: host tool installation failed; cozy was not replaced"; exit 13 }
    $tools = (& uv tool dir --bin).Trim()
    $runtime = ((& (Join-Path $tools "cozy-runtime.exe") version) -match '^distribution:') -replace '^distribution:\s*', ''

    # Sweep strays a PREVIOUS side-by-side replacement left behind; one still held open
    # by a running process simply stays for the next sweep.
    Get-ChildItem -Path $bin -Filter ".cozy-old-*.exe" -ErrorAction SilentlyContinue |
        ForEach-Object { Remove-Item $_.FullName -ErrorAction SilentlyContinue }

    if (Test-Path $target) {
        $aside = Join-Path $bin (".cozy-old-" + [System.IO.Path]::GetRandomFileName() + ".exe")
        Move-Item -Path $target -Destination $aside
        Remove-Item $aside -ErrorAction SilentlyContinue
    }
    Move-Item -Path $new[0].FullName -Destination $target
}
finally {
    Remove-Item -Recurse -Force $stage -ErrorAction SilentlyContinue
}

$now = (& $target -v 2>$null).Trim()
if ($Asset) { Write-Output "verified: sha256:$got" }
Write-Output "prefix:   $Prefix"
Write-Output "was:      $was"
Write-Output "now:      $now"
Write-Output "runtime:  cozy-runtime $runtime"
Write-Output "note:     add $bin and $tools to PATH if they are not already there"
