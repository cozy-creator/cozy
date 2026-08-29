# Install or upgrade the cozy CLI from a release asset — install.sh's Windows twin (#449).
#
#   scripts\install.ps1 -Asset <cozy-*.zip|.tar.gz> [-Sums <SHA256SUMS> | -Sha256 <hex>]
#                       [-Prefix <dir>]
#
# THE CHECKSUM IS VERIFIED BEFORE ANYTHING IS REPLACED. A corrupted or substituted asset
# refuses with the two digests named and leaves the installed binary exactly as it was.
#
# Replacement is SIDE-BY-SIDE, because Windows will not overwrite a running executable:
# the live cozy.exe is RENAMED aside (renaming a running exe is allowed), the new one
# takes its name, and the stray is deleted when nothing runs it any more — this install
# sweeps any strays a previous install left. A reader holding the old file keeps running
# it, exactly as the Unix rename promises.
#
# There is no download and no channel here on purpose — same rule as install.sh.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Asset,
    [string]$Sums = "",
    [string]$Sha256 = "",
    [string]$Prefix = "$(if ($env:COZY_PREFIX) { $env:COZY_PREFIX } else { Join-Path $env:LOCALAPPDATA 'cozy' })"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Asset)) { Write-Error "refusing: no such asset: $Asset"; exit 4 }

if (-not $Sha256) {
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

$got = (Get-FileHash -Algorithm SHA256 $Asset).Hash.ToLowerInvariant()
if ($got -ne $Sha256.ToLowerInvariant()) {
    Write-Error ("refusing: {0} does not match its declared checksum`n  declared sha256:{1}`n  actual   sha256:{2}`nnothing was replaced" -f (Split-Path -Leaf $Asset), $Sha256, $got)
    exit 13
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
    if ($Asset -match '\.zip$') { Expand-Archive -Path $Asset -DestinationPath $stage }
    else { tar -xzf $Asset -C $stage; if ($LASTEXITCODE -ne 0) { throw "tar failed" } }
    $new = @(Get-ChildItem -Path $stage -Recurse -Include cozy.exe, cozy | Select-Object -First 1)
    if (-not $new) { Write-Error "refusing: the asset carries no cozy binary"; exit 13 }

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
Write-Output "verified: sha256:$got"
Write-Output "prefix:   $Prefix"
Write-Output "was:      $was"
Write-Output "now:      $now"
Write-Output "note:     add $bin to PATH if it is not already there"
