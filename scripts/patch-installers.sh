#!/usr/bin/env bash
# patch-installers.sh <dir>: dist's installers in <dir>, as the release ships them (release.yml's
# global job and release-smoke's mirror both run it). A computer with no download (musl, glibc
# before 2.28) is told what cozy runs on, and `irm … | iex` puts cozy on that PowerShell's PATH at
# once. Fails when dist's text moved and a patch no longer applies.
set -euo pipefail
sh=$1/cozy-installer.sh ps=$1/cozy-installer.ps1
supported='cozy runs on Linux x86_64 or aarch64 with glibc 2.28 or newer (not musl), macOS, and Windows x86_64 or arm64'
sed -i "s|for your platform \$_true_arch\"|for your platform \$_true_arch. $supported\"|" "$sh"
test "$(grep -c "$supported" "$sh")" = 2
sed -i 's|^    if (Add-Path $dest_dir) {$|&\n        $env:Path = "$dest_dir;$env:Path"|; s|To add $dest_dir to your PATH, either restart your shell or run:|cozy works in this PowerShell now; in terminals already open, restart them or run:|' "$ps"
test "$(grep -c -e 'cozy works in this PowerShell now' -e '^        $env:Path = "$dest_dir;$env:Path"$' "$ps")" = 2
