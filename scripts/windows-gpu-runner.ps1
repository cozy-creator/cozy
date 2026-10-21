<#
Paul: this lets cozy-creator's windows-gpu workflow run ONE job on this PC, then removes itself.
  Installs: GitHub's Actions runner in %LOCALAPPDATA%\cozy-gpu-runner, deleted when the job ends.
  The job (scripts/windows-gpu-proof.ps1) installs cozy as `irm https://tensorhub.com/install.ps1 | iex` does,
  enables this PC as a cozy machine, downloads its models into ~\.cozy and runs them on your GPU, as you.
  Remove: the runner unregisters itself after its job; `cozy uninstall` removes cozy, its models and its login entry.
Without GitHub: powershell -ExecutionPolicy Bypass -File windows-gpu-proof.ps1, then send the zip it names.

  powershell -ExecutionPolicy Bypass -File windows-gpu-runner.ps1 -Token <registration token> [-Label windows-rtx]
The token: gh api -X POST repos/cozy-creator/cozy-creator/actions/runners/registration-token --jq .token
#>
param(
    [Parameter(Mandatory = $true)][string]$Token,
    [string]$Label = 'windows-rtx',
    [string]$Repo = 'cozy-creator/cozy-creator'
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
$dir = "$env:LOCALAPPDATA\cozy-gpu-runner"
$release = Invoke-RestMethod https://api.github.com/repos/actions/runner/releases/latest
$version = $release.tag_name.TrimStart('v')
$zip = "$env:TEMP\actions-runner-win-x64-$version.zip"
Invoke-WebRequest "https://github.com/actions/runner/releases/download/v$version/actions-runner-win-x64-$version.zip" -OutFile $zip
$sha256 = [regex]::Match($release.body, '<!-- BEGIN SHA win-x64 -->([0-9a-f]{64})').Groups[1].Value
if ((Get-FileHash $zip -Algorithm SHA256).Hash -ne $sha256) { throw "$zip does not match the release's sha256 $sha256" }
if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
Expand-Archive $zip $dir
Remove-Item $zip
Push-Location $dir
try {
    & .\config.cmd --unattended --ephemeral --url "https://github.com/$Repo" --token $Token --labels $Label --name "$Label-$env:COMPUTERNAME" --replace
    if ($LASTEXITCODE) { throw "config.cmd exited $LASTEXITCODE" }
    # Takes one job, then GitHub removes this ephemeral runner and run.cmd returns.
    & .\run.cmd
    if ($LASTEXITCODE) { throw "run.cmd exited $LASTEXITCODE" }
} finally {
    Pop-Location
    Remove-Item -Recurse -Force $dir
}
