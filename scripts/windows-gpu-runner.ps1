<#
Paul: this lets cozy-creator's windows-gpu workflow run ONE job on this PC, then removes itself.
  Installs: GitHub's Actions runner in %LOCALAPPDATA%\cozy-gpu-runner, deleted when it stops. It is a single-use
  (JIT) runner under a random label: it takes the one job it was made for, and stops if none comes in 15 min.
  The job (scripts/windows-gpu-proof.ps1) installs cozy as `irm https://tensorhub.com/install.ps1 | iex` does,
  enables this PC as a cozy machine, downloads its models into ~\.cozy and runs them on your GPU, as you.
  Remove: the runner is gone when this script ends; `cozy uninstall` removes cozy, its models and its login entry.
Without GitHub: powershell -ExecutionPolicy Bypass -File windows-gpu-proof.ps1, then send the zip it names.

  powershell -ExecutionPolicy Bypass -File windows-gpu-runner.ps1 -JitConfig <config>

The lead, first (the dispatch queues the job; the config is single-use, for that label only):
  tag=$(openssl rand -hex 6); label=windows-rtx-$tag
  gh workflow run windows-gpu.yaml -R cozy-creator/cozy-creator -f runner=$label -f tag=$tag [-f version=<release>]
  gh api -X POST repos/cozy-creator/cozy-creator/actions/runners/generate-jitconfig -f name=$label -F runner_group_id=1 -f "labels[]=$label" --jq .encoded_jit_config
#>
param(
    [string]$JitConfig,
    # The config in a file, read and deleted: a scheduled task's arguments then carry no secret.
    [string]$JitConfigFile
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
if ($JitConfigFile) { $JitConfig = (Get-Content -Raw $JitConfigFile).Trim(); Remove-Item $JitConfigFile }
if (-not $JitConfig) { throw 'give -JitConfig (or -JitConfigFile): the lead makes it with generate-jitconfig' }
$dir = "$env:LOCALAPPDATA\cozy-gpu-runner"
$release = Invoke-RestMethod https://api.github.com/repos/actions/runner/releases/latest
$version = $release.tag_name.TrimStart('v')
$zip = "$env:TEMP\actions-runner-win-x64-$version.zip"
Invoke-WebRequest -UseBasicParsing "https://github.com/actions/runner/releases/download/v$version/actions-runner-win-x64-$version.zip" -OutFile $zip
$sha256 = [regex]::Match($release.body, '<!-- BEGIN SHA win-x64 -->([0-9a-f]{64})').Groups[1].Value
if ((Get-FileHash $zip -Algorithm SHA256).Hash -ne $sha256) { throw "$zip does not match the release's sha256 $sha256" }
if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
Expand-Archive $zip $dir
Remove-Item $zip
try {
    $listener = Start-Process "$dir\bin\Runner.Listener.exe" -ArgumentList 'run', '--jitconfig', $JitConfig -WorkingDirectory $dir -NoNewWindow -PassThru
    $null = $listener.Handle
    $started = Get-Date
    # A job has begun once its worker writes a log; with none in 15 min, the runner stops (the owner's idle rule).
    while (-not $listener.WaitForExit(10000)) {
        if (-not (Test-Path "$dir\_diag\Worker_*.log") -and ((Get-Date) - $started).TotalMinutes -ge 15) {
            Stop-Process -Id $listener.Id
            throw 'no job reached this runner in 15 min: it stopped'
        }
    }
    if ($listener.ExitCode) { throw "the runner exited $($listener.ExitCode)" }
} finally {
    Remove-Item -Recurse -Force $dir
}
