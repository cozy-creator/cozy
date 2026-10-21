<#
Windows + NVIDIA proof (xs-061): this computer's facts, cozy installed by its real install line, the
machine enabled with GPU 0 listed, each function run here with a unique prompt and seed, and each
output checked as a real image. Transcript, outputs, logs and report.json land in -Out, zipped beside
it. Exits 1 when any step fails. Runs as you, in the default ~/.cozy home (Windows PowerShell 5.1+):

  powershell -ExecutionPolicy Bypass -File windows-gpu-proof.ps1 [-DownloadUrl <release dir>] [-Runs 'org/pkg/fn [key=value ...]', ...]
#>
param(
    [string]$Out = (Join-Path $HOME ('cozy-windows-gpu-proof-' + (Get-Date -Format yyyyMMdd-HHmmss))),
    # A candidate release (COZY_DOWNLOAD_URL, as release-smoke serves one); empty: tensorhub.com's install line.
    [string]$DownloadUrl,
    # Each entry: a function, then optional key=value inputs. A prompt and a seed are added where the
    # function takes them, and an image input is the previous run's image.
    [string[]]$Runs = @('fidika/sdxl/generate', 'fidika/birefnet/remove_background',
        'fidika/qwen-image-2/generate_image', 'fidika/flux-3-klein/generate')
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
Add-Type -AssemblyName System.Drawing
if ($env:COZY_HOME) { throw 'COZY_HOME is set: this proof runs in the default ~/.cozy home' }
$Out = (New-Item -ItemType Directory -Force $Out).FullName
New-Item -ItemType Directory -Force "$Out\logs" | Out-Null
Start-Transcript -Path "$Out\transcript.txt" | Out-Null
$cozy = "$HOME\.cozy\bin\cozy.exe"
$failures = New-Object Collections.Generic.List[string]
$report = [ordered]@{ facts = [ordered]@{}; cozy = $null; gpus = @(); runs = New-Object Collections.Generic.List[object]; failures = $failures }

# A step: its failure is recorded and printed, and the proof goes on to what does not depend on it.
function Step([string]$Name, [scriptblock]$Body) {
    Write-Host "`n== $Name  [$(Get-Date -Format HH:mm:ss)]"
    try { & $Body | Out-Host; Write-Host "== ${Name}: ok"; $true }
    catch { $failures.Add("${Name}: $($_.Exception.Message)"); Write-Host "== ${Name}: FAILED: $($_.Exception.Message)"; $false }
}
# Runs a native command, its stdout and stderr into the transcript; a non-zero exit throws.
function Exec([scriptblock]$Command) {
    $ErrorActionPreference = 'Continue'; $global:LASTEXITCODE = 0
    & $Command 2>&1 | ForEach-Object { Write-Host "$_" }
    if ($LASTEXITCODE) { throw "exited ${LASTEXITCODE}: $Command" }
}
# A native command's output into logs\<Name>; a non-zero exit throws.
function Save([string]$Name, [scriptblock]$Command) {
    $ErrorActionPreference = 'Continue'; $global:LASTEXITCODE = 0
    & $Command 2>&1 | ForEach-Object { "$_" } | Set-Content -Encoding UTF8 "$Out\logs\$Name"
    if ($LASTEXITCODE) { throw "exited ${LASTEXITCODE}: $Command" }
}
# cozy --json <words>, parsed; its stderr into the transcript; a non-zero exit throws.
function CozyJson {
    $ErrorActionPreference = 'Continue'; $global:LASTEXITCODE = 0
    $lines = & $cozy --json @args 2>&1 | ForEach-Object {
        if ($_ -is [Management.Automation.ErrorRecord]) { Write-Host "$_" } else { "$_" }
    }
    if ($LASTEXITCODE) { $lines | Write-Host; throw "cozy --json $args exited $LASTEXITCODE" }
    ($lines -join "`n") | ConvertFrom-Json
}
# An image decoded (WebP through cozy's ffmpeg: System.Drawing reads PNG, JPEG, BMP, GIF and TIFF), with
# its size and the spread of its channel values over a 32x32 grid of pixels (0 for one colour).
function Measure-Image([string]$Path) {
    if ($Path -match '\.webp$') {
        if (-not $script:ffmpeg) {
            Exec { & $cozy tools install ffmpeg --if-missing }
            $script:ffmpeg = ((CozyJson tools list).tools | Where-Object { $_.name -eq 'ffmpeg' }).path
            if (-not $script:ffmpeg) { throw 'cozy tools list names no ffmpeg after installing it' }
        }
        Exec { & $script:ffmpeg -v error -y -i $Path "$Path.png" }
        $Path = "$Path.png"
    }
    $bitmap = New-Object Drawing.Bitmap $Path
    try {
        $values = foreach ($y in 0..31) { foreach ($x in 0..31) {
            $p = $bitmap.GetPixel([math]::Floor(($x + .5) * $bitmap.Width / 32), [math]::Floor(($y + .5) * $bitmap.Height / 32))
            $p.R; $p.G; $p.B; $p.A
        } }
        $mean = ($values | Measure-Object -Average).Average
        $variance = ($values | ForEach-Object { ($_ - $mean) * ($_ - $mean) } | Measure-Object -Average).Average
        [pscustomobject]@{ Width = $bitmap.Width; Height = $bitmap.Height; Spread = [math]::Round([math]::Sqrt($variance), 1) }
    } finally { $bitmap.Dispose() }
}

Step 'machine facts' {
    $f = $report.facts
    $os = Get-CimInstance Win32_OperatingSystem
    $memory = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
    $f.os = "$($os.Caption) $($os.Version)"
    $f.cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
    $f.elevated = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole('Administrators')
    $f.ram_gb = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
    # \Memory\Committed Bytes and \Memory\Commit Limit, read locale-independently.
    $f.committed_gb = [math]::Round($memory.CommittedBytes / 1GB, 1)
    $f.commit_limit_gb = [math]::Round($memory.CommitLimit / 1GB, 1)
    $f.disk_free_gb = [math]::Round((New-Object IO.DriveInfo (Split-Path $HOME -Qualifier)).AvailableFreeSpace / 1GB, 1)
    $f.display_adapters = @(Get-CimInstance Win32_VideoController | ForEach-Object { "$($_.Name) [$($_.PNPDeviceID)] driver $($_.DriverVersion)" })
    $smi = @((Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue).Source, "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe") |
        Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
    $script:smiNote = 'nvidia-smi: absent, so no NVIDIA driver'
    if ($smi) {
        Exec { & $smi }
        Save nvidia-smi-q.txt { & $smi -q }
        $global:LASTEXITCODE = 0
        $rows = & $smi --query-gpu=name,driver_version,compute_cap,memory.total,driver_model.current --format=csv,noheader
        if ($LASTEXITCODE) { throw "nvidia-smi --query-gpu exited ${LASTEXITCODE}: $rows" }
        $f.nvidia = @($rows | ForEach-Object {
            $name, $driver, $cc, $mem, $model = $_ -split ',\s*'
            [ordered]@{ name = $name; driver = $driver; compute_capability = $cc; memory = $mem; driver_model = $model }
        })
        $script:smiNote = 'nvidia-smi: ' + (($f.nvidia | ForEach-Object { "$($_.name), driver $($_.driver) $($_.driver_model)" }) -join '; ') +
            '; cozy needs driver 580.88 or newer'
    }
    $f | ConvertTo-Json -Depth 4 | Write-Host
} | Out-Null

$installed = Step 'install cozy' {
    $line = if ($DownloadUrl) { "`$env:COZY_DOWNLOAD_URL = '$DownloadUrl'; irm '$DownloadUrl/cozy-installer.ps1' | iex" }
            else { 'irm https://tensorhub.com/install.ps1 | iex' }
    Write-Host "> $line"
    Exec { powershell.exe -NoProfile -ExecutionPolicy Bypass -Command $line }
    $report.cozy = & $cozy --version
    Write-Host $report.cozy
}

$enabled = $installed -and (Step 'enable the machine' {
    Exec { & $cozy machine enable }
    Exec { & $cozy machine show local --once }
})

$gpu = $enabled -and (Step 'GPU listed' {
    Exec { & $cozy doctor }
    $report.gpus = @((CozyJson machine show local --once).status.gpus | Where-Object { $_ })
    if (-not $report.gpus) { throw "no NVIDIA GPU: machine local lists none ($smiNote)" }
    $report.gpus | ForEach-Object { Write-Host "GPU $($_.index): $($_.name), $($_.family), $([math]::Round($_.memory_bytes / 1GB, 1)) GB, driver $($_.driver)" }
})

$subjects = 'a red fox curled up in fresh snow', 'a lighthouse on a basalt cliff', 'a bowl of ramen on a worn wooden table',
    'an astronaut tending tomatoes in a greenhouse', 'a vintage motorcycle in a narrow alley', 'a heron in a misty marsh',
    'a glass teapot of blooming tea', 'a fishing village after rain' | Get-Random -Count 8
$styles = 'watercolor', '35mm photograph', 'oil painting', 'ink and wash', 'studio photograph', 'isometric illustration'
$lights = 'at golden hour', 'under overcast light', 'at blue hour', 'by candlelight', 'in harsh noon sun', 'under neon light'
$image = $null
if (-not $Runs) { $failures.Add('runs: none given') }
if (-not $gpu) { Write-Host "`nruns not attempted: the steps above failed" }
for ($i = 0; $gpu -and $i -lt $Runs.Count; $i++) {
    $target, $rest = "$($Runs[$i])".Trim() -split '\s+', 2
    $entry = [ordered]@{ target = $target; inputs = @("$rest" -split '\s+' | Where-Object { $_ }) }
    $report.runs.Add($entry)
    Step "run $target" {
        $fields = @{}
        (CozyJson run $target --describe).fields | ForEach-Object { $fields[$_.name] = $_ }
        $inputs = $entry.inputs
        if ($fields.prompt -and -not ($inputs -like 'prompt=*')) {
            $inputs += "prompt=$($subjects[$i % 8]), $(Get-Random -InputObject $styles), $(Get-Random -InputObject $lights)"
        }
        if ($fields.seed -and -not ($inputs -like 'seed=*')) { $inputs += "seed=$(Get-Random -Maximum 2147483647)" }
        $source = $null
        foreach ($field in $fields.Values | Where-Object { $_.type.asset -eq 'image' -and -not $_.PSObject.Properties['default'] }) {
            if (-not $image) { throw "its input $($field.name) is an image, and no earlier run made one" }
            $source = $image
            $inputs += '--asset', "$($field.name)=$($image.path)"
        }
        $aspect = "$(($inputs -like 'aspect_ratio=*') -replace '^aspect_ratio=')"
        if (-not $aspect -and $fields.aspect_ratio) { $aspect = $fields.aspect_ratio.default }
        $entry.inputs = $inputs
        $dir = "$Out\outputs\$($i + 1)-$($target -replace '/', '-')"
        $clock = [Diagnostics.Stopwatch]::StartNew()
        $run = CozyJson run $target @inputs --out $dir --detach
        $entry.run = $run.number; $entry.id = $run.id
        Write-Host "run $($run.number) ($($run.id)): $target $inputs"
        try { Exec { & $cozy run watch $run.number } }
        finally {
            $entry.wall_s = [math]::Round($clock.Elapsed.TotalSeconds, 1)
            Exec { & $cozy run show $run.number --once }
            $done = CozyJson run show $run.number --once
            $entry.status = $done.status; $entry.execution_s = $done.execution_ms / 1000
        }
        $images = @($done.output | Where-Object { $_.media_type -like 'image/*' })
        if (-not $images) { throw 'it made no image' }
        $entry.outputs = @(foreach ($item in $images) {
            if (-not $item.path) { throw "$($item.name) was not saved here" }
            $m = Measure-Image $item.path
            $size = "$($m.Width)x$($m.Height)"
            Write-Host "$($item.name): $($item.path) $size, spread $($m.Spread)"
            $checked = 0
            if ($done.result.width) {
                $checked++
                if ($size -ne "$($done.result.width)x$($done.result.height)") { throw "$($item.name) is $size; its result says $($done.result.width)x$($done.result.height)" }
            }
            if ($source) {
                $checked++
                if ($size -ne $source.size) { throw "$($item.name) is $size; its input image is $($source.size)" }
            }
            if ($aspect -match '^(\d+):(\d+)$') {
                $checked++
                if ([math]::Abs($m.Width / $m.Height * $Matches[2] / $Matches[1] - 1) -gt .02) { throw "$($item.name) is $size, not $aspect" }
            }
            if (-not $checked) { throw "nothing states the size $($item.name) should have" }
            if ($m.Spread -lt 4) { throw "$($item.name) is one colour (spread $($m.Spread))" }
            [ordered]@{ name = $item.name; path = $item.path; size = $size; spread = $m.Spread }
        })
        $script:image = $entry.outputs[0]
    } | Out-Null
}

if (Test-Path $cozy) {
    Step 'collect logs' {
        Save status.txt { & $cozy status }
        Save calcifer.log { & $cozy logs calcifer }
        foreach ($log in 'tensord', 'tensorfs', 'memory') { Save "machine-$log.log" { & $cozy machine logs local $log -n 0 } }
    } | Out-Null
}

Write-Host "`n== summary"
$report.runs | ForEach-Object { Write-Host ("{0}: run {1}, {2}, {3} s wall, {4} s executing" -f $_.target, $_.run, $_.status, $_.wall_s, $_.execution_s) }
if ($failures.Count) { $failures | ForEach-Object { Write-Host "FAILED $_" } } else { Write-Host 'every step passed' }
$report | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 "$Out\report.json"
Stop-Transcript | Out-Null
Compress-Archive -Force -Path "$Out\*" -DestinationPath "$Out.zip"
Write-Host "bundle: $Out ($Out.zip)"
if ($failures.Count) { exit 1 }
