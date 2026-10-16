param(
    [switch]$Foreground
)

$ErrorActionPreference = 'Stop'
$a10 = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$root = $a10
$state = Join-Path $a10 'state'
$logs = Join-Path $a10 'logs'
$python = 'C:\Users\Admin Riset\miniconda3\envs\quantum_env\python.exe'
$entry = Join-Path $a10 'run_article10.py'

New-Item -ItemType Directory -Force -Path $state, $logs | Out-Null

$pidFile = Join-Path $state 'pipeline.pid'
if (Test-Path $pidFile) {
    $oldPid = [int](Get-Content $pidFile -Raw)
    $oldProcess = Get-Process -Id $oldPid -ErrorAction SilentlyContinue
    if ($null -ne $oldProcess) {
        Write-Output "Article 10 pipeline is already running with PID $oldPid."
        exit 0
    }
}

$stdout = Join-Path $logs 'pipeline.stdout.log'
$stderr = Join-Path $logs 'pipeline.stderr.log'
$arguments = @($entry, '--stage', 'all', '--resume')

if ($Foreground) {
    & $python @arguments
    exit $LASTEXITCODE
}

$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $root -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
Set-Content -LiteralPath $pidFile -Value $process.Id -Encoding ascii
$launch = @{
    pid = $process.Id
    launched_at = (Get-Date).ToUniversalTime().ToString('o')
    python = $python
    entry = $entry
    stdout = $stdout
    stderr = $stderr
    resume = $true
} | ConvertTo-Json
Set-Content -LiteralPath (Join-Path $state 'launch.json') -Value $launch -Encoding utf8
Write-Output "Article 10 pipeline launched with PID $($process.Id)."
Write-Output "It is checkpointed and will not be polled automatically."

