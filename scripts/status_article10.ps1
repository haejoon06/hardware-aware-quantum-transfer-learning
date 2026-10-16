$ErrorActionPreference = 'Stop'
$a10 = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$root = $a10
$state = Join-Path $a10 'state'
$pidFile = Join-Path $state 'pipeline.pid'

if (Test-Path $pidFile) {
    $pipelinePid = [int](Get-Content $pidFile -Raw)
    $process = Get-Process -Id $pipelinePid -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        Write-Output "Process $pipelinePid is not running."
    } else {
        Write-Output "Process $pipelinePid is running (CPU seconds: $([math]::Round($process.CPU, 1)))."
    }
} else {
    Write-Output 'No PID file exists.'
}

$statusPath = Join-Path $state 'pipeline_status.json'
if (Test-Path $statusPath) {
    Get-Content $statusPath
}

Get-ChildItem $state -Filter '*.json' -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '^(screen|ablate|final)_' } |
    Sort-Object Name |
    ForEach-Object { Get-Content $_.FullName -Raw }

if (Test-Path (Join-Path $state 'SUCCESS')) { Write-Output 'PIPELINE SUCCESS' }
if (Test-Path (Join-Path $state 'FAILED')) { Write-Output 'PIPELINE FAILED'; Get-Content (Join-Path $state 'FAILED') -Tail 30 }

