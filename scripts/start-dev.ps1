$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$backendDir = Join-Path $root "ml_platform"
$frontendDir = Join-Path $root "ml_platform_web"
$backendLauncher = Join-Path $PSScriptRoot "run_backend_dev.py"
$pythonExe = (Get-Command python).Source
$npmCmd = (Get-Command npm.cmd).Source

function Stop-PortProcess {
    param([int]$Port)

    $connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    if (-not $connections) {
        return
    }

    $processIds = $connections | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($processId in $processIds) {
        try {
            Stop-Process -Id $processId -Force -ErrorAction Stop
        } catch {
        }
    }
}

function Wait-ForUrl {
    param(
        [string]$Url,
        [string]$Name,
        [int]$TimeoutSeconds = 60
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing $Url -TimeoutSec 5
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                Write-Host "$Name ready: $Url"
                return
            }
        } catch {
        }

        Start-Sleep -Seconds 1
    }

    throw "$Name failed to become ready: $Url"
}

foreach ($port in 3000, 8000) {
    Stop-PortProcess -Port $port
}

$backendProcess = Start-Process `
    -FilePath $pythonExe `
    -ArgumentList $backendLauncher `
    -WorkingDirectory $backendDir `
    -WindowStyle Hidden `
    -PassThru

$frontendProcess = Start-Process `
    -FilePath cmd.exe `
    -ArgumentList '/k', ('"' + $npmCmd + '" run dev -- --host 127.0.0.1 --port 3000') `
    -WorkingDirectory $frontendDir `
    -WindowStyle Minimized `
    -PassThru

try {
    Wait-ForUrl -Url "http://127.0.0.1:8000/health" -Name "Backend"
    Wait-ForUrl -Url "http://127.0.0.1:3000" -Name "Frontend"
} catch {
    throw
}

Write-Host ""
Write-Host "Backend PID: $($backendProcess.Id)"
Write-Host "Frontend PID: $($frontendProcess.Id)"
Write-Host "Frontend URL: http://127.0.0.1:3000"
Write-Host "Backend Health: http://127.0.0.1:8000/health"
