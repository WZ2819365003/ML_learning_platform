$ErrorActionPreference = "Stop"

foreach ($port in 3000, 8000) {
    $connections = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if (-not $connections) {
        continue
    }

    $processIds = $connections | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($processId in $processIds) {
        try {
            Stop-Process -Id $processId -Force -ErrorAction Stop
            Write-Host "Stopped process $processId on port $port"
        } catch {
            Write-Host "Failed to stop process $processId on port $port"
        }
    }
}
