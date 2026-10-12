<#
.SYNOPSIS
    Ejecuta el plan de la GPU y lo recupera solo si la GPU se cuelga dentro de WSL.
.DESCRIPTION
    .\vigilar_gpu.ps1                      plan completo, con recuperación automática
    .\vigilar_gpu.ps1 -Hasta 23:30         igual, con el límite de tiempo de lanzar_gpu.ps1
    .\vigilar_gpu.ps1 -Paralelo 2          menos tareas a la vez (menos memoria de vídeo)
    .\vigilar_gpu.ps1 -MinutosSinAvance 15 umbral para considerar que está colgado

    Lanza .\lanzar_gpu.ps1 -SoloPlan y vigila resultados\<plan>\_registros. Si el registro de
    alguna tarea en curso no se escribe en MinutosSinAvance minutos (cuelgue del controlador
    NVIDIA en WSL: la tarea sigue viva pero la GPU no le responde), para el contenedor, reinicia WSL y Docker Desktop y vuelve a lanzar el plan,
    que sigue donde se quedó. Termina cuando el plan acaba, se pausa por el límite de tiempo
    o falla de otra forma. Un Ctrl+C lo detiene todo.
    Registro de recuperaciones: resultados\<plan>\_registros\vigilante.log
#>
param(
    [string]$Plan = "planes/plan_gpu.json",
    [string]$Hasta = "",
    [int]$Paralelo = 0,
    [int]$MinutosSinAvance = 10,
    [int]$MaxReinicios = 6
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$salida = (Get-Content $Plan -Raw | ConvertFrom-Json).salida
$registros = Join-Path $PSScriptRoot (Join-Path $salida "_registros")
New-Item -ItemType Directory -Force $registros | Out-Null
$log = Join-Path $registros "vigilante.log"

function Nota([string]$texto) {
    $linea = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $texto"
    Write-Host "[VIGILANTE] $linea" -ForegroundColor Magenta
    Add-Content -Path $log -Value $linea -Encoding utf8
}

function Escritura-Mas-Antigua {
    # La de la tarea en curso que lleva más tiempo sin escribir en su registro: basta con que una
    # se cuelgue (la GPU puede bloquear un proceso y dejar seguir a los demás).
    $estado = Join-Path (Split-Path $registros) "estado.json"
    $mas_antigua = Get-Date
    if (Test-Path $estado) {
        $tareas = (Get-Content $estado -Raw | ConvertFrom-Json).PSObject.Properties | Where-Object Value -eq "en curso"
        foreach ($t in $tareas) {
            $f = Get-Item (Join-Path $registros "$($t.Name).log") -ErrorAction SilentlyContinue
            if ($f -and $f.LastWriteTime -lt $mas_antigua) { $mas_antigua = $f.LastWriteTime }
        }
    }
    $mas_antigua
}

function Guardar-Diagnostico {
    # Lo que se pierde al reiniciar WSL: mensajes del núcleo (dxgkrnl), estado de la GPU y
    # errores recientes del controlador en Windows. Va a _registros\diagnostico_<fecha>.txt.
    $archivo = Join-Path $registros "diagnostico_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
    $partes = @("=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')", "", "=== nvidia-smi")
    $partes += (nvidia-smi 2>&1 | Out-String)
    $partes += "=== Errores del controlador (nvlddmkm, Display, WHEA) en la última hora"
    $partes += (Get-WinEvent -FilterHashtable @{LogName = 'System'; StartTime = (Get-Date).AddHours(-1)} -ErrorAction SilentlyContinue |
        Where-Object { $_.ProviderName -match 'nvlddmkm|Display|WHEA|dxgkrnl' } |
        ForEach-Object { "$($_.TimeCreated.ToString('HH:mm:ss.fff')) $($_.ProviderName) $($_.Id) $(([xml]$_.ToXml()).Event.EventData.Data -join ' | ')" } | Out-String)
    $partes += "=== dmesg de WSL (docker-desktop)"
    $partes += (wsl -d docker-desktop -e dmesg 2>&1 | Select-Object -Last 400 | Out-String)
    $partes += "=== docker ps"
    $partes += (docker ps 2>&1 | Out-String)
    Set-Content -Path $archivo -Value $partes -Encoding utf8
    Nota "Diagnóstico guardado en $archivo"
}

function Reiniciar-Docker {
    docker ps -q --filter name=cfl-practicas | ForEach-Object { docker kill $_ *> $null }
    # Todos los procesos de Docker Desktop: si queda alguno, bloquea su socket y no vuelve a arrancar.
    Get-Process | Where-Object { $_.ProcessName -match '^(Docker Desktop|com\.docker\..*|docker-.*)$' } |
        Stop-Process -Force -ErrorAction SilentlyContinue
    wsl --shutdown
    Start-Sleep 10
    Remove-Item "$env:LOCALAPPDATA\docker-secrets-engine\engine.sock*" -Force -ErrorAction SilentlyContinue
    Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep 5
        docker info *> $null
        if ($LASTEXITCODE -eq 0) { Start-Sleep 90; return $true }  # margen para que termine de arrancar
    }
    return $false
}

$args_plan = @("-SoloPlan", "-Plan", $Plan)
if ($Hasta) { $args_plan += @("-Hasta", $Hasta) }
if ($Paralelo -gt 0) { $args_plan += @("-Paralelo", "$Paralelo") }
$reinicios = 0

while ($true) {
    Nota "Lanzando el plan ($Plan)"
    $inicio = Get-Date
    $proc = Start-Process (Get-Process -Id $PID).Path -ArgumentList (@("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $PSScriptRoot "lanzar_gpu.ps1")) + $args_plan) `
        -NoNewWindow -PassThru
    $colgado = $false
    while (-not $proc.HasExited) {
        Start-Sleep 60
        $ultima = Escritura-Mas-Antigua
        if ($ultima -lt $inicio) { $ultima = $inicio }
        if (((Get-Date) - $ultima).TotalMinutes -ge $MinutosSinAvance) { $colgado = $true; break }
    }
    if (-not $colgado) {
        $code = $proc.ExitCode
        Nota "El plan ha terminado (código $code)"
        exit $code
    }
    $reinicios++
    Nota "Sin avance en $MinutosSinAvance min: GPU colgada. Reinicio $reinicios de $MaxReinicios"
    try { Guardar-Diagnostico } catch { Nota "No se pudo guardar el diagnóstico: $_" }
    if ($reinicios -gt $MaxReinicios) { Nota "Demasiados reinicios; me detengo"; exit 1 }
    docker ps -q --filter name=cfl-practicas | ForEach-Object { docker kill $_ *> $null }
    Start-Sleep 5
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
    if (-not (Reiniciar-Docker)) { Nota "Docker no vuelve a arrancar; me detengo"; exit 1 }
    Nota "Docker reiniciado"
}
