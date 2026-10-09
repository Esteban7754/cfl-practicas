<#
.SYNOPSIS
    Prepara la GPU NVIDIA y ejecuta el plan de experimentos completo, sin intervención.
.DESCRIPTION
    .\lanzar_gpu.ps1                    todo: comprobar GPU, construir imagen, tests y plan
    .\lanzar_gpu.ps1 -SoloPlan          salta la construcción y los tests (p. ej. para reanudar)
    .\lanzar_gpu.ps1 -Simular           muestra las tareas pendientes del plan sin ejecutarlas
    .\lanzar_gpu.ps1 -Plan planes\otro.json -Paralelo 2

    El plan es reanudable: si se corta (Ctrl+C, apagado, error), vuelve a lanzar el mismo
    comando y sigue donde se quedó. Un solo Ctrl+C detiene todas las tareas en curso.
    Mientras se ejecuta, el equipo no entra en suspensión (solo durante este script).
    Resultados e informe: resultados\plan_gpu\INFORME.md
#>
param(
    [string]$Plan = "planes/plan_gpu.json",
    [int]$Paralelo = 0,
    [switch]$SoloPlan,
    [switch]$Simular
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$run = Join-Path $PSScriptRoot "run.ps1"

function Paso([string]$texto) { Write-Host "`n=== $texto" -ForegroundColor Cyan }
function Fallo([string]$texto) { Write-Host "[ERROR] $texto" -ForegroundColor Red; exit 1 }

# ------------------------------------------------------------ 1. GPU y Docker
Paso "1/4 Comprobando la GPU NVIDIA y Docker"
try { $gpu = (nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader) 2>$null } catch { $gpu = $null }
if (-not $gpu) {
    Fallo "No se detecta ninguna GPU NVIDIA. Instala el controlador más reciente de nvidia.com/drivers, reinicia y vuelve a probar."
}
Write-Host "GPU: $gpu"
docker info *> $null
if ($LASTEXITCODE -ne 0) { Fallo "Docker Desktop no está en ejecución. Ábrelo, espera a que arranque y reintenta." }

# Evita la suspensión mientras dure este script (no cambia la configuración de energía).
Add-Type -Namespace Cfl -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][Cfl.Power]::SetThreadExecutionState([uint32]"0x80000001")   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

if (-not $SoloPlan -and -not $Simular) {
    # -------------------------------------------------------- 2. imagen GPU
    Paso "2/4 Construyendo la imagen con CUDA (la primera vez descarga varios GB; tarda un rato)"
    & $run -gpu construir
    if ($LASTEXITCODE -ne 0) { Fallo "Falló la construcción de la imagen GPU." }

    # -------------------------------------------------------- 3. PyTorch ve la GPU y los tests pasan en ella
    Paso "3/4 Comprobando que PyTorch usa la GPU y pasando los tests en ella"
    $info = (& $run -gpu info) | Out-String
    Write-Host $info
    if ($info -notmatch '"cuda_available":\s*true') {
        Fallo "PyTorch no ve la GPU dentro de Docker. Comprueba que Docker Desktop usa WSL 2 y que el controlador NVIDIA está actualizado."
    }
    & $run -gpu tests
    if ($LASTEXITCODE -ne 0) { Fallo "Los tests fallan en la GPU; no se lanza el plan. Copia el error para corregirlo." }
}

# ------------------------------------------------------------ 4. plan
Paso "4/4 Ejecutando el plan $Plan"
$planArgs = @("plan", $Plan)
if ($Simular) { $planArgs += "--dry-run" }
if ($Paralelo -gt 0) { $planArgs += @("--paralelo", "$Paralelo") }
& $run -gpu @planArgs
$code = $LASTEXITCODE
[void][Cfl.Power]::SetThreadExecutionState([uint32]"0x80000000")   # vuelve al comportamiento normal

if ($Simular) { exit $code }
if ($code -eq 0) {
    Write-Host "`nPlan terminado. Informe: resultados\plan_gpu\INFORME.md" -ForegroundColor Green
} else {
    Write-Host "`nEl plan terminó con problemas (código $code). Mira resultados\plan_gpu\INFORME.md y _registros\." -ForegroundColor Yellow
    Write-Host "Para reanudar lo que falte: .\lanzar_gpu.ps1 -SoloPlan" -ForegroundColor Yellow
}
exit $code
