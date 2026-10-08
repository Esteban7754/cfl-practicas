<#
.SYNOPSIS
    Lanzador único del proyecto: construye la imagen Docker si hace falta y ejecuta en ella.
.DESCRIPTION
    .\run.ps1                         menú interactivo
    .\run.ps1 construir               (re)construye la imagen con la versión actual del código
    .\run.ps1 info                    entorno, datos, GPU y versión de la imagen
    .\run.ps1 tests                   tests unitarios
    .\run.ps1 rapido                  prueba funcional con CIFAR-100 real + verificación
    .\run.ps1 comparar --seeds 42 43 44 --buffers 0 5 10 20 [--replay-mix balanced] [--distill-weight 1]
    .\run.ps1 semillas domainnet.py --seeds 42 43 44 --buffers 0 10 20
    .\run.ps1 python cifar100_three_buffer.py --quick      cualquier comando dentro del contenedor

    Opciones del lanzador (antes del comando):
      -dev   usa tu copia del código (montada) en vez de la congelada en la imagen
      -gpu   usa la imagen con GPU NVIDIA (constrúyela con: .\run.ps1 -gpu construir)
      -cpu   fuerza la imagen de CPU aunque haya una GPU
#>

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$compose = @("compose", "-f", (Join-Path $root "docker-compose.yml"), "--project-directory", $root)

# ------------------------------------------------------------ opciones del lanzador
$mode = "cfl"; $variant = "auto"; $rest = @()
foreach ($arg in $args) {
    if ($rest.Count -eq 0 -and $arg -eq "-dev") { $mode = "dev"; continue }
    if ($rest.Count -eq 0 -and $arg -eq "-gpu") { $variant = "gpu"; continue }
    if ($rest.Count -eq 0 -and $arg -eq "-cpu") { $variant = "cpu"; continue }
    $rest += [string]$arg
}

docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Docker Desktop no está en ejecución. Ábrelo, espera a que arranque y reintenta." -ForegroundColor Red
    exit 1
}

function Get-Revision {
    try {
        $rev = (git -C $root rev-parse --short HEAD 2>$null)
        if (-not $rev) { return "desconocida" }
        if (git -C $root status --porcelain --untracked-files=no 2>$null) { $rev += "-modificado" }
        return $rev
    } catch { return "desconocida" }
}

function Test-Image([string]$tag) {
    docker image inspect $tag *> $null
    return ($LASTEXITCODE -eq 0)
}

function Test-NvidiaGpu {
    try { nvidia-smi -L *> $null; return ($LASTEXITCODE -eq 0) } catch { return $false }
}

function Build-Image([string]$which) {
    $env:CFL_REVISION = Get-Revision
    $env:BUILD_DATE = (Get-Date -Format "yyyy-MM-ddTHH:mm:sszzz")
    Write-Host "Construyendo cfl-practicas:$which (código $($env:CFL_REVISION)). La primera vez tarda unos minutos;" -ForegroundColor Cyan
    Write-Host "después solo se reconstruye lo que cambia." -ForegroundColor Cyan
    $service = if ($which -eq "gpu") { "gpu" } else { "cfl" }
    $profileArgs = if ($which -eq "gpu") { @("--profile", "gpu") } else { @() }
    docker @compose @profileArgs build $service
    if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] Falló la construcción de la imagen." -ForegroundColor Red; exit $LASTEXITCODE }
}

# ------------------------------------------------------------ CPU o GPU
if ($variant -eq "auto") {
    $variant = "cpu"
    if (Test-NvidiaGpu) {
        if (Test-Image "cfl-practicas:gpu") { $variant = "gpu" }
        else { Write-Host "[INFO] GPU NVIDIA detectada. Para usarla: .\run.ps1 -gpu construir" -ForegroundColor Yellow }
    }
}
$tag = "cfl-practicas:$variant"

# ------------------------------------------------------------ menú
if ($rest.Count -eq 0) {
    Write-Host "=============================================================================" -ForegroundColor Cyan
    Write-Host "  APRENDIZAJE FEDERADO CONTINUO ($tag)" -ForegroundColor Cyan
    Write-Host "=============================================================================" -ForegroundColor Cyan
    Write-Host "  [1] Información del entorno          [2] Tests unitarios"
    Write-Host "  [3] Prueba rápida con CIFAR-100      [4] Comparación 0 % / 20 % (5 épocas)"
    Write-Host "  [5] Comparación 3 semillas, buffers 0/5/10/20"
    Write-Host "  [6] Igual con lotes equilibrados     [7] Igual con lotes equilibrados + destilación"
    Write-Host "  [8] Reconstruir la imagen            [9] Consola dentro del contenedor"
    Write-Host "  [0] Salir"
    switch (Read-Host "Opción") {
        "1" { $rest = @("info") }
        "2" { $rest = @("tests") }
        "3" { $rest = @("rapido") }
        "4" { $rest = @("comparar") }
        "5" { $rest = @("comparar", "--seeds", "42", "43", "44", "--buffers", "0", "5", "10", "20") }
        "6" { $rest = @("comparar", "--seeds", "42", "43", "44", "--buffers", "0", "5", "10", "20", "--replay-mix", "balanced") }
        "7" { $rest = @("comparar", "--seeds", "42", "43", "44", "--buffers", "0", "5", "10", "20", "--replay-mix", "balanced", "--distill-weight", "1") }
        "8" { $rest = @("construir") }
        "9" { $rest = @("bash") }
        default { exit 0 }
    }
}

if ($rest[0] -eq "construir") { Build-Image $variant; exit 0 }
if (-not (Test-Image $tag)) {
    if ($variant -eq "gpu") { Write-Host "[ERROR] Falta la imagen GPU: .\run.ps1 -gpu construir" -ForegroundColor Red; exit 1 }
    Build-Image $variant
}

# Con el código congelado en la imagen, avisa si la imagen es de otra versión del código.
if ($mode -eq "cfl") {
    $built = docker image inspect $tag --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}' 2>$null
    $current = Get-Revision
    if ($built -and $current -ne "desconocida" -and $built -ne $current) {
        Write-Host "[AVISO] La imagen tiene el código $built y tu carpeta está en $current." -ForegroundColor Yellow
        Write-Host "        Ejecuta '.\run.ps1 construir' para usar el código actual, o '-dev' para probar sin reconstruir." -ForegroundColor Yellow
    }
}

$service = if ($variant -eq "gpu") { "gpu" } elseif ($mode -eq "dev") { "dev" } else { "cfl" }
if ($variant -eq "gpu" -and $mode -eq "dev") { Write-Host "[INFO] -dev no está disponible con GPU; se usa el código de la imagen." -ForegroundColor Yellow }
$profileArgs = if ($variant -eq "gpu") { @("--profile", "gpu") } else { @() }
$runArgs = @("run", "--rm")
if ([Console]::IsInputRedirected -or [Console]::IsOutputRedirected) { $runArgs += "-T" }
New-Item -ItemType Directory -Force -Path (Join-Path $root "resultados") | Out-Null

docker @compose @profileArgs @runArgs $service @rest
exit $LASTEXITCODE
