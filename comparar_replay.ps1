<#
.SYNOPSIS
    Comparación controlada de replay en CIFAR-100 con los datos completos, verificada desde los pesos.
.DESCRIPTION
    Por defecto entrena la fase A en la propia ejecución con 5 épocas locales, el régimen de los
    resultados originales (A ≈ 52 % antes de B). La versión anterior reutilizaba
    global_model_phase1.pt, entrenado con 1 época (A ≈ 27 %), y en ese régimen todas las variantes
    caen a A = 0 %, también en los registros originales.

    Todas las variantes parten de los mismos pesos A, usan buffers anidados y hacen las mismas
    actualizaciones por cliente y ronda (--controlled-replay). Con varias semillas se ejecuta una
    carpeta por semilla y se resume con agregar_semillas.py.
.EXAMPLE
    .\comparar_replay.ps1
    .\comparar_replay.ps1 -Seeds 42,43,44 -Buffers 0,5,10,20
    .\comparar_replay.ps1 -ReuseCheckpoint -LocalEpochs 1 -OutputDir resultados/repeticion_1_epoca
#>
param(
    [string]$OutputDir = '',
    [ValidateRange(1, 50)][int]$LocalEpochs = 5,
    [ValidateRange(1, 50)][int]$Rounds = 5,
    [int[]]$Seeds = @(42),
    [ValidateSet(0, 5, 10, 15, 20, 25)][int[]]$Buffers = @(0, 20),
    [switch]$ReuseCheckpoint,
    [ValidateRange(1, 64)][int]$Threads = 4
)

$projectRoot = $PSScriptRoot
if (-not ($Buffers -contains 0)) { throw 'Buffers debe incluir 0: la referencia sin replay se mide.' }
if ($ReuseCheckpoint -and ($LocalEpochs -ne 1 -or ($Seeds | Where-Object { $_ -ne 42 }))) {
    throw 'global_model_phase1.pt se entrenó con 1 época local y semilla 42; -ReuseCheckpoint solo es coherente con -LocalEpochs 1 -Seeds 42.'
}
if (-not $OutputDir) {
    $OutputDir = "resultados/cifar100_completo_${LocalEpochs}epocas_" + (Get-Date -Format 'yyyy-MM-dd_HHmmss')
}
if ([System.IO.Path]::IsPathRooted($OutputDir)) { throw 'OutputDir debe ser una ruta relativa dentro del proyecto.' }
$OutputDir = $OutputDir.Replace('\', '/')
$targetDirectory = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $OutputDir))
if (-not $targetDirectory.StartsWith($projectRoot + [System.IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'OutputDir debe permanecer dentro del proyecto.'
}
if (Test-Path $targetDirectory) { throw 'La carpeta ya existe. Elige otro OutputDir para conservar sus resultados.' }

docker info > $null 2>&1
if ($LASTEXITCODE -ne 0) { throw 'Abre Docker Desktop antes de ejecutar la comparación.' }
New-Item -ItemType Directory -Force -Path $targetDirectory | Out-Null

$containerOptions = @('--rm', '-e', "OMP_NUM_THREADS=$Threads", '-e', "MKL_NUM_THREADS=$Threads",
    '-v', "${projectRoot}:/workspace", '-v', 'cfl-hf-cache:/cache/huggingface',
    '-w', '/workspace', 'cfl-practicas')
$bufferArguments = $Buffers | ForEach-Object { "$_" }

foreach ($seed in $Seeds) {
    $runDir = if ($Seeds.Count -gt 1) { "$OutputDir/seed_$seed" } else { $OutputDir }
    New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot $runDir) | Out-Null
    $experimentArguments = @('python', 'cifar100_three_buffer.py', '--controlled-replay', '--audit',
        '--seed', "$seed", '--buffers') + $bufferArguments + @(
        '--local-epochs', "$LocalEpochs", '--rounds', "$Rounds", '--output-dir', $runDir)
    if ($ReuseCheckpoint) { $experimentArguments += @('--phase1-checkpoint', 'global_model_phase1.pt') }

    Write-Host "=== Semilla $seed, $LocalEpochs épocas locales -> $runDir" -ForegroundColor Cyan
    docker run @containerOptions @experimentArguments 2>&1 | Tee-Object -FilePath (Join-Path $projectRoot "$runDir/ejecucion.log")
    if ($LASTEXITCODE -ne 0) { throw "El experimento falló (semilla $seed). Consulta $runDir/ejecucion.log." }
    docker run @containerOptions python verificar_cifar100.py $runDir 2>&1 |
        Tee-Object -FilePath (Join-Path $projectRoot "$runDir/verificacion.log")
    if ($LASTEXITCODE -ne 0) { throw "La verificación falló (semilla $seed). Consulta $runDir/verificacion.log." }
}

if ($Seeds.Count -gt 1) {
    docker run @containerOptions python agregar_semillas.py $OutputDir 2>&1 |
        Tee-Object -FilePath (Join-Path $targetDirectory 'resumen_semillas.log')
    if ($LASTEXITCODE -ne 0) { throw 'No se pudo generar el resumen de semillas.' }
}
Write-Host "Comparación finalizada y verificada en $targetDirectory" -ForegroundColor Green
