<#
.SYNOPSIS
    Repite un experimento con varias semillas en Docker y resume media ± desviación típica.
.EXAMPLE
    .\repetir_semillas.ps1 -Script three_buffer_v2.py
    .\repetir_semillas.ps1 -Script domainnet.py -Seeds 42,43,44,45,46 -ExtraArgs '--equal-steps'
    .\repetir_semillas.ps1 -Script cifar100_three_buffer.py -ExtraArgs '--local-epochs','5','--buffers','0','20'
.NOTES
    Cada semilla se guarda en <OutputDir>\seed_<n>. Al final se ejecuta agregar_semillas.py,
    que escribe resumen_semillas.csv y perdida_A_media_semillas.png en <OutputDir>.
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('step_six_v1.py', 'three_buffer_size.py', 'three_buffer_v2.py', 'cifar100_three_buffer.py',
                 'domainnet.py', 'mvtecad_all_experiments.py')]
    [string]$Script,
    [int[]]$Seeds = @(42, 43, 44),
    [string]$OutputDir = '',
    [string[]]$ExtraArgs = @(),
    [ValidateRange(1, 64)][int]$Threads = 4
)

$projectRoot = $PSScriptRoot
if (-not $OutputDir) {
    $OutputDir = 'resultados/' + [System.IO.Path]::GetFileNameWithoutExtension($Script) + '_semillas_' + (Get-Date -Format 'yyyy-MM-dd_HHmmss')
}
if ([System.IO.Path]::IsPathRooted($OutputDir)) { throw 'OutputDir debe ser una ruta relativa dentro del proyecto.' }
$OutputDir = $OutputDir.Replace('\', '/')
$targetDirectory = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $OutputDir))
if (Test-Path $targetDirectory) { throw "La carpeta $OutputDir ya existe. Elige otra para conservar los resultados anteriores." }

docker info > $null 2>&1
if ($LASTEXITCODE -ne 0) { throw 'Abre Docker Desktop antes de ejecutar los experimentos.' }
New-Item -ItemType Directory -Force -Path $targetDirectory | Out-Null

$containerOptions = @('--rm', '-e', "OMP_NUM_THREADS=$Threads", '-e', "MKL_NUM_THREADS=$Threads",
    '-v', "${projectRoot}:/workspace", '-v', 'cfl-hf-cache:/cache/huggingface', '-w', '/workspace', 'cfl-practicas')

foreach ($seed in $Seeds) {
    $seedDir = "$OutputDir/seed_$seed"
    New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot $seedDir) | Out-Null
    Write-Host "=== $Script, semilla $seed -> $seedDir" -ForegroundColor Cyan
    docker run @containerOptions python $Script --seed $seed --output-dir $seedDir @ExtraArgs 2>&1 |
        Tee-Object -FilePath (Join-Path $projectRoot "$seedDir/ejecucion.log")
    if ($LASTEXITCODE -ne 0) { throw "Falló la semilla $seed. Consulta $seedDir/ejecucion.log." }
}

docker run @containerOptions python agregar_semillas.py $OutputDir 2>&1 |
    Tee-Object -FilePath (Join-Path $targetDirectory 'resumen_semillas.log')
if ($LASTEXITCODE -ne 0) { throw 'No se pudo generar el resumen de semillas.' }
Write-Host "Terminado. Resumen en $OutputDir/resumen_semillas.csv" -ForegroundColor Green
