<#
.SYNOPSIS
    Ejecuta los experimentos del proyecto dentro del contenedor Docker oficial.
.DESCRIPTION
    Sin argumentos abre un menú interactivo. Con argumentos, el primero es el script
    (o "test" / "unittest") y el resto se pasa tal cual al script de Python.
    Monta siempre la carpeta del proyecto, aunque se lance desde otra carpeta.
.EXAMPLE
    .\run.ps1
    .\run.ps1 test
    .\run.ps1 unittest
    .\run.ps1 cifar100_three_buffer.py --quick --output-dir resultados/prueba
    .\run.ps1 legacy/step_four.py
#>
param (
    [Parameter(Position = 0)][string]$Script = "",
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ScriptArgs = @()
)

$projectRoot = $PSScriptRoot

function Show-Menu {
    Clear-Host
    Write-Host "==============================================================================" -ForegroundColor Cyan
    Write-Host "          APRENDIZAJE FEDERADO CONTINUO - MENU DE EJECUCION                   " -ForegroundColor Cyan
    Write-Host "==============================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  [1] Verificación rápida del entorno (verificar_entorno.py)" -ForegroundColor Green
    Write-Host "  [2] Tests unitarios (python -m unittest)" -ForegroundColor Green
    Write-Host "  [3] CIFAR-100: comparativa de replay (cifar100_three_buffer.py)"
    Write-Host "  [4] CIFAR-100: prueba funcional rápida (cifar100_three_buffer.py --quick)"
    Write-Host "  [5] DomainNet: cambio de dominio a bocetos (domainnet.py)"
    Write-Host "  [6] MVTec AD: clasificación de categorías (mvtecad_all_experiments.py)"
    Write-Host "  [7] Scripts históricos (carpeta legacy/)"
    Write-Host "  [9] Consola interactiva en el contenedor" -ForegroundColor Yellow
    Write-Host "  [0] Salir"
    Write-Host ""
    $opt = Read-Host "Introduce una opción"
    switch ($opt) {
        "1" { return @("verificar_entorno.py") }
        "2" { return @("-m", "unittest", "-v") }
        "3" { return @("cifar100_three_buffer.py") }
        "4" { return @("cifar100_three_buffer.py", "--quick") }
        "5" { return @("domainnet.py") }
        "6" { return @("mvtecad_all_experiments.py") }
        "7" {
            Write-Host "  [a] legacy/step_four.py   [b] legacy/step_six_v1.py   [c] legacy/three_buffer_size.py   [d] legacy/three_buffer_v2.py"
            switch (Read-Host "Script histórico") {
                "a" { return @("legacy/step_four.py") }
                "b" { return @("legacy/step_six_v1.py") }
                "c" { return @("legacy/three_buffer_size.py") }
                "d" { return @("legacy/three_buffer_v2.py") }
                default { return Show-Menu }
            }
        }
        "9" { return @("interactive") }
        "0" { exit 0 }
        default { Write-Host "Opción inválida." -ForegroundColor Red; Start-Sleep -Seconds 1; return Show-Menu }
    }
}

docker info > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Docker Desktop no está en ejecución. Ábrelo y reintenta." -ForegroundColor Red
    exit 1
}

if ([string]::IsNullOrWhiteSpace($Script)) {
    $command = Show-Menu
} elseif ($Script -eq "test") {
    $command = @("verificar_entorno.py") + $ScriptArgs
} elseif ($Script -eq "unittest") {
    $command = @("-m", "unittest", "-v") + $ScriptArgs
} else {
    $command = @($Script) + $ScriptArgs
}

$options = @("--rm", "-v", "${projectRoot}:/workspace", "-v", "cfl-hf-cache:/cache/huggingface", "-w", "/workspace")
foreach ($name in "OMP_NUM_THREADS", "MKL_NUM_THREADS") {
    if (Test-Path "env:$name") { $options += @("-e", "$name") }
}
if (-not [System.Console]::IsInputRedirected) { $options += "-it" }

if ($command[0] -eq "interactive") {
    docker run @options cfl-practicas bash
} else {
    Write-Host "Ejecutando: python $($command -join ' ')" -ForegroundColor Green
    docker run @options cfl-practicas python @command
}
exit $LASTEXITCODE
