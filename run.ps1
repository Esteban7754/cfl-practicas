<#
.SYNOPSIS
    Script de ejecución para el proyecto de Aprendizaje Federado Continuo.
.DESCRIPTION
    Permite ejecutar cualquier script de experimento dentro del contenedor Docker oficial,
    o abrir un menú interactivo si no se especifican parámetros.
.EXAMPLE
    .\run.ps1
    .\run.ps1 step_four.py
    .\run.ps1 cifar100_three_buffer.py
    .\run.ps1 test
#>

param (
    [Parameter(Position=0, Mandatory=$false)]
    [string]$Script = ""
)

function Show-Menu {
    Clear-Host
    Write-Host "==============================================================================" -ForegroundColor Cyan
    Write-Host "          APRENDIZAJE FEDERADO CONTINUO - MENU DE EJECUCION                   " -ForegroundColor Cyan
    Write-Host "==============================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Selecciona el experimento que deseas ejecutar:"
    Write-Host ""
    Write-Host "  [1] Verificación rápida del entorno (test_env.py)" -ForegroundColor Green
    Write-Host "  [2] CIFAR-100: Inicial sin replay (step_four.py)"
    Write-Host "  [3] CIFAR-100: Buffers 0%, 5%, 10%, 15% y 20% (step_six_v1.py)"
    Write-Host "  [4] CIFAR-100: Buffers 0%, 5%, 10%, 15%, 20% y 25% (three_buffer_size.py)"
    Write-Host "  [5] CIFAR-100: Fase A una vez, buffers 0% a 25% (three_buffer_v2.py)"
    Write-Host "  [6] CIFAR-100: Comparativa completa 0%, 5%, 10%, 15%, 20% y 25% (cifar100_three_buffer.py)"
    Write-Host "  [7] DomainNet: Cambio de dominio a bocetos (domainnet.py)"
    Write-Host "  [8] MVTec AD: Clasificación de anomalías (mvtecad_all_experiments.py)"
    Write-Host "  [9] Abrir consola interactiva de Python en el contenedor" -ForegroundColor Yellow
    Write-Host "  [0] Salir"
    Write-Host ""
    Write-Host "==============================================================================" -ForegroundColor Cyan
    
    $opt = Read-Host "Introduce una opción [1-9, 0]"
    switch ($opt) {
        "1" { return "test_env.py" }
        "2" { return "step_four.py" }
        "3" { return "step_six_v1.py" }
        "4" { return "three_buffer_size.py" }
        "5" { return "three_buffer_v2.py" }
        "6" { return "cifar100_three_buffer.py" }
        "7" { return "domainnet.py" }
        "8" { return "mvtecad_all_experiments.py" }
        "9" { return "interactive" }
        "0" { exit 0 }
        default { 
            Write-Host "Opción inválida." -ForegroundColor Red
            Start-Sleep -Seconds 1
            return Show-Menu
        }
    }
}

# Comprobar Docker
try {
    $dockerInfo = docker info 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Docker Desktop no está en ejecución. Por favor, ábrelo y reintenta." -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "[ERROR] Docker no está disponible en este sistema." -ForegroundColor Red
    exit 1
}

# Obtener script a ejecutar
if ([string]::IsNullOrWhiteSpace($Script)) {
    $target = Show-Menu
} else {
    if ($Script -eq "test") {
        $target = "test_env.py"
    } else {
        $target = $Script
    }
}

if ($target -eq "interactive") {
    Write-Host "Abriendo terminal interactiva en el contenedor..." -ForegroundColor Yellow
    docker run --rm -it -v "${PWD}:/workspace" -v cfl-hf-cache:/cache/huggingface -w /workspace cfl-practicas bash
} else {
    Write-Host "==============================================================================" -ForegroundColor Cyan
    Write-Host "Ejecutando: $target" -ForegroundColor Green
    Write-Host "==============================================================================" -ForegroundColor Cyan
    Write-Host ""
    
    # Comprobar si stdin es TTY o no
    if ([System.Console]::IsInputRedirected) {
        docker run --rm -v "${PWD}:/workspace" -v cfl-hf-cache:/cache/huggingface -w /workspace cfl-practicas python $target
    } else {
        docker run --rm -it -v "${PWD}:/workspace" -v cfl-hf-cache:/cache/huggingface -w /workspace cfl-practicas python $target
    }
    
    Write-Host ""
    Write-Host "==============================================================================" -ForegroundColor Cyan
    Write-Host "Ejecución finalizada." -ForegroundColor Cyan
    Write-Host "==============================================================================" -ForegroundColor Cyan
}
