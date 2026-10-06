@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: Si se pasa un argumento (ej: run.bat step_four.py), se ejecuta directamente
if not "%~1"=="" (
    if "%~1"=="test" (
        set SCRIPT=test_env.py
    ) else (
        set SCRIPT=%~1
    )
    goto direct_execute
)

:menu
cls
echo ==============================================================================
echo           APRENDIZAJE FEDERADO CONTINUO - MENU DE EJECUCION
echo ==============================================================================
echo.
echo Selecciona el experimento que deseas ejecutar:
echo.
echo   [1] Verificacion rapida del entorno (test_env.py)
echo   [2] CIFAR-100: Inicial sin replay (step_four.py)
echo   [3] CIFAR-100: Buffers 0%%, 5%%, 10%%, 15%% y 20%% (step_six_v1.py)
echo   [4] CIFAR-100: Buffers 0%%, 5%%, 10%%, 15%%, 20%% y 25%% (three_buffer_size.py)
echo   [5] CIFAR-100: Fase A una vez, buffers 0%% a 25%% (three_buffer_v2.py)
echo   [6] CIFAR-100: Comparativa completa 0%%, 5%%, 10%%, 15%%, 20%% y 25%% (cifar100_three_buffer.py)
echo   [7] DomainNet: Cambio de dominio a bocetos (domainnet.py)
echo   [8] MVTec AD: Clasificacion de anomalias (mvtecad_all_experiments.py)
echo   [9] Abrir consola interactiva de Python en el contenedor
echo   [0] Salir
echo.
echo ==============================================================================
set /p OPTION="Introduce una opcion [1-9, 0]: "

if "%OPTION%"=="1" (
    set SCRIPT=test_env.py
    goto execute
)
if "%OPTION%"=="2" (
    set SCRIPT=step_four.py
    goto execute
)
if "%OPTION%"=="3" (
    set SCRIPT=step_six_v1.py
    goto execute
)
if "%OPTION%"=="4" (
    set SCRIPT=three_buffer_size.py
    goto execute
)
if "%OPTION%"=="5" (
    set SCRIPT=three_buffer_v2.py
    goto execute
)
if "%OPTION%"=="6" (
    set SCRIPT=cifar100_three_buffer.py
    goto execute
)
if "%OPTION%"=="7" (
    set SCRIPT=domainnet.py
    goto execute
)
if "%OPTION%"=="8" (
    set SCRIPT=mvtecad_all_experiments.py
    goto execute
)
if "%OPTION%"=="9" goto interactive_shell
if "%OPTION%"=="0" exit /b 0

echo Opcion invalida.
timeout /t 2 >nul
goto menu

:execute
cls
echo ==============================================================================
echo Ejecutando: %SCRIPT%
echo ==============================================================================
echo.

docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker no esta en ejecucion. Por favor, abre Docker Desktop y reintenta.
    pause
    exit /b 1
)

docker run --rm -e OMP_NUM_THREADS -e MKL_NUM_THREADS -v "%~dp0:/workspace" -v cfl-hf-cache:/cache/huggingface -w /workspace cfl-practicas python %SCRIPT%

echo.
echo ==============================================================================
echo Ejecucion finalizada.
echo ==============================================================================
pause
goto menu

:direct_execute
docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker no esta en ejecucion. Por favor, abre Docker Desktop y reintenta.
    exit /b 1
)
if "%~1"=="test" (
    docker run --rm -e OMP_NUM_THREADS -e MKL_NUM_THREADS -v "%~dp0:/workspace" -v cfl-hf-cache:/cache/huggingface -w /workspace cfl-practicas python test_env.py
) else (
    docker run --rm -e OMP_NUM_THREADS -e MKL_NUM_THREADS -v "%~dp0:/workspace" -v cfl-hf-cache:/cache/huggingface -w /workspace cfl-practicas python %*
)
exit /b %ERRORLEVEL%

:interactive_shell
cls
echo Abriendo terminal interactiva con Python en Docker...
docker run --rm -e OMP_NUM_THREADS -e MKL_NUM_THREADS -it -v "%~dp0:/workspace" -v cfl-hf-cache:/cache/huggingface -w /workspace cfl-practicas bash
goto menu
