@echo off
chcp 65001 >nul
setlocal

:: Lanzador para CMD / doble clic. Sin argumentos abre el menú; con argumentos,
:: el primero es el script (o "test" / "unittest") y el resto se pasa a Python.
::   run.bat test
::   run.bat cifar100_three_buffer.py --quick --output-dir resultados\prueba
::   run.bat legacy\step_four.py

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "DOCKER_OPTS=--rm -e OMP_NUM_THREADS -e MKL_NUM_THREADS -v "%ROOT%:/workspace" -v cfl-hf-cache:/cache/huggingface -w /workspace"

docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker no esta en ejecucion. Abre Docker Desktop y reintenta.
    if "%~1"=="" pause
    exit /b 1
)

if "%~1"=="" goto menu
if /i "%~1"=="test" (
    docker run %DOCKER_OPTS% cfl-practicas python verificar_entorno.py
    exit /b %ERRORLEVEL%
)
if /i "%~1"=="unittest" (
    docker run %DOCKER_OPTS% cfl-practicas python -m unittest -v
    exit /b %ERRORLEVEL%
)
:: Las rutas con "\" se convierten a "/" para el contenedor Linux.
set "ARGS=%*"
set "ARGS=%ARGS:\=/%"
docker run %DOCKER_OPTS% cfl-practicas python %ARGS%
exit /b %ERRORLEVEL%

:menu
cls
echo ==============================================================================
echo           APRENDIZAJE FEDERADO CONTINUO - MENU DE EJECUCION
echo ==============================================================================
echo.
echo   [1] Verificacion rapida del entorno (verificar_entorno.py)
echo   [2] Tests unitarios
echo   [3] CIFAR-100: comparativa de replay (cifar100_three_buffer.py)
echo   [4] CIFAR-100: prueba funcional rapida (--quick)
echo   [5] DomainNet: cambio de dominio a bocetos (domainnet.py)
echo   [6] MVTec AD: clasificacion de categorias (mvtecad_all_experiments.py)
echo   [7] legacy\step_four.py         [8] legacy\three_buffer_v2.py
echo   [9] Consola interactiva en el contenedor
echo   [0] Salir
echo.
set "OPTION="
set /p OPTION="Introduce una opcion: "
set "CMD="
if "%OPTION%"=="1" set "CMD=verificar_entorno.py"
if "%OPTION%"=="2" set "CMD=-m unittest -v"
if "%OPTION%"=="3" set "CMD=cifar100_three_buffer.py"
if "%OPTION%"=="4" set "CMD=cifar100_three_buffer.py --quick"
if "%OPTION%"=="5" set "CMD=domainnet.py"
if "%OPTION%"=="6" set "CMD=mvtecad_all_experiments.py"
if "%OPTION%"=="7" set "CMD=legacy/step_four.py"
if "%OPTION%"=="8" set "CMD=legacy/three_buffer_v2.py"
if "%OPTION%"=="0" exit /b 0
if "%OPTION%"=="9" (
    docker run %DOCKER_OPTS% -it cfl-practicas bash
    goto menu
)
if not defined CMD (
    echo Opcion invalida.
    timeout /t 2 >nul
    goto menu
)
cls
echo Ejecutando: python %CMD%
echo.
docker run %DOCKER_OPTS% cfl-practicas python %CMD%
echo.
echo Ejecucion finalizada.
pause
goto menu
