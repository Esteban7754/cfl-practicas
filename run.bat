@echo off
:: Lanzador para CMD o doble clic: delega en run.ps1 (misma lógica y mismas opciones).
::   run.bat                 menú
::   run.bat comparar --seeds 42 43 44
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
set "CODE=%ERRORLEVEL%"
if "%~1"=="" pause
exit /b %CODE%
