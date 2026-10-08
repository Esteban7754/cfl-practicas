<#
.SYNOPSIS
    Compatibilidad: traduce los parámetros antiguos a ".\run.ps1 semillas ...".
.EXAMPLE
    .\repetir_semillas.ps1 -Script domainnet.py -Seeds 42,43,44 -ExtraArgs '--buffers','0','10','20'
    # equivale a: .\run.ps1 semillas domainnet.py --seeds 42 43 44 --buffers 0 10 20
#>
param(
    [Parameter(Mandatory = $true)][string]$Script,
    [int[]]$Seeds = @(42, 43, 44),
    [string]$OutputDir = '',
    [string[]]$ExtraArgs = @(),
    [int]$Threads = 0
)
$cli = @("semillas", $Script.Replace('\', '/'), "--seeds") + ($Seeds | ForEach-Object { "$_" })
if ($OutputDir) { $cli += @("--output-dir", $OutputDir.Replace('\', '/')) }
if ($Threads -gt 0) { $cli += @("--threads", "$Threads") }
& (Join-Path $PSScriptRoot "run.ps1") @($cli + $ExtraArgs)
exit $LASTEXITCODE
