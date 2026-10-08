<#
.SYNOPSIS
    Compatibilidad: traduce los parámetros antiguos a ".\run.ps1 comparar ...".
.EXAMPLE
    .\comparar_replay.ps1 -Seeds 42,43,44 -Buffers 0,5,10,20 -ReplayMix balanced
    # equivale a: .\run.ps1 comparar --seeds 42 43 44 --buffers 0 5 10 20 --replay-mix balanced
#>
param(
    [string]$OutputDir = '',
    [int]$LocalEpochs = 5,
    [int]$Rounds = 5,
    [int[]]$Seeds = @(42),
    [int[]]$Buffers = @(0, 20),
    [switch]$ReuseCheckpoint,
    [ValidateSet('concat', 'balanced')][string]$ReplayMix = 'concat',
    [double]$ReplayBatchFraction = 0.5,
    [double]$DistillWeight = 0,
    [int]$Threads = 0
)
$culture = [System.Globalization.CultureInfo]::InvariantCulture
$cli = @("comparar", "--local-epochs", "$LocalEpochs", "--rounds", "$Rounds", "--seeds") + ($Seeds | ForEach-Object { "$_" }) +
       @("--buffers") + ($Buffers | ForEach-Object { "$_" }) +
       @("--replay-mix", $ReplayMix, "--replay-batch-fraction", $ReplayBatchFraction.ToString($culture),
         "--distill-weight", $DistillWeight.ToString($culture))
if ($OutputDir) { $cli += @("--output-dir", $OutputDir.Replace('\', '/')) }
if ($Threads -gt 0) { $cli += @("--threads", "$Threads") }
# El checkpoint antiguo está en tu carpeta, no en la imagen: hace falta el modo -dev.
if ($ReuseCheckpoint) { $cli = @("-dev") + $cli + @("--reuse-checkpoint") }
& (Join-Path $PSScriptRoot "run.ps1") @cli
exit $LASTEXITCODE
