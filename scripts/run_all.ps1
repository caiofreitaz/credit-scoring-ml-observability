# Executa o pipeline completo: download -> preparação -> treino -> lotes -> drift -> relatórios.
# Uso:  .\scripts\run_all.ps1            (completo)
#       .\scripts\run_all.ps1 -SkipTrain (reaproveita o modelo já treinado)
param([switch]$SkipTrain)
$ErrorActionPreference = "Stop"
$args_list = @("-m", "src.cli", "run-all")
if ($SkipTrain) { $args_list += "--skip-train" }
& .\.venv\Scripts\python.exe @args_list
exit $LASTEXITCODE
