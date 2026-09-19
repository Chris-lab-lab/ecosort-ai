param(
  [int]$Camera = 0
)

$ErrorActionPreference = "Stop"
python (Join-Path $PSScriptRoot "collect_training_data.py") `
  --camera $Camera `
  --output-dir (Join-Path $PSScriptRoot "data\collected")
