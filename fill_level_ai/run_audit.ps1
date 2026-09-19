param(
  [string]$DataDir = "",
  [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
if (-not $DataDir) { $DataDir = Join-Path $PSScriptRoot "data\prepared-custom-data" }
if (-not $OutputDir) { $OutputDir = Join-Path $PSScriptRoot "evaluation\audit" }
$Model = Join-Path $PSScriptRoot "model\waste_bin_custom_v1_baseline.torchscript.pt"
python (Join-Path $PSScriptRoot "audit_dataset.py") --data-dir $DataDir --model $Model --output-dir $OutputDir
