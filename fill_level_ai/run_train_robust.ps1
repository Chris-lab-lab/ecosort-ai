param(
  [int]$Epochs = 14,
  [int]$BatchSize = 24,
  [int]$Seed = 20260919
)

$ErrorActionPreference = "Stop"
$Data = Join-Path $PSScriptRoot "data\robust-data-v2"
$Base = Join-Path $PSScriptRoot "model\waste_bin_fill_level_mobilenet_v3_small.pth"
python (Join-Path $PSScriptRoot "train_robust.py") --data-dir $Data --base-model $Base --output-dir (Join-Path $PSScriptRoot "model") --epochs $Epochs --batch-size $BatchSize --seed $Seed
