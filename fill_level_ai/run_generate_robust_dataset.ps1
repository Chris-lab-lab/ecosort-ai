param(
  [int]$TargetPerClass = 600,
  [int]$Seed = 20260919
)

$ErrorActionPreference = "Stop"
$Source = Join-Path $PSScriptRoot "data\prepared-custom-data"
$Output = Join-Path $PSScriptRoot "data\robust-data-v2"
python (Join-Path $PSScriptRoot "generate_robust_dataset.py") --source-dir $Source --output-dir $Output --manual-review-list (Join-Path $PSScriptRoot "manual_review_list.txt") --target-per-class $TargetPerClass --seed $Seed
