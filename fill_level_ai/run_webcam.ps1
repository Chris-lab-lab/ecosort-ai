param(
  [int]$Camera = 0
)

$ErrorActionPreference = "Stop"
$RobustModel = Join-Path $PSScriptRoot "model\waste_bin_robust_v2.torchscript.pt"
if (-not (Test-Path -LiteralPath $RobustModel)) {
  throw "Missing trained model: $RobustModel"
}
Write-Output "Using model: $RobustModel"
python (Join-Path $PSScriptRoot "webcam.py") --camera $Camera --model $RobustModel
