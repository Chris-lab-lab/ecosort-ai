param(
  [double]$ConfidenceThreshold = 0.58
)

$ErrorActionPreference = "Stop"
$Test = Join-Path $PSScriptRoot "data\robust-data-v2\test"
$Baseline = Join-Path $PSScriptRoot "model\waste_bin_custom_v1_baseline.torchscript.pt"
$Improved = Join-Path $PSScriptRoot "model\waste_bin_robust_v2.torchscript.pt"
python (Join-Path $PSScriptRoot "evaluate_robustness.py") --test-dir $Test --baseline-model $Baseline --improved-model $Improved --output-dir (Join-Path $PSScriptRoot "evaluation") --confidence-threshold $ConfidenceThreshold
python (Join-Path $PSScriptRoot "create_report_assets.py") --training-report (Join-Path $PSScriptRoot "model\robust_v2_training_report.json") --robustness-report (Join-Path $PSScriptRoot "evaluation\robustness_report.json") --examples-dir (Join-Path $PSScriptRoot "evaluation\examples") --output-dir (Join-Path $PSScriptRoot "evaluation")
