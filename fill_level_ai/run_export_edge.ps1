$ErrorActionPreference = "Stop"
python (Join-Path $PSScriptRoot "export_edge.py") --checkpoint (Join-Path $PSScriptRoot "model\waste_bin_robust_v2.pth") --output-dir (Join-Path $PSScriptRoot "model")
