param(
  [Parameter(Mandatory = $true)]
  [string]$ImagePath,
  [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"
$Arguments = @((Join-Path $PSScriptRoot "predict.py"), $ImagePath)
if ($OutputPath) {
  $Arguments += @("--output", $OutputPath)
}
python @Arguments
