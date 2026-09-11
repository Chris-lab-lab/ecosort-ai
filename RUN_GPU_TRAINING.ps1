[CmdletBinding()]
param(
    [ValidateRange(1, 1000)]
    [int]$Epochs = 15,

    [ValidateRange(0, 1000)]
    [int]$FineTuneEpochs = 5,

    [ValidateRange(1, 1024)]
    [int]$BatchSize = 64,

    [string]$DataDirectory = "dataset",

    [string]$OutputDirectory = "artifacts_gpu"
)

$ErrorActionPreference = "Stop"
$projectDirectory = $PSScriptRoot
$trainingImage = "ecosort-training:tf2.18-gpu"

Set-Location -LiteralPath $projectDirectory

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker is not installed. Install or repair Docker Desktop, then run this script again."
}

docker info *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Docker Desktop is not running. Start Docker Desktop, wait until it is ready, then run this script again."
}

if (-not (Test-Path -LiteralPath (Join-Path $projectDirectory $DataDirectory))) {
    throw "Dataset directory not found: $DataDirectory"
}

docker image inspect $trainingImage *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Building the TensorFlow 2.18 GPU training image (first run only)..."
    docker build --file Dockerfile.gpu --tag $trainingImage .
    if ($LASTEXITCODE -ne 0) {
        throw "Could not build the GPU training image."
    }
}

Write-Host "Checking that TensorFlow can see the NVIDIA GPU..."
docker run --rm --gpus all $trainingImage `
    python -c "import tensorflow as tf; g=tf.config.list_physical_devices('GPU'); print('TensorFlow GPUs:', g); raise SystemExit(0 if g else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "The container cannot see your GPU. In Docker Desktop, enable the WSL2 engine and update WSL with: wsl --update"
}

$workspaceMount = "${projectDirectory}:/workspace"
Write-Host "Starting GPU training from $DataDirectory into $OutputDirectory..."
docker run --rm --gpus all --shm-size 2g `
    --env TF_CPP_MIN_LOG_LEVEL=1 `
    --volume $workspaceMount `
    --workdir /workspace `
    $trainingImage `
    python train.py `
        --data $DataDirectory `
        --output $OutputDirectory `
        --allow-no-other `
        --require-gpu `
        --epochs $Epochs `
        --fine-tune-epochs $FineTuneEpochs `
        --batch-size $BatchSize

if ($LASTEXITCODE -ne 0) {
    throw "GPU training failed. Review the TensorFlow error printed above."
}

Write-Host "Training complete. Model: $OutputDirectory/waste_classifier_int8.tflite"
