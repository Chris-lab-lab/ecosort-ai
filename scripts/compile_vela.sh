#!/bin/sh
set -eu

MODEL="${1:-artifacts/waste_classifier_int8.tflite}"
OUTPUT="${2:-artifacts/vela}"

mkdir -p "$OUTPUT"
vela "$MODEL" --accelerator-config ethos-u65-256 --optimise Performance --output-dir "$OUTPUT"
echo "Vela model written under: $OUTPUT"
