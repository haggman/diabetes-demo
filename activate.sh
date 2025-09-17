#!/usr/bin/env bash
set -euo pipefail

# Auto-detect or reuse PROJECT_ID
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value core/project 2>/dev/null || true)}"
if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo "ERROR: PROJECT_ID is not set. Run: export PROJECT_ID=<your-project-id>" >&2
  return 1 2>/dev/null || exit 1
fi
export PROJECT_ID

# Dataset and model configuration
export BQ_DATASET="${BQ_DATASET:-demo_diabetes}"
export BQ_LOCATION="${BQ_LOCATION:-US}"
export GCS_URI="${GCS_URI:-gs://class-demo/diabetes_prediction_dataset.csv}"

# Table and model names
export RAW_TABLE="${BQ_DATASET}.diabetes_raw"
export TRAIN_TABLE="${BQ_DATASET}.diabetes_train"
export MODEL_NAME="${BQ_DATASET}.diabetes_model"
export PREDICTION_VIEW="${BQ_DATASET}.predict_diabetes"

echo "✓ Environment configured:"
echo "  PROJECT_ID: ${PROJECT_ID}"
echo "  DATASET: ${BQ_DATASET}"
echo "  LOCATION: ${BQ_LOCATION}"
echo ""
echo "Ready to proceed with demo setup!"
