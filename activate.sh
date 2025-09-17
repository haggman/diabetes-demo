#!/usr/bin/env bash

# Detect if script is being sourced
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  __ACTIVATE_SOURCED=1
else
  __ACTIVATE_SOURCED=0
fi

# Use strict mode only when executed, not when sourced
if [[ "${__ACTIVATE_SOURCED}" -eq 0 ]]; then
  set -euo pipefail
fi

# Auto-detect or reuse PROJECT_ID
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value core/project 2>/dev/null || true)}"
if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo "ERROR: PROJECT_ID is not set. Run: export PROJECT_ID=<your-project-id>" >&2
  if [[ "${__ACTIVATE_SOURCED}" -eq 1 ]]; then
    return 1
  else
    exit 1
  fi
fi
export PROJECT_ID

# Dataset and model configuration
export BQ_DATASET="${BQ_DATASET:-demo_diabetes}"
export BQ_LOCATION="${BQ_LOCATION:-US}"
export GCS_URI="${GCS_URI:-gs://class-demo/diabetes_prediction_dataset.csv}"

# Table and model names
export RAW_TABLE="${BQ_DATASET}.diabetes_raw"

echo "✓ Environment configured:"
echo "  PROJECT_ID: ${PROJECT_ID}"
echo "  DATASET: ${BQ_DATASET}"
echo "  LOCATION: ${BQ_LOCATION}"
echo ""
echo "Ready to proceed with demo setup!"
