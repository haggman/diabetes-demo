bq --location="${BQ_LOCATION}" mk \
  --dataset \
  --description "Healthcare demo: Diabetes prediction with BQML and ADK" \
  "${PROJECT_ID}:${BQ_DATASET}" || echo "Dataset already exists"

# Load CSV with automatic schema detection
bq --location="${BQ_LOCATION}" load \
  --autodetect \
  --skip_leading_rows=1 \
  --source_format=CSV \
  --replace \
  "${RAW_TABLE}" \
  "${GCS_URI}"