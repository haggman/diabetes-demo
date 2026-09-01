#!/usr/bin/env bash
#
# Deploy the diabetes agent to Agent Runtime (the service formerly called
# Vertex AI Agent Engine).
#
#   source activate.sh
#   source 4_deploy.sh
#
# Re-running this creates a NEW deployment each time unless you set
# AGENT_ENGINE_ID to an existing resource id, in which case it updates in place.

: "${PROJECT_ID:?PROJECT_ID is not set. Run: source activate.sh}"
: "${AGENT_REGION:?AGENT_REGION is not set. Run: source activate.sh}"

echo "=== 1/4  Enabling APIs ==="
gcloud services enable \
  aiplatform.googleapis.com \
  cloudresourcemanager.googleapis.com \
  --project "${PROJECT_ID}"

echo ""
echo "=== 2/4  Granting BigQuery access to the agent's service account ==="
# A deployed agent does NOT run as you. It runs as the Reasoning Engine service
# agent, which starts with no BigQuery access at all. Without these grants the
# agent deploys fine, answers general questions fine, and fails only when a user
# asks a data question.
gcloud beta services identity create \
  --service=aiplatform.googleapis.com \
  --project="${PROJECT_ID}" >/dev/null 2>&1 || true

PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
export AGENT_SA="service-${PROJECT_NUMBER}@gcp-sa-aiplatform-re.iam.gserviceaccount.com"
echo "Service account: ${AGENT_SA}"

# jobUser lets it run queries; dataViewer lets it read the tables, the model and
# the prediction TVF. Read-only, which matches WriteMode.BLOCKED in agent.py.
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${AGENT_SA}" \
  --role="roles/bigquery.jobUser" \
  --condition=None >/dev/null

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${AGENT_SA}" \
  --role="roles/bigquery.dataViewer" \
  --condition=None >/dev/null

echo "Granted roles/bigquery.jobUser and roles/bigquery.dataViewer"

echo ""
echo "=== 3/4  Checking the agent folder ==="
# `adk deploy agent_engine` installs diabetes_agent/requirements.txt into the
# container. That path is a symlink to the repo-root requirements.txt, and the
# deploy follows symlinks when it stages the agent, so the root file stays the
# single source of truth. If the symlink is missing (for example on a machine
# that cannot create them), fall back to a copy.
if [[ ! -e diabetes_agent/requirements.txt ]]; then
  echo "diabetes_agent/requirements.txt missing - recreating"
  ln -sfn ../requirements.txt diabetes_agent/requirements.txt 2>/dev/null \
    || cp requirements.txt diabetes_agent/requirements.txt
fi
if [[ ! -f diabetes_agent/.env ]]; then
  echo "diabetes_agent/.env missing - run: source activate.sh"
  return 1 2>/dev/null || exit 1
fi
echo "requirements.txt -> $(readlink diabetes_agent/requirements.txt 2>/dev/null || echo 'local copy')"
echo "Environment the agent will be deployed with:"
sed 's/^/  /' diabetes_agent/.env

echo ""
echo "=== 4/4  Deploying to Agent Runtime ==="
DEPLOY_ARGS=(
  --project="${PROJECT_ID}"
  --region="${AGENT_REGION}"
  --display_name="${AGENT_DISPLAY_NAME:-Diabetes Risk Agent}"
  --description="Educational diabetes risk assessment demo (BigQuery ML + ADK)"
)
if [[ -n "${AGENT_ENGINE_ID:-}" ]]; then
  echo "Updating existing deployment: ${AGENT_ENGINE_ID}"
  DEPLOY_ARGS+=(--agent_engine_id="${AGENT_ENGINE_ID}")
fi

adk deploy agent_engine "${DEPLOY_ARGS[@]}" diabetes_agent

echo ""
echo "Deployment finished. Copy the resource id from the output above, then:"
echo ""
echo "  export AGENT_ENGINE_ID=<resource-id>   # re-run this script to update in place"
echo ""
echo "Test it from the console (Agent Runtime), or query it directly:"
echo "  https://${AGENT_REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${AGENT_REGION}/reasoningEngines/<resource-id>:query"
