#!/usr/bin/env bash
#
# Prepare a shell for running the diabetes agent.
#
# Source this (it is sourced automatically by the `adk-web` terminal):
#   source .cursor/activate-env.sh
#
# It activates the Python virtualenv and, when Google Cloud credentials are
# supplied as Cloud Agent secrets, wires up Application Default Credentials and
# writes diabetes_agent/.env. It is intentionally safe to source even when no
# credentials are present - `adk web` still starts, agents are just unable to
# reach Gemini/BigQuery until credentials exist.

__REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "${__REPO_ROOT}"

# shellcheck disable=SC1091
source venv/bin/activate

# Materialize the service-account key (if provided) so google.auth.default()
# and gcloud both authenticate as that service account.
if [[ -n "${GCP_SERVICE_ACCOUNT_JSON:-}" ]]; then
  mkdir -p "${HOME}/.config/gcloud"
  __KEY_FILE="${HOME}/.config/gcloud/adk-sa-key.json"
  printf '%s' "${GCP_SERVICE_ACCOUNT_JSON}" > "${__KEY_FILE}"
  chmod 600 "${__KEY_FILE}"
  export GOOGLE_APPLICATION_CREDENTIALS="${__KEY_FILE}"
  if command -v gcloud >/dev/null 2>&1; then
    gcloud auth activate-service-account --key-file="${__KEY_FILE}" >/dev/null 2>&1 \
      || echo "warning: could not activate service account from GCP_SERVICE_ACCOUNT_JSON" >&2
    if [[ -n "${GOOGLE_CLOUD_PROJECT:-}" ]]; then
      gcloud config set project "${GOOGLE_CLOUD_PROJECT}" >/dev/null 2>&1 || true
    fi
  fi
else
  echo "note: GCP_SERVICE_ACCOUNT_JSON is not set - the agent will start but"
  echo "      cannot call Gemini or BigQuery until credentials are provided." >&2
fi

# activate.sh keys off PROJECT_ID; prefer the provided project id.
if [[ -n "${GOOGLE_CLOUD_PROJECT:-}" ]]; then
  export PROJECT_ID="${PROJECT_ID:-${GOOGLE_CLOUD_PROJECT}}"
fi

# Write diabetes_agent/.env so `adk web` and `adk deploy` see the project.
if [[ -n "${PROJECT_ID:-}" ]]; then
  # shellcheck disable=SC1091
  source activate.sh || echo "warning: activate.sh reported an error" >&2
fi
