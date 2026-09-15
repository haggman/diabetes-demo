#!/usr/bin/env bash
#
# Idempotent bootstrap for the diabetes demo Cloud Agent environment.
#
# Installs the system toolchain the demo needs (Python venv support and the
# Google Cloud SDK, which provides `gcloud` and `bq`), then creates the project
# virtualenv and installs the pinned Python dependencies (ADK 2.x + BigQuery
# toolset). It is safe to run repeatedly.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export DEBIAN_FRONTEND=noninteractive

echo "=== Installing system packages (python venv, curl, gnupg) ==="
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip curl gnupg

echo "=== Installing Google Cloud SDK (gcloud + bq) ==="
if ! command -v gcloud >/dev/null 2>&1; then
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list >/dev/null
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | sudo gpg --dearmor --batch --yes -o /usr/share/keyrings/cloud.google.gpg
  sudo apt-get update -qq
  sudo apt-get install -y -qq google-cloud-cli
else
  echo "gcloud already installed: $(gcloud --version | head -1)"
fi

echo "=== Creating Python virtual environment ==="
if [[ ! -d venv ]]; then
  python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

echo "=== Installing Python dependencies ==="
pip install --upgrade pip wheel
pip install -r requirements.txt

# ADK asks for telemetry consent on first run, which blocks non-interactive
# startup of `adk web`. Opt out so the terminal comes up cleanly.
adk telemetry disable >/dev/null 2>&1 || true

echo ""
echo "Install complete."
echo "  python : $(python --version)"
echo "  adk    : $(adk --version)"
echo "  gcloud : $(gcloud --version | head -1)"
