#!/bin/sh
# Re-runs the three scanners used in the report from the repository root.
cd "$(dirname "$0")/.."
bandit -r . -x ./data,./models
semgrep scan --config p/python --config p/flask --config p/secrets --config p/security-audit --config p/trailofbits \
    --metrics=off --exclude data --exclude models
pip-audit -r requirements.txt
