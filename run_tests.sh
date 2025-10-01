#!/usr/bin/env bash
set -euo pipefail

# Optional: clean previous artifacts
rm -rf .pytest_cache coverage.xml htmlcov >/dev/null 2>&1 || true

# Ensure local modules are importable
export PYTHONPATH="$(pwd)/ForceSMIP:$(pwd)/cope_methods:${PYTHONPATH:-}"

echo "Running pytest..."
pytest -q --maxfail=1 --disable-warnings "$@"

echo "All tests completed."