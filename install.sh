#!/usr/bin/env bash
set -euo pipefail
PLUGIN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$PLUGIN_DIR/scripts/install_plugin.py" "$@"
