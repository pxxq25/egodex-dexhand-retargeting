#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=/dev/null
source /opt/egodex/project/activate.sh
exec "$@"
