#!/usr/bin/env bash
set -Eeuo pipefail

exec /usr/bin/docker exec vidur-agz-spot-coordinator \
    /home/ubuntu/vidur-classical-search/.venv/bin/python3 "$@"
