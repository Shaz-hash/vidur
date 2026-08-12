#!/usr/bin/env bash
set -euo pipefail

umask 002

APP_HOME="${APP_HOME:-/home/ubuntu/vidur-classical-search}"
cd "${APP_HOME}"

mode="${1:-smoke}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "${mode}" in
    smoke)
        exec /usr/local/bin/agz-container-smoke "$@"
        ;;
    worker)
        exec python -m vidur.AlphaGoZero.worker_daemon "$@"
        ;;
    spot-worker)
        exec python -m vidur.AlphaGoZero.spot_worker_daemon "$@"
        ;;

    arena)
        exec python -m vidur.bellman_v4_adv.arena_mcts_value_runnerCPP "$@"
        ;;
    shell)
        exec /bin/bash "$@"
        ;;
    *)
        exec "${mode}" "$@"
        ;;
esac
