#!/bin/zsh
set -eu

cd /Users/jarvis/project/laya-mlx
set -a
source /Users/jarvis/project/laya-mlx/.env
set +a

service_venv="${LAYA_SERVICE_VENV:-/Users/jarvis/project/laya-mlx/.venv}"
exec "$service_venv/bin/laya-service"
