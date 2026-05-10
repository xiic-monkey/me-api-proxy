#!/bin/bash
set -euo pipefail

REGISTRY="${REGISTRY:-ghcr.io}"
OWNER="${OWNER:-xiic-monkey}"
IMAGE_REPO="${IMAGE_REPO:-me-api-proxy}"
IMAGE_NAME="${IMAGE_NAME:-$REGISTRY/$OWNER/$IMAGE_REPO}"
TAG="${TAG:-latest}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
COMPOSE_CMD="${COMPOSE_CMD:-docker compose}"

require_env() {
  local var_name="$1"
  if [ -z "${!var_name:-}" ]; then
    echo "Missing required environment variable: $var_name" >&2
    exit 1
  fi
}

compose() {
  $COMPOSE_CMD -f "$COMPOSE_FILE" "$@"
}

login_ghcr() {
  require_env GHCR_USERNAME
  require_env GHCR_TOKEN
  echo "Logging in to $REGISTRY as $GHCR_USERNAME..."
  printf '%s' "$GHCR_TOKEN" | docker login "$REGISTRY" -u "$GHCR_USERNAME" --password-stdin
}

case "${1:-}" in
  login)
    login_ghcr
    ;;
  build)
    echo "Building image: $IMAGE_NAME:$TAG"
    docker build -t "$IMAGE_NAME:$TAG" .
    ;;
  push)
    login_ghcr
    echo "Pushing image: $IMAGE_NAME:$TAG"
    docker push "$IMAGE_NAME:$TAG"
    ;;
  pull)
    compose pull
    ;;
  up|start)
    require_env ADMIN_USERNAME
    require_env ADMIN_PASSWORD
    compose up -d
    echo "Service started on port ${HOST_PORT:-38001}"
    ;;
  restart)
    require_env ADMIN_USERNAME
    require_env ADMIN_PASSWORD
    compose up -d --force-recreate
    echo "Service restarted on port ${HOST_PORT:-38001}"
    ;;
  down|stop)
    compose down
    ;;
  logs)
    compose logs -f
    ;;
  ps|status)
    compose ps
    ;;
  release)
    login_ghcr
    docker build -t "$IMAGE_NAME:$TAG" .
    docker push "$IMAGE_NAME:$TAG"
    ;;
  *)
    cat <<'EOF'
Usage: ./deploy.sh <command>

Commands:
  login    Login to GitHub Container Registry
  build    Build image locally
  push     Push IMAGE_NAME:TAG to GHCR
  pull     Pull image from compose file
  up       Start service with docker compose
  restart  Recreate service with docker compose
  down     Stop service
  logs     Tail service logs
  ps       Show service status
  release  Login, build, and push

Defaults:
  IMAGE_NAME=ghcr.io/xiic-monkey/me-api-proxy
  TAG=latest
EOF
    exit 1
    ;;
esac
