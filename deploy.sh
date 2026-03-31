#!/bin/bash
set -e

IMAGE_NAME="xiicu/me-api-proxy"
TAG="latest"
DOCKER_CONFIG_DIR="${DOCKER_CONFIG_DIR:-/tmp/docker-config-no-creds}"

prepare_docker_config() {
  mkdir -p "$DOCKER_CONFIG_DIR"

  if [ -f "$HOME/.docker/config.json" ]; then
    cp "$HOME/.docker/config.json" "$DOCKER_CONFIG_DIR/config.json"
    ruby -rjson -e '
      path = ARGV[0]
      json = JSON.parse(File.read(path))
      json.delete("credsStore")
      File.write(path, JSON.pretty_generate(json))
    ' "$DOCKER_CONFIG_DIR/config.json"
  elif [ ! -f "$DOCKER_CONFIG_DIR/config.json" ]; then
    echo '{"auths":{}}' > "$DOCKER_CONFIG_DIR/config.json"
  fi
}

case "$1" in
  build)
    echo "Building image..."
    prepare_docker_config
    DOCKER_CONFIG="$DOCKER_CONFIG_DIR" docker build -t $IMAGE_NAME:$TAG .
    echo "Build complete: $IMAGE_NAME:$TAG"
    ;;
  push)
    prepare_docker_config
    echo "Logging in as xiicu (enter password interactively)..."
    DOCKER_CONFIG="$DOCKER_CONFIG_DIR" docker login -u xiicu
    echo "Pushing image..."
    DOCKER_CONFIG="$DOCKER_CONFIG_DIR" docker push $IMAGE_NAME:$TAG
    echo "Push complete"
    ;;
  start)
    docker-compose up -d
    echo "Service started on port 38001"
    ;;
  stop)
    docker-compose down
    echo "Service stopped"
    ;;
  logs)
    docker-compose logs -f
    ;;
  all)
    $0 build
    $0 push
    $0 start
    ;;
  *)
    echo "Usage: $0 {build|push|start|stop|logs|all}"
    echo ""
    echo "Commands:"
    echo "  build  - Build Docker image"
    echo "  push   - Login and push to DockerHub"
    echo "  start  - Start service with docker-compose"
    echo "  stop   - Stop service"
    echo "  logs   - Show service logs"
    echo "  all    - Build, push and start"
    ;;
esac
