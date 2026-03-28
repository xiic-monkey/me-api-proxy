#!/bin/bash
set -e

IMAGE_NAME="xiicu/me-api-proxy"
TAG="latest"

case "$1" in
  build)
    echo "Building image..."
    docker build -t $IMAGE_NAME:$TAG .
    echo "Build complete: $IMAGE_NAME:$TAG"
    ;;
  push)
    echo "Logging in as xiicu (enter password interactively)..."
    docker login -u xiicu
    echo "Pushing image..."
    docker push $IMAGE_NAME:$TAG
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
