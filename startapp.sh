#!/bin/sh

set -e

echo "Starting local development environment"

# 1. Create network only if needed
NETWORK_NAME="whitman-network"

if ! docker network ls | grep -q "$NETWORK_NAME"; then
  echo "Creating Docker network: $NETWORK_NAME"
  docker network create $NETWORK_NAME
else
  echo "Docker network '$NETWORK_NAME' already exists"
fi

# 2. Stop existing containers gracefully
echo "Stopping existing docker-compose stack..."
docker compose -f docker/docker-compose.yml down --remove-orphans

# 3. Start the stack in background (recommended for dev)
echo "Starting docker services..."
docker compose -f docker/docker-compose.yml up --build --remove-orphans
