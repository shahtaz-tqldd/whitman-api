#!/bin/sh

set -eu

echo "Starting Whitman development environment"

COMPOSE_FILE="docker/docker-compose.yml"
NETWORK_NAME="whitman-network"

# Ensure the script runs from the project root.
PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$PROJECT_DIR"

# Create the network only if it doesn't exist.
if ! docker network inspect "$NETWORK_NAME" >/dev/null 2>&1; then
    echo "Creating Docker network: $NETWORK_NAME"
    docker network create "$NETWORK_NAME"
fi

# Stop the previous Compose stack without deleting database volumes.
echo "Stopping existing Compose stack..."
docker compose -f "$COMPOSE_FILE" down --remove-orphans

# Build the API image before running setup commands.
echo "Building API image..."
docker compose -f "$COMPOSE_FILE" build api

# Start PostgreSQL first.
echo "Starting PostgreSQL..."
docker compose -f "$COMPOSE_FILE" up -d db

# Wait for PostgreSQL to become ready.
echo "Waiting for PostgreSQL..."
attempt=0
max_attempts=60

until docker compose -f "$COMPOSE_FILE" exec -T db 
    pg_isready -U whitman -d whitman >/dev/null 2>&1
do
    attempt=$((attempt + 1))

    if [ "$attempt" -ge "$max_attempts" ]; then
        echo "PostgreSQL did not become ready in time."
        exit 1
    fi

    sleep 2
done

echo "PostgreSQL is ready."

# Download the embedding model only when needed.
echo "Checking embedding model..."
docker compose -f "$COMPOSE_FILE" run --rm api 
    python -m scripts.download_model

# Create the admin user if it doesn't already exist.
echo "Checking admin user..."
docker compose -f "$COMPOSE_FILE" run --rm api sh -c '
    set -eu

    : "${ADMIN_NAME:?ADMIN_NAME is required in .env}"
    : "${ADMIN_USERNAME:?ADMIN_USERNAME is required in .env}"
    : "${ADMIN_EMAIL:?ADMIN_EMAIL is required in .env}"
    : "${ADMIN_PASSWORD:?ADMIN_PASSWORD is required in .env}"

    python -m scripts.create_user \
        --name "$ADMIN_NAME" \
        --username "$ADMIN_USERNAME" \
        --email "$ADMIN_EMAIL" \
        --password "$ADMIN_PASSWORD"
'

# Start the complete stack in the background.
echo "Starting Whitman services..."
docker compose -f "$COMPOSE_FILE" up -d --build --remove-orphans

echo ""
echo "Whitman is running."
echo "API: http://localhost:8020"
echo "PostgreSQL: localhost:5452"
docker compose -f "$COMPOSE_FILE" ps
