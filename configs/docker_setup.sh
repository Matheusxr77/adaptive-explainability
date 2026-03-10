#!/bin/bash
# Docker Model Runner Setup for Adaptive Explainability Project
# This script sets up local SLMs using Docker Model Runner

echo "=== Docker Model Runner Setup ==="
echo "Setting up Qwen2.5 and IBM Granite 4.0 Nano models"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

echo "Docker detected: $(docker --version)"
echo ""

# Pull Qwen2.5 model (primary choice)
echo "--- Pulling Qwen2.5 model ---"
docker pull ai/qwen2.5:latest
if [ $? -ne 0 ]; then
    echo "Warning: Failed to pull ai/qwen2.5. Trying alternative..."
    docker pull qwen/qwen2.5:latest
fi

# Pull IBM Granite 4.0 Nano (backup choice)
echo ""
echo "--- Pulling IBM Granite 4.0 Nano model ---"
docker pull ai/granite-4.0-h-nano:latest
if [ $? -ne 0 ]; then
    echo "Warning: Failed to pull ai/granite-4.0-h-nano"
fi

echo ""
echo "--- Starting Qwen2.5 container ---"
# Run Qwen2.5 on port 8080
docker run -d --name qwen2.5-server \
    -p 8080:8080 \
    --restart unless-stopped \
    ai/qwen2.5:latest

echo ""
echo "--- Starting Granite 4.0 Nano container (backup) ---"
# Run Granite on port 8081 as backup
docker run -d --name granite-nano-server \
    -p 8081:8080 \
    --restart unless-stopped \
    ai/granite-4.0-h-nano:latest

echo ""
echo "=== Setup Complete ==="
echo "Qwen2.5 running on: http://localhost:8080"
echo "Granite 4.0 Nano running on: http://localhost:8081"
echo ""
echo "Test with:"
echo "curl -X POST http://localhost:8080/v1/completions \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"prompt\": \"Hello, world!\", \"max_tokens\": 50}'"
echo ""
echo "To stop containers:"
echo "docker stop qwen2.5-server granite-nano-server"
echo ""
echo "To remove containers:"
echo "docker rm qwen2.5-server granite-nano-server"
