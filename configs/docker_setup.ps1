# Docker Model Runner Setup for Adaptive Explainability Project (PowerShell)
# This script sets up local SLMs using Docker Model Runner

Write-Host "=== Docker Model Runner Setup ===" -ForegroundColor Cyan
Write-Host "Setting up Qwen2.5 and IBM Granite 4.0 Nano models" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is installed
try {
    $dockerVersion = docker --version
    Write-Host "Docker detected: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Docker is not installed. Please install Docker first." -ForegroundColor Red
    Write-Host "Visit: https://docs.docker.com/get-docker/"
    exit 1
}

Write-Host ""

# Pull Qwen2.5 model (primary choice)
Write-Host "--- Pulling Qwen2.5 model ---" -ForegroundColor Yellow
docker pull ai/qwen2.5:latest
if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: Failed to pull ai/qwen2.5. Trying alternative..." -ForegroundColor Yellow
    docker pull qwen/qwen2.5:latest
}

# Pull IBM Granite 4.0 Nano (backup choice)
Write-Host ""
Write-Host "--- Pulling IBM Granite 4.0 Nano model ---" -ForegroundColor Yellow
docker pull ai/granite-4.0-h-nano:latest
if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: Failed to pull ai/granite-4.0-h-nano" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "--- Starting Qwen2.5 container ---" -ForegroundColor Yellow
# Run Qwen2.5 on port 8080
docker run -d --name qwen2.5-server `
    -p 8080:8080 `
    --restart unless-stopped `
    ai/qwen2.5:latest

Write-Host ""
Write-Host "--- Starting Granite 4.0 Nano container (backup) ---" -ForegroundColor Yellow
# Run Granite on port 8081 as backup
docker run -d --name granite-nano-server `
    -p 8081:8080 `
    --restart unless-stopped `
    ai/granite-4.0-h-nano:latest

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host "Qwen2.5 running on: http://localhost:8080" -ForegroundColor Green
Write-Host "Granite 4.0 Nano running on: http://localhost:8081" -ForegroundColor Green
Write-Host ""
Write-Host "Test with PowerShell:"
Write-Host 'Invoke-RestMethod -Uri "http://localhost:8080/v1/completions" -Method POST -ContentType "application/json" -Body (@{prompt="Hello, world!"; max_tokens=50} | ConvertTo-Json)'
Write-Host ""
Write-Host "To stop containers:"
Write-Host "docker stop qwen2.5-server granite-nano-server"
Write-Host ""
Write-Host "To remove containers:"
Write-Host "docker rm qwen2.5-server granite-nano-server"
