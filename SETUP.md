# Setup Guide - Adaptive Explainability Project

Complete step-by-step guide to set up and run the adaptive explainability analysis.

## Prerequisites

### System Requirements
- **OS**: Windows 10/11, macOS, or Linux
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 5GB free space
- **Python**: 3.9 or higher
- **Docker**: Latest version

### Check Prerequisites

**Python:**
```bash
python --version
# Should show Python 3.9.x or higher
```

**Docker:**
```bash
docker --version
# Should show Docker version 20.x or higher
```

**Install Docker if needed:**
- Windows/Mac: [Docker Desktop](https://www.docker.com/products/docker-desktop)
- Linux: `sudo apt-get install docker.io` (Ubuntu/Debian)

## Step 1: Python Environment Setup

### Option A: Using Virtual Environment (Recommended)

**Windows:**
```powershell
# Create virtual environment
python -m venv venv

# Activate
.\venv\Scripts\Activate.ps1

# If you get execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Linux/Mac:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate
```

### Option B: Using Conda

```bash
# Create conda environment
conda create -n adaptive-explain python=3.9

# Activate
conda activate adaptive-explain
```

### Install Dependencies

```bash
# Navigate to project root
cd adaptive-explainability

# Install requirements
pip install -r requirements.txt

# Verify installation
pip list | grep -E "(xgboost|shap|lime)"
```

**Expected output:**
```
lime                     0.2.0.1
shap                     0.42.1
xgboost                  2.0.3
```

## Step 2: Docker SLM Setup

### Windows PowerShell

```powershell
# Navigate to configs
cd configs

# Run setup script
.\docker_setup.ps1

# Wait for downloads (may take 5-10 minutes first time)
```

### Linux/Mac Bash

```bash
# Navigate to configs
cd configs

# Make executable
chmod +x docker_setup.sh

# Run setup
./docker_setup.sh
```

### Verify Docker Containers

```bash
# Check running containers
docker ps

# Expected output:
# CONTAINER ID   IMAGE                        PORTS                    NAMES
# xxxxxxxxxxxx   ai/qwen2.5:latest           0.0.0.0:8080->8080/tcp   qwen2.5-server
# xxxxxxxxxxxx   ai/granite-4.0-h-nano:latest 0.0.0.0:8081->8080/tcp  granite-nano-server
```

### Test SLM Connection

**PowerShell:**
```powershell
$response = Invoke-RestMethod -Uri "http://localhost:8080/v1/completions" `
    -Method POST `
    -ContentType "application/json" `
    -Body (@{prompt="Test"; max_tokens=10} | ConvertTo-Json)
$response
```

**Bash/Curl:**
```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Test", "max_tokens": 10}'
```

**Expected response:**
```json
{
  "choices": [
    {
      "text": " message..."
    }
  ]
}
```

## Step 3: Verify Data

```bash
# Navigate to project root
cd ..

# Check data file exists
ls data/credit_risk_dataset.csv

# Windows:
dir data\credit_risk_dataset.csv

# Verify file is not empty
wc -l data/credit_risk_dataset.csv
# Should show thousands of lines
```

## Step 4: Launch Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Alternative: JupyterLab
jupyter lab

# Notebook will open in browser at http://localhost:8888
```

### Open the Main Notebook

1. Navigate to `notebooks/`
2. Open `adaptive_explainability_analysis.ipynb`
3. Run cells sequentially (Shift+Enter)

**Or run all cells at once:**
```bash
jupyter nbconvert --to notebook --execute \
    notebooks/adaptive_explainability_analysis.ipynb \
    --output adaptive_explainability_analysis_executed.ipynb
```

## Step 5: Initial Test Run

For a quick test (5-10 minutes):

1. Open the notebook
2. Find the cell with:
```python
n_test_instances = 50
n_adaptive_samples = min(5, len(test_instances))
n_aggregation_samples = min(5, len(test_instances))
```

3. Keep these small values for initial test
4. Run all cells (Cell → Run All)
5. Check outputs folder for generated plots

## Step 6: Full Analysis

For complete analysis (1-2 hours):

1. Update sample sizes in notebook:
```python
n_test_instances = 200  # Or len(X_test) for all
n_adaptive_samples = 50
n_aggregation_samples = 50
```

2. Optionally adjust for faster results:
```python
# In AdaptivePerturbationSelector initialization
stability_runs=2  # Instead of 3

# Use binary search
search_strategy="binary"  # Instead of "sequential"
```

3. Run analysis (Kernel → Restart & Run All)
4. Check `outputs/` for all results

## Troubleshooting Common Issues

### Issue 1: Python Module Not Found

```
ModuleNotFoundError: No module named 'lime'
```

**Solution:**
```bash
pip install --upgrade -r requirements.txt
```

### Issue 2: Docker Not Running

```
Cannot connect to Docker daemon
```

**Solution:**
- Windows/Mac: Start Docker Desktop
- Linux: `sudo systemctl start docker`

### Issue 3: Port Already in Use

```
docker: Error response from daemon: port is already allocated
```

**Solution:**
```bash
# Find process using port
netstat -ano | findstr :8080  # Windows
lsof -i :8080                  # Mac/Linux

# Stop existing container
docker stop qwen2.5-server
docker rm qwen2.5-server

# Re-run setup
```

### Issue 4: Out of Memory

```
Killed (memory error)
```

**Solution:**
1. Increase Docker memory limit (Docker Desktop → Settings → Resources)
2. Reduce sample sizes in notebook
3. Use smaller perturbation levels

### Issue 5: Jupyter Kernel Dies

```
Kernel Restarting: The kernel appears to have died
```

**Solution:**
1. Reduce sample sizes
2. Clear outputs and restart kernel
3. Check Docker container logs: `docker logs qwen2.5-server`
4. Restart Docker containers

### Issue 6: Slow SLM Responses

**Solution:**
1. Check Internet connection (if pulling models)
2. Verify Docker has sufficient resources
3. Use backup model if primary is slow:
```python
slm = SLMInterface(
    primary_url="http://localhost:8081",  # Try backup
    backup_url="http://localhost:8080"
)
```

## Performance Optimization

### For Faster Results

1. **Reduce sample sizes**: 5-10 instances for demo
2. **Use binary search**: Faster than sequential
3. **Fewer stability runs**: Set to 2 instead of 3
4. **Smaller perturbation range**: `[10, 50, 100, 500]`
5. **Cache SLM responses**: Already implemented in code

### For Better Quality

1. **Increase sample sizes**: 200+ instances
2. **Use sequential search**: More thorough
3. **More stability runs**: 3-5 runs
4. **Wider perturbation range**: `[5, 10, 25, 50, 100, 250, 500, 1000]`

## Verification Checklist

- [ ] Python 3.9+ installed
- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip list` shows required packages)
- [ ] Docker running (`docker ps` shows containers)
- [ ] SLM responding (`curl` test successful)
- [ ] Data file exists and has content
- [ ] Jupyter notebook opens
- [ ] First few cells run without errors
- [ ] Outputs folder populated after run

## Next Steps

Once setup is complete:

1. **Run initial test**: 5 samples to verify everything works
2. **Review outputs**: Check `outputs/` folder for plots
3. **Read summary**: Open `outputs/summary_report.txt`
4. **Full analysis**: Increase sample sizes and re-run
5. **Experiment**: Modify parameters and compare results

## Getting Help

### Check Logs

**Docker logs:**
```bash
docker logs qwen2.5-server
docker logs granite-nano-server
```

**Jupyter logs:**
Check terminal where you started Jupyter

### Verify Setup

**Test script:**
```python
# test_setup.py
import sys
print(f"Python: {sys.version}")

try:
    import numpy, pandas, sklearn, xgboost, lime, shap
    print("✓ All core packages imported")
except Exception as e:
    print(f"✗ Import error: {e}")

import requests
try:
    r = requests.get("http://localhost:8080/health", timeout=5)
    print(f"✓ SLM responding: {r.status_code}")
except Exception as e:
    print(f"✗ SLM not available: {e}")
```

Run: `python test_setup.py`

## Cleanup

After analysis is complete:

```bash
# Stop Docker containers
docker stop qwen2.5-server granite-nano-server

# Remove containers (optional)
docker rm qwen2.5-server granite-nano-server

# Remove images (optional, will need to re-download)
docker rmi ai/qwen2.5:latest ai/granite-4.0-h-nano:latest

# Deactivate Python environment
deactivate  # or: conda deactivate
```

---

**Ready to start?** Run through Steps 1-4, then open the notebook and begin the analysis!
