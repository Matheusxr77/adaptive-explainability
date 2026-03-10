"""
Test Setup Script
Verifies that all components are properly installed and configured
"""

import sys
import os

def test_python_version():
    """Test Python version"""
    print("=" * 60)
    print("TESTING PYTHON VERSION")
    print("=" * 60)
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 9:
        print("✓ Python version is compatible")
        return True
    else:
        print("✗ Python 3.9+ required")
        return False

def test_imports():
    """Test all required imports"""
    print("\n" + "=" * 60)
    print("TESTING PACKAGE IMPORTS")
    print("=" * 60)
    
    packages = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'scikit-learn'),
        ('xgboost', 'XGBoost'),
        ('lime', 'LIME'),
        ('shap', 'SHAP'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('plotly', 'Plotly'),
        ('requests', 'Requests'),
        ('jupyter', 'Jupyter')
    ]
    
    all_passed = True
    for module, name in packages:
        try:
            __import__(module)
            print(f"✓ {name} imported successfully")
        except ImportError as e:
            print(f"✗ {name} import failed: {e}")
            all_passed = False
    
    return all_passed

def test_data_file():
    """Test data file exists"""
    print("\n" + "=" * 60)
    print("TESTING DATA FILE")
    print("=" * 60)
    
    data_path = os.path.join('data', 'credit_risk_dataset.csv')
    
    if os.path.exists(data_path):
        size = os.path.getsize(data_path)
        print(f"✓ Data file found: {data_path}")
        print(f"  Size: {size:,} bytes")
        
        # Try to read first few lines
        try:
            import pandas as pd
            df = pd.read_csv(data_path, nrows=5)
            print(f"  Columns: {len(df.columns)}")
            print(f"  Sample rows read successfully")
            return True
        except Exception as e:
            print(f"✗ Error reading data file: {e}")
            return False
    else:
        print(f"✗ Data file not found: {data_path}")
        return False

def test_slm_connection():
    """Test SLM connection"""
    print("\n" + "=" * 60)
    print("TESTING SLM CONNECTION")
    print("=" * 60)
    
    import requests
    
    urls = [
        ("http://localhost:8080", "Primary SLM (Qwen2.5)"),
        ("http://localhost:8081", "Backup SLM (Granite)")
    ]
    
    any_connected = False
    
    for url, name in urls:
        try:
            # Try health endpoint
            response = requests.get(f"{url}/health", timeout=2)
            print(f"✓ {name} responding at {url}")
            any_connected = True
        except requests.exceptions.ConnectionError:
            print(f"✗ {name} not available at {url}")
        except requests.exceptions.Timeout:
            print(f"⚠ {name} timeout at {url}")
        except Exception as e:
            print(f"✗ {name} error: {e}")
    
    if not any_connected:
        print("\n⚠ No SLM servers available")
        print("  Run docker_setup.ps1 to start SLM containers")
        print("  Tasks 1.1 and 1.2 require SLM connection")
    
    return any_connected

def test_src_modules():
    """Test src modules can be imported"""
    print("\n" + "=" * 60)
    print("TESTING SRC MODULES")
    print("=" * 60)
    
    # Add src to path
    sys.path.insert(0, 'src')
    
    modules = [
        'model_trainer',
        'slm_interface',
        'explainer_wrapper',
        'adaptive_selector',
        'explanation_aggregator',
        'metrics'
    ]
    
    all_passed = True
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}.py imported successfully")
        except ImportError as e:
            print(f"✗ {module}.py import failed: {e}")
            all_passed = False
    
    return all_passed

def test_docker():
    """Test Docker availability"""
    print("\n" + "=" * 60)
    print("TESTING DOCKER")
    print("=" * 60)
    
    import subprocess
    
    try:
        result = subprocess.run(
            ['docker', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print(f"✓ Docker installed: {result.stdout.strip()}")
            
            # Check running containers
            result = subprocess.run(
                ['docker', 'ps', '--format', '{{.Names}}'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                containers = result.stdout.strip().split('\n')
                containers = [c for c in containers if c]
                
                print(f"✓ Docker running: {len(containers)} container(s)")
                for container in containers:
                    print(f"  - {container}")
                
                # Check for our SLM containers
                slm_containers = [c for c in containers if 'qwen' in c.lower() or 'granite' in c.lower()]
                if slm_containers:
                    print(f"✓ SLM containers found: {', '.join(slm_containers)}")
                    return True
                else:
                    print("⚠ SLM containers not found")
                    return False
            else:
                print("⚠ Docker not running containers")
                return False
        else:
            print("✗ Docker command failed")
            return False
            
    except FileNotFoundError:
        print("✗ Docker not installed or not in PATH")
        return False
    except Exception as e:
        print(f"✗ Docker test error: {e}")
        return False

def main():
    """Run all tests"""
    print("\n")
    print("*" * 60)
    print("ADAPTIVE EXPLAINABILITY - SETUP TEST")
    print("*" * 60)
    print("\n")
    
    results = {}
    
    results['Python Version'] = test_python_version()
    results['Package Imports'] = test_imports()
    results['Data File'] = test_data_file()
    results['Src Modules'] = test_src_modules()
    results['Docker'] = test_docker()
    results['SLM Connection'] = test_slm_connection()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {test}")
    
    total = len(results)
    passed = sum(results.values())
    
    print("\n" + "-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    print("-" * 60)
    
    if passed == total:
        print("\n🎉 All tests passed! Ready to run the analysis.")
        print("\nNext steps:")
        print("1. jupyter notebook")
        print("2. Open notebooks/adaptive_explainability_analysis.ipynb")
        print("3. Run all cells")
    elif results['SLM Connection'] is False:
        print("\n⚠ Core components OK, but SLM not available")
        print("\nYou can:")
        print("1. Run docker_setup.ps1 to start SLM containers")
        print("2. Or run notebook sections 1-3 and 7 without SLM")
        print("   (Tasks 1.1 and 1.2 require SLM)")
    else:
        print("\n❌ Some tests failed. Please fix errors before proceeding.")
        print("\nSee SETUP.md for troubleshooting steps.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
