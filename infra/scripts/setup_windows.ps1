# RVC Real-time Voice Conversion - Windows Setup Script
# Run this in PowerShell as Administrator (for some features)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "RVC Real-time Voice Conversion - Windows Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "[1/4] Checking Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Python not found!" -ForegroundColor Red
    Write-Host "  Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "  Make sure to check 'Add Python to PATH' during installation" -ForegroundColor Yellow
    exit 1
}

# Check pip
Write-Host ""
Write-Host "[2/4] Checking pip..." -ForegroundColor Yellow
try {
    $pipVersion = pip --version 2>&1
    Write-Host "  Found: $pipVersion" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: pip not found!" -ForegroundColor Red
    exit 1
}

# Install PyTorch with CUDA
Write-Host ""
Write-Host "[3/4] Installing PyTorch with CUDA support..." -ForegroundColor Yellow
Write-Host "  This may take a few minutes..." -ForegroundColor Gray
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Warning: PyTorch installation had issues" -ForegroundColor Yellow
}

# Install other dependencies
Write-Host ""
Write-Host "[4/4] Installing other dependencies..." -ForegroundColor Yellow

# Core dependencies
$packages = @(
    "numpy",
    "scipy",
    "librosa",
    "soundfile",
    "sounddevice",
    "websockets",
    "faiss-cpu",
    "fairseq",
    "praat-parselmouth",
    "pyworld",
    "torchcrepe"
)

foreach ($pkg in $packages) {
    Write-Host "  Installing $pkg..." -ForegroundColor Gray
    pip install $pkg --quiet
}

# Verify installation
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Verifying installation..." -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan

$testScript = @"
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"PyTorch: ERROR - {e}")

try:
    import sounddevice
    print(f"sounddevice: OK")
except Exception as e:
    print(f"sounddevice: ERROR - {e}")

try:
    import websockets
    print(f"websockets: OK")
except Exception as e:
    print(f"websockets: ERROR - {e}")

try:
    import librosa
    print(f"librosa: OK")
except Exception as e:
    print(f"librosa: ERROR - {e}")

try:
    import faiss
    print(f"faiss: OK")
except Exception as e:
    print(f"faiss: ERROR - {e}")
"@

python -c $testScript

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Start the RVC server:" -ForegroundColor White
Write-Host "     python main.py --mode api --model ./assets/models/BillCipher/BillCipher.pth --index ./assets/models/BillCipher/BillCipher.index" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. In another terminal, run the client:" -ForegroundColor White
Write-Host "     python examples/windows_full_client.py --input-device `"Jabra`" --output-device `"CABLE Input`"" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. In Discord, select 'CABLE Output' as your microphone" -ForegroundColor White
Write-Host ""
