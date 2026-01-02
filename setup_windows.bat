@echo off
REM RVC Real-time Voice Conversion - Windows Setup Script
REM Run this by double-clicking or from Command Prompt/PowerShell

echo ============================================================
echo RVC Real-time Voice Conversion - Windows Setup
echo ============================================================
echo.

REM Check Python
echo [1/4] Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   ERROR: Python not found!
    echo   Download from: https://www.python.org/downloads/
    echo   Make sure to check 'Add Python to PATH' during installation
    pause
    exit /b 1
)
python --version
echo.

REM Check pip
echo [2/4] Checking pip...
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   ERROR: pip not found!
    pause
    exit /b 1
)
pip --version
echo.

REM Install PyTorch with CUDA
echo [3/4] Installing PyTorch with CUDA support...
echo   This may take several minutes...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo.

REM Install other dependencies
echo [4/4] Installing other dependencies...
pip install numpy scipy librosa soundfile sounddevice websockets faiss-cpu fairseq praat-parselmouth pyworld torchcrepe
echo.

REM Verify installation
echo ============================================================
echo Verifying installation...
echo ============================================================
python -c "import sys; print(f'Python: {sys.version}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '')"
python -c "import sounddevice; print('sounddevice: OK')"
python -c "import websockets; print('websockets: OK')"
python -c "import librosa; print('librosa: OK')"
python -c "import faiss; print('faiss: OK')"
echo.

echo ============================================================
echo Setup Complete!
echo ============================================================
echo.
echo Next steps:
echo.
echo   1. Start the RVC server (Terminal 1):
echo      python main.py --mode api --model ./assets/models/BillCipher/BillCipher.pth --index ./assets/models/BillCipher/BillCipher.index
echo.
echo   2. Start the voice changer client (Terminal 2):
echo      python examples/windows_full_client.py --input-device "Jabra" --output-device "CABLE Input"
echo.
echo   3. In Discord, select 'CABLE Output' as your microphone
echo.
pause
