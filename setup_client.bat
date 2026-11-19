@echo off
REM One-time setup on the client PC.  Needs Python 3.11 (64-bit) installed,
REM "Add python.exe to PATH" ticked.  Run from the project folder.
REM NEVER copy the .venv folder between PCs - it contains absolute paths. Build it here.
python -m venv .venv || goto :err
call .venv\Scripts\activate || goto :err
python -m pip install --upgrade pip

REM --- PyTorch: pick ONE line.
REM NVIDIA GPU (check driver with nvidia-smi; choose the CUDA build from pytorch.org):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || goto :err
REM No GPU (slow):  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt || goto :err
python -c "import torch;print('CUDA available:', torch.cuda.is_available())"
echo.
echo Setup finished. Next: set the camera URL (see README) and run run_gate.bat
exit /b 0
:err
echo SETUP FAILED
exit /b 1
