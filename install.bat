@echo off
rem One-time setup for Windows: creates .venv and installs VideoGen with CUDA PyTorch.
rem Override the CUDA build with:  set TORCH_CUDA=cu126   (older NVIDIA drivers)
setlocal
cd /d "%~dp0"

set "PY=python"
where py >nul 2>nul
if %errorlevel%==0 set "PY=py -3"
if "%TORCH_CUDA%"=="" set "TORCH_CUDA=cu130"

%PY% -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)"
if errorlevel 1 (
  echo Python 3.10 - 3.13 is required. Install it from https://www.python.org/downloads/
  exit /b 1
)

echo ==^> Creating virtual environment in .venv
%PY% -m venv .venv || goto :error
call .venv\Scripts\activate.bat || goto :error
python -m pip install --upgrade pip wheel || goto :error

where nvidia-smi >nul 2>nul
if %errorlevel%==0 (
  echo ==^> Installing PyTorch with CUDA ^(%TORCH_CUDA%^)
  pip install torch torchvision --index-url https://download.pytorch.org/whl/%TORCH_CUDA% || goto :error
) else (
  echo ==^> No NVIDIA GPU found: installing CPU PyTorch ^(only the test engine will be usable^)
  pip install torch torchvision || goto :error
)

echo ==^> Installing VideoGen
pip install -e . || goto :error

echo.
echo VideoGen is installed.
echo.
echo Next steps:
echo   1. LTX-2.5 (best model) needs a free Hugging Face license click:
echo        open https://huggingface.co/Lightricks/LTX-2.5-Diffusers and click "Agree and access repository",
echo        then run:  .venv\Scripts\hf auth login
echo   2. Optional: pre-download weights:  .venv\Scripts\videogen download ltx
echo   3. Start the app:  run.bat
exit /b 0

:error
echo.
echo Installation failed. See the messages above.
exit /b 1
