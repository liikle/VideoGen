@echo off
rem Start the VideoGen web UI and open it in the browser.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run install.bat first.
  exit /b 1
)
.venv\Scripts\python.exe -m videogen serve --open %*
