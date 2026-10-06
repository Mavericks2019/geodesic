@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if errorlevel 1 goto failure
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failure
echo Ready. Double click open-animation.bat to launch.
pause
exit /b 0
:failure
echo Installation failed. Check the error above.
pause
exit /b 1
