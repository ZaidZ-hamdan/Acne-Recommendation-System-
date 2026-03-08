@echo off
cd /d "%~dp0"
REM Prefer Python 3.11 for TensorFlow; fall back to default python
set PYEXE=python
py -3.11 --version >nul 2>&1 && set PYEXE=py -3.11
if "%PYEXE%"=="python" if exist "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" set PYEXE=%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe
echo Using: %PYEXE%
%PYEXE% --version
echo.
echo Installing dependencies...
%PYEXE% -m pip install -r requirements.txt --quiet
if errorlevel 1 (
  echo.
  echo Pip install failed. TensorFlow requires Python 3.9-3.11.
  echo See INSTALL_PYTHON_311.md - install Python 3.11 then run:
  echo   py -3.11 -m pip install -r requirements.txt
  echo   py -3.11 train_model.py
  pause
  exit /b 1
)
echo.
echo Starting training...
%PYEXE% train_model.py
pause
