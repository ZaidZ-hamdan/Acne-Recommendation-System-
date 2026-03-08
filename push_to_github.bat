@echo off
REM Push project to GitHub using Git's full path (in case Git is not on PATH)
set GIT="C:\Program Files\Git\bin\git.exe"
cd /d "%~dp0"

echo Adding Git to PATH for this session...
set "PATH=%PATH%;C:\Program Files\Git\bin"

echo.
echo --- git init ---
git init
if errorlevel 1 goto err

echo.
echo --- git remote ---
git remote remove origin 2>nul
git remote add origin https://github.com/ZaidZ-hamdan/Acne-Recommendation-System-.git
if errorlevel 1 goto err

echo.
echo --- git add . ---
git add .
echo.
echo --- git status ---
git status

echo.
echo --- git commit ---
git commit -m "Initial commit: Acne AI Assistant - detection, chatbot, recommendations"
if errorlevel 1 (
  echo No changes to commit, or commit failed. Try: git status
  pause
  exit /b 0
)

echo.
echo --- git branch -M main ---
git branch -M main

echo.
echo --- git push (you may be asked to sign in) ---
git push -u origin main
if errorlevel 1 goto err

echo.
echo Done. Check: https://github.com/ZaidZ-hamdan/Acne-Recommendation-System-
pause
exit /b 0

:err
echo.
echo Something failed. Make sure Git is at: C:\Program Files\Git\bin\git.exe
echo If Git is elsewhere, edit this .bat and change the path.
pause
exit /b 1
