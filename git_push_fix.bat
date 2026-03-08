@echo off
cd /d "%~dp0"
set "PATH=%PATH%;C:\Program Files\Git\bin"

echo 1. Adding all files...
git add .
git status

echo.
echo 2. Creating first commit...
git commit -m "Initial commit: Acne AI Assistant"
if errorlevel 1 (
  echo Commit failed. If it says "nothing to commit", run: git status
  pause
  exit /b 1
)

echo.
echo 3. Renaming branch to main and pushing...
git branch -M main
git push -u origin main

echo.
echo Done.
pause
