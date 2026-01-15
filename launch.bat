@echo off
echo ===================================================
echo   NOA DeepThink Launcher
echo ===================================================
echo.

echo [1/3] Installing/Updating dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b %errorlevel%
)

echo.
echo [2/3] Committing and pushing changes to Git...
git add .
git commit -m "Auto-commit before launch"
git push
if %errorlevel% neq 0 (
    echo.
    echo [WARNING] Git push failed. Proceeding with launch anyway...
) else (
    echo [SUCCESS] Git changes pushed.
)

echo.
echo [3/3] Launching NOA App...
echo Access the app at: http://localhost:8000
echo.
python app.py
pause
