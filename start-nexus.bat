@echo off
REM NEXUS AI v4.0 — Ollama + NEXUS Startup Script
REM Starts both Ollama Docker container and NEXUS AI application

echo ============================================================
echo  NEXUS AI v4.0 — Startup Script
echo ============================================================
echo.

REM Check if Docker is running
echo Checking Docker...
docker ps >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker Desktop.
    pause
    exit /b 1
)
echo ✅ Docker is running
echo.

REM Start Ollama container if not running
echo Checking Ollama container...
docker ps | find "ollama" >nul 2>&1
if errorlevel 1 (
    echo Starting Ollama container...
    docker start ollama >nul 2>&1
    if errorlevel 1 (
        echo Creating new Ollama container...
        docker run -d --name ollama -p 11434:11434 -v ollama:/root/.ollama ollama/ollama:latest >nul 2>&1
    )
    timeout /t 5 /nobreak
)
echo ✅ Ollama container is running
echo.

REM Verify Ollama is ready
echo Verifying Ollama API...
python verify_ollama.py
if errorlevel 1 (
    echo ❌ Ollama verification failed
    pause
    exit /b 1
)
echo.

REM Load environment variables
echo Loading configuration from .env.ollama...
for /f "tokens=*" %%A in ('type .env.ollama ^| findstr /v "^#" ^| findstr "="') do set %%A
echo ✅ Configuration loaded
echo.

REM Start NEXUS AI
echo ============================================================
echo Starting NEXUS AI v4.0...
echo ============================================================
echo.
echo LLM Backend: %OLLAMA_HOST%
echo Primary Model: %OLLAMA_MODEL_PRIMARY%
echo.

python main.py

pause
