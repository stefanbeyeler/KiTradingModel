@echo off
REM ============================================================
REM KITradingModel - Restart All Services
REM ============================================================
REM Startet alle Dienste neu:
REM   1. Dashboard Container (Port 3001)
REM   2. FastAPI Backend (Port 3011)
REM   3. Ollama LLM Service (Port 11434)
REM ============================================================

setlocal enabledelayedexpansion

echo ============================================================
echo        KITradingModel - Dienste neustarten
echo ============================================================
echo.

cd /d "%~dp0"

REM ============================================================
REM Phase 1: Alle Dienste stoppen
REM ============================================================
echo ============================================================
echo                   Phase 1: Dienste stoppen
echo ============================================================
echo.

REM --- Dashboard Container ---
echo [1/3] Stoppe Dashboard Container...
docker info >nul 2>&1
if %errorlevel% equ 0 (
    docker ps --filter "name=ki-trading-dashboard" --format "{{.Names}}" | findstr "ki-trading-dashboard" >nul 2>&1
    if %errorlevel% equ 0 (
        docker stop ki-trading-dashboard >nul 2>&1
        echo       [OK] Dashboard Container gestoppt
    ) else (
        echo       [INFO] Dashboard Container war nicht aktiv
    )
) else (
    echo       [WARNUNG] Docker nicht verfuegbar
)
echo.

REM --- FastAPI Backend ---
echo [2/3] Stoppe FastAPI Backend...
set "found=0"
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":3011" ^| findstr "LISTENING"') do (
    set "pid=%%a"
    if not "!pid!"=="" (
        taskkill /F /PID !pid! >nul 2>&1
        set "found=1"
    )
)
if "!found!"=="1" (
    echo       [OK] FastAPI Backend gestoppt
) else (
    echo       [INFO] FastAPI Backend war nicht aktiv
)
echo.

REM --- Ollama ---
echo [3/3] Stoppe Ollama...
set "found=0"
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":11434" ^| findstr "LISTENING"') do (
    set "pid=%%a"
    if not "!pid!"=="" (
        taskkill /F /PID !pid! >nul 2>&1
        set "found=1"
    )
)
taskkill /F /IM ollama.exe >nul 2>&1
taskkill /F /IM ollama_llama_server.exe >nul 2>&1
if "!found!"=="1" (
    echo       [OK] Ollama gestoppt
) else (
    echo       [INFO] Ollama war nicht aktiv
)
echo.

REM Kurze Pause um Ports freizugeben
echo [INFO] Warte 3 Sekunden...
timeout /t 3 /nobreak >nul
echo.

REM ============================================================
REM Phase 2: Alle Dienste starten
REM ============================================================
echo ============================================================
echo                   Phase 2: Dienste starten
echo ============================================================
echo.

REM --- Ollama ---
echo [1/4] Starte Ollama LLM Service...
start "" ollama serve
timeout /t 5 /nobreak >nul

curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% equ 0 (
    echo       [OK] Ollama gestartet auf Port 11434
) else (
    echo       [WARNUNG] Ollama konnte nicht gestartet werden
)
echo.

REM --- TimescaleDB Check ---
echo [2/4] Pruefe TimescaleDB...
netstat -an | findstr ":5432" | findstr "LISTENING" >nul 2>&1
if %errorlevel% equ 0 (
    echo       [OK] TimescaleDB laeuft auf Port 5432
) else (
    echo       [WARNUNG] TimescaleDB nicht erreichbar
)
echo.

REM --- FastAPI Backend ---
echo [3/4] Starte FastAPI Backend...
if exist "venv\Scripts\python.exe" (
    start "KITradingModel-API" cmd /k "cd /d %~dp0 && venv\Scripts\python.exe run.py"
) else (
    start "KITradingModel-API" cmd /k "cd /d %~dp0 && python run.py"
)

echo       [INFO] Warte auf Backend-Start...
timeout /t 10 /nobreak >nul

curl -s http://localhost:3011/health >nul 2>&1
if %errorlevel% equ 0 (
    echo       [OK] FastAPI Backend gestartet auf Port 3011
) else (
    echo       [INFO] Backend startet noch...
)
echo.

REM --- Dashboard Container ---
echo [4/4] Starte Dashboard Container...
docker info >nul 2>&1
if %errorlevel% equ 0 (
    docker ps -a --filter "name=ki-trading-dashboard" --format "{{.Names}}" | findstr "ki-trading-dashboard" >nul 2>&1
    if %errorlevel% equ 0 (
        docker start ki-trading-dashboard >nul 2>&1
    ) else (
        docker images --format "{{.Repository}}" | findstr "ki-trading-dashboard" >nul 2>&1
        if %errorlevel% neq 0 (
            echo       [INFO] Baue Dashboard Image...
            docker build -t ki-trading-dashboard:latest ./dashboard
        )
        docker run -d --name ki-trading-dashboard -p 3001:80 --restart unless-stopped ki-trading-dashboard:latest >nul 2>&1
    )
    timeout /t 3 /nobreak >nul
    echo       [OK] Dashboard Container gestartet auf Port 3001
) else (
    echo       [WARNUNG] Docker nicht verfuegbar
)
echo.

REM ============================================================
REM Status-Zusammenfassung
REM ============================================================
echo ============================================================
echo                    Neustart abgeschlossen
echo ============================================================
echo.
echo   Service              Port      Status
echo   --------------------------------------------------

curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% equ 0 (
    echo   Ollama LLM           11434     [OK] Aktiv
) else (
    echo   Ollama LLM           11434     [--] Nicht erreichbar
)

netstat -an | findstr ":5432" | findstr "LISTENING" >nul 2>&1
if %errorlevel% equ 0 (
    echo   TimescaleDB          5432      [OK] Aktiv
) else (
    echo   TimescaleDB          5432      [--] Nicht erreichbar
)

curl -s http://localhost:3011/health >nul 2>&1
if %errorlevel% equ 0 (
    echo   FastAPI Backend      3011      [OK] Aktiv
) else (
    echo   FastAPI Backend      3011      [..] Startet noch
)

docker ps --filter "name=ki-trading-dashboard" --format "{{.Names}}" 2>nul | findstr "ki-trading-dashboard" >nul 2>&1
if %errorlevel% equ 0 (
    echo   Dashboard            3001      [OK] Aktiv
) else (
    echo   Dashboard            3001      [--] Nicht aktiv
)

echo.
echo   URLs:
echo   - Dashboard:   http://localhost:3001
echo   - API:         http://localhost:3011
echo   - Swagger:     http://localhost:3011/docs
echo.
echo ============================================================
echo.

pause
