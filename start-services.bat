@echo off
REM ============================================================
REM KITradingModel - Start All Services
REM ============================================================
REM Dienste:
REM   1. Ollama LLM Service (Port 11434)
REM   2. TimescaleDB (Port 5432) - muss extern laufen
REM   3. FastAPI Backend (Port 8000)
REM   4. Dashboard Container (Port 3001)
REM ============================================================

setlocal enabledelayedexpansion

echo ============================================================
echo        KITradingModel - Dienste starten
echo ============================================================
echo.

cd /d "%~dp0"

REM ============================================================
REM 1. Prüfe und starte Ollama
REM ============================================================
echo [1/4] Pruefe Ollama LLM Service...

REM Prüfe ob Ollama läuft
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% equ 0 (
    echo       [OK] Ollama laeuft bereits auf Port 11434
) else (
    echo       [INFO] Starte Ollama...
    start "" ollama serve
    timeout /t 5 /nobreak >nul

    curl -s http://localhost:11434/api/tags >nul 2>&1
    if %errorlevel% equ 0 (
        echo       [OK] Ollama erfolgreich gestartet
    ) else (
        echo       [WARNUNG] Ollama konnte nicht gestartet werden
        echo       Bitte starten Sie Ollama manuell: ollama serve
    )
)
echo.

REM ============================================================
REM 2. Prüfe TimescaleDB
REM ============================================================
echo [2/4] Pruefe TimescaleDB...

REM Prüfe ob TimescaleDB erreichbar ist (Port 5432)
netstat -an | findstr ":5432" | findstr "LISTENING" >nul 2>&1
if %errorlevel% equ 0 (
    echo       [OK] TimescaleDB laeuft auf Port 5432
) else (
    echo       [WARNUNG] TimescaleDB scheint nicht zu laufen
    echo       Bitte starten Sie TimescaleDB manuell
)
echo.

REM ============================================================
REM 3. Starte FastAPI Backend
REM ============================================================
echo [3/4] Starte FastAPI Backend...

REM Prüfe ob bereits läuft
curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo       [OK] FastAPI Backend laeuft bereits auf Port 8000
) else (
    echo       [INFO] Starte FastAPI Backend...

    REM Aktiviere Virtual Environment und starte
    if exist "venv\Scripts\python.exe" (
        start "KITradingModel-API" cmd /k "cd /d %~dp0 && venv\Scripts\python.exe run.py"
    ) else (
        start "KITradingModel-API" cmd /k "cd /d %~dp0 && python run.py"
    )

    echo       [INFO] Warte auf Backend-Start...
    timeout /t 10 /nobreak >nul

    curl -s http://localhost:8000/health >nul 2>&1
    if %errorlevel% equ 0 (
        echo       [OK] FastAPI Backend erfolgreich gestartet
    ) else (
        echo       [INFO] Backend startet noch... Bitte warten
    )
)
echo.

REM ============================================================
REM 4. Starte Dashboard Container
REM ============================================================
echo [4/4] Starte Dashboard Container...

REM Prüfe ob Docker läuft
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo       [WARNUNG] Docker ist nicht verfuegbar
    echo       Bitte starten Sie Docker Desktop
    goto :end
)

REM Prüfe ob Container bereits läuft
docker ps --filter "name=ki-trading-dashboard" --format "{{.Names}}" | findstr "ki-trading-dashboard" >nul 2>&1
if %errorlevel% equ 0 (
    echo       [OK] Dashboard Container laeuft bereits auf Port 3001
) else (
    REM Prüfe ob Container existiert aber gestoppt ist
    docker ps -a --filter "name=ki-trading-dashboard" --format "{{.Names}}" | findstr "ki-trading-dashboard" >nul 2>&1
    if %errorlevel% equ 0 (
        echo       [INFO] Starte existierenden Container...
        docker start ki-trading-dashboard >nul 2>&1
    ) else (
        REM Prüfe ob Image existiert
        docker images --format "{{.Repository}}" | findstr "ki-trading-dashboard" >nul 2>&1
        if %errorlevel% neq 0 (
            echo       [INFO] Baue Dashboard Image...
            docker build -t ki-trading-dashboard:latest ./dashboard
        )

        echo       [INFO] Erstelle und starte Container...
        docker run -d --name ki-trading-dashboard -p 3001:80 --restart unless-stopped ki-trading-dashboard:latest >nul 2>&1
    )

    timeout /t 3 /nobreak >nul
    echo       [OK] Dashboard Container gestartet
)
echo.

:end
REM ============================================================
REM Status-Zusammenfassung
REM ============================================================
echo ============================================================
echo                    Status-Zusammenfassung
echo ============================================================
echo.
echo   Service              Port      URL
echo   --------------------------------------------------
echo   Ollama LLM           11434     http://localhost:11434
echo   TimescaleDB          5432      localhost:5432
echo   FastAPI Backend      8000      http://localhost:8000
echo   Dashboard            3001      http://localhost:3001
echo.
echo   API Dokumentation:
echo   - Swagger UI:  http://localhost:8000/docs
echo   - ReDoc:       http://localhost:8000/redoc
echo.
echo ============================================================
echo.

pause
