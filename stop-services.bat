@echo off
REM ============================================================
REM KITradingModel - Stop All Services
REM ============================================================
REM Dienste:
REM   1. Dashboard Container (Port 3001)
REM   2. FastAPI Backend (Port 3011)
REM   3. Ollama LLM Service (Port 11434) - optional
REM   4. TimescaleDB (Port 5432) - wird nicht gestoppt
REM ============================================================

setlocal enabledelayedexpansion

echo ============================================================
echo        KITradingModel - Dienste stoppen
echo ============================================================
echo.

cd /d "%~dp0"

REM ============================================================
REM 1. Stoppe Dashboard Container
REM ============================================================
echo [1/3] Stoppe Dashboard Container...

docker info >nul 2>&1
if %errorlevel% equ 0 (
    docker ps --filter "name=ki-trading-dashboard" --format "{{.Names}}" | findstr "ki-trading-dashboard" >nul 2>&1
    if %errorlevel% equ 0 (
        docker stop ki-trading-dashboard >nul 2>&1
        echo       [OK] Dashboard Container gestoppt
    ) else (
        echo       [INFO] Dashboard Container laeuft nicht
    )
) else (
    echo       [INFO] Docker nicht verfuegbar
)
echo.

REM ============================================================
REM 2. Stoppe FastAPI Backend
REM ============================================================
echo [2/3] Stoppe FastAPI Backend...

REM Finde Python-Prozesse die run.py ausführen
set "found=0"

REM Methode 1: Suche nach Prozessen auf Port 3011
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":3011" ^| findstr "LISTENING"') do (
    set "pid=%%a"
    if not "!pid!"=="" (
        echo       [INFO] Beende Prozess mit PID: !pid!
        taskkill /F /PID !pid! >nul 2>&1
        set "found=1"
    )
)

if "!found!"=="0" (
    echo       [INFO] FastAPI Backend laeuft nicht
) else (
    echo       [OK] FastAPI Backend gestoppt
)
echo.

REM ============================================================
REM 3. Stoppe Ollama (optional)
REM ============================================================
echo [3/3] Ollama Status...

curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% equ 0 (
    echo       [INFO] Ollama laeuft noch auf Port 11434
    echo.
    set /p "stop_ollama=       Ollama auch stoppen? (j/n): "
    if /i "!stop_ollama!"=="j" (
        REM Finde Ollama Prozess
        for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":11434" ^| findstr "LISTENING"') do (
            set "pid=%%a"
            if not "!pid!"=="" (
                taskkill /F /PID !pid! >nul 2>&1
            )
        )
        REM Auch ollama.exe direkt beenden
        taskkill /F /IM ollama.exe >nul 2>&1
        taskkill /F /IM ollama_llama_server.exe >nul 2>&1
        echo       [OK] Ollama gestoppt
    ) else (
        echo       [INFO] Ollama bleibt aktiv
    )
) else (
    echo       [INFO] Ollama laeuft nicht
)
echo.

REM ============================================================
REM Hinweis zu TimescaleDB
REM ============================================================
echo ============================================================
echo                         Hinweis
echo ============================================================
echo.
echo   TimescaleDB wird NICHT automatisch gestoppt.
echo   Falls gewuenscht, stoppen Sie TimescaleDB manuell.
echo.
echo   Alle anwendungsspezifischen Dienste wurden gestoppt.
echo.
echo ============================================================
echo.

pause
