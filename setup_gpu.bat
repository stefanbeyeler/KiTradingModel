@echo off
echo ============================================
echo KI Trading Model - GPU Setup Script
echo System: Intel i9-13900K + RTX 3070 + 128GB RAM
echo ============================================
echo.

echo Aktiviere Virtual Environment...
call venv\Scripts\activate.bat

echo.
echo [1/3] Installiere PyTorch mit CUDA 12.4 Unterstuetzung...
pip uninstall torch torchvision -y 2>nul
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo.
echo [2/3] Installiere restliche Abhaengigkeiten...
pip install -r requirements.txt

echo.
echo [3/3] Verifiziere Installation...
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA verfuegbar: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Keine\"}'); print(f'CUDA Version: {torch.version.cuda}')"

echo.
echo ============================================
echo Setup abgeschlossen!
echo.
echo Hinweis: FAISS-GPU ist fuer Python 3.13 nicht verfuegbar.
echo          FAISS laeuft auf CPU, Embeddings auf GPU (RTX 3070).
echo.
echo Starte den Service mit: python run.py
echo Pruefe GPU-Status unter: http://localhost:3011/api/system/info
echo ============================================
pause
