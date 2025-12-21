#!/usr/bin/env bash
set -euo pipefail

echo "KI Trading Model - Jetson GPU setup helper"

ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
  echo "Warnung: Dieses Skript ist für NVIDIA Jetson (aarch64) gedacht. Aktuelle Architektur: $ARCH"
  echo "Fahre fort auf eigenes Risiko..."
fi

echo "1) Prüfe JetPack / CUDA (manuell prüfen, SDK Manager empfohlen)"
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version || true
else
  echo "nvcc nicht gefunden. Stelle sicher, dass JetPack / CUDA installiert ist."
fi

echo "2) Python-Umgebung vorbereiten"
PY=python3
if ! command -v $PY >/dev/null 2>&1; then
  echo "python3 nicht gefunden. Bitte installieren.";
  exit 1
fi

echo "Erzeuge Virtualenv in ./venv (falls nicht vorhanden)"
if [ ! -d venv ]; then
  $PY -m venv venv
fi

echo "Aktiviere venv..."
# shellcheck disable=SC1091
source venv/bin/activate

echo "Upgrade pip, wheel, setuptools"
pip install --upgrade pip setuptools wheel

echo "3) PyTorch: Jetson-spezifische Wheel installieren"
echo "-> WICHTIG: Verwende die für Deine JetPack-Version passenden PyTorch-Wheels/Container."
echo "Hinweis: Offizielle x86 CUDA-Wheels (download.pytorch.org/whl/cu*) sind NICHT kompatibel."
echo "Empfohlene Quellen und Anleitung:"
echo "  - NVIDIA Jetson Forum: https://forums.developer.nvidia.com/c/jetson/"
echo "  - Jetson PyTorch releases (z.B. https://developer.download.nvidia.com/compute/redist/jp/)"

read -p "Hast du bereits ein Jetson-kompatibles PyTorch-Wheel heruntergeladen? (y/N) " yn
if [ "${yn:-N}" = "y" ] || [ "${yn:-y}" = "Y" ]; then
  read -e -p "Pfad zum Wheel (lokal): " wheelpath
  pip install "$wheelpath"
else
  echo "Überspringe automatische PyTorch-Installation. Bitte installiere PyTorch manuell entsprechend JetPack-Version."
fi

echo "4) Abhängigkeiten (ohne torch/faiss) installieren"
# Erzeuge temporäre requirements ohne torch/faiss-Einträge
TMPREQ=$(mktemp)
grep -Ev '^\s*#' requirements.txt | sed '/^\s*$/d' | grep -Ev '^\s*(torch|faiss)' > "$TMPREQ"
echo "Installiere:"
cat "$TMPREQ"
pip install -r "$TMPREQ"
rm -f "$TMPREQ"

echo "5) FAISS: faiss-cpu ist auf Jetson meist praktikabler. Versuche Installation via pip."
if pip install faiss-cpu; then
  echo "faiss-cpu installiert"
else
  echo "Konnte faiss-cpu nicht per pip installieren. Du kannst faiss manuell bauen oder faiss-cpu aus anderen Quellen installieren."
fi

echo "6) sentence-transformers + transformer-abhängigkeiten installieren"
pip install sentence-transformers --no-deps || true
pip install transformers || true

echo "7) Umgebungsvariablen für niedrige Ressourcen setzen (optional)"
echo "export FAISS_USE_GPU=0  # falls faiss-gpu nicht verfügbar"
echo "export EMBEDDING_DEVICE=cpu  # setze 'cuda' nur wenn PyTorch GPU korrekt installiert ist"
echo "export NHITS_USE_GPU=0"

echo "8) Smoke-Tests (einzeln laufen lassen)"
echo "python3 -c \"import platform; print('arch:', platform.machine())\""
echo "python3 -c \"import torch; print('torch', getattr(torch,'__version__',None), 'cuda', torch.cuda.is_available())\""
echo "python3 -c \"from sentence_transformers import SentenceTransformer; print('s-t ok')\""
echo "python3 -c \"import faiss; print('faiss', faiss.__version__ if hasattr(faiss,'__version__') else 'ok')\" || true

echo "Fertig. Bitte folge dem README in docs/JETSON_SETUP.md für detaillierte Schritte."
