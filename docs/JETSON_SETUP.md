# Jetson (aarch64) Setup for KI Trading Model

Diese Anleitung beschreibt, wie das Projekt auf einem NVIDIA Jetson (aarch64) lauffähig gemacht werden kann.

Wichtig: Jetson benutzt eigene PyTorch/JetPack-Binaries — x86 CUDA-Wheels sind nicht kompatibel.

Voraussetzungen
- NVIDIA Jetson Gerät mit JetPack (inkl. CUDA, cuDNN) installiert
- Zugriff auf Jetson SDK Manager oder passende JetPack-Images
- Mindestens 8 GB Speicher empfohlen (mehr für große Modelle)

1) System prüfen

Führe auf dem Jetson aus:

```bash
uname -m
/usr/local/cuda/bin/nvcc --version || cat /usr/local/cuda/version.txt
```

2) Virtualenv & Python

Auf dem Jetson im Projektordner:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```

3) PyTorch für Jetson

Installiere das PyTorch-Wheel, das zu Deiner JetPack-Version passt. Beispiele und Links:

- Jetson Forum und Community Builds: https://forums.developer.nvidia.com/c/jetson/
- NVIDIA L4T / JetPack redistributables: https://developer.download.nvidia.com/compute/redist/jp/

Das bereitgestellte `setup_gpu.sh` fragt nach einem lokalen Wheel-Pfad. Alternativ nutze einen NVIDIA-Container, der PyTorch bereits enthält.

4) Projektabhängigkeiten (ohne torch/faiss)

```bash
# im aktivierten venv
grep -Ev '^\s*#' requirements.txt | sed '/^\s*$/d' | grep -Ev '^\s*(torch|faiss)' > /tmp/reqs_jetson.txt
pip install -r /tmp/reqs_jetson.txt
rm /tmp/reqs_jetson.txt
```

5) FAISS

FAISS-GPU ist auf Jetson oft schwer zu bekommen; `faiss-cpu` ist die einfache Option:

```bash
pip install faiss-cpu || echo "Falls pip-Installation fehlschlägt: faiss manuell bauen oder CPU-Variante suchen."
```

6) Sentence Transformers

```bash
pip install sentence-transformers transformers
```

7) Konfigurationsempfehlungen

Setze in Deiner Umgebung oder `.env` (falls verwendet):

```bash
FAISS_USE_GPU=0
EMBEDDING_DEVICE=cpu
NHITS_USE_GPU=0
# oder EMBEDDING_DEVICE=cuda wenn PyTorch GPU korrekt installiert und ausreichend VRAM vorhanden
```

8) Smoke-Tests

```bash
python3 -c "import platform; print('arch:', platform.machine())"
python3 -c "import torch; print('torch', getattr(torch,'__version__',None), 'cuda', torch.cuda.is_available())"
python3 -c "from sentence_transformers import SentenceTransformer; print('s-t ok')"
python3 -c "import faiss; print('faiss', faiss.__version__ if hasattr(faiss,'__version__') else 'ok')" || true
```

9) Ollama / LLM Hinweis

Große lokale LLMs sind i.d.R. nicht praxistauglich auf Jetson. Verwende stattdessen einen Remote-Host für Ollama oder einen dedizierten Server.

10) Dienst starten

```bash
python3 run.py
```

Weitere Optionen
- Nutze NVIDIA Jetson-optimierte Docker-Images (L4T) mit vorinstalliertem PyTorch.
- Wenn FAISS-GPU unbedingt gebraucht wird, plane einen manuellen Build von FAISS (mit CUDA-Toolkit) auf dem Gerät.

Support
Wenn du möchtest, erstelle ich ein Jetson-spezifisches `docker-compose`-Beispiel oder ein angepasstes `setup_gpu.sh`-Flow, das die Wheel-URLs automatisch abfragt.
Docker (AGX Thor)

Für den AGX Thor empfehle ich die Verwendung eines NVIDIA L4T / JetPack PyTorch-Containers. Das Repo enthält ein Beispiel-Dockerfile und eine Compose-Datei unter `docker/jetson/`.

Kurzer Ablauf (Beispiel):

```bash
# Baue das Image (wähle BASE_IMAGE passend zur JetPack-Version)
docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:latest -t ki-trading-jetson -f docker/jetson/Dockerfile ../../

# Oder mit docker-compose (im Verzeichnis docker/jetson):
docker compose -f docker/jetson/docker-compose.yml up --build -d
```

Hinweis:
- Setze `BASE_IMAGE` auf ein passendes L4T-PyTorch-Image für deine JetPack-Version.
- Die Compose-Datei setzt `EMBEDDING_DEVICE=cuda`, `NHITS_USE_GPU=1` und `FAISS_USE_GPU=1` als Defaults — passe diese Variablen an, falls Faiss-GPU nicht installiert ist.

Wenn du möchtest, passe ich das Compose-File an dein JetPack-Tag (z.B. `r35.2.1`) und prüfe, ob das Image auf dem AGX Thor verfügbar ist.
