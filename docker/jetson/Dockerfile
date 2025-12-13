ARG BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:latest
FROM ${BASE_IMAGE}

LABEL maintainer="KI Trading Model"

WORKDIR /app

# Copy project files
COPY . /app

# Ensure python venv available
RUN apt-get update && apt-get install -y python3-venv && rm -rf /var/lib/apt/lists/*

# Create and activate venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade packaging tools
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies except torch/faiss (use base image's PyTorch)
RUN python3 - <<'PY'
import re
req_lines = []
with open('requirements.txt') as f:
    for l in f:
        s = l.strip()
        if not s or s.startswith('#'):
            continue
        if s.startswith('torch') or s.startswith('faiss'):
            continue
        req_lines.append(s)
open('/tmp/reqs_jetson.txt','w').write('\n'.join(req_lines))
PY
RUN pip install -r /tmp/reqs_jetson.txt || true

# Expose service port
EXPOSE 3011

CMD ["python3", "run.py"]
