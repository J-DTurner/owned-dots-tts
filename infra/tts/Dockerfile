FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    OWNED_TTS_ENGINE=dots.tts \
    OWNED_TTS_VARIANT=soar \
    OWNED_TTS_MODEL_DIR=/models/dots/soar \
    PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-venv \
      ffmpeg \
      git \
      libsndfile1 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY worker.py /app/worker.py
COPY score.py /app/score.py
COPY beam_app.py /app/beam_app.py

# Weights are mounted or baked at /models/dots/{soar,mf}. Missing weights keep /healthz 503.
VOLUME ["/models/dots"]

EXPOSE 8000
CMD ["python", "/app/worker.py"]
