# GPU image for VideoGen. Needs the NVIDIA Container Toolkit on the host:
#   docker compose up --build        (see docker-compose.yml)
FROM python:3.12-slim

ARG TORCH_CUDA=cu130
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf \
    VIDEOGEN_CACHE=/data/videogen \
    VIDEOGEN_OUTPUT_DIR=/data/outputs \
    VIDEOGEN_HOST=0.0.0.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/${TORCH_CUDA}
COPY pyproject.toml README.md ./
COPY videogen ./videogen
RUN pip install .

VOLUME ["/data"]
EXPOSE 7860
CMD ["videogen", "serve"]
