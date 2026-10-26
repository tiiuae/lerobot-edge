# The pi0.5 model server (deployment/model_server/server_policy.py) as an image.
# Build context = the repository root (parent of `deployment/`):
#
#   docker build -t lerobot-pi05-server -f deployment/Dockerfile .
#
# Run (the checkpoint and the Hugging Face cache holding the PaliGemma tokenizer are mounted):
#
#   docker run --rm --gpus '"device=0"' --network host \
#     -v /path/to/pi05:/models/pi05:ro \
#     -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#     lerobot-pi05-server --ckpt_path /models/pi05/pretrained_model --port 8800
#
# Every argument after the image name goes to server_policy.py (see DEPLOYMENT.md).

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/lerobot/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=600 \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates libglib2.0-0 libgl1 ffmpeg \
    && curl -LsSf https://astral.sh/uv/0.8.17/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /lerobot

# Dependencies first, from the lockfile, so a source change does not reinstall torch.
COPY setup.py pyproject.toml uv.lock README.md MANIFEST.in ./
RUN uv venv && uv sync --locked --extra pi --no-install-project --no-cache

COPY src ./src
RUN uv sync --locked --extra pi --no-cache \
    && uv pip install --no-cache "websockets==17.0.1" "msgpack==1.2.1"

COPY deployment ./deployment

EXPOSE 8800
ENTRYPOINT ["python", "-m", "deployment.model_server.server_policy"]
CMD ["--help"]
