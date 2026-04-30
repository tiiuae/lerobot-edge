# Dockerfile for the LeRobot policy server.
# Build context = the `lerobot/` repo root (parent of `deployment/`).
#
# Build:
#   docker build -t lerobot_server -f deployment/Dockerfile .
# Run:
#   docker run --rm -it --gpus=all -p 8800:8800 \
#     -v /path/to/checkpoints:/models -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#     -e SERVER_ARGS="--ckpt_path /models/<ckpt> --camera_keys cam_high:primary,cam_left_wrist:secondary,cam_right_wrist:wrist --use_bf16 --port 8800" \
#     lerobot_server

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/bin:$PATH \
    PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev \
        git git-lfs build-essential ca-certificates \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create venv
RUN uv venv --python 3.10 ${UV_PROJECT_ENVIRONMENT}

# Install lerobot + pi extra (transformers fork) + websocket deps
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
COPY README.md /app/README.md
RUN UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=10 GIT_LFS_SKIP_SMUDGE=1 \
    uv pip install -e ".[pi]" && \
    uv pip install websockets msgpack

# Copy deployment server code
COPY deployment /app/deployment

EXPOSE 8800

# SERVER_ARGS overridable at run time, e.g.:
#   -e SERVER_ARGS="--ckpt_path /models/foo --port 8800 --use_bf16"
ENV SERVER_ARGS="--ckpt_path /models/checkpoint --camera_keys cam_high:primary,cam_left_wrist:secondary,cam_right_wrist:wrist --actions_per_chunk 25 --use_bf16 --device cuda:0 --port 8800 --idle_timeout -1"

CMD ["/bin/bash", "-c", "python -m deployment.model_server.server_policy $SERVER_ARGS"]
