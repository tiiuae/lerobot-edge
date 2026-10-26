# pi0.5 model server: run it from a local venv or in Docker.
# Run from `deployment/`. Override any variable on the command line:
#   make serve CKPT=/path/to/pretrained_model PORT=9000
#   make docker-up-d CKPT=/models/<dir>/pretrained_model GPU=1
# Server flags are listed in DEPLOYMENT.md; pass extra ones with EXTRA="--rtc".

LEROBOT_DIR  ?= $(abspath $(CURDIR)/..)
VENV         ?= $(LEROBOT_DIR)/.venv
PY           ?= $(VENV)/bin/python
UV           ?= uv

CKPT         ?= /home/ibrahim/storage/Stable_VLA/demo-day/pi05/pretrained_model
DEVICE       ?= cuda
PORT         ?= 8800
IDLE_TIMEOUT ?= -1
DEBUG_DIR    ?=
EXTRA        ?=

# Docker only. CKPT must then be a path inside the container, under /models.
DOCKER_IMAGE ?= lerobot-pi05-server
DOCKER_TAG   ?= latest
MODELS_DIR   ?= /home/ibrahim/storage
HF_CACHE     ?= $(HOME)/.cache/huggingface
GPU          ?= 0

SERVER_FLAGS = --ckpt_path $(CKPT) --device $(DEVICE) --port $(PORT) --idle_timeout $(IDLE_TIMEOUT)
ifneq ($(strip $(DEBUG_DIR)),)
SERVER_FLAGS += --debug_dir $(DEBUG_DIR)
endif
SERVER_FLAGS += $(EXTRA)

COMPOSE = DOCKER_IMAGE=$(DOCKER_IMAGE) DOCKER_TAG=$(DOCKER_TAG) CKPT=$(CKPT) PORT=$(PORT) \
	IDLE_TIMEOUT=$(IDLE_TIMEOUT) EXTRA="$(EXTRA)" MODELS_DIR=$(MODELS_DIR) HF_CACHE=$(HF_CACHE) \
	GPU=$(GPU) docker compose -f $(CURDIR)/docker-compose.yml

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "pi0.5 model server"
	@echo "  install          create $(VENV) (Python 3.12) with lerobot[pi], websockets, msgpack"
	@echo "  cache-tokenizer  download the PaliGemma tokenizer once (gated: needs a Hub login)"
	@echo "  serve            run the server from the venv"
	@echo "  serve-debug      the same, with per-call images and actions under /tmp/lerobot_debug"
	@echo "  show-vars        print the variables and the server flags"
	@echo "  docker-build     build $(DOCKER_IMAGE):$(DOCKER_TAG)"
	@echo "  docker-up-d      start the container (docker-up: in the foreground)"
	@echo "  docker-down      stop it"
	@echo "  docker-logs      follow its log"
	@echo "  docker-shell     a shell in the running container"
	@echo "  clean-venv       remove $(VENV)"

.PHONY: install
install:
	cd $(LEROBOT_DIR) && $(UV) venv --python 3.12 .venv
	cd $(LEROBOT_DIR) && . .venv/bin/activate && $(UV) pip install -e ".[pi]"
	cd $(LEROBOT_DIR) && . .venv/bin/activate && $(UV) pip install "websockets==17.0.1" "msgpack==1.2.1"

.PHONY: cache-tokenizer
cache-tokenizer:
	$(PY) -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224')"

.PHONY: serve
serve:
	cd $(LEROBOT_DIR) && $(PY) -m deployment.model_server.server_policy $(SERVER_FLAGS)

.PHONY: serve-debug
serve-debug:
	$(MAKE) serve DEBUG_DIR=/tmp/lerobot_debug

.PHONY: show-vars
show-vars:
	@echo "CKPT  = $(CKPT)"
	@echo "PORT  = $(PORT)"
	@echo "GPU   = $(GPU) (docker only)"
	@echo "FLAGS = $(SERVER_FLAGS)"

.PHONY: docker-build
docker-build:
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) -f $(CURDIR)/Dockerfile $(LEROBOT_DIR)

.PHONY: docker-up
docker-up:
	$(COMPOSE) up

.PHONY: docker-up-d
docker-up-d:
	$(COMPOSE) up -d

.PHONY: docker-down
docker-down:
	$(COMPOSE) down

.PHONY: docker-logs
docker-logs:
	$(COMPOSE) logs -f

.PHONY: docker-shell
docker-shell:
	$(COMPOSE) exec lerobot_server /bin/bash

.PHONY: clean-venv
clean-venv:
	rm -rf $(VENV)
