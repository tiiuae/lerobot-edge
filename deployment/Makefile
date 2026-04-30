# LeRobot policy server — Makefile
# Run from `lerobot/deployment/`. Override any variable on the command line:
#   make serve CKPT=/path/to/ckpt PORT=9000

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LEROBOT_DIR    ?= $(abspath $(CURDIR)/..)
VENV           ?= $(LEROBOT_DIR)/.venv
PY             ?= $(VENV)/bin/python
UV             ?= uv

# ---------------------------------------------------------------------------
# Defaults — override per invocation
# ---------------------------------------------------------------------------
CKPT           ?= /home/ibrahim/storage/VLA_MODELS/special_tests/Pi0/last/pretrained_model
POLICY_TYPE    ?=
CAM_KEYS       ?= cam_high:primary,cam_left_wrist:secondary,cam_right_wrist:wrist
ACTIONS        ?= 25
DEVICE         ?= cuda:0
PORT           ?= 8800
IDLE_TIMEOUT   ?= -1
DEBUG_DIR      ?=
EXTRA          ?=

# Compose toggles
DOCKER_IMAGE   ?= lerobot_server
DOCKER_TAG     ?= latest
MODELS_DIR     ?= /home/ibrahim/storage/VLA_MODELS
HF_CACHE       ?= $(HOME)/.cache/huggingface

# ---------------------------------------------------------------------------
# Common flags
# ---------------------------------------------------------------------------
SERVER_FLAGS = --ckpt_path $(CKPT) \
               --camera_keys $(CAM_KEYS) \
               --actions_per_chunk $(ACTIONS) \
               --device $(DEVICE) \
               --port $(PORT) \
               --idle_timeout $(IDLE_TIMEOUT) \
               --use_bf16
ifneq ($(strip $(POLICY_TYPE)),)
SERVER_FLAGS += --policy_type $(POLICY_TYPE)
endif
ifneq ($(strip $(DEBUG_DIR)),)
SERVER_FLAGS += --debug_dir $(DEBUG_DIR)
endif
SERVER_FLAGS += $(EXTRA)

# ===========================================================================
# Help
# ===========================================================================
# ANSI colours (no-ops if terminal does not support).
C_BOLD  := \033[1m
C_DIM   := \033[2m
C_GREEN := \033[32m
C_CYAN  := \033[36m
C_YEL   := \033[33m
C_RST   := \033[0m

.DEFAULT_GOAL := help

.PHONY: help
help:
	@printf "$(C_BOLD)LeRobot policy server — Makefile$(C_RST)\n"
	@printf "Run targets from this directory ($(C_DIM)$(CURDIR)$(C_RST)).\n"
	@printf "\n"
	@printf "$(C_BOLD)Usage:$(C_RST)\n"
	@printf "  make $(C_GREEN)<target>$(C_RST) [VAR=value ...]\n"
	@printf "\n"
	@printf "$(C_BOLD)Setup:$(C_RST)\n"
	@printf "  $(C_GREEN)install$(C_RST)            Create $(C_DIM)$(VENV)$(C_RST), install lerobot[pi] + websockets + msgpack\n"
	@printf "  $(C_GREEN)cache-tokenizer$(C_RST)    Pre-download paligemma tokenizer (offline use later)\n"
	@printf "\n"
	@printf "$(C_BOLD)Run (local venv):$(C_RST)\n"
	@printf "  $(C_GREEN)serve$(C_RST)              Run server with current vars\n"
	@printf "  $(C_GREEN)serve-pi05$(C_RST)         Run pi05 with starVLA-style cam mapping\n"
	@printf "  $(C_GREEN)serve-debug$(C_RST)        Run with debug artefacts under /tmp/lerobot_debug\n"
	@printf "  $(C_GREEN)show-vars$(C_RST)          Print resolved variables and final flags\n"
	@printf "\n"
	@printf "$(C_BOLD)Docker:$(C_RST)\n"
	@printf "  $(C_GREEN)docker-build$(C_RST)       Build $(C_DIM)$(DOCKER_IMAGE):$(DOCKER_TAG)$(C_RST) from $(C_DIM)$(LEROBOT_DIR)$(C_RST)\n"
	@printf "  $(C_GREEN)docker-up$(C_RST)          Compose up (foreground, --build)\n"
	@printf "  $(C_GREEN)docker-up-d$(C_RST)        Compose up -d --build (detached)\n"
	@printf "  $(C_GREEN)docker-down$(C_RST)        Compose down\n"
	@printf "  $(C_GREEN)docker-logs$(C_RST)        Tail compose logs\n"
	@printf "  $(C_GREEN)docker-shell$(C_RST)       Interactive shell inside running container\n"
	@printf "\n"
	@printf "$(C_BOLD)Cleanup:$(C_RST)\n"
	@printf "  $(C_GREEN)clean-venv$(C_RST)         Remove $(C_DIM)$(VENV)$(C_RST)\n"
	@printf "\n"
	@printf "$(C_BOLD)Variables (override on command line):$(C_RST)\n"
	@printf "  $(C_CYAN)CKPT$(C_RST)         = $(C_YEL)$(CKPT)$(C_RST)\n"
	@printf "                  Local dir or HF Hub repo id of pretrained policy.\n"
	@printf "  $(C_CYAN)POLICY_TYPE$(C_RST)  = $(C_YEL)$(if $(POLICY_TYPE),$(POLICY_TYPE),<auto>)$(C_RST)\n"
	@printf "                  act|diffusion|pi0|pi05|smolvla|tdmpc|vqbet|groot|xvla|wall_x. Auto from config.json.\n"
	@printf "  $(C_CYAN)CAM_KEYS$(C_RST)     = $(C_YEL)$(CAM_KEYS)$(C_RST)\n"
	@printf "                  CSV of <client_key>:<lerobot_key> or bare <key>. Lerobot key joins observation.images.<key>.\n"
	@printf "  $(C_CYAN)ACTIONS$(C_RST)      = $(C_YEL)$(ACTIONS)$(C_RST)\n"
	@printf "                  --actions_per_chunk slice length.\n"
	@printf "  $(C_CYAN)DEVICE$(C_RST)       = $(C_YEL)$(DEVICE)$(C_RST)\n"
	@printf "                  cuda, cuda:0, cpu, ...\n"
	@printf "  $(C_CYAN)PORT$(C_RST)         = $(C_YEL)$(PORT)$(C_RST)\n"
	@printf "  $(C_CYAN)IDLE_TIMEOUT$(C_RST) = $(C_YEL)$(IDLE_TIMEOUT)$(C_RST)\n"
	@printf "                  -1 means never auto-shutdown.\n"
	@printf "  $(C_CYAN)DEBUG_DIR$(C_RST)    = $(C_YEL)$(if $(DEBUG_DIR),$(DEBUG_DIR),<off>)$(C_RST)\n"
	@printf "                  Set to enable per-call debug dump.\n"
	@printf "  $(C_CYAN)EXTRA$(C_RST)        = $(C_YEL)$(EXTRA)$(C_RST)\n"
	@printf "                  Free-form flags appended to server command.\n"
	@printf "\n"
	@printf "$(C_BOLD)Docker-only variables:$(C_RST)\n"
	@printf "  $(C_CYAN)DOCKER_IMAGE$(C_RST) = $(C_YEL)$(DOCKER_IMAGE)$(C_RST)\n"
	@printf "  $(C_CYAN)DOCKER_TAG$(C_RST)   = $(C_YEL)$(DOCKER_TAG)$(C_RST)\n"
	@printf "  $(C_CYAN)MODELS_DIR$(C_RST)   = $(C_YEL)$(MODELS_DIR)$(C_RST)\n"
	@printf "                  Host dir mounted at /models inside container. CKPT must point INSIDE this mount.\n"
	@printf "  $(C_CYAN)HF_CACHE$(C_RST)     = $(C_YEL)$(HF_CACHE)$(C_RST)\n"
	@printf "                  Host dir mounted at /root/.cache/huggingface so tokenizer/weights persist.\n"
	@printf "\n"
	@printf "$(C_BOLD)Examples:$(C_RST)\n"
	@printf "  $(C_DIM)# First-time setup$(C_RST)\n"
	@printf "  make install\n"
	@printf "  make cache-tokenizer\n"
	@printf "\n"
	@printf "  $(C_DIM)# Run pi05 server with starVLA-style client$(C_RST)\n"
	@printf "  make serve-pi05\n"
	@printf "\n"
	@printf "  $(C_DIM)# Run with custom checkpoint, port and identity cam keys$(C_RST)\n"
	@printf "  make serve POLICY_TYPE=pi05 CKPT=/path/to/ckpt PORT=9000 \\\\\n"
	@printf "             CAM_KEYS=primary,secondary,wrist\n"
	@printf "\n"
	@printf "  $(C_DIM)# Capture debug artefacts$(C_RST)\n"
	@printf "  make serve-debug\n"
	@printf "\n"
	@printf "  $(C_DIM)# Docker — build then run detached$(C_RST)\n"
	@printf "  make docker-build\n"
	@printf "  make docker-up-d POLICY_TYPE=pi05 CKPT=/models/special_tests/Pi0/last/pretrained_model\n"
	@printf "  make docker-logs\n"
	@printf "  make docker-down\n"

# ===========================================================================
# Local install
# ===========================================================================
.PHONY: install
install:
	cd $(LEROBOT_DIR) && $(UV) venv --python 3.10 .venv
	cd $(LEROBOT_DIR) && . .venv/bin/activate && $(UV) pip install -e ".[pi]"
	cd $(LEROBOT_DIR) && . .venv/bin/activate && $(UV) pip install websockets msgpack

.PHONY: cache-tokenizer
cache-tokenizer:
	$(PY) -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224')"

# ===========================================================================
# Run
# ===========================================================================
.PHONY: serve
serve:
	cd $(LEROBOT_DIR) && $(PY) -m deployment.model_server.server_policy $(SERVER_FLAGS)

.PHONY: serve-pi05
serve-pi05:
	$(MAKE) serve POLICY_TYPE=pi05

.PHONY: serve-debug
serve-debug:
	$(MAKE) serve DEBUG_DIR=/tmp/lerobot_debug

.PHONY: show-vars
show-vars:
	@echo "LEROBOT_DIR  = $(LEROBOT_DIR)"
	@echo "VENV         = $(VENV)"
	@echo "CKPT         = $(CKPT)"
	@echo "POLICY_TYPE  = $(POLICY_TYPE)"
	@echo "CAM_KEYS     = $(CAM_KEYS)"
	@echo "ACTIONS      = $(ACTIONS)"
	@echo "DEVICE       = $(DEVICE)"
	@echo "PORT         = $(PORT)"
	@echo "IDLE_TIMEOUT = $(IDLE_TIMEOUT)"
	@echo "DEBUG_DIR    = $(DEBUG_DIR)"
	@echo "EXTRA        = $(EXTRA)"
	@echo "FLAGS        = $(SERVER_FLAGS)"

# ===========================================================================
# Docker
# ===========================================================================
COMPOSE = docker compose -f $(CURDIR)/docker-compose.yml

.PHONY: docker-build
docker-build:
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) -f $(CURDIR)/Dockerfile $(LEROBOT_DIR)

.PHONY: docker-up
docker-up:
	CKPT=$(CKPT) POLICY_TYPE=$(POLICY_TYPE) CAM_KEYS=$(CAM_KEYS) \
	ACTIONS=$(ACTIONS) DEVICE=$(DEVICE) PORT=$(PORT) IDLE_TIMEOUT=$(IDLE_TIMEOUT) \
	DEBUG_DIR=$(DEBUG_DIR) MODELS_DIR=$(MODELS_DIR) HF_CACHE=$(HF_CACHE) \
	$(COMPOSE) up --build

.PHONY: docker-up-d
docker-up-d:
	CKPT=$(CKPT) POLICY_TYPE=$(POLICY_TYPE) CAM_KEYS=$(CAM_KEYS) \
	ACTIONS=$(ACTIONS) DEVICE=$(DEVICE) PORT=$(PORT) IDLE_TIMEOUT=$(IDLE_TIMEOUT) \
	DEBUG_DIR=$(DEBUG_DIR) MODELS_DIR=$(MODELS_DIR) HF_CACHE=$(HF_CACHE) \
	$(COMPOSE) up -d --build

.PHONY: docker-down
docker-down:
	$(COMPOSE) down

.PHONY: docker-logs
docker-logs:
	$(COMPOSE) logs -f

.PHONY: docker-shell
docker-shell:
	$(COMPOSE) exec lerobot_server /bin/bash

# ===========================================================================
# Clean
# ===========================================================================
.PHONY: clean-venv
clean-venv:
	rm -rf $(VENV)
