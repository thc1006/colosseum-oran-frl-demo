# ─── Makefile ─────────────────────────────────────────────────────────────
# 全域參數 ────────────────────────────────────────────────────────────────
PYTHON     := .venv/Scripts/python.exe
PIP        := .venv/Scripts/pip.exe
WIN        := $(findstring Windows_NT,$(OS))
ifeq ($(WIN),Windows_NT)
    ACTIVATE = .venv\Scripts\activate
    PYTHON   = .venv\Scripts\python.exe
    PIP      = .venv\Scripts\pip.exe
else
    ACTIVATE = source .venv/bin/activate
    PYTHON   = .venv/bin/python
    PIP      = .venv/bin/pip
endif

# Parquet、模型、Notebook artefacts
RAW_DATA      := src/colosseum_oran_frl_demo/data/raw
PROCESSED_DIR := src/colosseum_oran_frl_demo/data/processed
OUTPUTS_DIR   := outputs

# ─── phony targets ────────────────────────────────────────────────────────
.PHONY: env install lint format test \
        data train clean docs \
        docker-build-dev docker-run-dev docker-build-prod

# 1. 建立虛擬環境並安裝
env:
	@if [ ! -d .venv ]; then \
		python -m venv .venv; \
	fi
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt -r requirements-dev.txt -e .

install: env  ## alias

# 2. Lint（ruff）+ format（black）+ Type Check (mypy) + Security Scan (bandit)
lint:
	@echo "Running Ruff checks..."
	$(PYTHON) -m ruff check src scripts tests
	@echo "Running Black formatting checks..."
	$(PYTHON) -m black --check src scripts tests
	@echo "Running Mypy type checks..."
	$(PYTHON) -m mypy src scripts
	@echo "Running Bandit security scan..."
	$(PYTHON) -m bandit -r src scripts

format:
	$(PYTHON) -m black src scripts tests

# 3. 執行單元測試 (含覆蓋率)
test:
	@echo "Running unit tests with Pytest and Coverage..."
	$(PYTHON) -m pytest --cov=src --cov-report=xml tests/

# 4. 轉換原始 CSV → Parquet
data:
	$(PYTHON) scripts/make_dataset.py --raw $(RAW_DATA) --out $(PROCESSED_DIR)

# 5. 執行離線 FRL 訓練
train:
	$(PYTHON) scripts/train.py --parquet $(PROCESSED_DIR) \
	                           --rounds 5 --clients 1,2,3 \
	                           --out $(OUTPUTS_DIR)

# 6. 文件（MkDocs）– 本地瀏覽
docs:
	$(PYTHON) -m mkdocs serve

# 7. 清理中間檔
clean:
	@echo "Cleaning processed data, outputs & __pycache__ ..."
	@rm -rf $(PROCESSED_DIR)/*
	@rm -rf $(OUTPUTS_DIR)/*
	@find . -name "__pycache__" -type d -exec rm -rf {} +
	@find . -name "*.pyc" -delete

# 8. Docker Targets
docker-build-dev:
	@echo "Building Docker development image..."
	docker build -t colosseum-oran-frl-demo-dev:latest --target development .

docker-run-dev:
	@echo "Running Docker development container..."
	docker run -it --rm -v "$(shell pwd)":/app colosseum-oran-frl-demo-dev:latest bash

docker-build-prod:
	@echo "Building Docker production image..."
	docker build -t colosseum-oran-frl-demo:latest --target production .

# 預設
default: test