.PHONY: help setup setup-server setup-client server client dev stop healthz \
        torgo-load torgo-smoke clean clean-data

ROOT := $(shell pwd)
SERVER := $(ROOT)/server
CLIENT := $(ROOT)/client
VENV := $(SERVER)/.venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
UVICORN := $(VENV)/bin/uvicorn
PORT_SERVER := 8000
PORT_CLIENT := 3000

help:
	@echo "Phonemer commands:"
	@echo "  make setup          install everything (brew deps + venv + npm)"
	@echo "  make dev            run backend + frontend together"
	@echo "  make server         run backend only (uvicorn, --reload)"
	@echo "  make client         run frontend only (next dev)"
	@echo "  make stop           kill anything bound to ports 8000 and 3000"
	@echo "  make healthz        curl the backend health endpoint"
	@echo "  make torgo-load     ingest server/data/torgo/ into manifest"
	@echo "  make torgo-smoke    stream 5 random TORGO clips through /ws/audio"
	@echo "  make clean          remove build artifacts (keeps data/)"
	@echo "  make clean-data     remove SQLite + saved audio (DESTRUCTIVE)"

# ===================================================================== setup

setup: setup-server setup-client
	@echo "✓ Setup complete. Run 'make dev'."

setup-server:
	@echo "→ checking system dependencies (espeak-ng, libsndfile)…"
	@command -v espeak-ng >/dev/null 2>&1 || \
		(echo "  installing espeak-ng + libsndfile via Homebrew" && brew install espeak-ng libsndfile)
	@test -d $(VENV) || (echo "→ creating venv" && python3 -m venv $(VENV))
	@echo "→ installing Python requirements"
	$(PIP) install -U pip
	$(PIP) install -r $(SERVER)/requirements.txt

setup-client:
	@echo "→ installing npm packages"
	cd $(CLIENT) && npm install

# ===================================================================== run

server:
	@test -x $(UVICORN) || (echo "✗ venv missing — run 'make setup' first" && exit 1)
	cd $(SERVER) && $(UVICORN) app.main:app --host 0.0.0.0 --port $(PORT_SERVER) --reload

client:
	cd $(CLIENT) && npm run dev

dev:
	@echo "→ starting backend on :$(PORT_SERVER) and frontend on :$(PORT_CLIENT)"
	@./scripts/dev.sh

stop:
	@./scripts/stop.sh

healthz:
	@curl -fsS http://localhost:$(PORT_SERVER)/healthz | python3 -m json.tool || \
		(echo "✗ backend not responding on :$(PORT_SERVER)" && exit 1)

# ==================================================================== torgo

torgo-load:
	@test -x $(PY) || (echo "✗ venv missing — run 'make setup' first" && exit 1)
	cd $(SERVER) && $(PY) -m scripts.torgo_loader --root data/torgo

torgo-smoke:
	@test -x $(PY) || (echo "✗ venv missing — run 'make setup' first" && exit 1)
	cd $(SERVER) && $(PY) -m scripts.torgo_smoke --n 5

# =================================================================== clean

clean:
	rm -rf $(CLIENT)/.next $(CLIENT)/out
	find $(SERVER) -type d -name __pycache__ -prune -exec rm -rf {} +

clean-data:
	@read -p "This will delete server/data/phonemer.db and saved audio. Continue? [y/N] " ans; \
	if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then \
		rm -f $(SERVER)/data/phonemer.db $(SERVER)/data/phonemer.db-journal; \
		rm -rf $(SERVER)/data/audio; \
		echo "✓ cleared"; \
	else echo "aborted"; fi
