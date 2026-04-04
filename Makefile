.PHONY: setup rust-build ingest index eval viz all clean

# Vars
PYTHON = python
SCRIPTS = scripts

setup:
	pip install -e .
	go mod tidy

rust-build:
	cd src/ingestion/semantic_chunker_rust && maturin develop

ingest:
	$(PYTHON) src/ingestion/document_loader.py
	$(PYTHON) src/ingestion/semantic_chunker.py

index:
	$(PYTHON) src/retrieval/qdrant_store.py

eval:
	$(PYTHON) $(SCRIPTS)/run_eval.py

viz:
	$(PYTHON) $(SCRIPTS)/visualize.py

# THE MASTER COMMAND
all: rust-build ingest index eval viz

# Start the dashboard 
dashboard:
	go run cmd/server/main.go