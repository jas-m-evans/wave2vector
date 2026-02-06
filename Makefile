PYTHON ?= python
VENV ?=

.PHONY: install run test

install:
	@if [ -n "$(VENV)" ]; then \
		if [ ! -d "$(VENV)" ]; then $(PYTHON) -m venv $(VENV); fi; \
		$(VENV)/bin/python -m pip install -r requirements.txt; \
	else \
		$(PYTHON) -m pip install -r requirements.txt; \
	fi

run:
	uvicorn app.main:app --reload

test:
	pytest
