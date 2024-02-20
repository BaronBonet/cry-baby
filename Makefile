#!/bin/bash

run:
	@set -a; . ./.env; set +a; PYTHONPATH="$$pwd:$$PYTHONPATH" poetry run python cry_baby/cmd/cli.py
