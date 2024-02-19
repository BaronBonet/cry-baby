define LOAD_ENV
	@bash -c "source .env; set +a";
endef

run:
	PYTHONPATH="$$pwd:$$PYTHONPATH" poetry run python cry_baby/cmd/cli.py
