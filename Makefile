define LOAD_ENV
	@bash -c "source .env; set +a";
endef

run-tf:
	PYTHONPATH="$$pwd:$$PYTHONPATH" poetry run python cry_baby/cmd/cli.py --run-continuously-tf

run-tf-lite:
	@$(LOAD_ENV) PYTHONPATH="$$pwd:$$PYTHONPATH" poetry run python cry_baby/cmd/cli.py --run-continuously-tf-lite
