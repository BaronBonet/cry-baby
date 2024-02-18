run-tf:
	PYTHONPATH="$$pwd:$$PYTHONPATH" poetry run python cry_baby/cmd/cli.py --run-continuously-tf
