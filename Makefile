.PHONY: cli
cli:
	@$(eval ARGS := $(filter-out $@,$(MAKECMDGOALS)))
	python3 src/cli.py $(ARGS)