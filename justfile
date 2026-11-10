# without installing just
# uvx --from rust-just just test

# with just installed
# just test

root := justfile_directory()

eval-train:
    uv run --project {{root}} python harness.py evaluate --model model.py

eval-test:
    uv run --project {{root}} python harness.py evaluate --model model.py --source file:data/test_demo.npz

eval-quick:
    uv run --project {{root}} python harness.py evaluate --model model.py --set time_budget_s=20

clean:
	@echo "Cleaning cache files..."
	find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".ruff_cache" \) -prune -exec rm -rf {} +

model-size model="latest":
    @echo "Getting {{model}} size:"
    @du -h "artifacts/{{model}}.joblib"

test target_test="":
    #!/usr/bin/env bash
    set -e
    trap 'rm -rf .pytest_cache' EXIT
    if [ -n "{{target_test}}" ]; then
        uv run --project {{root}} pytest -vv "{{target_test}}"
    else
        uv run --project {{root}} pytest -vv
    fi