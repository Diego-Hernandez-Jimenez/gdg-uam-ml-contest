# without installing just
# uvx --from rust-just just test

# with just installed
# just test

root := justfile_directory()

train data_file model_config:
    uv run --project {{root}} \
    python -m src.train --data data/{{data_file}}.npz --config configs/{{model_config}}.yaml

eval data_file="test_demo" model_path="artifacts/latest":
    @echo "Evaluating model {{model_path}} on data {{data_file}}..."
    @export MODEL_WEIGHTS="{{root}}/{{model_path}}.joblib"
    uv run --project {{root}} \
        python harness.py evaluate --model model.py --source file:data/{{data_file}}.npz

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
        uv run --project {{root}} pytest -vv "test/test_{{target_test}}.py"
    else
        uv run --project {{root}} pytest -vv
    fi

deps:
    cd {{root}} &&uv export --format requirements-txt --output-file requirements.txt
