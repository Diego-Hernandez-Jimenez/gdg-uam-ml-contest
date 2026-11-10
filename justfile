# without installing just
# uvx --from rust-just just eval-train

eval-train:
    uv run python harness.py evaluate --model model.py

eval-test:
    uv run python harness.py evaluate --model model.py --source file:data/test_demo.npz

eval-quick:
    uv run python harness.py evaluate --model model.py --set time_budget_s=20

clean:
	@echo "Cleaning cache files..."
	find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".ruff_cache" \) -prune -exec rm -rf {} +
