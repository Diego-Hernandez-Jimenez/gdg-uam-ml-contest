eval-train:
    uv run python harness.py evaluate --model model.py

eval-test:
    uv run python harness.py evaluate --model model.py --source file:data/test_demo.npz

eval-quick:
    uv run python harness.py evaluate --model model.py --set time_budget_s=20