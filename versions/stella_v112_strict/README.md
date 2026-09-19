# Stella v1.12 strict nested-CV experiment

Entry point:

```powershell
C:\Users\Stella\.conda\envs\best-route\python.exe -B versions\stella_v112_strict\run.py preflight --project-root .
```

The runner activates repository-local LightGBM/XGBoost/CatBoost packages explicitly. It does not modify the Python environment or the frozen baseline.

Formal seed42 outputs are isolated below `runs/stella_v112_nested_v1_split42_model42/`. Training is resumable only when the saved run identity and receipts match.

