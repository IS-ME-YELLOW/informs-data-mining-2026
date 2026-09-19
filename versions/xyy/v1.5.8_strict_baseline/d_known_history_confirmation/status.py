"""Read committed training receipts and completion markers; never fits models."""
from pathlib import Path
from datetime import datetime, timezone
import json

HERE = Path(__file__).resolve().parent
plan = json.loads((HERE/"confirmation_plan.json").read_text())
result = {"checked_at_utc":datetime.now(timezone.utc).isoformat(), "splits":[]}
for item in plan["splits"]:
    run = Path(item["baseline_dir"])
    receipts = list(run.glob("models/outer*/*.json"))
    state = HERE/f"split{item['split_seed']}"/"execution.json"
    attempt = json.loads(state.read_text())["attempts"][-1] if state.exists() else {}
    elapsed = (datetime.now(timezone.utc)-datetime.fromisoformat(attempt["started_at_utc"])).total_seconds() if attempt else 0
    result["splits"].append({"split_seed":item["split_seed"], "training_status":attempt.get("status","not_started"),
                            "elapsed_minutes":round(elapsed/60,1), "committed_outer_models":len(receipts),
                            "committed_fits":5*len(receipts), "total_outer_models":100, "total_fits":500,
                            "models_by_fold":{str(f):len(list((run/f"models/outer{f}").glob("*.json"))) for f in range(5)},
                            "cv_complete":(run/"CV_COMPLETE").exists(),
                            "split_complete":(state.parent/"completion.json").exists()})
(HERE/"progress.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
