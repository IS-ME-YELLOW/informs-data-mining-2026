"""Confirm the copied projection and evaluation are numerically unchanged."""
from pathlib import Path
import ast
import hashlib
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OLD = BASE/"d_known_history_projection"
sys.path.insert(0, str(BASE))
from protocol import HORIZONS, COMPONENTS, MODES, build_controls
from project_d import observed_history, known_bounds, project_components


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def functions(path):
    return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}


def main():
    assert sha(HERE/"project_d.py") == sha(OLD/"project_d.py")
    old_plan = json.loads((OLD/"experiment_plan.json").read_text())
    new_plan = json.loads((HERE/"experiment_plan.json").read_text())
    for key in old_plan.keys() | new_plan.keys():
        if key not in {"experiment", "baseline_run", "design_status"}:
            assert old_plan[key] == new_plan[key], key
    for old, new in [("run_experiment.py", "run_projection.py"), ("verify_experiment.py", "verify_projection.py")]:
        a, b = functions(OLD/old), functions(HERE/new)
        assert set(a) == set(b)
        for name in a:
            if name != "main": assert a[name] == b[name], (new, name)
    confirmation = json.loads((HERE/"confirmation_plan.json").read_text())
    for item in confirmation["splits"]:
        assert item["model_seed"] == 42
        assert Path(item["baseline_dir"]) == BASE/"runs"/item["run_id"]
        assert Path(item["projection_dir"]) == HERE/f"split{item['split_seed']}"/"projection"
        assert Path(item["logs_dir"]) == HERE/f"split{item['split_seed']}"/"logs"
    # Replay only predictions against the already evaluated seed42 reference.
    run = Path(confirmation["anchor_run"])
    reference = Path(confirmation["anchor_projection"])
    frozen = pd.read_parquet(run/"oof_component_predictions.parquet")
    direct = pd.read_parquet(run/"base_predictions_cv.parquet")
    meta = frozen[["fipsCode", "timestamp_et", "hour_idx", "fold"]]
    raw = pd.read_csv(BASE.parents[2]/"data/DM_Train.csv", usecols=["fipsCode", "timestamp_et", "P_t"], dtype={"fipsCode":str})
    bounds = known_bounds(meta, observed_history(raw))
    components = {h: {c:frozen[f"pred_{c}_target_{h.rsplit('_',1)[-1]}"].to_numpy() for c in COMPONENTS} for h in HORIZONS}
    result = project_components(components, bounds)
    controls, _ = build_controls(meta, {h:direct[f"raw_{h}"].to_numpy() for h in HORIZONS}, result)
    saved = pd.read_parquet(reference/"candidate_oof_predictions.parquet")
    for mode in MODES:
        for h in HORIZONS:
            np.testing.assert_array_equal(controls[mode][h], saved[f"candidate_{mode}_{h}"].to_numpy())
    for path in HERE.glob("*.py"):
        compile(path.read_text(), str(path), "exec")
    result = {"status":"PASS", "projection_byte_identical_to_seed42":True,
              "all_evaluation_and_verifier_helper_functions_AST_identical":True,
              "formula_routing_diagnostics_and_bootstrap_unchanged":True,
              "seed42_all_controls_replayed_exactly":True, "training_performed":False,
              "path_layout_checked":True, "projector_sha256":sha(HERE/"project_d.py")}
    (HERE/"implementation_verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
