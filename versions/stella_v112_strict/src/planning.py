from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

from .config import COMPONENTS, FOLDS, LONG_HORIZONS, TARGETS, component_target


@dataclass(frozen=True)
class ModelTask:
    role: str
    family: str
    target: str
    scope: tuple[int, ...]
    predict_folds: tuple[int, ...]

    @property
    def scope_id(self) -> str:
        return "folds_" + "_".join(str(fold) for fold in self.scope)

    @property
    def task_id(self) -> str:
        return f"{self.role}__{self.family}__{self.scope_id}__{self.target}"

    def as_dict(self) -> dict:
        result = asdict(self)
        result["scope"] = list(self.scope)
        result["predict_folds"] = list(self.predict_folds)
        result["scope_id"] = self.scope_id
        result["task_id"] = self.task_id
        return result


def outer_tasks() -> list[ModelTask]:
    tasks = []
    for family in ("xgboost", "catboost"):
        for outer_fold in FOLDS:
            scope = tuple(fold for fold in FOLDS if fold != outer_fold)
            for target in TARGETS:
                tasks.append(ModelTask("outer", family, target, scope, (outer_fold,)))
    return tasks


def inner_tasks() -> list[ModelTask]:
    tasks = []
    for scope in combinations(FOLDS, 3):
        excluded = tuple(fold for fold in FOLDS if fold not in scope)
        for horizon in LONG_HORIZONS:
            for component in COMPONENTS:
                tasks.append(ModelTask("inner_anchor", "lightgbm", component_target(component, horizon), scope, excluded))
    return tasks


def all_new_tasks() -> list[ModelTask]:
    return outer_tasks() + inner_tasks()


def dependency_plan() -> dict:
    new_tasks = all_new_tasks()
    edges = []
    for task in new_tasks:
        for validation_fold in task.scope:
            edges.append({
                "task_id": task.task_id,
                "kind": "probe",
                "train_folds": [fold for fold in task.scope if fold != validation_fold],
                "early_stop_folds": [validation_fold],
                "forbidden_folds": [fold for fold in FOLDS if fold not in task.scope],
            })
        edges.append({
            "task_id": task.task_id,
            "kind": "refit",
            "train_folds": list(task.scope),
            "early_stop_folds": [],
            "forbidden_folds": [fold for fold in FOLDS if fold not in task.scope],
        })
        edges.append({
            "task_id": task.task_id,
            "kind": "prediction",
            "prediction_folds": list(task.predict_folds),
        })
    threshold_sources = []
    for outer_q in FOLDS:
        for horizon in LONG_HORIZONS:
            for inner_r in FOLDS:
                if inner_r == outer_q:
                    continue
                scope = tuple(fold for fold in FOLDS if fold not in (outer_q, inner_r))
                threshold_sources.append({
                    "outer_fold": outer_q,
                    "horizon": horizon,
                    "predicted_fold": inner_r,
                    "scope": list(scope),
                    "targets": [component_target(component, horizon) for component in COMPONENTS],
                    "excludes_outer_and_predicted_labels": True,
                })
    return {
        "protocol": "stella_v112_nested_v1",
        "new_tasks": [task.as_dict() for task in new_tasks],
        "edges": edges,
        "threshold_sources": threshold_sources,
        "budget": {
            "new_fit_calls": sum(len(task.scope) + 1 for task in new_tasks),
            "new_saved_models": len(new_tasks),
            "referenced_models": 50,
            "outer_new_fit_calls": sum(len(task.scope) + 1 for task in outer_tasks()),
            "inner_new_fit_calls": sum(len(task.scope) + 1 for task in inner_tasks()),
        },
    }
