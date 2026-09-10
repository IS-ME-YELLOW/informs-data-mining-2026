"""为 v3.1 生成诊断级 v1.5.6 LightGBM 原始 OOF 与测试预测。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL_PACKAGES = ROOT / ".python_packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import numpy as np
import pandas as pd


def _provide_dense_only_scipy_stub() -> None:
    """LightGBM 的 dense DataFrame 路径不使用 SciPy，但导入时要求 sparse 类型存在。"""
    try:
        import scipy.sparse  # noqa: F401
        return
    except ImportError:
        scipy_module = types.ModuleType("scipy")
        sparse_module = types.ModuleType("scipy.sparse")

        class SparseMatrix:
            pass

        class CSRMatrix(SparseMatrix):
            pass

        class CSCMatrix(SparseMatrix):
            pass

        def unsupported(*_args, **_kwargs):
            raise RuntimeError("v3.1 仅允许 LightGBM dense DataFrame 输入")

        sparse_module.spmatrix = SparseMatrix
        sparse_module.csr_matrix = CSRMatrix
        sparse_module.csc_matrix = CSCMatrix
        sparse_module.hstack = unsupported
        scipy_module.sparse = sparse_module
        sys.modules["scipy"] = scipy_module
        sys.modules["scipy.sparse"] = sparse_module


_provide_dense_only_scipy_stub()
import lightgbm as lgb

from config import (
    BASELINE_METADATA,
    BASELINE_PREDICTIONS,
    CHECKPOINT_DIR,
    CV_ASSIGNMENTS,
    CV_VERSION,
    FEATURE_TEST,
    FEATURE_TRAIN,
    HORIZONS,
    LGBM_EARLY_STOPPING_ROUNDS,
    LGBM_NUM_BOOST_ROUND,
    LGBM_PARAMS,
    META_TRAIN,
    N_FOLDS,
    SEED,
    TARGET_TRAIN,
    ensure_artifact_dirs,
)
from evaluate import metrics, postprocess


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_fips(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)


def load_row_folds(meta: pd.DataFrame) -> np.ndarray:
    assignment = pd.read_csv(CV_ASSIGNMENTS, dtype={"fipsCode": str})
    assignment["fipsCode"] = normalize_fips(assignment["fipsCode"])
    if assignment["fipsCode"].duplicated().any():
        raise ValueError("CV assignment 出现重复县")
    if set(assignment["fold"]) != set(range(N_FOLDS)):
        raise ValueError("CV assignment 折编号错误")
    fold_map = assignment.set_index("fipsCode")["fold"]
    row_folds = normalize_fips(meta["fipsCode"]).map(fold_map)
    if row_folds.isna().any():
        raise ValueError("至少一个训练行没有县级折分配")
    return row_folds.to_numpy(dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    ensure_artifact_dirs()
    if BASELINE_PREDICTIONS.exists() and not args.overwrite:
        print(f"已存在，跳过训练: {BASELINE_PREDICTIONS}")
        return

    X_train = pd.read_parquet(FEATURE_TRAIN)
    X_test = pd.read_parquet(FEATURE_TEST)
    targets = pd.read_parquet(TARGET_TRAIN)
    meta = pd.read_parquet(META_TRAIN)
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("v1.5.6 训练和测试特征列不一致")
    if list(targets.columns) != HORIZONS:
        raise ValueError("v1.5.6 目标列不一致")

    row_folds = load_row_folds(meta)
    oof_raw = np.full((len(X_train), len(HORIZONS)), np.nan, dtype=np.float64)
    test_by_fold = np.empty((N_FOLDS, len(X_test), len(HORIZONS)), dtype=np.float64)
    fold_records = []

    for horizon_idx, horizon in enumerate(HORIZONS):
        y = targets[horizon].to_numpy(dtype=float)
        valid = np.isfinite(y)
        for fold in range(N_FOLDS):
            train_mask = valid & (row_folds != fold)
            val_mask = valid & (row_folds == fold)
            if not train_mask.any() or not val_mask.any():
                raise ValueError(f"{horizon} fold {fold} 没有合法样本")

            train_data = lgb.Dataset(X_train.loc[train_mask], label=y[train_mask])
            val_data = lgb.Dataset(
                X_train.loc[val_mask], label=y[val_mask], reference=train_data
            )
            booster = lgb.train(
                dict(LGBM_PARAMS),
                train_data,
                num_boost_round=LGBM_NUM_BOOST_ROUND,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(LGBM_EARLY_STOPPING_ROUNDS, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            val_prediction = booster.predict(X_train.loc[val_mask])
            test_prediction = booster.predict(X_test)
            oof_raw[val_mask, horizon_idx] = val_prediction
            test_by_fold[fold, :, horizon_idx] = test_prediction

            fold_metric = metrics(y[val_mask], postprocess(val_prediction))
            fold_records.append({
                "horizon": horizon,
                "fold": fold,
                "best_iteration": int(booster.best_iteration),
                **fold_metric,
            })
            model_path = CHECKPOINT_DIR / f"baseline_{horizon}_fold{fold}.txt"
            booster.save_model(str(model_path))
            print(
                f"{horizon} fold={fold}: best_iter={booster.best_iteration}, "
                f"RMSE={fold_metric['rmse']:.6f}, MAE={fold_metric['mae']:.6f}"
            )

    for horizon_idx, horizon in enumerate(HORIZONS):
        expected_valid = targets[horizon].notna().to_numpy()
        if not np.isfinite(oof_raw[expected_valid, horizon_idx]).all():
            raise ValueError(f"{horizon} OOF 预测不完整")
        if np.isfinite(oof_raw[~expected_valid, horizon_idx]).any():
            raise ValueError(f"{horizon} 无效目标行出现 OOF 预测")

    test_raw = test_by_fold.mean(axis=0)
    np.savez_compressed(
        BASELINE_PREDICTIONS,
        oof_raw=oof_raw,
        test_raw=test_raw,
        test_by_fold=test_by_fold,
        row_folds=row_folds,
        horizons=np.asarray(HORIZONS, dtype="U32"),
    )

    pooled = {}
    for horizon_idx, horizon in enumerate(HORIZONS):
        pooled[horizon] = metrics(
            targets[horizon].to_numpy(dtype=float), postprocess(oof_raw[:, horizon_idx])
        )
    metadata = {
        "version": "v3.1",
        "evaluation_status": "diagnostic_only",
        "baseline_version": "v1.5.6",
        "cv_version": CV_VERSION,
        "seed": SEED,
        "lightgbm_version": lgb.__version__,
        "prediction_semantics": {
            "oof_raw": "五折 LightGBM 后处理前折外预测；折内早停，供M3诊断使用",
            "test_raw": "五个折模型对测试集原始预测的算术平均",
        },
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in [FEATURE_TRAIN, FEATURE_TEST, TARGET_TRAIN, META_TRAIN, CV_ASSIGNMENTS]
        },
        "output_sha256": sha256_file(BASELINE_PREDICTIONS),
        "fold_metrics": fold_records,
        "pooled_postprocessed_metrics": pooled,
        "params": LGBM_PARAMS,
    }
    BASELINE_METADATA.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(pooled, ensure_ascii=False, indent=2))
    print(f"已保存: {BASELINE_PREDICTIONS}")


if __name__ == "__main__":
    main()

