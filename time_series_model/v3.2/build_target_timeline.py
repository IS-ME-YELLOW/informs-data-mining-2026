"""构建 v3.2 唯一目标小时轴及与原提交结构的双向映射。"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import (
    HORIZON_HOURS,
    HORIZONS,
    N_TARGET_HOURS,
    OBSERVED_HOURS,
    TARGET_EQUALITY_ATOL,
    TARGET_HOURS,
    TIMELINE_MANIFEST,
    TIMELINE_TEST,
    TIMELINE_TRAIN,
    V31_DIR,
    V31_SEQUENCE_MANIFEST,
    V31_TEST_SEQUENCE,
    V31_TRAIN_SEQUENCE,
    VERSION,
    ensure_artifact_dirs,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class SourceBundle:
    county_fips: np.ndarray
    timestamps: np.ndarray
    observed: np.ndarray
    future_weather: np.ndarray
    static_numeric: np.ndarray
    static_category: np.ndarray
    last_state: np.ndarray
    row_index: np.ndarray
    targets: np.ndarray


def load_source(path: Path) -> SourceBundle:
    with np.load(path, allow_pickle=False) as data:
        return SourceBundle(**{key: data[key].copy() for key in data.files})


def mapping_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 target->origin、候选 mask 和 origin/horizon->target 的索引。"""
    target_to_origin = np.full((N_TARGET_HOURS, len(HORIZONS)), -1, dtype=np.int16)
    candidate_mask = np.zeros_like(target_to_origin, dtype=bool)
    origin_horizon_to_target = np.full((144, len(HORIZONS)), -1, dtype=np.int16)
    for target_idx, target_hour in enumerate(TARGET_HOURS):
        for horizon_idx, horizon_hours in enumerate(HORIZON_HOURS):
            origin_hour = target_hour - horizon_hours
            if OBSERVED_HOURS <= origin_hour <= 215:
                origin_idx = origin_hour - OBSERVED_HOURS
                target_to_origin[target_idx, horizon_idx] = origin_idx
                candidate_mask[target_idx, horizon_idx] = True
                origin_horizon_to_target[origin_idx, horizon_idx] = target_idx
    return target_to_origin, candidate_mask, origin_horizon_to_target


def collapse_targets(
    source_targets: np.ndarray,
    target_to_origin: np.ndarray,
    candidate_mask: np.ndarray,
) -> np.ndarray:
    unique = np.full((source_targets.shape[0], N_TARGET_HOURS), np.nan, dtype=np.float32)
    for target_idx in range(N_TARGET_HOURS):
        candidates = []
        for horizon_idx in range(len(HORIZONS)):
            if candidate_mask[target_idx, horizon_idx]:
                origin_idx = int(target_to_origin[target_idx, horizon_idx])
                candidates.append(source_targets[:, origin_idx, horizon_idx])
        stacked = np.stack(candidates, axis=1).astype(np.float64)
        finite = np.isfinite(stacked)
        if not finite.all():
            bad = np.argwhere(~finite)[0]
            raise ValueError(
                f"目标缺失结构异常: county={bad[0]}, target_hour={TARGET_HOURS[target_idx]}"
            )
        spread = np.ptp(stacked, axis=1)
        if np.any(spread > TARGET_EQUALITY_ATOL):
            county_idx = int(np.argmax(spread))
            raise ValueError(
                "同一目标小时的 horizon 标签不一致: "
                f"county={county_idx}, target_hour={TARGET_HOURS[target_idx]}, "
                f"values={stacked[county_idx].tolist()}"
            )
        unique[:, target_idx] = stacked[:, 0].astype(np.float32)
    return unique


def build_one(source_path: Path, output_path: Path, has_targets: bool) -> dict:
    source = load_source(source_path)
    target_to_origin, candidate_mask, origin_horizon_to_target = mapping_arrays()
    unique_targets = (
        collapse_targets(source.targets, target_to_origin, candidate_mask)
        if has_targets
        else np.full((len(source.county_fips), N_TARGET_HOURS), np.nan, dtype=np.float32)
    )
    target_timestamps = source.timestamps[np.asarray(TARGET_HOURS, dtype=int)]
    np.savez_compressed(
        output_path,
        county_fips=source.county_fips,
        target_hours=np.asarray(TARGET_HOURS, dtype=np.int16),
        target_timestamps=target_timestamps,
        unique_targets=unique_targets,
        target_to_origin=target_to_origin,
        candidate_mask=candidate_mask,
        origin_horizon_to_target=origin_horizon_to_target,
        row_index=source.row_index,
    )
    return verify_one(output_path, len(source.county_fips), has_targets)


def verify_one(path: Path, counties: int, has_targets: bool) -> dict:
    with np.load(path, allow_pickle=False) as data:
        expected_keys = {
            "county_fips", "target_hours", "target_timestamps", "unique_targets",
            "target_to_origin", "candidate_mask", "origin_horizon_to_target", "row_index",
        }
        if set(data.files) != expected_keys:
            raise ValueError(f"{path.name} 数组键错误: {data.files}")
        if data["unique_targets"].shape != (counties, N_TARGET_HOURS):
            raise ValueError(f"{path.name} unique_targets 形状错误")
        if data["row_index"].shape != (counties, 144):
            raise ValueError(f"{path.name} row_index 形状错误")
        counts = data["candidate_mask"].sum(axis=1)
        expected_counts = np.r_[np.ones(5), np.full(18, 2), np.full(24, 3), np.full(96, 4)]
        if not np.array_equal(counts, expected_counts):
            raise ValueError(f"{path.name} 候选数量阶段错误")
        if int(data["candidate_mask"].sum()) != 497:
            raise ValueError(f"{path.name} 每县映射应对应497个有效提交单元格")
        if has_targets and not np.isfinite(data["unique_targets"]).all():
            raise ValueError(f"{path.name} 唯一标签不完整")
        if not has_targets and np.isfinite(data["unique_targets"]).any():
            raise ValueError(f"{path.name} 测试集不应包含标签")
        return {
            "sha256": sha256_file(path),
            "counties": counties,
            "unique_positions": counties * N_TARGET_HOURS,
            "mapped_submission_cells": counties * 497,
            "candidate_counts_by_target_hour": {
                str(k): int(np.sum(counts == k)) for k in range(1, 5)
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    ensure_artifact_dirs()
    for source in [V31_TRAIN_SEQUENCE, V31_TEST_SEQUENCE, V31_SEQUENCE_MANIFEST]:
        if not source.exists():
            raise FileNotFoundError(source)

    if args.overwrite or not TIMELINE_TRAIN.exists():
        train = build_one(V31_TRAIN_SEQUENCE, TIMELINE_TRAIN, True)
    else:
        train = verify_one(TIMELINE_TRAIN, 239, True)
    if args.overwrite or not TIMELINE_TEST.exists():
        test = build_one(V31_TEST_SEQUENCE, TIMELINE_TEST, False)
    else:
        test = verify_one(TIMELINE_TEST, 63, False)

    source_manifest = json.loads(V31_SEQUENCE_MANIFEST.read_text(encoding="utf-8"))
    manifest = {
        "version": VERSION,
        "milestone": "M1",
        "status": "passed",
        "source": "v3.1 县级序列缓存，只读复用",
        "target_hour_range": [73, 215],
        "target_hours": N_TARGET_HOURS,
        "horizons": HORIZONS,
        "horizon_hours": HORIZON_HOURS,
        "train": train,
        "test": test,
        "source_sha256": {
            str(V31_TRAIN_SEQUENCE.relative_to(V31_DIR.parents[1])): sha256_file(V31_TRAIN_SEQUENCE),
            str(V31_TEST_SEQUENCE.relative_to(V31_DIR.parents[1])): sha256_file(V31_TEST_SEQUENCE),
            str(V31_SEQUENCE_MANIFEST.relative_to(V31_DIR.parents[1])): sha256_file(V31_SEQUENCE_MANIFEST),
        },
        "v31_manifest_version": source_manifest.get("version"),
        "contracts": {
            "same_target_label_tolerance": TARGET_EQUALITY_ATOL,
            "one_prediction_per_county_target_hour": True,
            "postprocess_before_broadcast": True,
        },
    }
    TIMELINE_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.verify:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    else:
        print(f"M1 完成: {TIMELINE_MANIFEST}")


if __name__ == "__main__":
    main()
