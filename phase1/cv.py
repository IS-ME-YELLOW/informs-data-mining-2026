# ============================================================
# cv.py — 分层分组交叉验证
# ============================================================
# 职责: 生成5折CV, 确保:
#   - 同一县(fipsCode)的所有144行样本在同一折(GroupKFold, 防信息泄漏)
#   - 各折的州×severity_tier分布均衡(Stratified)
#
# 优先用 sklearn 的 StratifiedGroupKFold, 未安装则用手动实现
# 手动实现: 在每个stratum内随机洗牌县, 轮询分配到5折
# ============================================================

import numpy as np
from collections import defaultdict

from config import SEED, N_FOLDS


def _manual_stratified_group_kfold(groups, strata_list, n_splits, seed):
    """
    手动实现的分层分组KFold(sklearn不可用时的fallback)
    - groups: 每行的fipsCode(县ID)
    - strata_list: 每行对应的stratum(州_severity层级)
    - 在每个stratum内随机洗牌县, 轮询分配到n折
    - 返回: [(train_idx, val_idx), ...] 共n折
    """
    rng = np.random.RandomState(seed)

    # 建立 县→stratum 映射
    group_stratum = {}
    for g, s in zip(groups, strata_list):
        group_stratum[g] = s

    unique_groups = list(set(groups))

    # 按 stratum 分组县
    stratum_groups = defaultdict(list)
    for g in unique_groups:
        stratum_groups[group_stratum[g]].append(g)

    # 在每个stratum内随机洗牌, 轮询分配到各折
    fold_of_group = {}
    for stratum in sorted(stratum_groups.keys()):
        gl = stratum_groups[stratum]
        rng.shuffle(gl)
        for i, g in enumerate(gl):
            fold_of_group[g] = i % n_splits

    # 根据县→折映射, 生成行级索引
    n = len(groups)
    folds = []
    for k in range(n_splits):
        train_idx = np.array([i for i in range(n) if fold_of_group[groups[i]] != k])
        val_idx = np.array([i for i in range(n) if fold_of_group[groups[i]] == k])
        folds.append((train_idx, val_idx))
    return folds


def get_cv_folds(meta_df, n_splits=N_FOLDS, seed=SEED):
    """
    生成CV折数
    输入: meta_df (含 fipsCode, stateAbbr, severity_tier 列)
    输出: [(train_idx, val_idx), ...] 5折

    stratum = 州_严重度层级 (如 'IN_3', 'OH_1')
    优先用 sklearn, 未安装则用手动实现
    """
    groups = meta_df['fipsCode'].values.tolist()

    # 构建县→stratum映射
    if 'severity_tier' in meta_df.columns:
        unique_fips = meta_df[['fipsCode', 'stateAbbr', 'severity_tier']].drop_duplicates()
        stratum_map = {}
        for _, row in unique_fips.iterrows():
            stratum_map[row['fipsCode']] = f"{row['stateAbbr']}_{int(row['severity_tier'])}"
        strata_list = [stratum_map.get(g, g) for g in groups]
    else:
        strata_list = groups

    # 优先用sklearn的StratifiedGroupKFold
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        unique_groups = list(set(groups))
        unique_strata = [stratum_map.get(g, g) for g in unique_groups] if 'stratum_map' in dir() else strata_list
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        raw_folds = list(sgkf.split(np.zeros(len(unique_groups)), unique_strata, unique_groups))
        # 将县级fold转换为行级fold
        group_to_idx = {g: i for i, g in enumerate(unique_groups)}
        group_array = np.array([group_to_idx[g] for g in groups])
        result = []
        for train_g_idx, val_g_idx in raw_folds:
            train_mask = np.isin(group_array, [unique_groups[i] for i in train_g_idx])
            val_mask = np.isin(group_array, [unique_groups[i] for i in val_g_idx])
            result.append((np.where(train_mask)[0], np.where(val_mask)[0]))
        return result
    except ImportError:
        return _manual_stratified_group_kfold(groups, strata_list, n_splits, seed)
