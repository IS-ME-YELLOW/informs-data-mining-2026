# dem_v158_nested_v2

本目录是新的、与历史 `outputs` 隔离的 DEM-GAT 实验协议。协议使用
`versions/xyy/v1.5.6` 的冻结 163 列 Phase-1 特征；“v158”表示本协议和
v1.8-compatible 组件目标路线。v1.5.8 配置明确把 `FEATURE_VERSION` 固定为
v1.5.6；仓库不存在独立的 1.5.8 特征表。旧 209 维
checkpoint、旧 OOF 和 `component_v21` 别名均不参与新运行。

## 题目合规边界

- 停电侧输入只来自 `hour_idx=0..71`；公开天气可以使用全部小时。
- 训练县与测试县不交叉，预测起点为 `72..215`。
- 目标时刻超出 215 小时的目标/提交单元保留 NaN。
- `severity_tier`、州标识和其他 metadata 只用于对齐、分组和分折。
- 固定后处理为 `clip([0,0.65])` 后将小于 `0.001` 的值置零。

## 输入和维度

默认数据包必须同时包含五张 Parquet、`feature_names_v1.5.6.json` 和
`manifest.json`。默认 Parquet 引擎为 PyArrow；使用 fastparquet 时必须
显式传参并和 PyArrow 做等价校验。

GAT 有序输入为：

```text
1 base prediction + 163 frozen Phase-1 features
+ 2 coordinates + 7 DEM features
+ 16 neighbor means + 16 neighbor deltas = 205
```

当前 `make_county_time_view()` 返回的 base 通道必须等于
`features[..., 0]`。合法 NaN 只按冻结缺失白名单补 0；Inf、未知缺失和
失败预测直接失败。

## 训练隔离

`fit_base(S)` 只读取 S 折标签，先在 S 内逐折 early stopping，再在 S 上
固定轮数 refit。图输入遵守：S 内折 r 使用 `fit_base(S-{r})`，S 外节点
使用 `fit_base(S)`。GAT residual、标准化、residual scale、early stopping
和缓存身份均受同一个 S 约束。

外层折的 alpha 由完整重建的 inner stack 选择，不能只改变 GAT mask 而
复用 outer base/residual。最终 alpha 使用 `select_alpha(F)` 的完整训练集
内部 CV，不使用五个 outer alpha 的众数。

## CLI

从项目根目录运行：

```bash
python code_phase2_dem_eval_v158/main.py --stage preflight --base-mode direct
python code_phase2_dem_eval_v158/main.py --stage cv \
  --base-mode component_v158 \
  --run-id dem_v158_nested_v2_component_seed42 --seed 42 \
  --device auto --epochs 220 --patience 35 --time-stride 1 --k 8
python code_phase2_dem_eval_v158/main.py --stage final \
  --base-mode component_v158 \
  --run-id dem_v158_nested_v2_component_seed42 --resume \
  --seed 42 --device auto --epochs 220 --patience 35 --time-stride 1 --k 8
```

`preflight` 不创建训练输出、不训练模型、不写缓存。每次正式运行写入：

```text
code_phase2_dem_eval_v158/outputs/runs/<run_id>/
```

已存在的 run 必须使用 `--resume`，且 manifest 必须完全一致。`cv` 完成后
写入不可变 `CV_COMPLETE`；最终的 `COMPLETE` 只能由独立核验脚本生成：

```bash
python code_phase2_dem_eval_v158/verify_artifacts.py \
  code_phase2_dem_eval_v158/outputs/runs/<run_id>
```

## 证据和历史结果

正式运行保存逐行 `outer_oof.parquet`、`inner_oof.parquet`、
`alpha_selection.parquet`、`base_fit_manifest.parquet`、最终图 base、
测试预测、模型 checkpoint、环境和输入 manifest。历史
`EVALUATION_REPORT_CN.md` 只保留历史结果说明；新结果必须从当前运行
manifest 和逐行产物生成，不能回写旧成绩。
