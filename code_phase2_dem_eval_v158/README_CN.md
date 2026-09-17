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

## 2026-09-17 第1–6项修复说明

- 冻结特征名称 JSON 的 manifest 哈希对应 LF 字节，并通过根目录
  `.gitattributes` 固定该文件的检出换行；五张 Parquet 未修改。
- pandas 3 的缺失 mask 使用可写副本；训练/测试行映射按互斥且联合覆盖
  全图检查，不要求两个映射各自覆盖全部节点。
- 239 县 OOF 有效行数为 34177/32982/28680/22944；63 县提交的有效行数
  仍为 9009/8694/7560/6048，两者分开校验。
- `feature_side_hash` 表示特征及元数据侧身份；`feature_package_hash`
  表示包含 OSI 标签身份的完整包身份。基模型和 GAT 的保存/加载保持一致。
- 模型文件中的 `t01h` 等 horizon 不再重复添加 `t`、`h`。
- `base_fit_manifest.parquet` 只记录并冻结 CV 模型；最终训练另写
  `base_fit_manifest_final.parquet`，它保留全部 CV 记录并加入全五折模型。
  完整 final 清单优先用于重载，否则读取 CV 清单。未登记的中断产物
  不被加载；登记模型缺失或哈希错误则失败。最终清单按值验证 CV 记录
  未变，CV 清单的文件字节保持不变。

无训练回归检查可直接运行，不需要 pytest 或 PyTorch：

```bash
python -B code_phase2_dem_eval_v158/tests/test_repairs.py
```

这些检查覆盖数据、图行映射、模拟 CV 汇总、两种模式模型清单，以及历史
LightGBM 文本权重的重载；历史权重仅作为文件格式测试材料，不作新协议的
成绩证据。**它们不替代内外层完整标签隔离、真实 GAT 或最终端到端验收。**
后续 P1 修复已补齐第7项 verifier 的数值验收链路，具体见下节；恢复流程和
真实模型标签隔离验收仍待处理，当前不能仅据这些无训练检查声明已可正式训练。

## 2026-09-17 剩余P1修复：独立产物验收

`artifact_checks.py` 实现无需 Torch 的数值检查，`verify_artifacts.py` 负责
完整运行验收和真实模型独立重载：

- 外层 alpha 是5折×4 horizon×8候选，共160条；检查完整候选网格、每个
  候选的样本量/SSE/RMSE/MAE、最小RMSE选择及同分时取更小alpha。
- 所有县时连接统一FIPS为五位字符串，并按完整键集合对齐；缺行、重复、
  错县、错折和错误horizon都失败，不以预测是否有限缩小计分集合。
- 逐行OOF标签与冻结的官方标签核对；内层预测的base/stack来源、允许标签
  集合、各基模型probe的训练/早停折及完整模型清单均须一致。
- 独立重算summary、fold、county与固定县级bootstrap，数值容差为绝对
  `1e-12`；最终alpha只与全训练内部CV对应，不改写外层alpha或分数。
- 提交CSV逐行核对全部标识列、值和精确NaN位置；最终全图base须保留训练县
  的交叉拟合输入，并与保存的OOF/test base来源一致。
- 独立重载阶段禁止通过项目训练接口拟合模型，禁止通过项目标签接口和
  已知文件读取路径读取监督标签。重新计算的全图base、测试base/correction、
  最终预测与提交CSV都须一致，预测重载容差为绝对 `1e-7`。
- 只有数值验收、checkpoint身份/依赖核验、无标签独立重载全部成功，才写
  `verification.json` 的通过标记和 `COMPLETE`。

同时修正内层产物：`base_source_id` 现在记录实际LightGBM模型编号；GAT编号
单独保存在 `stack_id`。旧文件误将这两个字段写为相同值，不应改写已冻结的
CV产物以绕过校验；新正式run应由当前代码生成。

无训练验收回归测试（标准库unittest，不需要pytest）：

```bash
python -B code_phase2_dem_eval_v158/tests/test_artifact_checks.py
python -B code_phase2_dem_eval_v158/tests/test_repairs.py
```

两套分别18项和11项，覆盖正常direct/component产物、错误候选/择优、错误
scope/标签/缺行、县码类型、指标与提交篡改、标签读取/训练调用禁止及失败时
不生成COMPLETE。checkpoint和重载流程的部分测试使用替代对象，不代表已在
本机执行真实GAT推理。完整验收命令仍为 `verify_artifacts.py <run_dir>`，须在
安装PyTorch的远程环境运行；不提供跳过真实模型重载就生成COMPLETE的模式。

上述P2的代码修改已在后续完成，当前状态与远程验收方法见下节。

## P2实现与远程验收

完整远程说明：`REMOTE_ACCEPTANCE_CN.md`。本轮新增：

- `run_identity.py`：冻结源代码、完整配置、输入哈希和运行环境；恢复前逐项
  比较，原始静态文件不覆盖，每次启动仅追加 `attempts.jsonl`。
- `model_records.py`：每个模型完成后写 `.complete.json`，它是可恢复状态的
  提交点；先校验模型再提交记录。缺记录的模型文件按未完成处理，有记录但
  文件损坏则失败。GAT还绑定其所有上游base模型的实际哈希。
- `final_rounds` 保留“请求轮数”的含义，新增 `actual_rounds`。保存、预测和
  重载按实际轮数；核验要求实际值与文件一致且不大于请求值。
- `--stage all --resume` 可从中断CV继续，也可在CV完成后直接进入final；
  已有FINAL_READY不重复生成最终产物。只恢复完整模型，不恢复半途的优化器。
- `validate_runtime.py`：独立进程的外折/内折/cross-fit标签扰动、正向对照、
  中断恢复、无标签重载及真实短树模型边界检查。每折固定少量县，不改五折CSV。

正式方法协议仍为 `dem_v158_nested_v2`；不可变身份与模型完成记录采用新格式。
旧run没有该身份/记录时拒绝自动恢复，请使用新的run_id。诊断模型输出与正式
`outputs/runs` 隔离，不能把诊断产物作为正式训练缓存。

本地新增检查：

```bash
python -B code_phase2_dem_eval_v158/tests/test_p2_contract.py
```

正式完整模型验收仍需要远程PyTorch环境。本地替代函数检查或 `--backend mock`
报告不会把 `real_acceptance_passed` 标成true。请按远程说明完成CPU完整检查、
目标GPU补充检查，以及正式训练结束后的 `verify_artifacts.py` 独立验收。
