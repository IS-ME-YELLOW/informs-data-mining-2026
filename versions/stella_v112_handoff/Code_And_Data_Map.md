# 代码依据与文件传递

所有路径相对 `informs-data-mining-2026` 项目根；`B` 代表 `versions/xyy/v1.5.8_strict_baseline`。本包为 `B/stella_v112_handoff`。机器绝对路径不是协议的一部分。

## 1. 哪些代码参考什么

| 文件/函数 | 应复用的含义 | 不能直接照搬的地方 |
|---|---|---|
| `B/protocol.py`: `load_data`, `expected_mask`, `post_process`, `clip_component`, `compose_osi` | 冻结数据验证、标签与时间、后处理 | 保持163列；不要让别的目录同名 `config` 被误导入 |
| `B/scoped_training.py`: `scope_data`, `provenance`, `fit_scoped_base` | 标签作用域、内部选轮、完整scope重训 | `scope_data`只接受4/5折；新三折实现不能直接调用 |
| 同文件 `validate_receipt`, `load_fit`, `get_or_fit` | 回执、树数、模型哈希和schema验证 | `get_or_fit`命名假设只有一个外折；三折scope会命名冲突 |
| `B/run_artifacts.py`: `data_identity`, `prepare_run`, `freeze_stage`, `check_stage` | 数据/代码/参数/环境身份、原子写入、完成标志 | 原身份包含平台；不能把旧run当本机新run恢复 |
| `B/train_v18.py`: `output_frames` | raw预测及逐行来源、control列的定义 | 旧runner默认20个目标，不能意外全跑或跑final |
| `B/verify_artifacts.py` | 新进程重载模型、逐行复算的验证模式 | 默认检查当前环境身份、4/5折范围及完整旧实验；不是新实验的直接验收器 |
| `B/evaluation.py`: `_paired_bootstrap` | 县整轨迹配对重抽样 | 不把逐小时当独立样本，不平均折RMSE |
| `versions/v1.12_tail_protected_ensemble/run_tail_protected.py`: `fit_model`, `predict` | 固定q95及`>`的门控算术 | `run()`从全局旧OOF算阈值，还会跑test/final；不能作为新入口 |
| `versions/v1.11_tree_ensemble/run_ensemble.py`: `CANDIDATES`, `load_oof_inputs` | 六路含义与顺序、长时LGB C1 | `INPUTS`中旧`versions/v1.8`地址不适用，不导入旧OOF训练 |
| `versions/v2/adapters.py`: `XGBoostAdapter`, `CatBoostAdapter` | 模型结构/损失、保存和best_iteration语义 | 换成scope内选轮；单线程；不要继承旧LGB的`eval_X/eval_y`接口 |
| `versions/v1.9_xgboost/run_xgboost.py`, `versions/v1.10/run_catboost.py`, `versions/v2/v2.4/run_xgboost.py`, `versions/v2/v2.5/run_catboost.py` | 各模型的max_rounds/patience | 不调用其旧训练入口 |
| `versions/gbdt_experiment.py`, `versions/v2/component_experiment.py` | 审阅旧float32转换、后处理及标签语义 | 外折早停、自动final、旧cache路径和`.python_packages`注入不迁移 |

建议新实现放 `versions/stella_v112_strict/`，包含独立配置、scope适配层、训练入口、验证入口、tests与runs。该目录只是建议目标，本次尚未实现。冻结的 `B/` 与 Stella 历史版本只读；不要在原文件中打补丁来使新验证通过。

不要靠广泛修改 `sys.path` 导入多个同名 `protocol/config`。使用独立模块命名或显式加载路径；预检打印每个引用模块的 `__file__`、SHA-256 和版本。必要的迁移副本在新目录记录来源哈希及差异。

## 2. 数据与对照产物

`source_inventory.json` 是精确传递清单；文件字节以 SHA-256 为准。哈希不匹配时不要自行刷新清单。Git 换行转换也可能改哈希；优先保留文件原字节打包传递，记录原因后由负责人确认新快照，不放宽为忽略差异。

| 清单组 | 必需内容/用途 |
|---|---|
| `sources` | 上表代码、相关计划及结果、官方evaluation PDF；用于阅读和实现依据 |
| `inputs` | 原始train/test、submission模板、冻结163列/特征/标签/meta、分量标签、三份CV文件 |
| `reference_seed42` | seed42严格基线非模型产物，以及F2固定control预测、身份、验证和完成记录 |
| `reuse_seed42` | seed42严格基线完整已保存模型/回执，保证原完整性记录可核查；本次实际仅使用其中50个24/48h模型 |
| `confirmation` | 另两套同类对照/模型/完成记录，后续确认才需传输 |

首轮核心路径：

```text
B/runs/v18_tree_nested_v2_split42_model42/
    run_manifest.json, input_manifest.json, environment.json, CV_COMPLETE
    base_predictions_cv.parquet, oof_predictions.parquet
    model_manifest_cv.json, round_selection_cv.csv, verification_cv.json
    models/outer{0..4}/{target}.{txt,json}
B/p_information_increment/runs/v18_pinfo_v1_split42_model42/
    control_predictions.parquet, run_manifest.json, source_manifest.json
    verification.json, audit_finalization.json, INFO_CV_COMPLETE
```

精确参考身份见清单的 `reference_runs`。基线seed42：`144c278fa2029b96f2e844f15f9b5c90719ce290c80fbc9e779c7f2f7e3cfa5c`。F2seed42：`e39b76d75f191891abda1939dc47933815cc4fa15796227b0c200eb1b96bfcac`。

F2 主预测列为 `pred_F2_v18_rule_osi_target_t01h` / `...t06h`；基线 C1 列为 `pred_C1_component_osi_osi_target_t24h` / `...t48h`，C0 列中同样有两个 `osi`，不要凭直觉拼错。用当前实际schema断言，不模糊匹配列名。

F2 在这里被当作**已验证、带来源身份的固定短期对照产物**。为保持1/6h不变，无须迁移整个F2训练器；其递归依赖涉及E2、D及绝对路径。附带F2验证记录不能让新验证器声称重新验证了所有F2上游模型。若要独立重训F2，则是另一个完整依赖传递任务。

## 3. 跨平台环境与身份

已有strict LGB产物环境记录为 Python3.13.7、LightGBM4.7.0、NumPy2.5.3、Pandas3.0.5、PyArrow25.0.1，Linux。Stella历史XGB/Cat记录为XGBoost3.2.0、CatBoost1.2.10，Windows。这些是来源版本，不代表对方当前环境已安装或可用；对方需要在开训前记录真实环境并冻结。

新runner显式接受 `--project-root`、`--cv-file`、`--run-root`、`--model-seed`、`--stage`；首轮stage=cv，模型seed42。旧manifest中的绝对路径通过**外部路径映射**解析，原文件保持原字节。

区分三件事：

1. 来源完整性：严格基线完整run须重算保存identity的digest、原完成记录覆盖的全部文件hash、模型回执与scope；依据原记录验证，不用Windows平台字符串替换Linux字符串。F2仅验证已传递文件、保存identity及原完成记录中`control_predictions.parquet`对应的hash；不声称检查了F2整个完成记录闭包或全部上游模型。
2. 本地来源重放：在本机重载选中的50个LGB模型，按原列序/数值类型预测，与已保存raw OOF逐行核对。单独保存 `replay_environment`，默认`atol=1e-12, rtol=0`且NaN位置一致；失败先定位库/精度，不私自放宽或覆盖原预测。
3. 新实验身份：新XGB/Cat和三折LGB模型具有本地环境、代码、参数、输入、作用域及来源依赖，产生新的run identity。跨平台重新训练不承诺字节级相同；保证相同实验定义，并对本次保存模型进行独立重载复算。

不要直接在Windows执行原 `verify_artifacts.py` 后把环境拒绝当成来源模型损坏，更不要改原identity“修复”这种拒绝。新的来源验证模式应避免训练或写入旧目录。

## 4. 新实验的路径规则

建议输出结构：

```text
versions/stella_v112_strict/
  Protocol_Alignment.md
  src/                        # 新实现；代码冻结后开始正式run
  tests/                      # 隔离与算术检查
  preflight/                  # 输入/依赖/环境和dry-run报告
  runs/<独立run_id>/
    run_manifest.json, resolved_config.json, environment.json
    models/{family}/{scope_id}/{target}.*
    receipts/, fit_records/, predictions/, metrics/, logs/
    verification.json, CV_COMPLETE
  Results_<日期>.md
```

scope ID必须包含实际fold集合；probe/refit、family、target需区分。先完整保存模型再原子提交回执；无回执的孤儿文件不能复用。已有run代码/数据/环境变化则开新run；不修改manifest冒充resume。若进行小型真实隔离测试，写入单独diagnostic run，不能混进正式820次fit或正式成绩。
