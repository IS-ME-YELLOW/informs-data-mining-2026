# v1.8 前置条件详细修改方案（待实施，2026-09-17）

**本文件是修改规格，不是已完成的代码修复。当前仅新增了两套CV文件、审查脚本、证据及文档；既有训练代码未修改，未启动训练。** 检查事实见同目录 `V18_Precondition_Audit_2026-09-17.md`。

目标是在主项目 `versions/xyy/v1.5.8` 的树基线上建立可复现、可切换分折且外层标签隔离的实验入口，为之后D约束等小实验提供共同基准。新协议建议命名 `v18_tree_nested_v2`，不要把它与历史 v1.8 的原五折早停成绩混为同一验证口径。

## 1. 冻结范围：本次不顺带改模型问题

- 保留 v1.5.6 的163列特征、观测小时0–71、预测起点72–215、四个horizon及官方标签。
- 保留官方提供的N/R，包括小时71，保留观测OSI重组、下限clip和round(4)口径；未来监督标签不round或重写。
- 保留组件clip `[0,1]`、重组公式、当前OSI后处理 `[0,0.65]`及 `<0.001`置零；它们是冻结的实验规则，不据本轮检查重新调阈值。
- 默认保留C0直接、C1分量、C2固定等权、C3同目标小时分量对齐四套输出。新增明确的 `v18_rule` 输出：短期1/6h取C3，长期24/48h取C1。禁止每次新划分又按最低分改路由。
- 后续分量修改主要相对同协议C1对照，同时报告固定v18_rule结果；C0/C2作诊断。不要把“仅C1”标成完整的当前v1.8规则。
- 不在此次前置修复中加入D下限、气象shift、DEM、邻县特征、GAT或新损失函数。

## 2. 逐文件修改清单

| 文件 | 必须改动 | 验收证据 |
|---|---|---|
| `config.py` | 修正项目根、成包解析冻结特征；分离split/model seed；建立独立run输出根 | 从任意cwd解析到正确输入；不产生旧目录写入 |
| `protocol.py` | loader显式接收路径；严格键、mask、CV分层校验；指标拒绝缺预测 | 新三套CV均可读；坏键/错序/Inf/缺预测主动失败 |
| `train_v18.py` | 外层训练县内部选树数；支持stage/CV文件/run-id；分运行模型身份 | 外层标签扰动不改变对应模型；CV阶段不训练全量提交模型 |
| `verify_artifacts.py` | 验证指定新run；按新manifest校验模型、分折、列序与数值；默认不改历史产物 | 新进程复算通过，篡改任一身份或缺预测则非零退出 |
| `README.md`、实验说明 | 修复路径；区分原协议/新协议、准备/训练/验证阶段 | 命令与真实CLI一致，不把旧成绩当新协议成绩 |

若增加新模块，应只为分折/运行身份、严格指标或scoped训练等清晰职责服务。可以参考已写DEM修复规范中的隔离思想，但树基线无需导入GAT依赖，也无需实现GAT残差交叉拟合或alpha搜索。

## 3. 路径和数据包

当前 `V18_DIR=Path(__file__).resolve().parent`，因此新 `PROJECT_ROOT` 应为 `V18_DIR.parents[2]`。随后验证根目录确实含 `data/DM_Train.csv`、`cv` 等必要文件，失败应清楚报错。

默认完整特征包指向：

```text
PROJECT_ROOT/versions/xyy/v1.5.6/
  features_train_v1.5.6.parquet
  features_test_v1.5.6.parquet
  meta_train_v1.5.6.parquet
  meta_test_v1.5.6.parquet
  targets_train_v1.5.6.parquet
  feature_names_v1.5.6.json
```

分量目标默认读取 `V18_DIR/component_targets_v1.8.parquet`。文件已存在且本轮验证通过，preflight不应重新生成它。

- `--feature-dir` 指定完整包，不能逐文件混用cache与versions来源。相对路径按PROJECT_ROOT解析，绝对路径按指定路径解析。
- `--cv-file` 指定现有CSV；禁止训练入口在缺CV时自动创建一套替代分折。
- 固定pyarrow引擎并记录版本。本机已通过，不需要安装旧兼容层或引入 `_compat/scipy`。
- 内存中将FIPS规范化为五位字符串、时间解析为datetime、row_id重建为连续位置；保存提交时保持官方模板标识和顺序。
- 只验证原始和缓存的键，不自行重排一张无键features表来“猜测”与meta的对应；允许对有稳定键的目标表显式reindex，但须先验证唯一性。
- 所有源文件只读。需要生成新的组件目标时，单独显式构建到新数据包并验证；不得让preflight或普通verify隐式覆盖原标签。

## 4. 把CV seed与模型seed分开

三份清单已冻结：

```text
cv/cv_assignments_balanced_v1_seed42.csv
cv/cv_assignments_balanced_v1_seed20260917.csv
cv/cv_assignments_balanced_v1_seed20260918.csv
```

这些数字只定义县分配，不能自动传给LightGBM。初轮三个分折实验的 `model_seed` 均固定为42；其他超参数和线程设置也保持一致。若派生各target/inner模型的子seed，规则须预先固定且不依赖标签、误差、目标文件哈希或CV表现。

`load_cv_folds(meta, cv_file, n_folds=5)` 必须验证：

1. 恰好四列 `fipsCode,stateAbbr,severity_tier,fold`，类型合法，fold为0–4。
2. 239县各出现一次；与训练meta县集及州/severity逐县一致，训练/测试县无交集。
3. 五折大小差≤1，每个州×severity层内各折大小差≤1，包括该层某折为0的情形。
4. 一个县的全部小时、四个horizon、四分量始终属于同一个外折。
5. 读入的CV SHA-256与指定运行的manifest一致，resume不允许替换文件。

先保持原seed42作为开发参考，对事先选定的少数候选同时跑两套固定确认清单；不能看到其中一套较差就放弃报告、换seed，不能继续手工调整极端县。

## 5. 树模型所需的严格嵌套训练

### 5.1 只在允许县集合内选择树数

定义 `fit_scoped_base(S, target, X, scoped_labels, folds, params)`：S是允许使用该目标标签的县折集合。未来的目标值可以作为S内县的监督，但S外县的标签不能进入梯度训练或早停。

```python
def fit_scoped_base(S, target):
    best_rounds = []
    for inner_fold in sorted(S):
        inner_train = valid_rows(S - {inner_fold}, target)
        inner_valid = valid_rows({inner_fold}, target)
        assert both_nonempty(inner_train, inner_valid)
        probe = lightgbm_train_with_early_stopping(
            X[inner_train], y[inner_train],
            X[inner_valid], y[inner_valid],
            frozen_params_and_seed,
        )
        best_rounds.append(probe.best_iteration)
    rounds = max(1, int(np.mean(best_rounds)))
    model = lightgbm_train_fixed_rounds(X[valid_rows(S)], y[valid_rows(S)], rounds)
    return model, rounds, best_rounds, dependency_record(S)
```

固定最大2000轮、patience100、均值向下取整规则及原超参数。不要同时为严格验证修复做大规模调参。可以将线程数固定为1以便重现，但须为所有新基线/候选采用相同配置并记录；不要将并行设置变化造成的差异归给特征改进。

### 5.2 外层评分

对于外折k：

- 调用 `fit_scoped_base(F-{k}, target)`，即四个训练折内部做四次早停探测。
- 每个probe是三折训练、一折早停；随后以选定轮数重训全部四折。
- 只有四折重训模型在k上生成正式OOF。不得将probe在其早停县上的预测当作正式外层成绩。
- k的目标只在预测冻结后的评分器中读取；训练接口最好只接收裁切到S的监督对象，测试时对越界标签访问主动报错。

**因此，正式外折树模型仍使用约191县；三折只是内部选轮数的探测模型。最终提交树模型使用全部239县。** 与GAT方案不同，纯树实验不需要额外生成训练残差，也不需要两折残差基模型或内层alpha。

各标量目标使用同一县折。P/N/D/R分别按上述流程训练，再clip、重组；C2固定等权；C3只能合并同一县、同一目标小时、同一外层排除边界下的各horizon预测，不得混入旧seed42预测或全量训练模型的训练县预测。

### 5.3 最终模型

在CV检查完成、身份冻结后，为每个目标调用同一接口 `fit_scoped_base(F, target)`：五次内部早停（每次四折训练、一折早停），汇总轮数后在239县上固定轮数重训。它可以使用全部训练标签，但其内部选择分数不作为新的独立成绩。

这条最终轮数规则与旧代码“直接取五个外层best_iteration均值”不同，须在新协议中明确记录；不混用两种规则。正式比较始终使用此前冻结的外层OOF。

默认4个直接目标＋16个分量目标共20个标量目标。每套分折CV为每目标 `5×(4+1)=25` 次LightGBM拟合，共500次；最终阶段为每目标 `5+1=6` 次，共120次。这里是拟合次数，不是耗时估计，也不要求所有候选都训练最终模型。

## 6. 运行身份、恢复与阶段

建议目标CLI（尚未实现）：

```text
--stage preflight|cv|final|all       默认preflight
--feature-dir <complete_package>
--cv-file <existing_csv>
--model-seed 42
--run-id <unique_name>
--resume
```

现有 `--validate-only` 可废弃或映射到真正只读的preflight，不能沿用“先ensure_dirs、可能生成标签再退出”的行为。

所有新产物放在：

```text
versions/xyy/v1.5.8/runs/<run_id>/
```

不得覆盖原来的 `models/`、OOF、submission、verification、metadata或实验日志；旧文件作为历史参考保留。新日志放在新run内。

运行身份至少包含：协议版本、run-id、CV文件SHA-256/县分配、原始/特征/目标文件哈希、有序特征schema、模型参数及随机种子、后处理、C3/固定路由规则、代码哈希和运行依赖。模型身份还包含target、outer/inner/final角色、允许标签县集合、轮数选择规则与每次probe的训练/早停县集合。

resume应拒绝：不同CV、不同特征或标签、不同代码协议/参数、缺模型、模型哈希不符。**仅文件名含seed或fold号不够。** 旧模型数值可复现也不意味着能用于新划分。

建议只恢复已经完整保存并校验的目标/外折模型；中断中的fit从固定seed重跑，避免不完整训练状态被误当完成模型。

阶段边界：

1. preflight：只读输入及身份，不建模型目录、不生成目标、不训练。
2. cv：生成外层OOF与诊断；不训练最终全量提交模型。全部CV复核通过后写 `CV_COMPLETE` 和不可变OOF/manifest哈希。
3. final：仅接受同一run的CV_COMPLETE及完全一致的身份；训练全量模型、生成提交。
4. verify：新进程重载模型，复核数值与提交，验证报告写新run。默认不回写旧版本文件；全部新run验收完成才写COMPLETE。

## 7. 严格评分、C3与产物

### 7.1 评分集合由数据契约定义

对于每个horizon，先由县、起点小时和 `hour_idx+h<=215` 建立expected_mask，再断言该范围的官方目标、预测均有限，且OOF覆盖恰好一次。

```python
expected = is_selected_county & (hour_idx + h <= 215)
assert finite(y[expected]).all()
assert coverage_count[expected].eq(1).all()
assert finite(prediction[expected]).all()
metric = pooled_metric(y[expected], prediction[expected])
```

不能用预测是否finite决定哪些行计分。对真实标签有效但预测NaN/Inf的情况立即失败；不得少算n、跳过整县、置零或降级到另一模型隐藏错误。Bootstrap和各分组指标也使用同一完整集合。

C3对每个目标小时应核验候选horizon身份和期望数量，而不只接受“当前恰好有多少非NaN”。缺少某个本应存在的候选应报错；同一县同一目标小时的对齐分量与重组值必须一致。

### 7.2 必需新产物

- `run_manifest.json`、`environment.json`、`input_manifest.json`、精确CV副本及SHA-256。
- `model_manifest`：每个模型当前哈希、特征序、target、标签允许集合、训练/早停/评分角色、树数与seed。
- `round_selection`：逐outer/inner/target的best_iteration、训练和早停县名单/哈希，外层聚合轮数。
- `oof_predictions`：县时键、hour_idx、horizon、outer_fold、split_id、y、is_scoreable、C0/C1/C2/C3/v18_rule预测；137664个县时horizon行或等价宽表，有效行总数118783。
- `oof_component_predictions`：四组件实际值、预测值及模型来源；C3对齐后的唯一目标时刻组件与候选来源。
- pooled、逐折、逐县、逐时段指标，以及按县配对bootstrap。官方主要指标仍是各horizon pooled RMSE，不把折RMSE简单平均当官方指标。
- final阶段保存20个全量模型、相应预测、提交与结构审计。
- `verification.json`、CV_COMPLETE/COMPLETE和产物哈希。

提交必须逐行检查模板标识、原顺序、精确NaN位置、有效行数9009/8694/7560/6048、预测有限且在冻结输出范围内。不能仅检查总NaN数量或把异常写成日志后继续完成。

### 7.3 旧哈希问题的处理

本轮已有当前120个模型哈希与原记录的并列审计，以及预测完全一致的证据。保留旧清单，先追溯历史模型是否有重新序列化/修复/复制记录；本轮没有确定原因。

如果需要继续只读使用当前历史模型，建立独立的 `legacy_current_snapshot` 清单，明确它是当前文件快照、数值已复算、历史字节身份未确认。不要修改旧哈希伪装成原有验证已经通过。严格新run则从头训练并产生新的自洽身份，不能从这个历史快照恢复到新CV。

## 8. 修改后必须通过的验收

以下测试是未来实施后的要求，本轮没有训练验证新算法。

| ID | 验收内容 | 通过标准 |
|---|---|---|
| P1 | 从项目外cwd做preflight | 路径正确，原数据/代码/模型/日志哈希与mtime不变，无隐式新目标 |
| P2 | 三套CV分别加载 | 县覆盖、州/severity、全局/层内平衡通过；坏CV报错 |
| P3 | 特征因果与最后观测小时 | 163列重建及未来停电扰动不变；last_N/R等于官方小时71 |
| P4 | 标签与尾部 | OSI和16组件按同县t+h逐值匹配；精确mask/有效行数通过 |
| I1 | 标签访问记录型替代模型 | 每个outer流程的训练、早停、轮数选择仅访问非outer县；评分器分离 |
| I2 | 修改外折k的OSI及四组件监督 | 冻结features/CV，关闭缓存，对应k的训练轮数、参数、C0/C1/C3预测不变，评分允许变化 |
| I3 | 小规模真实LightGBM复核I2 | 固定CPU/线程/seed，数值在预定 `atol=1e-12, rtol=0` 内相同；不通过时定位依赖或非确定性，不事后放宽 |
| R1 | 新CV尝试resume旧模型 | 明确拒绝；同fold编号、同树数也不能绕过 |
| R2 | 同身份完整模型resume | 逐行预测和指标与无恢复运行一致 |
| M1 | 有效预测NaN/Inf、缺行、重复行 | 必须抛错；不得改变计分n |
| C1 | C3同目标时刻跨horizon | 所有来源具有相同外折排除集合；组件一致；候选完整 |
| A1 | 独立重载全部新run模型 | OOF、测试预测与提交在 `1e-12` 数值容差内复算；身份哈希均通过 |
| A2 | 未获新训练授权的审查模式 | 不调用 lgb.train，不覆盖历史文件，不生成最终提交 |

I2改变的是未来监督，不改观测输入或官方用于分层的severity/CV文件；不能重新分折后要求模型不变。只比较对应外折k的训练子流程，其他外折和最终全量模型可以合法依赖k标签。目标文件哈希会随扰动改变，不要求整个manifest字节相同；但该哈希不能进入随机种子而改变隔离模型。

## 9. 修改顺序与之后的运行方式

推荐实施顺序：路径/只读preflight → CV显式加载及身份 → scoped树数选择 → 严格计分/C3来源 → 新run保存和resume → 独立验证 → 小规模真实测试。前置验收通过后，才开始后续D约束等模型实验。

以下仅为**实现后的目标命令**，当前旧训练器不支持这些参数，本轮未执行：

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-2026
.venv/bin/python versions/xyy/v1.5.8/train_v18.py --stage preflight --cv-file cv/cv_assignments_balanced_v1_seed20260917.csv --model-seed 42
.venv/bin/python versions/xyy/v1.5.8/train_v18.py --stage cv --cv-file cv/cv_assignments_balanced_v1_seed42.csv --model-seed 42 --run-id v18_tree_nested_v2_split42_model42
```

对预先选定的同一候选，随后分别使用seed20260917、seed20260918及不同run-id确认，模型seed仍为42。只比较每套划分内部的候选-基线差，保存两套确认结果，不混合不同划分的OOF行作为更大“独立样本”。

即使严格隔离实现正确，重复看这些确认分折后继续调参也会使其变成开发数据。本轮冻结两套清单的目的，是提供有限的稳定性确认，而不是无限增加可供择优的分数。
