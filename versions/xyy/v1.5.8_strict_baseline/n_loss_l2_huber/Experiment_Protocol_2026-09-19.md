# 冻结特征的 N-L2 / Huber 对照（2026-09-19）

用户已授权执行。本轮按N诊断报告的下一步建议，先完成 **split seed42、四个horizon** 的首轮对照；不自动启动另外两套CV或全量提交训练。

全部新增文件位于：

`/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/n_loss_l2_huber`

## 1. 唯一训练变量

当前参照是同split的完整F2：strict v158＋D_G＋P1h a/b＋邻县F2。N四个模型仍是原163列输入的Huber LightGBM。

- 保持原163列特征、列序、官方N标签、全体239县、seed42的县级五折及模型seed42。
- 新N模型仅把 `objective=huber` 改为 `objective=regression`（L2平方损失）；不增删特征，不做目标变换、样本重加权、尾部加权、N71门控、时间平移或县例外。
- 保持 LightGBM 4.7.0、learning_rate=.05、num_leaves=31、min_child_samples=50、feature_fraction=.8、bagging_fraction=.8、bagging_freq=5、lambda_l1=.1、lambda_l2=1.0、单线程、deterministic及所有随机种子。
- 早停指标仍为未截断N预测的RMSE，最大2000轮、耐心100轮。轮数可因损失变化而不同，**选择规则保持相同**。
- 每个外折先在其他四折内做四次“3折训练＋1折早停”，取best_iteration均值向下取整，再用完整四折重训。该外折县不能进入梯度训练、分箱参考、早停或轮数选择。
- 每h新增25次拟合、5个外层模型；四h共100次拟合、20个模型。复用并重载既有20个N-Huber模型，不重训P/D/R/直接OSI，也不把oracle诊断结果输入训练。

## 2. 六个预定组合

| 编号 | 使用L2的来源N | 其他部分 |
|---|---|---|
| H0_Huber | 无 | 完整同CV F2 |
| L2_N01 | N1h | 其余来源N及P/D/R均复用 |
| L2_N06 | N6h | 同上 |
| L2_N24 | N24h | 同上 |
| L2_N48 | N48h | 同上 |
| L2_all | 四个来源N | P/D/R复用 |

主要比较预先固定为 **L2_all−H0_Huber 的四个最终OSI RMSE**。四个单来源组合用于解释传播和后续候选选择，不把首轮择优当作独立确认，不额外搜索全部16种来源组合。

组件clip到[0,1]；C3按同县同目标小时等权对齐；主规则1/6h用C3、24/48h用C1；OSI clip到[0,.65]并将<.001置零，全部保持原样。同一来源horizon可通过C3影响其他短时输出，单来源24/48h替换不应改变另一个长horizon的C1输出。

每县起点t=72–215，s=t+h≤215才有效；四horizon正式评分行数固定为34,177 / 32,982 / 28,680 / 22,944。固定停电历史截止71，保留官方中心三小时N，不读取隐藏未来停电作为特征，不删尾部官方有效标签。

## 3. 评价和解释

- 完整六组合的C0/C1/C2/C3及主规则均保存；正式结论使用各horizon全县全行pooled OSI RMSE，另报告SSE、MAE、bias。
- N模型分别评价model_raw、component_clipped、aligned；记录截断率。对真实N分组：0、(0,1e-4]、(1e-4,1e-3]、(1e-3,1e-2]、>1e-2，判断改善幅度与背景虚报的取舍。
- 固定目标窗口73–76、77–95、96–143、144–215；补充共同目标小时120–215，避免混淆horizon覆盖差异。保留全239县和5外折指标。
- 各L2组合相对H0的四horizon主OSI，以及L2_all的N raw/clip/aligned，做县轨迹配对bootstrap：2,000次，seed20260910、95%百分位区间。每次抽样保留整县轨迹，以SSE/样本数计算RMSE。
- 使用已完成N诊断冻结的7县作为解释案例，不按本轮收益重新选县；不能只报改善县或按FIPS启用新模型。
- 不把四个RMSE的简单均值称为官方分数；官方按各horizon名次平均，平局优先1h。首轮结果只在seed42上，其他CV确认须按结果单独安排。

## 4. 数据、模型与文件身份

原文件仅只读：`versions/xyy/v1.5.6`冻结特征；strict目录的官方分量标签；`cv/cv_assignments_balanced_v1_seed42.csv`；原strict的 `runs/v18_tree_nested_v2_split42_model42`；当前F2的 `p_information_increment/runs/v18_pinfo_v1_split42_model42`。

当前F2 identity固定为 `e39b76d75f191891abda1939dc47933815cc4fa15796227b0c200eb1b96bfcac`。训练新数据和模型不从N诊断的oracle、匹配未来结果或逐行误差读取输入。

新目录布局：

```text
n_loss_l2_huber/
  Experiment_Protocol_2026-09-19.md
  experiment_config.json
  loss_protocol.py / train_n_loss.py / evaluate_n_loss.py / verify_n_loss.py
  summarize_results.py
  reference/                     # 输入哈希、批准范围与预检记录；无旧模型复制
  runs/v18_nloss_l2_v1_split42_model42/
    run_manifest.json
    models/outer0..4/N_t_target_t{01,06,24,48}h.{txt,json}
    logs/                        # 内层曲线、逐行内层预测和进度
    n_l2_oof_predictions.parquet
    component_predictions.parquet
    control_predictions.parquet
    aligned_unique_components.parquet
    metrics/
    MODEL_CV_COMPLETE / EVALUATION_COMPLETE / VERIFIED_COMPLETE
  summary/
  Results_2026-09-19.md
  README.md / final_integrity_check.json / completion.json
```

模型receipt记录允许折、训练/早停县和行键、训练特征/标签哈希、四次内层best_iteration、最终轮数、模型文件哈希和模型ID。逐行产物保留县/起点/目标小时、fold、scoreable、候选和来源模型身份，不能只保存汇总。

只允许相同代码、配置、来源身份的未完成run恢复；完整run不覆盖。所有写入必须在本目录，不能调用会改写旧目录的训练或验收入口。

## 5. 执行验收

训练前：核对源manifest/完成标记、环境和列序；复现完整F2；对20个N目标/外折的训练作用域做排除检查，替换外折标签不能改变任何允许训练子集；L2与Huber参数仅objective不同。

训练后：独立进程重载20个L2和20个Huber模型，重新预测各自外折并对齐保存值；核对内层标签float32存储对应的RMSE、曲线和best_iteration；用独立县/目标小时实现复算C1/C3、OSI和bootstrap，不依赖生产重建函数自证。

有效行预测须全部finite，尾部掩码一致；复算预测/指标绝对误差≤1e-12。P/D/R及直接OSI逐行不变，未替换的N来源不变，单来源传播符合固定C3/C1规则。输入文件前后哈希一致。

通过后汇报四horizon整体变化、N高值组变化、县/时段收益集中性、是否值得重复CV确认。任何改善仅为首轮开发OOF证据；本轮不会直接替换正式提交模型。
