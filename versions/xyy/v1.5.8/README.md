# v1.8 树基线：v18_tree_nested_v2

当前代码保留 v1.5.6 的163列特征、P/N/D/R公式和既有后处理，将树数选择移入外层训练县内部。外层验证县不参与对应模型的梯度训练或早停。默认命令只进行只读预检，不训练、不生成标签或提交。

本目录原有 `models/`、OOF、CSV、`Report_v1.8.md` 和旧运行记录是历史实验，保持原样。新实验全部写入 `runs/<run-id>/`，不从旧模型恢复。旧模型数值已在9月17日只读复现，但旧模型文件哈希清单失配；新协议使用独立的完整身份校验。

## 环境和入口

本机使用项目 `.venv`；以下命令从 `informs-data-mining-2026` 根目录执行：

```bash
.venv/bin/python -B versions/xyy/v1.5.8/train_v18.py
.venv/bin/python -B versions/xyy/v1.5.8/train_v18.py --stage preflight --cv-file cv/cv_assignments_balanced_v1_seed20260917.csv
```

输入相对路径按项目根解析，因此也可以从其他cwd用解释器和脚本的绝对路径调用。`--validate-only` 是只读preflight的兼容别名。`requirements.txt` 固定了本次验证使用的库版本；不依赖torch、GAT或旧 `_compat/scipy`。

preflight检查：完整五表和163列schema、县时键、合法缺失、官方小时71组件、测试停电遮蔽、OSI/16组件目标对齐、CSV分层均衡和模板mask。不会逐文件混用不同特征包，也不会在输入缺失时自动重建数据。

## 先运行原seed42的严格CV

```bash
.venv/bin/python -B versions/xyy/v1.5.8/train_v18.py \
  --stage cv \
  --cv-file cv/cv_assignments_balanced_v1_seed42.csv \
  --model-seed 42 \
  --run-id v18_tree_nested_v2_split42_model42
```

CV阶段只生成100个外层模型及OOF，不训练最终20个提交模型。每个标量目标在每个外层训练集合内做4次早停探测，再固定轮数重训该集合：20目标×5外折×5次拟合，共500次LightGBM拟合。

部分CV中断可加 `--resume` 继续同一命令。已完整保存的模型按其receipt复核后复用；未提交receipt的中断拟合从固定seed重跑。不同分折、输入位置/内容、代码、参数或环境不能混用。CV完成后有 `CV_COMPLETE`，不得以CV阶段重跑覆盖。

```bash
.venv/bin/python -B versions/xyy/v1.5.8/verify_artifacts.py \
  versions/xyy/v1.5.8/runs/v18_tree_nested_v2_split42_model42
```

独立verifier默认只读。它重新加载模型，验证标签允许集合与每次早停记录，并重算预测、C3来源、指标、bootstrap和模板；不依据旧verification的passed字段判断成功。`--write-report` 只更新指定新run的验证记录。

## 最终阶段单独执行

在候选确认后才运行：

```bash
.venv/bin/python -B versions/xyy/v1.5.8/train_v18.py \
  --stage final --resume \
  --cv-file cv/cv_assignments_balanced_v1_seed42.csv \
  --model-seed 42 \
  --run-id v18_tree_nested_v2_split42_model42
```

最终阶段要求同run的CV_COMPLETE且所有哈希一致。每目标在全部训练县内部做5次早停探测，汇总树数后重训239县，共120次拟合，保存20个最终模型。它不改写外层OOF或CV模型清单。提交经过重载校验后生成 `FINAL_READY` 和 `COMPLETE`。

`--stage all` 显式运行CV和final两个阶段，但当前研究计划先执行CV。`--max-rounds` 或 `--early-stopping-rounds` 偏离冻结预算必须显式传 `--diagnostic`，此类run始终标为diagnostic，不能作为正式成绩。

## 三套固定分折

使用现有seed42，以及确认用seed20260917、20260918。split seed仅决定县分配，LightGBM的model seed仍固定42。不同分折使用不同run-id，基线和候选在各自相同分折内作比较，两个确认结果都报告。

## 固定控制与评分

- C0：直接OSI LightGBM。
- C1：四分量分别预测、clip到[0,1]、按公式重组。
- C2：直接预测与分量重组的固定等权平均。
- C3：每个分量对同县同目标小时的horizon预测等权平均，再重组OSI。
- `v18_rule`：1/6h取C3，24/48h取C1，规则不在新OOF上重新选择。

所有OSI输出沿用[0,0.65]裁剪和<0.001置零。数据预定的有效行必须全部有有限预测，不能删除失败预测后降低n；缺少C3候选、重复县时键或同县跨折都会失败。

这是固定模型/特征/规则下的嵌套县级验证。它不消除历史模型选择带来的不确定性，重复划分也不是独立风暴。新严格CV结果不能与原协议历史分数直接相减后归因于模型改进。

## 新run的主要产物

`run_manifest.json`、`input_manifest.json`、`environment.json`、CV副本；`model_manifest_cv/final.json` 和每模型receipt；`round_selection_cv/final.csv`；OOF/测试控制与分量预测；带模型ID的原始预测及C3对齐来源；逐折、逐县、逐时段和分量诊断、县级bootstrap；五套最终提交；验证记录与阶段标记。

模型身份包括CV与输入内容哈希、允许训练县、内层早停县、target、seed、参数、实际树数、schema、代码和依赖。不能仅因旧文件名的fold号相同而复用模型。完整数值复核容差为1e-12。

## 开发验收

```bash
.venv/bin/python -B versions/xyy/v1.5.8/tests/test_nested.py
```

测试包含：真实三套分折只读preflight；原C1/C3数值一致；错误CV/缺预测拒绝；20目标的外折标签扰动和真实LightGBM隔离；小型合成数据的完整CV、恢复、final、独立重载与篡改拒绝。合成拟合每模型最多2轮，产物仅放临时目录，不训练比赛数据。

正式训练安排见 `review/2026-09-17/v18_preconditions/V18_Training_Plan_2026-09-17.md`。后续候选应使用独立实现/协议和run身份，保持本严格基线的代码版本可恢复，不要就地修改后直接resume旧run。
