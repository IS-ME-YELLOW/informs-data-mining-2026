# DEM v158 剩余P1修复：独立产物验收

日期：2026-09-17。范围：上一轮剩余问题1–3，即alpha候选数量、FIPS类型及独立验收链路。沿用此前第1–6项补丁，不改分折、模型结构、损失、alpha网格或训练预算。

> 后续P2代码实现已完成，见 `DEM_v158_P2_Implementation_2026-09-17.md`；本文件末尾的待办保留为本阶段历史记录。真实模型运行结果仍待远程验收。

**三项P1已完成代码修复。本轮18项新增验收检查与11项既有回归检查均通过；未训练模型、未安装PyTorch/pytest，也未改动旧模型与旧结果。真实PyTorch产物的端到端验收仍需要远程执行。**

## 1. 修复内容

新增 `code_phase2_dem_eval_v158/artifact_checks.py`，将不需要Torch的数值复算与键/依赖校验集中到独立模块；重构 `verify_artifacts.py` 为“产物审核→禁止标签与训练的独立重载→生成通过标记”三步。

### P1-1：alpha候选计数与选择复算

- 外层候选数量固定为5折×4 horizons×8候选，共160条。
- 每个外折/horizon必须恰好包含完整网格；候选重复、缺失或额外值均失败。
- 从该外折对应的inner逐行base/correction和官方标签重新计算n、SSE、RMSE及MAE。
- 重新执行最小RMSE选择，同分选较小alpha，核对selected和每行外层alpha。
- 最终alpha单独从五个外层stack的原始base/correction池化计算，核对最终selected及test行的alpha；不会使用最终alpha改写外层成绩。

### P1-2：FIPS与行键

- 县码统一规范化为五位字符串；读取county CSV时显式指定dtype。
- 行表按完整 `(fipsCode,hour_idx,horizon)` 及需要的outer/inner fold集合核对；允许合法行重排，不允许缺行、重复、错县或多余行。
- 校验timestamp与hour_idx对应关系，标签与冻结官方目标逐值一致，scoreable来自时间契约。

### P1-3：数值、依赖与独立推理验收

1. 独立复算summary、fold、county指标以及固定seed的县级paired bootstrap，与保存表按键逐项比较，绝对容差 `1e-12`。
2. 核对inner的外折/内折、允许标签集合、模型编号和coverage；核对outer/test的base＋alpha×correction及后处理结果。
3. 核对每个base模型的完整scope清单、文件哈希、probe训练/早停折、有效行数、轮数汇总与种子；最终清单必须保留不可变CV记录。
4. 核对64个GAT checkpoint的完整scope集合、架构、seed、训练参数、schema、图、节点/时间顺序、缩放、有限权重，以及每类图节点的base来源。
5. 提交文件逐行核对全部标识列、值、顺序和精确NaN位置。生成提交的训练入口也使用同一严格检查，数量相同但位置错移不能通过。
6. final_graph_base必须保留训练县的cross-fit base及正确来源，并与相应outer/test base一致。
7. 新进程独立重载阶段不加载监督对象，并主动封锁项目的标签读取接口、已知监督文件读取路径、LightGBM/GAT训练入口。它重新核对全图base/source、测试base/correction、最终预测和提交CSV，重载预测容差 `1e-7`。
8. 任何一步失败均不生成新的COMPLETE；全部通过才写入明确的数值复算、依赖、重载、禁止标签和训练检查标记。

CV manifest的hash清单、run manifest身份、CV_COMPLETE及所有冻结产物字节也重新核验；不能靠一个空hash字典跳过冻结校验。

## 2. 配套产物修正

发现并修复了inner OOF的来源字段错误：原 `base_source_id` 保存的是GAT stack编号。现在 `AlphaSelection` 单独携带实际 `inner_base_source_ids`，写出真实LightGBM模型编号；`stack_id`仍单独记录GAT身份。

旧格式已经冻结的inner OOF不能直接冒充新格式产物。新正式run应由当前代码生成，不应编辑已冻结的CV证据来绕过新验收。

## 3. 测试与范围

使用项目解释器，运行以下两套标准库unittest测试，不需要本机pytest或PyTorch：

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-wendyxu
PYTHONDONTWRITEBYTECODE=1 /home/jacklo/XYY/INFORMS/informs-data-mining-2026/.venv/bin/python -B code_phase2_dem_eval_v158/tests/test_artifact_checks.py
PYTHONDONTWRITEBYTECODE=1 /home/jacklo/XYY/INFORMS/informs-data-mining-2026/.venv/bin/python -B code_phase2_dem_eval_v158/tests/test_repairs.py
```

新增18项覆盖：

- 正常direct/component逐行产物，合法行重排及FIPS整数/字符串混合来源。
- alpha缺失/重复/额外候选、分数错误、样本量错误、择优错误和同分规则。
- inner缺行/重复、错误官方标签、混入被留折的scope、base/stack来源混淆。
- outer/test错误alpha或预测关系、final graph错误值与错误cross-fit来源。
- summary/fold/county/bootstrap数值篡改，提交值、NaN位置和标识字段篡改。
- CV冻结文件或hash清单损坏、base probe包含被排除折、GAT图节点base来源错误。
- 独立重载实际经过禁止标签/训练的guard；错误提交不能通过；失败时不生成COMPLETE。

测试采用合成预测和临时产物，不拟合任何模型。涉及GAT checkpoint结构和独立重载的部分测试使用替代对象，不能表述为真实PyTorch反序列化/推理已通过。

另外实际执行了新审核器的真实数据引用加载：34416个训练起点、9072个测试起点，四个训练有效行数为34177/32982/28680/22944；冻结包、官方标签、CV、组件目标及地形的身份检查通过。

## 4. 尚未完成的工作

本轮解决的是已识别P1代码问题，不表示整个实验已经验收完成。下列原P2仍需继续：

- CV中断恢复与已完成集合模型的持久化复用。
- run恢复身份中的代码、完整参数和环境约束，避免覆盖原始环境记录。
- 关闭缓存的完整outer/inner/cross-fit标签扰动测试，以及远程真实GAT的重复性和端到端重载验收。
- 请求轮数与实际树数不一致时的LightGBM模型重载处理。

checkpoint依赖元数据核验能发现产物宣称的错误来源，但不能单独替代对实际训练数据流的标签扰动测试。完整验收仍须在远程安装PyTorch的环境执行 `verify_artifacts.py <run_dir>`；没有跳过真实重载就生成COMPLETE的选项。
