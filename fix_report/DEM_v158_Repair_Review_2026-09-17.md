# DEM v158 第1–6项修复与复审

日期：2026-09-17。范围：`informs-data-mining-wendyxu/code_phase2_dem_eval_v158`，对应上一轮审查的第1–6项。第7项 verifier 的其他错误、恢复流程及完整实验验收仍保留为下一步工作。

> 后续更新：下文是第1–6项补丁完成时的阶段记录。剩余P1中的R1/R2/R3已在后续补丁中完成代码修复与无训练回归检查，见同目录 `DEM_v158_P1_Verification_Update_2026-09-17.md`。P2恢复/身份与真实模型隔离验收仍未完成。

**第1–6项已修复，并通过11项无训练回归检查；当前仍不应标记为完整实验验收通过，也不建议直接投入正式长时间训练。** 全程未训练 LightGBM/GAT、未安装依赖、未改原始数据/Parquet/历史模型与旧结果。测试输出仅位于自动清理的临时目录。

## 1. 已完成修复

| 原问题 | 本次实现 | 验证 |
|---|---|---|
| 1. feature_names 哈希因换行不符 | 更新 `versions/xyy/v1.5.6/manifest.json` 为当前 LF 字节的真实 SHA-256；根目录 `.gitattributes` 固定该 JSON 和 manifest 的 LF 检出规则。继续严格校验哈希，不跳过校验 | 默认完整数据包与标签加载成功；所有六个登记文件哈希通过 |
| 2a. pandas 3 返回不可写数组 | 缺失值 mask 明确 `to_numpy(copy=True)` | 使用真实无正峰测试县的 NaN 输入验证，固定补0且原输入不变 |
| 2b. train/test 图映射错误 | 验证两种行映射互斥、联合覆盖全图、节点属于正确 split、每个源行恰好出现一次；同时拒绝县集合交叉及非法时间 | 正常混合图通过，缺行/重复/交叉输入被拒绝；全302县实际图通过 |
| 3. OOF 错用测试有效行数 | 新增 `TRAIN_SCOREABLE_ROWS`；CV与verifier使用训练计数，并按 hour_idx+h 验证 scoreable mask；提交继续使用测试计数 | 实际执行 CV 汇总/保存路径（模型和逐行输出生产由替代函数提供），137664行 OOF 汇总完成并产生临时 CV_COMPLETE |
| 4. 包身份口径不一致 | 基模型、GAT、manifest及上下文统一完整 `feature_package_hash`，另存并校验 `feature_side_hash` | 正确身份通过；错误完整包或特征侧身份被拒绝；上下文向两个模型传递同一身份 |
| 5. horizon 文件名解析错误 | `osi_t01h_S0-1.txt` 正确恢复为 `osi_target_t01h`；同时核验文件名与manifest中的 component/horizon/scope | direct和四组件、四horizon的保存/重载往返通过；旧LightGBM权重仅作为文本格式fixture，重载预测逐值相同 |
| 6. 最终模型未登记 | 保留不可变 `base_fit_manifest.parquet`；全量阶段另写完整 `base_fit_manifest_final.parquet`。重载优先完整final清单，否则使用CV清单；保留probe/轮数记录 | direct清单100→104、component清单400→416；CV清单字节不变；final清单与所有CV记录逐值一致 |

模型加载仅接受清单登记且哈希正确的完整模型。未登记的中断残留文件不被当作模型加载，后续有监督运行可以重建它们；登记文件缺失则失败。清单全部验证成功后才整体加入内存，避免半加载状态。

本轮改动涉及 `config.py`、`data.py`、`base_model.py`、`main.py`、`stacking.py`、`verify_artifacts.py`，以及冻结包 manifest、`.gitattributes`、两份 README 和新增回归测试。`verify_artifacts.py` 中仅修复与第3/4/6项直接相连的计数、身份及final清单引用，未顺带解决第7项全部问题。

修复前没有完整包/特征侧双身份字段的模型清单不应直接恢复到本轮代码中。应使用新的 run_id，不能手动修改旧清单以绕过身份检查。

## 2. 本次已执行的检查

解释器：`/home/jacklo/XYY/INFORMS/informs-data-mining-2026/.venv/bin/python`。

从 wendyxu 根目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/jacklo/XYY/INFORMS/informs-data-mining-2026/.venv/bin/python -B code_phase2_dem_eval_v158/tests/test_repairs.py
```

结果：**11 tests，全部通过**。该文件使用标准库 unittest，可直接执行，不需要安装 pytest 或 PyTorch。

具体覆盖：完整数据包及官方标签；pandas缺失mask；图节点的联合与互斥覆盖；缺行/重复/交叉拒绝；真实CV汇总代码在合成预测下完成输出；完整包与特征侧身份一致；direct/component清单数量与往返；CV冻结；未登记文件不加载与已登记文件缺失失败；模型篡改失败；四个真实历史LightGBM文本模型重载后预测相等。

补充实测：

- 真实冻结特征：train `(34416,163)`，test `(9072,163)`；固定五折可读取。
- 16列分量标签重组与官方OSI仍在 `5.1e-5` 容差内，最大差约 `5.001075e-5`。
- 全302县图含3028条有向边；四个horizon的图特征均为 `(144,302,205)`，全部有限。此处base为0占位，仅检查输入构造，不验证模型效果。
- 图中train/test行映射分别完整覆盖34416和9072个源行。
- 所有Python源文件AST解析及 `git diff --check` 通过。

模型清单大范围往返使用不学习标签的序列化替代对象；真实LightGBM检查只加载、保存、预测已有权重，不训练。它们验证文件协议，不能作为新嵌套模型性能或全部标签隔离的证据。

## 3. 剩余未修复问题

### R1 / P1：verifier 的 alpha 候选数量仍写错

位置：`verify_artifacts.py:94`。

代码仍要求 `5*4*8*4=640` 条外层alpha候选，实际每个外折和horizon池化一次内层预测，正确数量是 `5*4*8=160`。它后面的分组检查又要求每个 `(outer_fold,horizon)` 恰好8条，前后规则不一致。正常结果仍会被拒绝，无法完成验收。

修复要求：按外折×horizon×候选网格定义完整键集合；不仅改总数，还检查每组包含且仅包含完整网格、只有一个选中值、候选使用相同且正确的行数。

### R2 / P1：县级指标CSV的FIPS类型与OOF不一致

位置：`verify_artifacts.py:77` 及后续merge。

CSV默认读取把县码变为整数，Parquet OOF中县码为字符串；当前pandas会拒绝str/int64的merge。上一轮已用对应dtype最小复现；本轮仍未改动该路径。

修复要求：读取时固定dtype，并在所有键连接前统一为五位字符串。增加正常CSV/Parquet互相核验的回归测试。

### R3 / P1：独立验收仍不足以支持规范中的“通过”结论

位置：`verify_artifacts.py` 的 `_check_run()`、`_independent_reload()`。

当前有结构、部分哈希、模型数量和重载预测检查，但缺少以下关键核验：

1. 从outer逐行预测重新计算并核对 `cv_summary`、fold/county指标与bootstrap；目前部分文件仅检查存在或键数。
2. 从inner逐行base/correction重新计算每个alpha候选的SSE/RMSE，验证真正按最小RMSE、同分最小alpha选择；仅检查selected数量不足。
3. inner每个外折的完整县时覆盖、允许标签集合、每行base/stack来源以及无外折标签依赖，尚未形成完整验收。
4. 独立重载只和 `test_predictions.parquet` 比较，没有把重载预测按稳定键逐行核对到最终提交CSV。提交CSV只检查范围和有效数量，数值错配仍可能未被发现。
5. 最终提交NaN位置应逐行由模板/时间契约验证，不能只检查总有效数；final_graph_base的实际值与来源也需要和重载图对应。
6. 独立推理入口虽使用无监督对象，但尚无自动化检查主动禁止读取标签和调用fit；需要在测试中验证该边界。

修复要求：让 `COMPLETE` 表示规范要求的数值复算和无标签独立推理均通过，而不是仅完成现有结构检查。第1–6项补丁不声称已经完成此工作。

### R4 / P2：CV恢复仍忽略 `--resume`，中断复用未完成

位置：`main.py:885`，以及基模型清单在CV末尾才发布的流程。

上下文仍以 `resume=args.stage == "final"` 创建。因此 `--stage cv --resume` 或 `--stage all --resume` 不会加载已经完成的集合模型，而是重新训练。即使简单改为传递args.resume，首次CV中途没有完整base manifest时也无法正确恢复。

修复要求：按已完成集合原子发布模型身份/完成状态，恢复时只复用完整且一致的集合；中断中的单次fit从固定seed重跑。新回归测试覆盖清单往返，但不是实际中断续跑等价测试。

### R5 / P2：运行身份没有完整绑定代码、参数和环境

位置：`main.py:295` 的run identity、`main.py:383` 的静态产物写入、`stacking.py` 的checkpoint恢复。

当前run identity覆盖部分CLI参数和数据，但没有源代码哈希、完整LightGBM/GAT配置以及依赖版本/设备细节。恢复时重新写environment等静态文件，可能覆盖原始环境记录。GAT checkpoint保存seed等字段，但加载核验还未完整逐项对应预注册配置。

修复要求：把影响结果的完整配置、代码及运行环境纳入不可变身份；恢复只能在明确兼容规则下进行，原始记录不得被覆盖。尤其不能让改过代码或超参数的模型混入同一个scope缓存。

### R6 / P2：完整标签隔离和真实GAT验收仍缺少证据

现有新增11项测试针对本轮六项工程错误，不覆盖规范中的所有S1–S5。仍需完成：

- 外层k标签变动后，对应该外折的模型、alpha_k和预测不变。
- 内层j标签变动后，该次inner模型、缩放及base/correction不变；alpha允许变化。
- 残差交叉拟合折r的标签不影响预测r的基模型及其选轮数。
- component路径同时覆盖OSI与P/N/D/R监督；测试关闭缓存，固定原始特征和分折。
- 相同scope复用、真实GAT checkpoint损失/预测一致、诊断级重复性和完整独立重载。

这些应在远程训练环境补齐。缺少本机PyTorch不是代码缺陷，也不要求现在安装；完整GAT验收需要远程PyTorch环境，运行现有pytest套件时再准备pytest。

### R7 / P2（条件性边界）：请求轮数与实际树数仍被视为必须相等

位置：`base_model.py:435`。

加载器把 `model.current_iteration()` 与请求的 `final_rounds` 严格比较。如果LightGBM在固定轮数refit中因无法继续分裂而实际产生更少树，合法保存模型会被拒绝重载。当前11项回归使用实际树数等于请求轮数的fixture；本轮没有训练来触发该条件，因此这里只列为需要补测的代码边界，不声称已在正式模型上出现。

建议分别保存请求轮数与实际树数，按保存模型的实际结构验证完整性，并保持预测语义。此项是复审新增的条件性问题，不是第5项horizon解析错误的回归。

## 4. 后续顺序

先完成R1/R2和R3的独立验收链路，再完善R4/R5的恢复身份与R6的标签隔离测试；R7用小规模固定轮数案例补测。随后在远程环境执行preflight、诊断级真实模型验收，全部通过后再正式CV和最终训练。

本轮没有提交Git commit，也没有更改已有评估报告或模型结果；本文件记录当前补丁边界，不能作为正式训练的验收结果。
