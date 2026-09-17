# DEM v158 严格嵌套实验修改与验收规范

日期：2026-09-16。交付对象：空间模型贡献者 wendyxu。

本规范仅修改 `informs-data-mining-wendyxu/code_phase2_dem_eval_v158` 的后续实现；`code_phase2` 是历史版本，不纳入本次改造。本文中的新函数、测试和 CLI 是待实现的接口要求，不能当作当前已经存在的功能。

**交付目标：在固定数据、特征、图、县级分折和预注册超参数下，外层验证折 k 的标签不进入其预测所依赖的基模型、早停、残差训练、GAT、缩放或 alpha 选择；内层验证折 j 的标签不进入生成其 base/correction 的模型，只允许用于该层 alpha 选择。最终全训练集的参数选择不回写外层成绩。完整保存证据，并能在新进程中不训练、不读取监督标签地复现提交。**

该目标必须由实现和验收共同证明。完成本文描述但没有运行隔离测试，不能直接标记“无泄漏”。本规范不承诺模型提升，也不消除反复查看同一外层 OOF 后选模型造成的历史选择偏差。

本轮已完成只读数据核验，未修改模型代码、未训练、未安装依赖。机器可读核验记录见同目录 `DEM_v158_ReadOnly_Audit_2026-09-16.json`，包含受检代码和输入的 SHA-256。此前分析背景为 `Project_Assessment_2026-09-14.md` 第四部分及 `2026-09-15/Large_Error_Counties_Analysis.md`。下述行号对应本次受检代码，后续修改以函数名和内容定位。

## 1. 本次补查结果与训练前置条件

### 1.1 统一解释器

本机项目环境使用：

```text
/home/jacklo/XYY/INFORMS/informs-data-mining-2026/.venv/bin/python
```

可在 main 根目录激活该环境，再进入 wendyxu 根目录；也可始终显式调用上述解释器，避免误用默认 Python。不要把本机绝对路径写死在需跨机器交付的模型配置中。

本次实际导入版本：NumPy 2.5.3、pandas 3.0.5、PyArrow 25.0.1、fastparquet 2026.5.0、LightGBM 4.7.0、Shapely 2.1.2、scikit-learn 1.9.1、rasterio 1.5.1。

**该环境尚未安装 `torch`。** 因此当前数据加载和空间图构造可用，但 GAT 训练及完整 checkpoint 加载尚不能执行。贡献者应按训练机器的 CPU/CUDA 配置安装并锁定兼容的 PyTorch；安装完成后记录 Python、全部依赖、PyTorch/CUDA、设备、驱动及线程配置。不能仅以当前 `requirements.txt` 的 `>=` 下限作为复现环境。

已有 `county_terrain.csv` 时无需重算或重新下载 DEM；rasterio 用于地形构建，不是已有地形表训练路径的必要导入项。

### 1.2 本次已实际验证

| 对象 | 结果 |
|---|---|
| v1.5.6 五张 Parquet、v1.5.7 五张 Parquet、v1.8 分量目标 Parquet | 共 11 张；默认 PyArrow 与显式 fastparquet 均可读取 |
| v1.5.6 特征 | train `(34416,163)`，test `(9072,163)` |
| v1.5.7 特征，仅作历史补查 | train `(34416,211)`，test `(9072,211)`；不用于本次 v158 |
| 原始训练/测试 CSV | `(51624,63)` / `(13608,43)`；239 / 63 个县 |
| 缓存元数据 | 县时键唯一；每县 144 行，小时 72–215；全部缓存为从 0 开始的 RangeIndex |
| OSI 标签 | 与原始 CSV 同行目标及同县 `timestamp+h` 的 OSI 逐值一致，最大差 0 |
| 16 列 P/N/D/R 标签 | 表形状 `(34416,19)`；与同县未来时刻原始分量逐值一致，最大差 0；当前 loader 可读取 |
| 分量重组与官方 OSI | 最大差约 `5.001075e-5`，来自公开数据精度；校验容差用 `5.1e-5`，不能要求差为 0 |
| 固定五折 | 48 / 47 / 48 / 48 / 48 个县，完整覆盖 239 县 |
| 地形、坐标、空间图 | 地形覆盖 302 县；实际构图为 302 节点、3028 条有向边（含 302 自环）、4 维边特征 |
| 当前数据入口 | `load_cached_data()`、`load_cv_folds()`、`_load_v158_component_targets()` 实调成功 |
| 当前 GAT 输入构造 | 用真实特征、地形及图，base 以 0 占位，仅验证构造；实得 `(144,302,205)`，无非有限输入 |
| 历史四个 direct LightGBM 文本模型 | 新环境均可加载；163 列输入，轮数分别 206 / 275 / 237 / 127；未以此证明旧成绩或新版正确性 |
| 历史四个 GAT checkpoint | 静态读取 zip/pickle 操作码，首层 `gat1.linear.weight` 均为 `(96,209)`；未执行 pickle、未运行 Torch 推理 |

另对 302 县的 31 项观测摘要作了因果实测：重算并对比缓存全部 43,488 行，共 1,348,128 个单元格，`rtol=atol=1e-12` 下全部一致，最大差约 `6.22e-15`。将预测窗口停电、lag、target、delta 等列分别替换为大数及 NaN，重新预处理后摘要逐值不变。**该实测覆盖 31 项观测摘要，不能表述成全部 163 列均已通过扰动重建测试。**

训练有效行数与提交有效行数：

| Horizon | 全训练 OOF 有效行 | 63 县提交有效行 |
|---|---:|---:|
| t+1 | 34177 | 9009 |
| t+6 | 32982 | 8694 |
| t+24 | 28680 | 7560 |
| t+48 | 22944 | 6048 |

合法缺失值须保留语义：目标时刻天气超出 215 小时会缺失；测试县 42093 Montour 的 `hours_since_peak` 有 144 个 NaN，因为观测 OSI 按冻结口径重组并 round(4) 后没有正峰。不能要求原始特征缓存全有限。LightGBM 原生处理合法 NaN；GAT 对批准的原始特征 NaN 固定补 0；Inf、未声明缺失以及失败的基预测不得补 0 掩盖。

### 1.3 209 / 205 维究竟是什么问题

当前代码正确维度为：

```text
1 base prediction + 163 冻结特征 + 2 坐标 + 7 DEM + 32 邻县摘要 = 205
```

当前 `README_CN.md` 的维度段落也已写成 205。旧 `outputs/run_metadata.json` 仍为 209，且四个旧 checkpoint 的首层权重确实需要 209 个输入，与旧版多出的四列州 one-hot 一致。

因此，**不是当前代码还算出 209，也不是仅把文档中的数字改掉就能解决；这是历史模型/结果与已修改的 205 维代码之间的版本不匹配。** 不应补回四列州输入，不应裁剪旧权重或把旧 metadata 改成 205 冒充新结果。保留旧产物历史身份，新协议从头训练；加载器必须同时校验维数与有序特征 schema。205 维并不单独证明没有泄漏。

## 2. 冻结本轮实验边界

1. 协议命名为 `dem_v158_nested_v2`；与现有产物和缓存分开。这里的 v158 使用 **v1.5.6 的 163 列特征**，并支持 v1.8-compatible 的 direct C0 或 component C1 监督路线；不等同于完整 v1.8 的短期 C3 或 v1.12 融合。
2. 固定现有 `balanced_v1_seed42` 县折，不为改善结果重新分散高误差县。标签扰动检查也不得重建 severity 或分折。
3. 固定 k=8、现有共享县界图、hidden=24、heads=4、dropout=0.12、GAT MSE、epochs 上限 220、patience 35、time_stride=1。LightGBM 参数沿用当前配置，除种子和训练轮数按下文管理。
4. 固定 alpha 网格 `[0,0.05,0.10,0.20,0.35,0.50,0.75,1.0]`；只按 pooled RMSE 选择，相同值选更小 alpha。MAE 仅诊断。
5. 明确传入 `--base-mode direct` 或 `--base-mode component_v158`，每个模式独立 run。新协议不加载旧 component_v21 OOF 或最终 test 预测；旧别名应拒绝或显式映射后记录唯一的新模式身份，不得形成第三套旧产物路径。
6. 所有停电侧输入仅来自小时 0–71。官方提供的最后观测小时 N/R 可直接使用；不能额外读取原始小时 72 的停电来重新做中心平滑。天气可用全部公开小时。
7. 固定后处理 `P(x)=where(clip(x,0,0.65)<0.001,0,clip(x,0,0.65))`。组件预测先 clip `[0,1]`，按 `0.40P+0.35N+0.25D-0.10R` 重组后执行 P。OSI 残差和评价仍以官方 OSI 为监督目标，不能改用分量重组标签替代官方 OSI。

如据外层 OOF 在 direct/component、k、loss、seed 或阈值之间挑赢家，必须如实记录这是额外模型选择。每个固定方案的隔离成立，并不使反复选出的赢家成绩自动成为无偏独立评估。

## 3. 数据契约与唯一行键

### 3.1 使用完整冻结数据包

默认五表根目录固定为 `PROJECT_ROOT/versions/xyy/v1.5.6`。不要继续逐个文件判断 `cache` 存在就优先使用，避免五表混合来源。

允许 `--feature-dir` 显式指定完整替代数据包，但必须一次解析五表、feature_names 和 manifest，验证版本、schema、哈希与行键后才使用；不完整时失败，不能单文件回退。默认 Parquet 引擎固定为 pyarrow；本次环境已兼容，不需要静默异常回退。若支持 fastparquet，必须显式记录引擎并做数值、键和顺序等价检查。

监督与特征须分开存储。`features` 接口不得附带全部未来目标供下游任意访问。允许用于模型输入的 schema 使用冻结白名单，不能仅用原始禁止列名黑名单：`state_IN` 等改名或编码后的州标识同样不得加入。

### 3.2 行键和断言

- 规范化 `fipsCode` 为五位字符串，时间转换为一致的可比较时间值；稳定键为 `(fipsCode, timestamp_et)`，long format 再加 horizon。
- 验证 `timestamp_et = 2026-03-11 00:00 + hour_idx 小时`，保持题目时间标签，不额外变换时区。
- 五表必须来自同一冻结包，features/meta/targets 行数和索引一致；不依赖任意 DataFrame index label 作为位置。建图前重建明确的 `row_id=0..n-1`。
- 元数据与分量目标县时键必须唯一；所有连接使用 one-to-one/many-to-one 验证和显式 reindex，不能只检查覆盖。
- 原始、冻结与分量标签有效性都必须满足 `hour_idx+h<=215`；有效窗口内若目标缺失，失败，不允许默默缩小评分集合。
- 固定分折必须精确覆盖 239 县，state/severity 元数据与文件一致；这些字段只用于对齐和分组，不进入模型张量。
- train/test 县集合不交叉，每县小时为 72–215，分别 34416 / 9072 行。提交严格采用原模板的标识列和行顺序。

后续无训练验收应为完整特征构建提供纯函数入口：扰动小时 72–215 的停电/lag/target/delta 后重建特征，全部合法输入应不变；不得顺带写回冻结缓存。既有冻结包继续以本次哈希为基准。

## 4. 必须实现的标签隔离接口

记五折集合为 `F={0,1,2,3,4}`。下文 `S` 表示“该模型允许访问监督标签的县折集合”，不是仅指梯度拟合行。训练、早停、轮数汇总、缩放、停止规则和缓存都受此集合约束。

| 接口 | 可访问的监督标签 | 返回内容 |
|---|---|---|
| `fit_base(S,h,target)` | 仅 S 的该目标 | 固定轮数重训模型、轮数选择记录、依赖元数据 |
| `build_graph_inputs(S,h,mode)` | 仅 S 的监督，用于其训练残差 | 全图 features/base、S 监督 mask/residual、节点/行映射、schema、缩放参数 |
| `fit_stack(S,h,mode)` | 仅 S | GAT、图输入身份、未乘 alpha 的 base/correction 预测器 |
| `select_alpha(S,h,mode)` | S；每个内部预测器只接触 S 去掉被预测折 | alpha、各候选分数、配套 inner OOF |
| `evaluate_outer(k,...)` | k，仅用于最后评价 | 带逐行证据的外折分数，不回流训练接口 |

训练接口优先接收已按 S 裁切的监督对象；至少在开发/测试中使用会拒绝读取 S 外标签的访问器。不要将全标签张量传入后仅靠 loss mask 自觉隔离。对验证标签的读取统一留给评估器。

接口必须断言集合范围：`fit_base` 接受 2–5 折；`build_graph_inputs/fit_stack` 接受 3–5 折；`select_alpha` 接受 4–5 折。其余大小直接拒绝，不得隐式回退到单折模型或借用更大集合的轮数。

### 4.1 `fit_base(S,h,target)`

适用于 direct OSI 及四个组件，最小 S 大小为两折：

```python
def fit_base(S, h, target):
    S = canonical_fold_set(S)
    best_iterations = []
    for q in sorted(S):
        tr = valid_rows(S - {q}, h, target)
        va = valid_rows({q}, h, target)
        require_nonempty(tr, va)
        probe = fit_lgbm_with_early_stop(tr, va, scoped_seed(S, h, target, q))
        best_iterations.append(probe.best_iteration)

    rounds = max(1, int(np.mean(best_iterations)))
    model = fit_lgbm_fixed_rounds(valid_rows(S, h, target), rounds,
                                  scoped_seed(S, h, target, "refit"))
    return model, rounds, provenance(S, best_iterations)
```

关键要求：

- 参数中的任何早停折都属于 S。不能从全数据或较大集合继承轮数；那会使当前被排除折间接参与选择。
- `probe` 只用于选轮数。对外预测统一由全 S 固定轮数重训模型产生；不能把 `probe` 直接当作其早停折的无接触标签预测器。
- 两折集合分别用一折训练、一折早停，再在两折上重训，无须无限递归。记录两折探测的波动，不额外根据外层分数调整聚合规则。
- 目标、特征、随机种子、所有训练参数和允许标签集合都纳入模型身份。
- 每个模型为全图所有 144 时刻生成有限基预测，包括该 horizon 尾部不计分时刻。尾部用时间 mask 排除监督/评分，不能让“是否存在标签”决定节点特征是否生成。

### 4.2 `build_graph_inputs(S,h,mode)`

每个节点的基预测来源必须为：

- 节点属于 S 内的折 r：使用 `fit_base(S-{r},h,target)`；形成训练节点的交叉拟合基预测。
- 节点在 S 外：使用 `fit_base(S,h,target)`；适用于内层验证、外层验证以及 63 个真正测试县。

例如 outer=0、inner validation=1，此时 S={2,3,4}：

| 图节点 | 基模型使用的标签折 |
|---|---|
| 折 0、折 1、真正测试县 | {2,3,4} |
| 折 2 | {3,4} |
| 折 3 | {2,4} |
| 折 4 | {2,3} |

以上模型各自的轮数也只能在其表中集合内部选择。四组件必须全部遵守同一来源规则，再重组为一个 OSI base 通道。

对 S 内有效行建立 `residual=y_official-base`；S 外不读取真实 residual。初始化残差为 0 只是非监督占位，必须与强监督 mask 同时使用。交叉拟合只要求被预测折不参与其基预测器；S 内其他训练标签可用于该训练流程。

完整 schema 按顺序保存：base、冻结 163 列、latitude/longitude、七项 DEM、16 项邻居均值、对应 16 项 `neighbor_delta=neighbor-own`，总计 205。邻县摘要只从批准的 Phase-1 原始特征切片聚合，不把 base、真实 residual、标签、severity 纳入邻县统计。图结构、边顺序、节点顺序固定并哈希。

原 `make_county_time_view` 返回的独立 `base` 数组一直为 0，当前 runner 忽略它。新接口应删除此误导返回值，或保证它等于 `features[...,0]`，并有一致性断言。

### 4.3 标准化与缺失处理

为避免口径歧义，新协议固定如下顺序：

1. 先检查冻结 schema 与允许缺失 mask；拒绝 Inf 和未知缺失。
2. GAT 原始合法 NaN 固定补 0；base、DEM、坐标和 correction 不允许 NaN/Inf。LightGBM 保留其合法原生 NaN。
3. 构造全图 205 列后，仅在 **S 节点且 `hour_idx+h<=215`** 的单元格拟合列均值/标准差，`ddof=0`；std 小于 `1e-6` 则置 1。
4. 使用该 mean/std 变换全图；归一化后必须全有限。
5. 残差尺度只在 S 的有效监督行计算，`scale=max(std(residual,ddof=0),0.003)`。空监督集直接失败，取消 `0.01` 兜底。
6. GAT 拟合 `residual/scale`，推理 correction 乘回 scale，保存的逐行 correction 以 OSI 单位表示。

这是对现有“标准化使用训练节点全部 144 小时”的明确调整，必须写入协议版本。各层和最终训练执行同一规则。推理只加载已保存 mean/std/scale，不重新拟合。

## 5. alpha、外层评价与最终训练

### 5.1 内层必须重建整个两阶段模型

对外折 k，令 A=F-{k}。对每个 j∈A：

1. 调用 `fit_stack(A-{j},h,mode)`，使用独立构造的全图输入和残差。
2. 在 j 上同时取得 `inner_base` 和 `inner_correction`。
3. 写入 `inner_oof_base` 与 `inner_oof_correction`，同时记录生成它们的训练集合和模型身份。

禁止仅改变 GAT supervision mask 而复用外层 `fold_features/fold_residual_grid`；禁止新 correction 仍配旧 `fold_base_rows`。当前需要替换的位置是 `main.py:258–311` 的内层流程及其上游基预测构造。

为 A 的有效行先建立 expected_mask，要求配套 base/correction 全部有限且每行恰好产生一次。然后对每个候选 alpha 计算 `P(inner_base+alpha*inner_correction)` 的 pooled RMSE。所有候选必须使用同一个完整行集，禁止过滤失败预测后继续选择。得到 `alpha_k`。

调用 `fit_stack(A,h,mode)` 预测 k，应用固定的 `alpha_k`，最后才用 k 的官方标签评分。外层输出必须保存 alpha_k，不能使用最终全量 alpha 回填。

### 5.2 最终训练与内层流程同构

新协议明确采用 `alpha_final=select_alpha(F,h,mode)`，替换现有五个 alpha 取众数的汇总法。其内部预测器是五个 `fit_stack(F-{j})`，可复用刚完成的外层 stack 的原始 base/correction。

随后训练 `fit_stack(F,h,mode)`，以 alpha_final 生成测试提交。最终阶段允许使用全部训练县标签，这是正常的最终参数选择；**alpha_final 在用于选择它的 OOF 上所得最优分数不得冒充独立外层成绩**。

```python
for h in HORIZONS:
    for k in F:
        A = F - {k}
        alpha_k = select_alpha(A, h, mode)
        stack_k = get_or_fit_stack(A, h, mode)
        b, c = stack_k.predict_counties(fold=k)
        save_outer_rows(k, h, b, c, alpha_k, P(b + alpha_k * c))
    verify_and_score_outer_oof(h)  # 每行使用自己的 alpha_k

# CV 产物冻结后，进入最终阶段
for h in HORIZONS:
    alpha_final = select_alpha(F, h, mode)
    final_stack = get_or_fit_stack(F, h, mode)
    b, c = final_stack.predict_counties(test_counties)
    save_final_predictions(h, P(b + alpha_final * c))
```

`alpha_selection` 元数据区分 `outer_train_inner_cv` 与 `full_train_cv_for_final`。不要继续用含义模糊的一个布尔量宣称所有 alpha 从不使用任何 outer OOF 标签：最终阶段使用全训练内部 CV 标签是允许的，关键是不能改变已经冻结的外层评分。

## 6. 同时修复 GAT 训练与返回值的一致性

这些问题不等于外折标签泄漏，但应在新正式训练前修复：

1. 当前 `gat_model.py:121–127` 用更新前、带 dropout 的训练 loss 判断是否保存更新后的权重。改为每次 optimizer.step 后，在 `model.eval()`、`no_grad()` 下，对同一个 S 的训练 mask 计算确定性训练 MSE，再同时保存 `best_loss/best_epoch/state_dict`；下一 epoch 恢复 train 模式。
2. patience 仍只看 S 内训练损失。不得为了方便调用者而增加 inner/outer 验证标签参与 GAT early stopping。epochs/patience 本轮预注册固定，不增加搜索。
3. 当前 `fit_gat` 返回预测时未传 `block_attr`，而 `predict_gat` 会传。统一预测入口，保证返回预测与再次调用 `predict_gat` 在相同图和同设备上相等。
4. 验证 epochs≥1、patience≥1、time_stride≥1、1≤k<节点数；正式 run time_stride=1。loss、梯度、参数或预测出现非有限值立即失败，不能保存 COMPLETE。
5. seed 同时控制 Python、NumPy、PyTorch、LightGBM 所有随机字段；修复当前 `--seed` 与 config 固定 LightGBM seed 脱节的问题。保持线程数固定并记录。
6. CPU 小规模重复性验收启用确定性设置。CUDA 算子如不能满足确定性要求，应明确报告而不是静默忽略；只承诺已验证的同环境容差，不承诺跨硬件位级一致。

## 7. 缓存与运行隔离

规范化 S 使用排序整数元组。随机种子通过稳定摘要派生自 `(global_seed,stage,S,horizon,component,probe_fold)`；不使用进程随机化的 Python `hash()`，不依赖调用顺序或 outer/inner 的有序身份。

缓存 key 至少包含：协议/代码版本、mode、S、horizon/component、超参数、轮数聚合规则、种子、特征有序 schema、数据包/CV/目标/图哈希、预处理策略、依赖版本与影响结果的设备/线程设置。保存允许标签集合、每个 probe 的 train/early-stop 集合、最终训练集合。只比较梯度训练集合不足以证明缓存安全。

不加载当前历史 OOF/模型缓存进入新协议。标签扰动测试强制禁用缓存且清空进程内 memoization；否则缓存命中会掩盖标签依赖。

一个 S 同时服务不同外层/内层角色时可复用模型，但绝不能按“相关系数接近”或“列数相同”复用不同身份的缓存。写缓存使用临时文件加原子 rename，并仅在完整校验后写 COMPLETE 标识；resume 必须校验全部身份与哈希。

恢复粒度固定为已完整保存的集合模型。中断中的单次 fit 不从仅保存的 state_dict 继续，而应放弃其临时产物并从固定 seed 重跑；若以后支持 epoch 级恢复，必须另行保存 optimizer、全部 RNG、patience 计数和 best checkpoint 等完整状态。当前规范不要求实现这种额外恢复方式。

预算：每个标量目标需要训练集合大小 2/3/4/5 的 10/10/5/1 个最终 base 模型，共 26 个；每个集合内部选轮数再 refit，共 `10×3+10×4+5×5+1×6=101` 次 booster 拟合。四 horizon 的 direct 为 404 次，四组件为 1616 次。GAT 按 S 去重后每 horizon 为 10 个三折、5 个四折、1 个全量，共 16 个，四 horizon 合计 64 个。上述是单模式、单 seed 数量，不是耗时估计。

静态特征、地形、邻县摘要和图只构造一次或按 horizon 共用；缓存标签相关 base 数组和缩放，不重复保存大量相同的完整 205 维张量。探测模型可在记录其轮数与依赖后释放；最终 base 和各 scope GAT 要保存。

## 8. 路径、CLI 与文档同步

必须修复的路径：

- `OUTPUT_ROOT=Path(__file__).resolve().parent/'outputs'`，不再写 `code_phase2_dem_eval_v156/outputs`。
- 每次运行写 `outputs/runs/<run_id>/`，run_id 包含协议和模式；禁止直接覆盖现有旧 `outputs/*.csv` 或模型。
- 原始数据和 geo 仍使用相对 `PROJECT_ROOT` 的路径；五表按第 3 节成包解析；分量目标明确指向 `versions/xyy/v1.5.8/component_targets_v1.8.parquet`。
- 修正 README 的 `code_phase2_dem/main.py`、requirements、build_terrain 和输出路径；统一写 `code_phase2_dem_eval_v158`。
- `--run-id` 已存在且完整时拒绝写入；只有 `--resume` 且 manifest 完全一致才允许恢复。不同 mode、seed 或参数不得写入同一 run。

建议新增 `--stage preflight|cv|final|all`、`--run-id`、`--resume`、`--feature-dir`。正式运行固定五折，不提供能关闭隔离的快捷模式。preflight 必须在创建训练输出目录之前完成，不训练模型、不改缓存；缺依赖、数据不完整或 schema 不符时返回非零退出码。

CV 覆盖和计分复算通过后，写入不可变 `CV_COMPLETE` 及 outer_oof、inner_oof、外层 alpha 和 CV manifest 哈希。`--stage final` 只接受该状态且逐项核验哈希；全量阶段不得修改这些外层证据。最终训练、提交和独立重载验收全部完成后，才写整次运行的 `COMPLETE`。

README 和新 `REPORT.md` 从同一运行 manifest/逐行产物生成。旧 EVALUATION_REPORT_CN 保留历史说明，不把新配置写回旧成绩。新报告明确运行身份、模式、特征版本、协议、依赖和证据路径。

## 9. 每次正式运行的必需产物

以下是建议固定的交付目录；文件格式可用 Parquet/JSON/NPZ，但字段和可复核性要求不能省略：

```text
outputs/runs/<run_id>/
  run_manifest.json
  environment.json
  inputs_manifest.json
  feature_schema.json
  graph.npz
  folds.csv
  base_fit_manifest.parquet
  inner_oof.parquet
  outer_oof.parquet
  alpha_selection.parquet
  final_graph_base.parquet
  test_predictions.parquet
  cv_summary.csv
  fold_metrics.csv
  county_metrics.csv
  submission_phase2_dem_gat.csv
  submission_audit.json
  verification.json
  REPORT.md
  models/base/...txt
  models/gat/...pt
  CV_COMPLETE
  COMPLETE
```

### 9.1 逐行字段

`outer_oof.parquet` 以 `(fipsCode,timestamp_et,horizon)` 唯一，共 `34416×4=137664` 行，包含所有预测起点；其中有效行合计 118783。至少包含：

```text
run_id, protocol, base_mode, fipsCode, timestamp_et, hour_idx,
target_timestamp, horizon, outer_fold, is_scoreable,
y_true, base_prediction, correction_osi, alpha,
prediction_before_postprocess, prediction,
base_source_id, stack_id, inner_selection_id
```

外层无效目标行 `y_true/prediction` 为 NaN，`is_scoreable=false`；base/correction 仍可保存有限全图值，不能混入评分。组件模式额外保存四项分量预测与来源，便于 P/D 诊断。分数计算使用浮点原值，报告才舍入。

`inner_oof.parquet` 保存每个外折 k 的各内层验证行，包含 outer_fold、inner_fold、allowed_folds、base、correction、is_scoreable、y_true 和模型身份。其 `(outer_fold,inner_fold,fipsCode,timestamp_et,horizon)` 唯一；每个外折的训练县每个有效 horizon 恰好覆盖一次。

`alpha_selection.parquet` 区分 `outer_train_inner_cv` 和 `full_train_cv_for_final`，保存每个候选的 alpha、n、SSE、RMSE、MAE、selected 及输入预测身份。所有候选 n 相同且等于预期值。MAE 不参与选择。

`base_fit_manifest` 保存 mode/horizon/component/S、有效行数、各 probe 训练及早停折、best_iterations、final_rounds、种子、参数、模型路径与哈希；标签来源要可追踪到最终图每类节点。

`final_graph_base.parquet` 保存全部 302 县×144 起点×4 horizon，共 173952 行：县时键、horizon、base_prediction、base_source_id、节点所属 train/test、原始 county_fold；**不需要任何真实监督标签**。

`test_predictions.parquet` 保存 9072×4 行配套 base/correction/alpha/final，随后严格按官方模板生成宽表。

### 9.2 模型与独立推理

保存 26 个集合/标量目标的最终 base，以及全部各集合 GAT。分量模式每个 horizon 每个集合保存 P/N/D/R 四个模型；修复当前分量模型只存在内存而未 save_model 的问题。

GAT checkpoint 包含架构参数、输入/边维度、state_dict、feature schema 哈希、all_fips/node_order、time_order、mean/std/residual_scale、训练集合、seed、best_epoch/best_loss、协议和图身份。alpha 另存并关联其选择记录。模型结构和标准化数组必须与 schema 一致，禁止 `strict=False` 忽略旧维数不匹配。

**最终推理必须保留训练县节点的 cross-fit base。** 最终 GAT 中折 r 的训练县使用 `fit_base(F-{r})`，测试县使用 `fit_base(F)`。不能在重新加载时用全 F base 给全部 302 县统一预测，那会改变 GAT 的实际输入。

交付应同时提供 `final_graph_base` 及必要 base 模型以复核：独立预测入口只读取合法特征、图、保存的 base grid/模型、缩放和 GAT/alpha，不读取 OSI 或组件监督表，也不 fit 任何模型。新进程重载后逐行复现提交。

## 10. 严格评分与失败处理

当前 `metrics.py:9` 会过滤非有限预测，`main.py` 也以 correction 是否 finite 缩小评分 mask。新协议必须改为：

```python
expected = fold_mask & (hour_idx + h <= 215)
assert np.isfinite(y[expected]).all()
assert coverage_count[expected].eq(1).all()
assert np.isfinite(base[expected]).all()
assert np.isfinite(correction[expected]).all()
assert np.isfinite(alpha_by_row[expected]).all()
assert np.isfinite(pred[expected]).all()
score = pooled_metrics(y[expected], pred[expected])
```

原始合法缺失和失败预测必须区分。不得用 `nan_to_num`、减少行数、跳过折或返回 alpha=0 掩盖失败模型。alpha=0 是正常候选，不是异常兜底。

提交校验必须是硬断言而非只写布尔日志：9072 行；标识列逐值等于原模板；顺序一致；每 horizon 的有效/NaN 位置逐行一致；有效行数为 9009/8694/7560/6048；所有有效预测有限且位于 `[0,0.65]`。任一失败则无 COMPLETE。

baseline 与 GAT 用同一批 outer 行、同一后处理，baseline 就是该次 scope 的 base_prediction，不能拿旧缓存 baseline 代替。已有 v1.8/v1.12 数值可列为历史参考，不能把不同训练协议的差异直接归因为 GAT 的增益。

逐县 SSE、RMSE、外折方向和县级配对 bootstrap 均从 `outer_oof` 重算。bootstrap 以县为单位保留其所有小时，用 SSE 与样本数计算 pooled RMSE；固定 seed 和次数（建议 2000），不能按行独立重采样。空间相关仍是区间限制，应写入报告。

## 11. 必须通过的验收测试

实现新的 `tests/` 与 `verify_artifacts.py`，测试报告区分“只读/替代模型检查”“小规模真实训练检查”“完整训练后复算”。本文制定要求，本轮未执行任何真实训练测试。

### 11.1 不训练模型的检查

| ID | 操作 | 通过标准 |
|---|---|---|
| D1 | 导入依赖、读全部输入 | 包括 torch；环境已记录；读取不写缓存 |
| D2 | schema/键/范围/目标对齐 | 第 3 节所有断言通过；合法 NaN 与原始缺失模式匹配 |
| D3 | 205 维真实输入构造 | 每个 horizon 均恰好 205 列，名称/顺序固定；禁止字段或其编码不在白名单 |
| D4 | 全特征因果扰动 | 冻结天气/静态/前72h，替换未来停电及目标后，纯函数重建的合法输入不变；不写回数据 |
| S1 | 枚举所有 S 的替代 base | 记录拟合、早停、轮数的标签访问，全部在声明集合内 |
| S2 | outer k 标签替换 | 对应该外折的训练依赖、alpha_k、预测不变；评分可以变化 |
| S3 | inner j 标签替换 | 该次 inner pack/base/GAT 输入及预测不变；最终 alpha 可以变化 |
| S4 | cross-fit 预测折 r 标签替换 | `fit_base(S-{r})` 的轮数及对 r 的 base 不变；残差可随 y 改变 |
| S5 | 交换 outer/inner 身份 | 同一规范化 S、参数、seed 生成相同模型身份和输入 |
| F1 | 缺一行/重复一行/有效预测NaN/Inf | 立即失败，不能少算 n 或退回 alpha=0 |
| F2 | 打乱标签/元数据输入顺序 | 通过键重排恢复正确结果，或明确拒绝无法验证的未配套特征包 |
| F3 | 换特征顺序/旧209 checkpoint/错误缓存 | schema/身份校验拒绝，不允许静默加载 |

S2/S3/S4 关闭磁盘缓存和进程内缓存、冻结原始特征及 CV 文件，只改变监督数据。component 模式同时覆盖 OSI 和全部 P/N/D/R 标签；仅换 OSI 不足以检查组件路径。读取“禁止标签”的替代对象应主动抛错，不只在日志中声称未使用。

扰动验收比较模型数值参数、选轮数、缩放、预测和允许标签集合，不要求整个 manifest 字节相同。目标文件哈希会合法反映扰动，但它不得进入 scoped seed 或影响该集合的数值训练结果。

S2 只要求外折 k 对应的训练子流程不变；其他外折和最终全量模型合法使用 k 的标签，不应错误要求整个五折运行所有模型不变。S3 的 alpha 允许变化，因为 j 正是用于 alpha 选择的验证数据。

### 11.2 正式训练前的小规模真实模型检查

由贡献者在实现完成后执行，输出必须标为 diagnostic，不能当作正式成绩：

- 保留五折结构，每折预先按固定规则抽取少量县，保留四 horizon 和时间边界；用小轮数/epoch 验证上述隔离测试确实覆盖真实 LightGBM/GAT。
- 在独立进程、关闭缓存的条件下执行标签扰动；CPU 固定线程与随机种子，同一运行配置下预测差容差先固定为 `atol=1e-7, rtol=0`，若失败须定位非确定性或依赖，不能按测试结果不断放宽。
- 验证 `fit_gat` 返回预测与 `predict_gat` 一致，包含 edge_attr；best_loss 与 best checkpoint 的重算 train MSE 一致。
- 验证相同 S 的复用等价、训练中断恢复等价、缺预测硬失败、组件模型保存与重载。
- 所有诊断产物与正式缓存隔离，不能恢复到正式训练中。

### 11.3 完整训练后的复算

- 全部 outer/inner 覆盖、来源集合和 NaN mask 合格。
- 从保存 OOF 重算 summary/fold/county/alpha 选择，CPU float64 指标绝对误差不超过 `1e-12`。
- 新进程加载所有正式模型及输入身份，在无标签、禁止 fit 的独立推理入口复现提交；同设备最终 OSI 最大绝对误差不超过 `1e-7`。后处理导致阈值跳变也按最终预测检查，不能忽略。
- 新生成的所有 GAT first linear input_dim 为 205；旧209模型不参与运行。四组件 final boosters 和最终图训练节点的 cross-fit base 均可重载。
- 核验脚本退出码为 0，`verification.json` 全项通过，才生成 COMPLETE。

不应只依赖替代模型测试宣称整个神经网络实现已证明无泄漏，也不应只依赖一次训练分数“看起来正常”跳过依赖测试。

## 12. 贡献者的修改顺序与运行示例

按以下顺序提交可审查的改动：

1. 数据包解析、唯一键、205 schema、路径及 preflight；补环境 torch，固定依赖。
2. `fit_base(S)`、集合种子、来源记录、组件保存及替代模型隔离测试。
3. `build_graph_inputs(S)`、缩放、完整配套 base/correction、`fit_stack/select_alpha`。
4. GAT checkpoint/loss 对齐、edge_attr 返回一致、有限值失败处理。
5. OOF/模型/最终图输入保存、严格计分、独立推理和报告生成。
6. 无训练测试 → 小规模真实模型检查 → 完整外层 CV → 冻结 CV 结果 → 最终训练 → 独立重载复算。

以下命令是**实现上述目标 CLI 后**的示例，本轮没有运行。全部从 wendyxu 根目录执行，显式使用已安装环境：

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-wendyxu
/home/jacklo/XYY/INFORMS/informs-data-mining-2026/.venv/bin/python code_phase2_dem_eval_v158/main.py --stage preflight --base-mode component_v158
/home/jacklo/XYY/INFORMS/informs-data-mining-2026/.venv/bin/python code_phase2_dem_eval_v158/main.py --stage cv --base-mode component_v158 --run-id dem_v158_nested_v2_component_seed42 --seed 42 --device auto --epochs 220 --patience 35 --time-stride 1 --k 8
/home/jacklo/XYY/INFORMS/informs-data-mining-2026/.venv/bin/python code_phase2_dem_eval_v158/main.py --stage final --base-mode component_v158 --run-id dem_v158_nested_v2_component_seed42 --resume --seed 42 --device auto --epochs 220 --patience 35 --time-stride 1 --k 8
```

direct 模式使用独立 run_id；示例不意味着已经根据结果选定 component 为赢家。`--stage final` 必须校验并沿用已冻结的 CV manifest，允许读取其未乘 alpha 的配套预测进行最终 alpha 选择，不能改写 outer_oof。

贡献者应交回：代码 diff、新协议 README、完整运行目录、环境锁定信息、验收日志、逐行 OOF 与独立推理命令。只有第 11 节全部通过，才可将该次结果表述为“固定协议下通过标签隔离验收的嵌套县级实验”；不能以文档更新、维度变为205或旧训练产物存在代替这一结论。
