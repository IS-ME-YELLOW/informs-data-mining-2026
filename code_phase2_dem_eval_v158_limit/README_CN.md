# code_phase2_dem_eval_v158_limit

## 目录定位

本目录实现的是 dem_v158_nested_v2 的“无邻县摘要”GAT 输入变体。它使用
v1.5.6 的冻结 Phase-1 特征和 v1.5.8 的分量目标路线，但不等同于原来的
205 维 GAT 实验，也不读取原 205/209 维模型、OOF 或旧 component_v21
产物。

当前代码中的身份字段为：

~~~
protocol             = dem_v158_nested_v2
experiment_version   = v1.5.8
feature_version      = v1.5.6
graph_schema_version = direct_gat_no_neighbor_summary_v1
graph_input_dim      = 173
~~~

PROTOCOL 为兼容现有训练框架仍保持 dem_v158_nested_v2；173 维变体由
graph_schema_version、输入 schema 哈希和独立 run_id 区分。正式报告中必须
明确写出这是无邻县摘要的 173 维变体，不能把它的结果表述为原 205 维协议的
结果。

当前仓库只提供可运行代码和验收接口，尚无本目录生成的正式完整训练结果。
没有 COMPLETE、独立重载验收和逐行 OOF 证据时，不应宣称实验已经完成或
通过全部标签隔离验收。

## 1. 数据来源与题目边界

默认数据位置相对于项目根目录解析：

~~~
versions/xyy/v1.5.6/
  features_train_v1.5.6.parquet
  features_test_v1.5.6.parquet
  meta_train_v1.5.6.parquet
  meta_test_v1.5.6.parquet
  targets_train_v1.5.6.parquet
  feature_names_v1.5.6.json
  manifest.json 或 manifest_v1.5.6.json

versions/xyy/v1.5.8/component_targets_v1.8.parquet
cv/cv_assignments_balanced_v1_seed42.csv
data/geo/county_terrain.csv
data/geo/c_16ap26.dbf
data/geo/c_16ap26.shp
data/sample_submission.csv
~~~

五张 v1.5.6 Parquet、特征名称文件和 manifest 必须作为同一个完整冻结包
解析。默认引擎为 PyArrow；使用 --parquet-engine fastparquet 时，代码会
额外进行 PyArrow 等价检查。不能因为某一个缓存文件存在就混合读取不同来源
的表。

固定数据契约如下：

- 239 个训练县、63 个测试县；训练和测试县集合不交叉。
- 每县 144 行，hour_idx=72..215；时间标签保持题目定义的 timestamp_et。
- 停电侧输入只能使用小时 0..71；天气特征可使用公开的全部小时。
- 超出 hour_idx+horizon > 215 的目标保留为合法 NaN，并从相应评分集合
  排除。合法缺失不能被当成失败预测。
- fipsCode、时间、州标识、severity_tier 等只用于键对齐、分折、分组和
  证据记录，不进入模型张量。州标识不能通过改名、编码或 one-hot 重新加入。
- LightGBM 保留批准的原生 NaN；GAT 对批准的原始特征 NaN 固定补 0。Inf、
  未声明缺失和失败的 base/correction 预测直接报错，不能用 0 掩盖。

## 2. 173 维 GAT 输入

当前 GAT 的有序输入为：

~~~
1  base prediction
+ 163 冻结的 v1.5.6 Phase-1 特征
+ 2  坐标：latitude, longitude
+ 7  DEM/地形特征
--------------------------------
= 173 维
~~~

7 个地形字段为：

~~~
elevation_mean_m
elevation_std_m
elevation_min_m
elevation_max_m
slope_mean_deg
slope_std_deg
terrain_ruggedness
~~~

本变体明确不包含以下 32 列：

~~~
16 个 neighbor_mean_* + 16 个 neighbor_delta_*
~~~

因此，邻县统计量不再作为节点输入的显式重复特征。空间关系仍然通过 GAT
的图消息传播建模：图固定为 302 个县节点、3028 条有向边（包括自环），
边特征为 4 维。删除邻县摘要并不等于删除图，也不等于取消邻县消息传播。

make_county_time_view() 必须生成形状 [144, 302, 173] 的有限输入，且
返回的 base 通道必须与 features[..., 0] 完全一致。所有输入名称和顺序由
冻结白名单、feature_names_v1.5.6.json、坐标和地形字段共同确定；不能通过
只检查维数来加载旧 checkpoint。旧 205 维和旧 209 维 GAT checkpoint 必须被
拒绝。

## 3. 两阶段模型与标签隔离

五折集合记为 F={0,1,2,3,4}。所有允许读取监督标签的接口都以规范化集合
S 为边界；训练、早停、轮数聚合、标准化、残差尺度和缓存身份均不能访问
S 之外的标签。

### 3.1 LightGBM base

对目标 horizon 和目标列调用 fit_base(S,h,target)：

1. 对每个 q in S，用 S-{q} 训练 probe，并只在 q 上 early stopping。
2. 汇总这些 probe 的 best iteration，取固定的平均轮数。
3. 使用 S 的全部有效训练行，以该固定轮数重新拟合最终 base 模型。
4. 记录 S、probe 训练折、probe 早停折、轮数、种子、参数和模型身份。

所有 base 模型都会为全图的 144 个时刻生成有限预测；horizon 尾部只通过
时间 mask 排除监督和评分，不通过标签是否存在来决定是否生成节点输入。

### 3.2 图输入和 residual

对 build_graph_inputs(S,h,mode)：

- S 内折 r 的节点使用 fit_base(S-{r},h,target)，形成 cross-fit
  base prediction；
- S 外节点，包括外层验证县和测试县，使用 fit_base(S,h,target)；
- 仅对 S 内有效行读取官方目标并计算 residual = y_official - base；
  S 外不读取真实 residual；
- GAT 的监督 mask 只覆盖 S 内且 hour_idx+horizon<=215 的行；
- 标准化均值/标准差只从 S 节点的有效监督时刻拟合，std<1e-6 时置 1；
- residual scale 只从 S 的有效 residual 计算，使用
  max(std(residual, ddof=0), 0.003)。

component 模式分别对 P_t/N_t/D_t/R_t 执行同样的 base 和图隔离流程，
先将各分量预测 clip 到 [0,1]，再按

~~~
0.40 P + 0.35 N + 0.25 D - 0.10 R
~~~

重组为 OSI base。OSI 残差和最终评价仍使用官方 OSI 目标，不使用分量重组
标签替代官方 OSI 监督目标。

GAT 结构参数固定为：

~~~
k=8                 hidden=24
heads=4             dropout=0.12
loss=MSE             epochs 上限=220
patience=35         time_stride=1
~~~

每次 optimizer 更新后，以 eval() 和无梯度模式在同一个 S 训练 mask 上
计算确定性 MSE，依据该损失保存 best checkpoint；不使用 inner/outer 验证
标签做 GAT early stopping。训练 loss、梯度、参数或预测出现非有限值时立即
失败。

## 4. 嵌套外层评价和 alpha

固定候选网格为：

~~~
[0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]
~~~

后处理函数固定为：先将结果 clip 到 [0,0.65]，再把小于 0.001 的值
置零。baseline 和 GAT 预测必须使用同一后处理和同一批外层评分行。

对外折 k，令 A=F-{k}。alpha 选择必须重新构建整个 inner stack：

1. 对每个 j in A，调用 fit_stack(A-{j},h,mode)；
2. 在 j 上同时得到配套的 inner base 和 correction；
3. 用同一完整行集计算每个 alpha 的 pooled RMSE；
4. 取 RMSE 最小者，同分取更小 alpha；
5. 用 fit_stack(A,h,mode) 预测外折 k，应用该 alpha_k，最后才读取
   k 的官方标签评分。

不能只改变 GAT supervision mask 后复用另一套 base/residual，也不能用最终
全量 alpha 回填 outer OOF。最终阶段使用
alpha_final=select_alpha(F,h,mode) 的全训练内部 CV；该分数只能作为最终
参数选择证据，不能冒充独立 outer 评价。

## 5. 严格评分和提交约束

训练 OOF 的预期有效行数为：

| horizon | 239 县 OOF 有效行 |
|---|---:|
| t+1 | 34177 |
| t+6 | 32982 |
| t+24 | 28680 |
| t+48 | 22944 |

测试提交使用 63 个县、9072 行模板，预期有效行数为：

| horizon | 有效提交行 |
|---|---:|
| t+1 | 9009 |
| t+6 | 8694 |
| t+24 | 7560 |
| t+48 | 6048 |

评分前必须断言：预期行的真实值、base、correction、alpha 和最终预测全部
有限，且每行恰好覆盖一次。不能过滤非有限预测来缩小评分集合，不能跳过折，
不能把 alpha=0 当作失败兜底。

提交生成还必须硬断言：

- 9072 行，标识列和原模板逐值、逐行顺序一致；
- 每个 horizon 的有效/NaN 位置与模板规则一致；
- 有效预测全部有限且位于 [0,0.65]；
- baseline、GAT、外层 OOF 和逐县指标来自当前 run，不读取旧成绩。

县级 bootstrap 按县抽样并保留该县所有有效小时；它是报告诊断，不参与模型
或 alpha 选择。空间相关性仍然限制其置信区间的解释。

## 6. 运行命令

从项目根目录运行。远程 GPU 环境中将 --device 设为 cuda；只有 CUDA、
PyTorch 和相关依赖通过 preflight 后才可以正式训练。

### 6.1 预检查

~~~bash
python -u code_phase2_dem_eval_v158_limit/main.py \
  --stage preflight --base-mode component_v158 --device cuda
~~~

preflight 只读取和验证输入，不训练模型、不创建正式 run 目录、不写训练
缓存。若环境没有 CUDA，应使用 --device cpu；不能用 auto 掩盖期望的
运行设备。

### 6.2 外层 CV

~~~bash
python -u code_phase2_dem_eval_v158_limit/main.py \
  --stage cv \
  --base-mode component_v158 \
  --run-id dem_v158_nested_v2_no_neighbor_summary_component_seed42_cuda \
  --seed 42 --device cuda --epochs 220 --patience 35 --time-stride 1 --k 8
~~~

direct 路线使用独立的 --run-id 和 --base-mode direct，不能与 component
路线共用一个目录。CV 成功并通过覆盖和计分检查后，代码写入不可变的
CV_COMPLETE。

### 6.3 最终训练和测试预测

~~~bash
python -u code_phase2_dem_eval_v158_limit/main.py \
  --stage final \
  --base-mode component_v158 \
  --run-id dem_v158_nested_v2_no_neighbor_summary_component_seed42_cuda \
  --resume --seed 42 --device cuda --epochs 220 --patience 35 --time-stride 1 --k 8
~~~

final 只接受身份完全一致且已经存在 CV_COMPLETE 的 run；它不得修改
outer OOF、外层 alpha 或 CV manifest。--stage all 可以按同一身份连续执行
CV 和 final；中断恢复只能加载带完整完成记录的模型，不能从半途的
state_dict 恢复优化器状态。

可使用完整替代特征包：

~~~bash
python -u code_phase2_dem_eval_v158_limit/main.py \
  --stage preflight --feature-dir <完整v1.5.6特征包目录> \
  --base-mode component_v158 --parquet-engine pyarrow
~~~

替代目录必须一次性包含五张表、特征名称文件和 manifest，并通过版本、schema、
哈希、行键、缺失模式和顺序检查；不支持单文件回退。

## 7. 运行产物

产物根目录为：

~~~
code_phase2_dem_eval_v158_limit/outputs/runs/<run_id>/
~~~

正式 run 至少应包含：

~~~
run_manifest.json
environment.json
inputs_manifest.json
feature_schema.json
graph.npz
folds.csv
attempts.jsonl
base_fit_manifest.parquet
base_fit_manifest_final.parquet
inner_oof.parquet
outer_oof.parquet
alpha_selection.parquet
cv_summary.csv
fold_metrics.csv
county_metrics.csv
county_bootstrap.csv
cv_manifest.json
CV_COMPLETE
final_graph_base.parquet
test_predictions.parquet
alpha_selection_final.parquet
submission_phase2_dem_gat.csv
submission_audit.json
verification.json
REPORT.md
FINAL_READY
models/base/...
models/gat/...
~~~

独立 verifier 成功后才会新增 COMPLETE。其中：

- outer_oof.parquet 保存所有训练县、四个 horizon 的逐行外层预测和来源；
- inner_oof.parquet 保存 alpha 选择所需的 inner 预测、允许折集合和模型身份；
- final_graph_base.parquet 保存最终 GAT 的全图 base 输入，必须保留训练县的
  cross-fit base 来源；
- test_predictions.parquet 保存测试县的 base、correction、alpha 和最终预测；
- component 模式还保存四个分量的 base prediction 及其来源；
- GAT checkpoint 必须包含 173 维架构、schema 哈希、节点/时间顺序、图身份、
  mean/std、residual scale、训练集合、seed、best epoch/loss 及上游 base 身份。

## 8. 独立验收

在最终训练进程之外启动新进程执行：

~~~bash
python -u code_phase2_dem_eval_v158_limit/verify_artifacts.py \
  code_phase2_dem_eval_v158_limit/outputs/runs/<run_id>
~~~

验收包括输入和 schema、折覆盖、逐行 OOF、alpha 候选、指标重算、提交模板、
checkpoint 维度和身份、最终图 base、无标签独立重载以及禁止训练调用。只有
以下条件同时成立才可以把 run 视为完整结果：

1. CV_COMPLETE 存在且 CV manifest 和所有冻结证据哈希一致；
2. FINAL_READY 的最终产物完整；
3. verifier 在新进程中重载模型并逐行复现提交；
4. verification.json 全部检查通过并生成 COMPLETE。

本目录没有正式训练结果时，不能用旧 outputs、旧 205/209 checkpoint、
历史报告、替代模型或仅通过静态检查来代替上述验收。

## 9. 当前变体的解释边界

删除显式邻县摘要只是改变节点输入 schema，并不保证 GAT 一定优于 LightGBM
base。它用于检验“邻县摘要特征”和“图消息传播”之间的重复信息是否影响残差
修复效果。若要比较 173 维与 205 维，必须使用独立 run、相同数据包、相同
固定折、相同后处理和预注册的比较方案；不能把两个版本的 OOF 或模型缓存
交叉使用，也不能依据外层 OOF 反复挑选赢家后仍称为无偏独立评价。

