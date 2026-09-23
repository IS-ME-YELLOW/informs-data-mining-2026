# I3完整基线预测包：Wendy入口

这是冻结I3的基线预测与依赖包，不含已训练GAT、不含比赛提交。不要把普通五折OOF直接喂给GAT。本包已提供正确作用域的预测和读取器。

解压后保留目录结构。只需要NumPy/Pandas/PyArrow；读取预测不需要LightGBM/XGBoost/CatBoost或本地绝对路径。首先运行：

```bash
python -B verify_package.py
```

该命令只读文件、核对哈希和作用域，不训练或改文件。`PACKAGE_COMPLETE.json`只说明这个预测包经过验收，不代表你的GAT已经训练/验收。

接收端接口兼容Python 3.10的标准库API；Parquet按Arrow物理schema读取，忽略生成端pandas 3的专属dtype元数据，避免与pandas 2.3的差异。实际接收环境仍以verify_package.py通过为准。

## 固定配方

T=NB_P24：P1 a/b+F2邻县183列；P6 a/b163列；P24直接P+邻县183列；P48原严格P。D1仅K>0时用K+max(U,0)重建并裁剪；其他D和全部N/R维持原严格树。

T1/T6使用同县、同目标小时四horizon来源分量平均C3；T24/T48是本horizon四分量组合。S=Stella严格L2，三家模型各自直接OSI和分量OSI共六路、正值LGB锚点q95门控。

最终I3：1/6h=T；24h=H((T24+S24)/2)；48h=S48。H裁剪[0,.65]后严格小于.001置零。门控L0>theta用L0，否则用六路均值L1。所有配方与现有外层结果对齐，不重新选择权重/阈值。

## 最小读取示例

```python
from prediction_reader import PredictionPackage
import numpy as np

package = PredictionPackage()  # 默认本文件所在的解压目录
# 外层留出fold0，GAT允许监督训练的县来自1/2/3/4折：
g = package.graph_context(42, (1, 2, 3, 4), 1)
residual, train_mask = package.residual_supervision(42, (1, 2, 3, 4), 1)
X = package.model_features('base')  # 冻结的163列，顺序与g一致
```

`g.base_prediction`是最终I3的原尺度OSI；训练残差=官方OSI−base。`g.base_graph_input`只把无效尾行填0供图输入，不能拿来评分。`train_mask`只包含允许scope内的训练县和有效目标时刻；图里其他县及测试节点不监督。

GAT最终预测应为H(I3_base+alpha×原尺度残差校正)，每个horizon独立。I3_base已经完成C3或Stella组合，不能对GAT修正再做一次C3，也不能从T四分量重算一个不同base却称为I3。

**必须按fipsCode和hour_idx显式重排到你的图节点顺序**。包内行序是训练县块后接测试县块，每块按县/小时排序；不要假定等于你当前302节点图的顺序。`timestamp_et`为官方无时区本地小时标签，不额外转换时区。

可用`aligned = package.align_context(g, your_meta.fipsCode, your_meta.hour_idx)`按你的顺序或训练/测试子集重排。`aligned.package_row_index`给出包内原始行号，用于同步重排`X.iloc[...]`、residual和train_mask；不要把带34416起始索引的测试Series直接赋给0起始索引的DataFrame。该接口会拒绝缺失键或重复键。

## 内层alpha、外层评分、最终训练

三套CV seed为42/20260917/20260918，与模型seed42是不同概念。四horizon可分别读取和训练。

三套预测分别保存，不做跨CV平均。每套的全训练模型也使用该套内部选轮/门控规则；GAT必须配套使用同一CV的分折与预测，不能混用不同seed的scope。

接入时必须调用`package.assert_cv_assignment(split_seed, your_train_fips, your_row_folds)`，确认你现有代码没有继续使用默认seed42分折。CV seed选择分折，GAT的模型随机seed单独配置；不能把两者当成同一个参数。

- 外层留出k：允许scope S=其余4折。用`graph_context(seed,S,h)`训练GAT；只在k折有效行评价外层结果。
- 为该外层选择alpha，内层留出j∈S：用`graph_context(seed,S\{j},h)`训练独立GAT，只取j折预测；汇总所有j的内层结果选择alpha。不能用外层四折GAT的预测冒充内层预测。
- 最终训练：用S=(0,1,2,3,4)训练GAT；最终alpha由四折GAT的内层OOF确定。完整最终base已在本包，测试未来标签仍全部缺失。
- `graph_context`会自动给属于S中r折的节点使用完整B(S\{r})，其他节点使用B(S)。该完整B包含所有C3来源、Stella模型和门控依赖，不能只换最外层base文件。
- 所有标准化均值/方差、残差尺度、epoch/alpha或其他选择，只能使用各自允许的scope。外层评分标签不得影响它们。`labels_for_rows`会拒绝scope外、测试县及无效尾行标签请求。

## 需要改你的哪些接口

现有`BaseModelStore.scope_predictions()`自行训练/读取旧树。新增一个外部预测provider，把上述`graph_context`提供的base按原行键映射到`make_county_time_view()`。保留原GAT残差尺度、图构建和所选结构的语义，训练前核对本包的base逐行相等。

`StackContext._base_dependencies()`也需改为记录本包身份、scope文件哈希和完整recipe依赖；不要继续假设一个base只对应4个旧分量模型。GAT checkpoint应绑定包的identity_hash、CV seed、scope、horizon和新特征schema。不能借用旧base或旧GAT检查点的身份。

接入后的第一项实数检查是强制alpha=0：外层所有有效行必须复现本包I3，pooled RMSE应匹配`provenance/frozen_I3_scores.csv`。如果仍得到旧component_v158基线成绩，说明旧base路径尚未完全替换，不能开始解释GAT增益。

本包提供全部分量供核验和后续研究，但**不意味着这次自动把全部预测追加为GAT输入列**。若首轮只考察新基线，建议保持你冻结的GAT输入定义，只替换base通道。追加分量、改变邻县摘要或删除特征是独立实验因素，应显式记录。

## 文件

- `inputs/`：302节点行键、163/183列冻结特征、三套分折、官方监督标签、已知历史K。labels与输入分开，测试未来标签全NaN。
- `inputs/cv_seed*.csv`保留原分折审计字段，其中severity_tier只用于解释既有分折，绝不能作为GAT输入；读取器只提取fold，`model_features()`只返回已冻结特征白名单。
- `splits/seed*/scopes/S*.parquet`：B(S)对S外县和测试县的全套预测；S内行NaN，避免误当OOF。每套26个scope。
- `splits/seed*/anchors/S*.parquet`：单原始折LGB锚点，仅服务最内层Stella q95。不是完整I3来源。
- 相邻JSON：模型身份、门控依赖、阈值和文件哈希。
- `reference/`：三套既有外层T、Stella、I3结果，仅用于复现/评分比较，不直接用于GAT训练残差。
- `provenance/`：训练来源与独立验收摘要。模型权重及全部探测检查点留在生成端供审计，Wendy训练无需携带它们。
- `provenance/generator_code/`是生成与验收源码快照，供审阅作用域和计算逻辑；它不是接收端训练入口，运行入口仍是根目录的verify_package.py和prediction_reader.py。

完整字段解释见`SCHEMA_CN.md`。本包没有GAT收益结论；你的GAT/MLP训练和独立验收完成后，才可以比较新增收益。

严格作用域排除保证本次训练依赖隔离，不消除此前反复观察开发集OOF选择模型的选择偏差。三套CV是同一事件重分折，不是三个独立事件。
