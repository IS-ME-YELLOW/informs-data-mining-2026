# 字段与作用域口径

`fipsCode`是五位字符串；`hour_idx`是预测原点t=72..215；`timestamp_et`与t对应；`split`为train/test；`fold`为对应CV中的0..4，测试县为−1。目标小时s=t+h，有效条件s≤215。

每套CV有26个完整模型作用域：所有2折组合10个、3折组合10个、4折组合5个、全5折1个。S012表示只能利用0/1/2折县未来监督标签训练及选择该基线。各scope是完整模型配方的训练范围，而不是拿普通OOF删几行。

## scope文件

- `eligible_prediction`：节点县不属于S（含测试县）。S内行的预测为NaN，必须通过其他scope获得其交叉拟合预测。
- `valid_01/06/24/48`：eligible_prediction且目标小时≤215。不同h有不同尾部缺失。
- `T_source_{P_t/N_t/D_t/R_t}_{01/06/24/48}`：T各独立来源的部署分量，已按各分量规则重建/裁剪到[0,1]，不是未经处理的原始树输出。D_G已经按K>0应用。
- `T_aligned_*`：同县/同目标时刻各有效h来源的分量平均。最终短时用它；最终24/48h不使用这组C3分量。
- `T_OSI_*`：树方案最终OSI。各分量先按.4P+.35N+.25D−.1R组合，再冻结后处理。
- `raw_P_a_01/raw_P_b_01/raw_P_a_06/raw_P_b_06`：对应树分支原始预测；初始p=0的a、p=1的b是无贡献分支，记录0。
- `raw_U_01`：D未知贡献原始预测；只在K>0的行实际进入D_G。
- `S_raw_{lgb/xgboost/catboost}_{component}_{24/48}`：Stella原始四分量树预测，不等于T的结构化P/D。
- `S_{family}_direct_{24/48}`：各家直接OSI经H后的预测；`S_{family}_component_*`：该家四分量先各clip[0,1]、加权组合后经H的OSI。
- `S_L0_*`：原LGB分量OSI锚点；`S_L1_*`：六路已处理OSI按冻结顺序平均后经H；`S_L2_*`：L0>theta时用L0，否则用L1。
- `S_theta_*`：scope内交叉拟合L0的正值q95（linear）。同一scope/h的标量；它不是由scope外预测或未来标签估计的。
- `I3_OSI_*`：Wendy训练残差应使用的完整最终候选base。

## reader输出

`source_scope`是该行**完整base配方**的真实训练范围。训练节点r∈S时它为S去掉r；其他节点为S。`supervised`是允许GAT监督的scope内有效训练行；`outer_or_inner_holdout`是scope外有效训练行，不包括测试县。

`base_prediction`保留真实有效窗口及NaN尾行。`base_graph_input`仅为图张量构造在无效尾行填0；不会增加有效评分行。`residual_supervision`返回原始OSI单位残差及掩码，掩码外的0是占位符，不能进入loss、残差标准差或任何模型选择指标。

全302节点有效县小时数：1h43186、6h41676、24h36240、48h28992。其中239训练县分别34177/32982/28680/22944；63测试县分别9009/8694/7560/6048。测试标签未知，不能计算测试RMSE。

## 身份与数值

包身份绑定配方、冻结数据、三套CV、生成代码和来源清单；每个模型身份绑定作用域和目标。复用模型保留原model_id，同时记录当前字节哈希与原收据。Stella部分LightGBM文本经Git换行归一化，已在内存重建Windows Python追加的pandas_categorical尾行CRLF并验证原哈希；原文件未被改写。

已有外层OOF复现使用绝对容差1e−12。保存模型重载与本包逐行原始预测要求完全一致。某些LightGBM探测的原生最佳轮验证缓存与训练结束后保存前缀的推理有可复现差异：审计保留二者并通过完整选轮流程重放分别核验，未修改模型、最佳轮数或预测；详见provenance中的模型验收说明。
