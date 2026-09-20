# P6 a/b迁移：另外两套CV确认（2026-09-19）

用户已授权执行seed20260917、seed20260918。只确认首轮AB_P06：P6改为a/b、原163列输入，其余P1h＋F2、D_G、N/R、P24/P48和组合规则固定。模型seed仍为42，不随CV seed变化。

全部新增文件限定在本目录：

`/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/p_structure_horizon_transfer/confirmation`

## 1. 参照与实验边界

每套CV使用自己的完整F2，不能复用seed42模型或仅替换fold列：

| CV seed | F2 run | identity |
|---|---|---|
| 20260917 | `v18_pinfo_v1_split20260917_model42` | `64f8263f5356378124c5e981fcb7c1117ad808c97b46c383b64b7cdbe109484d` |
| 20260918 | `v18_pinfo_v1_split20260918_model42` | `a3f16d3463442c01b1e1315df498317eed53298a1f89ef087b016ab8bd0c9532` |

这些run位于strict baseline的`p_information_increment/runs/`。特征来自原v1.5.6的163列；新P6不增加邻县特征。候选只有B0和AB_P06，不重训P1，不训练P24/P48，不做联合来源、混合权重、时间门控或县例外。

每套训练a/b×5外折，每外折4次内层probe＋1次完整四折refit，新增50次拟合、10个外层模型。两套共100次拟合、20个模型；加上已完成seed42的P6部分，三套P6累计150次拟合、30个模型。

## 2. 完全冻结的数值规则

严格沿用首轮协议：p=P71；a=min(y,p)/p（p>0）、b=max(y-p,0)/(1-p)（p<1）；原始贡献权重p²/(1-p)²，仅按该次训练支持集均值归一化；验证Dataset使用训练均值。早停指标为原始分支预测的P贡献RMSE，分母包含该验证折全部有效P行。四次best_iteration均值向下取整后refit。

分支支持与端点、float64重建、[0,1]截断、LightGBM L2参数、2000轮上限、100轮早停、单线程deterministic、全部模型seed42，均不更改。先筛允许折/有效小时/已知p支持集，再读取未来标签。外折标签不能参与训练、分箱或轮数选择。

P6覆盖目标小时78–215，预测起点t+h≤215。C3同县同目标小时平均；主输出1/6h用C3、24/48h用C1。最终24/48h必须逐行不变；输出1h目标小时73–77也必须不变。P1h/N/D/R及未替换来源始终沿用同CV F2。

## 3. 代码适配和来源检查

在本目录创建独立入口，父目录seed42代码、模型和报告只读。适配仅限CV选择、F2身份、来源路径、输出目录、允许候选集合和计数说明；对关键数学、训练及独立验证函数做AST一致性检查并保存适配记录。

所有入口显式要求`--split-seed 20260917`或`--split-seed 20260918`；禁止默认回退seed42。baseline loader必须显式使用本CV文件，独立verifier也遵守同样选择。模型随机种子不被CV seed覆盖。

每套先验证F2完成标记与哈希、行键/fold、p历史和163列一致，再做分支支持及外折标签扰动检查。恢复仅允许代码、配置、来源和CV身份一致的未完成run。

## 4. 固定评价与三套汇总

每套报告AB_P06−B0的四项pooled OSI RMSE/SSE/MAE/bias；来源P6的clip与C3后P误差；全部239县、五折、P71分组以及73–76、77–95、96–143、144–215时段，共同目标区间120–215。

县轨迹配对bootstrap仍为2000次、seed20260910；四个OSI输出加P6 clip/aligned，共每套6组区间。首轮P6“源P更准、C3后P略差但整体OSI改善”的现象须在两套中复核，并保留P与其他分量误差交叉项的拆解。

三套结果逐套报告，不能平均/拼接三套OOF构造额外候选。按同县同h汇总方向一致性，并沿用已冻结7县作解释。CV是同事件、同239县的重复开发分折，bootstrap不消除空间依赖或此前选择偏差。

若1/6h总体方向复现、24/48h不变，可支持P6作为后续优先组合；若出现退化或收益主要依赖少数不稳定县，明确说明。不能把完成确认直接视为已经训练了全量提交模型。

## 5. 目录与验收

```text
confirmation/
  Confirmation_Protocol_2026-09-19.md
  confirmation_runtime.py
  prepare_confirmation.py
  transfer_protocol.py / train_transfer.py / evaluate_transfer.py / verify_transfer.py
  summarize_confirmation.py
  reference/                    # 父文件哈希、适配审计
  split20260917/
    experiment_config.json
    reference/
    runs/v18_ptransfer_ab_v1_split20260917_model42/
    summary/
  split20260918/                 # 同结构
  summary/                      # 三CV汇总，包含只读seed42结果
  Results_2026-09-19.md
  README.md / final_integrity_check.json / completion.json
```

每套独立进程重载10个新模型、核对40个probe的标签/权重/曲线和轮数；独立重建a/b、C3及OSI，复算40组控制指标、6组区间及全部县/折/时段指标。预测/指标绝对容差保持1e-12，掩码和来源不变量保持严格。原始输入及父目录完成清单前后字节不变。

正式执行为每个seed依次运行preflight、train、evaluate、verify；两套完成后统一汇总。本轮没有其他模型线训练、特征搜索、外部下载或提交操作。
