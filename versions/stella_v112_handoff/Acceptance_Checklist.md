# 迁移验收清单

本清单验收的是“实验定义是否被正确执行”。低RMSE不能替代依赖验证，包哈希通过也不能证明模型已通过真实验收。

## A. 开训前交付

- `Protocol_Alignment.md`：逐项列出规格条目、新实现文件/函数、测试/产物、偏差状态。明确哪些复用、哪些新增；不得只写“参考了严格协议”。
- `resolved_config.json`：六路顺序、163列及dtype、三family参数、CPU/线程、round caps、patience、floor规则、模型seed42、CV路径/hash、q=.95及linear、后处理、预测和bootstrap规则。
- `dependency_plan.json`：枚举5外折、每个目标的probe/refit、三折anchor去重scope、阈值来源、全部预测县。证明没有 q 标签路径进入 q 预测；三折anchor额外排除 r。
- 文件清单检查结果，输入的行数/县数/键/列顺序/标签时域结果，以及当前环境、模块路径。既有文件不可变性检查。
- 无训练检查实际调用训练入口的依赖规划分支；不能只是手工生成一份看似正确的JSON。
- 提交准确的拟执行seed42命令、预计820次新fit/180模型＋50个引用模型、输出目录、恢复策略及停止条件。当前尚未提供已实现的训练命令，不得把示意CLI写成已经存在的入口。

完成这些后汇报，正式拟合待负责人授权。普通实现细节自行处理；修改核心实验定义或缺少必需输入时报告差异，不自选替代实验。

## B. 防泄漏检查

数据预检可读取原始标签核实一致性；**进入训练消费者后**，作用域外标签不得影响任何决策。

1. 所有fit通过唯一的scope接口取得数据。对每次probe/refit记录训练折、早停折、被预测折、县名单、行数、排序键hash、target、列hash、seed和参数。
2. 用spy/sentinel覆盖所有作用域，验证fit和早停收到的县/行恰等于计划。测试必须接入真实生产调用链，不能只验证独立的伪代码。
3. 在预检后的内存对象中毒化外折q标签，强制fresh路径：q的预测、选轮和theta数值应不变；在单个(q,r)内毒化q/r标签，该T生成r的anchor及选轮应不变。变更评分真值后分数可以变化，全局run hash也可能变化，不要求二者不变。
4. 不能改磁盘冻结标签来做该测试——那只会触发输入一致性拒绝；不能复用缓存预测来获得虚假的“完全相同”。无训练阶段至少完成scope拦截测试；正式授权后安排每family最小真实fresh-fit隔离检查，独立保存scope/轮数/预测对比，诊断轮数上限明确标为缩小版，不能声称它等同正式模型性能。
5. q95覆盖：每个(q,h)拼接S内四折的有效行，键无重复无遗漏，且每行四分量模型都排除q及该行所在r。theta的输入集合与拟合来源逐行可追溯。

这里的隔离结论以固定CV划分为条件。三套CV与候选经过项目研究选择，不能把整条研究过程宣称为完全未接触数据的最终测试。

## C. 必须保存的逐行产物

所有逐行预测表保存 `fipsCode, timestamp_et, hour_idx, fold, split_id`，以及horizon、run/dependency identity；可用规范long格式或明确宽格式。阈值、轮数和回执等聚合表使用`outer_fold/horizon/model_id/probe_id`等对应的自然键。引用模型保留原model_id，并记录本次dependency边，不重新冒充新训练。

| 产物 | 至少包含 |
|---|---|
| outer原始预测 | 六路对应的raw direct或raw四分量、source model IDs、外折 |
| inner anchor预测 | outer context q、被预测折r、四个raw与clip分量、composed/processed anchor、四个model IDs、scope T |
| threshold表 | outer_fold、horizon、quantile_probability=.95、method=linear、有效/正预测数、theta、inner预测hash、来源清单 |
| 候选逐行表 | 六路processed OSI、L0/L1/L2、theta、`anchor > theta`标志、固定scoreable、truth（只作评分） |
| 轮数和模型回执 | 每次probe原始best index及轮数、requested/actual rounds、params、列、scope、训练/早停行hash、文件hash、完成状态 |
| 指标和解释 | pooled/县/折/目标日、bootstrap、六路误差相关、预测门控分组及事后真实尾部诊断 |

没有内层逐行anchor或依赖不完整时，即使有theta JSON也不能验收q95严格性。

## D. 独立新进程验收

- 来源模型：读取原回执、原run身份和完整性记录；确认原文件未变。为便于这一步，传递严格基线完整run（虽只用50个长期模型），而不是只拿一个汇总CSV。
- 所有100个新外层模型、80个新inner模型以及50个引用模型从磁盘重载。重新计算raw → clip/compose → 六路均值 → inner q95 → gate → L0/L1/L2，逐行与保存值比较。不要复用训练进程中的内存model或只读保存的metrics。
- raw预测/处理后预测/指标默认`atol=1e-12, rtol=0`；键、scope、mask、列序、整数轮数必须精确一致。同一文件的固定复制短期预测要求数组精确一致（NaN位置相同）。不得为了PASS静默放宽容差。
- 必须证明 L0长期等于同split严格C1，且L0/L1/L2的1/6h都精确等于同split固定F2。不能只比较最终RMSE相同。
- 有效行完全覆盖：1/6/24/48分别34177/32982/28680/22944；评分mask不能依赖候选是否产出。NaN尾部须一致，真实零值不被丢弃。
- 验证实际轮数/目标/列序/参数，特别是XGB/Cat的+1和refit无早停；XGB重载用固定refit全部树，不能套probe的best_iteration截断范围。
- 校验 `protocol_examples.json` 的等于阈值、零、clip顺序、mean后post、无正预测报错及尾部边界。模型数、fit数与dry-run预算逐项对账，失败/重试/diagnostic独立计数。
- 重新计算所有指标与县级bootstrap。新增 `CV_COMPLETE` 仅在独立验证通过后写入；完成记录覆盖代码/输入/回执/模型/预测/指标/验证报告哈希，不能仅以日志出现“done”为准。

F2短期只做固定产物身份/字节/行值检查，报告中写明它是已有验证对照；本轮没有重新训练或重新独立验收F2完整模型链。

## E. 最终回传给负责人的材料

最少回传：`Protocol_Alignment.md`、本地resolved config与环境、依赖计划、输入/代码hash、fit/round/model清单、全部逐行outer与inner预测、threshold表、metrics、独立verification、完成记录以及Results报告。模型文件也应可取得，以支持我们复核重载。

Results至少先回答：L2相对L0在24/48h是否改善；1/6h是否精确不变；L1和L2差异是否来自尾部保护；改善是否依赖少数县或单折；不确定性多大；哪些验证是包检查/算术/小型诊断，哪些是真实正式模型验收。

不得只回传“训练成功，RMSE下降x%”。不得将seed42探索分数当测试集成绩，或在未经下一阶段授权时自动跑其他分折/最终训练。
