# DEM v158 P2实现与远程验收交接

日期：2026-09-17。本轮按照已确认的顺序实现“运行身份→请求/实际轮数→模型级恢复→完整测试入口”。保持现有五折、特征、模型结构及正式训练预算。

**四项P2的实现已完成；真实模型运行验收仍待wendy在远程执行。** 本机未训练LightGBM/GAT、未安装PyTorch或pytest，未更改原始数据、折分或旧模型结果。

详细远程步骤位于：`informs-data-mining-wendyxu/code_phase2_dem_eval_v158/REMOTE_ACCEPTANCE_CN.md`。新脚本为 `validate_runtime.py`。

## 已实现

1. 不可变运行身份：包括核心源代码哈希、实际配置、数据/CV/地形/地理文件/模板哈希、依赖和设备设置。恢复时给出具体不一致字段；静态记录验证后保留原字节，启动事件写attempts.jsonl。随机种子不依赖身份或标签哈希。
2. LightGBM轮数：final_rounds表示内部选择的请求轮数，actual_rounds来自实际Booster。保存和预测使用实际轮数，文件重载核验实际结构，允许合法的actual<requested。原有probe轮数和来源记录保留。
3. 模型级恢复：每个base/GAT模型先写临时文件并验证，发布模型后最后提交完成JSON。CV结束前也能复用已完成模型；孤立模型文件重建，已完成模型损坏报错。GAT记录绑定上游base文件哈希。CV/最终总清单仍分开且CV不可变。
4. 调度：all+resume按CV_COMPLETE/FINAL_READY继续下一阶段；完成阶段不重写。中断中的单个模型从固定seed重跑，不做epoch级恢复。
5. 本地数据流测试：算术替代函数实际经过fit_base、BaseModelStore、StackContext、图消息输入和alpha选择，检查外层/内层/cross-fit隔离与正向对照，关闭跨实验缓存。
6. 远程诊断：固定每折少量县，实际使用正式训练函数；每个原始/扰动/恢复/重载案例独立进程执行。包含两模式、四horizon、五外折位置，以及短树模型与篡改拒绝案例。诊断目录禁止位于正式outputs/runs。

## 本地检查

- 既有11项repair回归通过。
- 既有18项artifact数值/拒绝错误产物检查通过；新增懒加载Torch后，独立推理guard相关检查再次通过。
- P2累计15项检查通过，包括数据流隔离、记录提交点、禁止重训已完成模型、身份拒绝、静态文件不可变、请求/实际轮数、缺少真实依赖时不能报告验收成功、正式预检仍要求真实Torch导入、分horizon图schema，以及确保四个组件监督都实际改变的扰动构造。
- 使用validate_runtime的mock后端检查独立进程执行路径：连续运行、外/内层/cross-fit扰动、正向对照、注入中断、恢复、禁止标签读取的重载及短树边界。mock只能报告替代函数检查通过，不能报告真实模型通过。
- Python语法与git diff --check通过。

模型级恢复检查比较原始与恢复后的完成记录/模型字节，并在恢复过程中拦截针对已完成模型的训练调用。标签扰动比较模型、轮数、缩放、base/correction和外层alpha；数据身份哈希允许变化，但不得改变scoped seed。

复审补充修复：四个horizon的邻县统计列名不同，`feature_schema.json`现在分别保存 `graph_schemas[horizon]`，checkpoint按自己的horizon核验。原先共用t+1 schema会错误拒绝t+6/t+24/t+48模型；该问题已加入回归检查。

## wendy必须完成的远程部分

请按REMOTE_ACCEPTANCE_CN依次执行：环境导入与NumPy/Torch互操作、preflight、三套无训练回归、真实CPU快速诊断、真实CPU完整诊断；正式使用CUDA时再做目标GPU补充诊断。正式模型训练后另跑verify_artifacts。

完整CPU诊断报告必须满足：backend=real、passed=true、full_coverage=true、real_acceptance_passed=true。缺依赖、任一案例失败或仅运行子集都不能当作完整真实验收。报告记录核心及诊断源代码哈希、输入哈希、环境、预算与已通过案例。

回传diagnostic_report.json、失败案例log（如有）、rounds/rounds.json、一组prepared.json与resume_calls.json，以及正式run最终verification.json/COMPLETE。只有收到真实运行结果后，才能关闭“真实模型验收执行”这一项；代码及替代函数测试不能替代它。

旧的无身份/无完成记录run不作自动迁移；需新run_id。任何代码/参数/环境变化后的继续训练都须通过身份比较，不能手工修改manifest绕过。修改README或tests不改变正式模型代码身份。
