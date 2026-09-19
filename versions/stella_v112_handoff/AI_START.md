# 给 Stella 的 AI：可复制的首轮提示词

将下面整段发给对方AI；同时让它能访问本目录和`source_inventory.json`列出的实际文件。不要只贴提示词而省略数据、对照产物和代码。

---

你正在参与INFORMS县级停电预测项目。请在本机仓库中，将Stella v1.12的24h/48h六模型等权＋q95尾部保护方案，迁移到我们已有的严格县级嵌套CV协议。

本轮授权：阅读、实现新实验代码、运行无训练预检和scope/sentinel检查；不得调用`lightgbm.train`、`xgboost.train`或任何真实estimator的`.fit`。**暂不正式训练，不启动真实小模型诊断、不推理测试集、不生成提交；完成实现和预检后汇报供负责人确认。** 不因为旧脚本有默认run/final而自动执行它。

请先定位项目根目录，不要求与别人机器的绝对路径相同。交接目录为：
`versions/xyy/v1.5.8_strict_baseline/stella_v112_handoff/`。

按以下顺序读取：
1. `README.md`
2. `Experiment_Contract.md`
3. `Code_And_Data_Map.md`
4. `Acceptance_Checklist.md`
5. `source_inventory.json`及其中指定的代码/输入/对照产物。

先运行该目录的`verify_handoff_sources.py --project-root <项目根目录>`，记录结果。它只验证交接文件，不代表迁移完成。若缺文件或hash不同，精确报告缺失路径/差异和影响；不要刷新清单、猜替代数据或伪造已通过状态。

核心实验定义：
- 首轮split_seed42，model_seed42，冻结163列，既有五个县折。之后的20260917/20260918仅在负责人另行确认后执行。
- L0=长期严格LGB分量C1；L1=六路processed OSI等权平均后post；L2=当L0>inner正anchor的q95时用L0，否则L1。q固定.95，linear插值，等于阈值走L1。
- 六路顺序：LGB direct、LGB component、XGB direct、Cat direct、XGB component、Cat component。
- 三个候选的1h/6h都直接复制同split当前F2已保存的主预测；不要把新长期结果送回C3改变短期。
- 每个外折q的模型必须在其余四折内轮换早停，再floor平均轮数、四折全量refit。q标签不能影响梯度、早停、阈值上游模型或任何决策。
- q95需要额外inner OOF：对每个r，用另外三折T训练anchor，选轮也只能在T内部进行；完整排除q/r标签。旧global OOF删q行不能替代这个过程。
- 三折内层只训练LGB的P/N/D/R anchor；相同T按严格依赖身份去重，不重训所有六路inner。已有合规LGB外层长期模型按回执和身份复用。
- 本包明确的max rounds、Huber损失、单线程CPU、best_iteration语义、dtype、后处理顺序和mask均不可私改。

请实际阅读现有严格`protocol.py/scoped_training.py/run_artifacts.py/verify_artifacts.py`以及Stella adapters，引用其明确函数作为依据。注意原strict helper只允许4/5折，新inner需要独立3折scope实现和不会冲突的模型ID；不能修改冻结旧文件。旧Stella训练runner含评分外折早停和自动final，不能直接拿来充当strict runner。

建议所有新实现/输出放在新目录`versions/stella_v112_strict/`。保留既有文件与用户当前改动。数据和旧模型只读，输出按独立run隔离。不要扫描archive/articles。代码参数和环境在开训前固定；可移植路径映射不要改写旧manifest。

Windows上特别注意：旧Linuxrun身份保留原样，来源完整性验证与本机重放环境分开记录；新训练生成新身份。不要因环境hash拒绝就修改旧identity或降低验证标准。需要复用F2短期时读固定control预测，不必迁移整个F2训练链。

请自主完成范围内的实现与预检，结束时提供：
1. `Protocol_Alignment.md`：规格→实现函数→检查证据→差异/限制逐条对应。
2. 新改文件列表、resolved config、本地依赖版本与模块路径、输入/代码哈希。
3. 从生产调用链生成的dependency dry-run：所有scope/probe/refit/prediction边，尤其q95三折依赖；预计820次正式新fit、180个新模型、50个被引用模型。
4. 无训练测试报告：数据/schema/mask、独立算术案例、spy/sentinel作用域隔离；明确没有进行真实learner.fit。
5. 拟执行的seed42命令、输出目录、恢复及独立验收流程，和任何尚未解决的阻塞。

不要为了获得更低RMSE自行添加特征、调q/权重、重分折、删县或改模型损失。普通工程细节可自行决定；改变实验定义需要先提出具体差异。只有实际证据支持的检查才标PASS，未执行的训练和模型重载不能标完成。

---

## 后续授权怎么给

收到上述实现和预检后，负责人检查对齐表、代码差异与依赖图，再明确发出“按已审阅的代码/配置运行seed42及必要的隔离诊断，完成独立验收并汇报”的指令。不要现在就把这一句当成已经发出的授权。

对方不用重读整个聊天历史；如本包和后续用户明确指令冲突，以后续指令为准，同时在对齐表记录协议版本变化，不能静默混入同一run。
