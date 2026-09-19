# Stella v1.12 严格实验交接包（2026-09-19）

建议交给 Stella 的是本目录和 `source_inventory.json` 列出的仓库文件，再将 `AI_START.md` 的提示词交给她的 AI。只给聊天记录、旧 README 或一句“参考严格基线”不足以保证实验一致。

本包把**实验问题、数据边界、代码依据和验收证据**分开写清。它是待审阅的迁移规格，不是已完成的迁移；本次没有训练、改变旧代码或生成新模型成绩。

## 1. 本次建议检验的问题

在同一严格县级 CV 下，Stella v1.12 的六路等权集成及 q95 尾部保护，能否改善当前方案的 24h/48h？1h/6h 固定使用当前 F2 的已保存预测。先准备 seed42；训练授权与另两套分折确认分阶段进行。

这里的“严格”要求外折标签同时不能进入梯度训练、早停选轮、阈值的上游模型或其他学习环节。不是只在最后计算阈值时删除外折行。

## 2. 阅读及交付顺序

| 文件 | 用途 |
|---|---|
| [Experiment_Contract.md](Experiment_Contract.md) | 实验定义、折依赖图、六个学习器、后处理和比较规则 |
| [Code_And_Data_Map.md](Code_And_Data_Map.md) | 精确代码锚点、数据/模型传递范围、跨平台注意事项 |
| [Acceptance_Checklist.md](Acceptance_Checklist.md) | 实现前后必须提交的证据、隔离测试和结果验收 |
| [AI_START.md](AI_START.md) | 可直接复制给 Stella 的 AI 的首轮提示词；默认只实现和预检，不训练 |
| [source_inventory.json](source_inventory.json) | 项目相对路径、文件大小、SHA-256 和参考 run 身份 |
| [verify_handoff_sources.py](verify_handoff_sources.py) | 无训练的交接文件完整性检查；不等同于实验验收 |
| [protocol_examples.json](protocol_examples.json) | 手工可复算的边界案例，供新实现独立对照 |
| [Package_Verification.json](Package_Verification.json) | 本地文件清单核验记录；没有迁移训练的通过结论 |

建议先发送四份 Markdown 和文件清单，让对方 AI 提交“逐条对齐表＋拟修改文件＋dry-run 依赖图”；再完成实现、无训练预检，最后由负责人授权 seed42 正式训练。中间不需要反复确认普通工程细节，只有改变实验定义、缺少必要数据或检查失败时才需报告阻塞。

## 3. 实际怎么传文件

最稳妥的是共享同一个仓库快照，并按清单补齐未纳入 Git 的数据、严格基线产物和 F2 预测。**不能假定 push 代码就已包含这些文件。** 如果分开传输，保持项目内相对目录，或让对方使用显式 `--project-root` 映射。

默认交接检查覆盖 `sources`、`inputs`、`reference_seed42`、`reuse_seed42`：包含首轮需要的代码、冻结数据、对照预测以及 50 个可复用的严格 LightGBM 长期模型及回执。另两套分折的文件在 `confirmation` 组；只有后续确认才需要，不因清单里有文件就自动授权训练。

在 Stella 的项目根目录执行（`python` 指她已配置的项目解释器）：

```bash
python -B versions/xyy/v1.5.8_strict_baseline/stella_v112_handoff/verify_handoff_sources.py --project-root .
```

本机可将 `python` 换成 `.venv/bin/python`。Windows 的项目路径可以不同。文件检查器只读取，结果输出到终端；若需保存，重定向到新实验的 `preflight/`，不要改本包的冻结记录。

## 4. 协作中最重要的约定

1. 本规格是新实验的目标；旧代码是算法/实现参考，不因为旧 README 写了 strict 就直接继承其训练流程。
2. 精确复用已有严格 LightGBM 产物，XGBoost/CatBoost 重新按严格协议训练；旧 Stella OOF 只用于历史解释。
3. q95 要有独立的内层交叉预测。当前严格基线 helper 仅支持 4/5 折作用域，不能原样拿来做三折模型。
4. 先给出“我实现了哪些不变量”的证据，再看 RMSE；训练成功和无泄漏不是同义词。
5. 结果不理想也是有效实验。不得为得到提升而私自更换 CV、删县、选 q、搜权重或改变早停规则。

本包的校验脚本证明“对方拿到的文件和这里一致”，迁移后的独立验证证明“新代码按这些规则运行”；二者结合能显著降低偏差，但不能用一份提示词替代审阅与验收。
