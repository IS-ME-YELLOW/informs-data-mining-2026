# N 冻结特征 L2 / Huber 对照

**状态：seed42四horizon训练、评估及独立验收完成。100次新拟合、20个L2模型；与Huber的树结构、轮数、OOF及全部指标精确相同。**

[结果与解释](/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/n_loss_l2_huber/Results_2026-09-19.md)。原Huber alpha=0.9，已保存外层树前缀的残差保守界最高约0.264，仍处于二次损失区域。没有采用新模型，也没有启动另外两套CV。

输入参照是完整F2。只将N的objective从huber改为regression；163列特征、标签、CV、种子和内层早停规则冻结。六个预定组合均有完整逐行产物。

首次执行顺序如下（完成stage拒绝覆盖；只可恢复身份相同的未完成训练）：

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-2026
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/n_loss_l2_huber/train_n_loss.py --stage preflight
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/n_loss_l2_huber/train_n_loss.py --stage train
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/n_loss_l2_huber/evaluate_n_loss.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/n_loss_l2_huber/verify_n_loss.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/n_loss_l2_huber/audit_loss_equivalence.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/n_loss_l2_huber/summarize_results.py
```

`--resume`仅适用于同身份未完成train；不复用不同配置或不同分折模型。重验可单独运行verify_n_loss.py，不拟合任何模型。

- `reference/`：输入SHA-256与20个目标/外折的预检。
- `runs/v18_nloss_l2_v1_split42_model42/models/`：20个新模型及允许折、特征/标签哈希、轮数与来源receipt。
- 同run的`logs/probes/`：80个内层probe逐行真值、float32指标标签与最佳轮次预测；完整指标曲线在模型receipt中。
- 同run的`n_l2_oof_predictions.parquet`、`component_predictions.parquet`、`control_predictions.parquet`：全县逐行预测及来源；`metrics/`：完整六组合、县/折/窗口、N分组及区间。
- `summary/`：可直接阅读的汇总、固定7县表、损失等价性和保守残差界。
- `MODEL_CV_COMPLETE / EVALUATION_COMPLETE / VERIFIED_COMPLETE`：训练、评估、验收身份；顶层completion.json为本轮完成清单。

没有复制旧Huber模型到本目录；验收从原路径只读重载。训练或核验脚本不写旧日志、旧报告或缓存。
