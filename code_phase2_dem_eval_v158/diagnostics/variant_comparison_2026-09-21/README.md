# M0–M3 轻量结果与源码审查

报告：`Results_and_Code_Audit_2026-09-21.md`。

复算命令：

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-wendyxu
/home/jacklo/XYY/INFORMS/informs-data-mining-2026/.venv/bin/python -u -B \
  code_phase2_dem_eval_v158/diagnostics/variant_comparison_2026-09-21/analyze_results.py
```

只读 `code_phase2_experiment` 中 manifest 确认的四个 run root，写入本诊断子目录。没有模型训练、权重加载或推理重放；原 `COMPLETE`、预测、源码和报告保持原状。CSV 复读使用 `float_precision="round_trip"`。

该脚本的产物核查通过，不等于原独立 verifier 通过。后者当前有 variant 传递与路径规范化问题，见报告第 6 节。

用户指定的 `code_phase2_dem_eval_v158_limit` 是独立 173 维变体，不是本轮 M0–M3 的源码。其只读预检查命令为：

```bash
/home/jacklo/XYY/INFORMS/informs-data-mining-2026/.venv/bin/python -u -B \
  code_phase2_dem_eval_v158_limit/main.py --stage preflight \
  --base-mode component_v158 --device cpu
```

本次预检查日志保存在 `limit_preflight.log`；不进入 cv/final 阶段。所有审查文件的最终清单见 `delivery_manifest.json`。
