# 真实模型重载诊断（2026-09-19）

本目录保存旧 DEM v158 模型的 CPU 推理复现和 Newton 固定权重机制诊断。入口报告为 `Results_2026-09-19.md`；本目录的 PASS 不写回原模型包，也不代表新模型训练或原 CUDA 训练验收。

## 运行

依赖已安装在主仓库 `.venv`：torch 2.7.1+cpu。以下命令只覆盖本诊断子目录内对应输出，不训练模型，不改原模型/代码/提交。

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-wendyxu

/home/jacklo/XYY/INFORMS/informs-data-mining-2026/.venv/bin/python -u -B \
  code_phase2_dem_eval_v158/diagnostics/checkpoint_reload_2026-09-19/replay_models.py \
  --stage all \
  > code_phase2_dem_eval_v158/diagnostics/checkpoint_reload_2026-09-19/replay.log 2>&1

/home/jacklo/XYY/INFORMS/informs-data-mining-2026/.venv/bin/python -u -B \
  code_phase2_dem_eval_v158/diagnostics/checkpoint_reload_2026-09-19/probe_newton.py \
  > code_phase2_dem_eval_v158/diagnostics/checkpoint_reload_2026-09-19/Newton_probe.log 2>&1

/home/jacklo/XYY/INFORMS/informs-data-mining-2026/.venv/bin/python -u -B \
  code_phase2_dem_eval_v158/diagnostics/checkpoint_reload_2026-09-19/verify_diagnostics.py
```

按上面顺序执行；`probe_newton.py` 依赖重放生成的 `Newton_outer2_inputs.npz`。全部重放本次耗时约 209 秒，Newton 探针约 6 秒，具体耗时随机器变化。仅检查保存产物时只运行最后一步。

`replay_models.py --stage metadata` 只核对元数据并加载权重；`--stage newton` 只重放 Newton 所属外层上下文。它们不能替代完整 `--stage all`。脚本固定读取 `diagnostic_config.json` 指定的历史运行，缺文件或身份不符即报错，不自动重训。

## 文件索引

| 文件 | 用途 |
|---|---|
| `Results_2026-09-19.md` | 中文结论及限制 |
| `diagnostic_config.json` | 冻结来源、运行身份与事先声明的容差 |
| `environment_before_install.json`、`environment_after_install.json` | 安装前后包版本与导入验证 |
| `source_snapshot.json` | 本次开始时 1,055 个已有文件的 SHA256 |
| `replay_models.py`、`replay.log` | 全部 416 个树模型/64 个 GAT 的加载和上下文重放 |
| `metadata_verification.json`、`replay_verification.json` | 元数据及推理复现结果 |
| `Newton_outer2_inputs.npz` | 完整节点输入、标准化输入及该检查点的 CPU 校正 |
| `probe_newton.py`、`Newton_probe.log`、`Newton_findings.json` | 固定权重干预与分解 |
| `verify_diagnostics.py`、`verification.json` | 独立复算、覆盖检查、提交对齐、文件保护核验 |
| `diagnostic_manifest.json` | 本次交付文件哈希；只适用于当前快照，重跑覆盖产物后会改变 |
| `tables/checkpoint_metadata.csv` | 64 个检查点的 scope、训练轮次、尺度等 |
| `tables/CPU_replay_metrics.csv` | 64 个上下文的误差与数值匹配 |
| `tables/CPU_replayed_outer_and_test.parquet` | 原始保存列与重载 CPU 列并列，173,952 行 |
| `tables/outer_horizon_replay.csv` | 重载后四个 horizon 的外层 RMSE |
| `tables/test_submission_replay.csv` | 重载与旧提交逐 horizon 的差异 |
| `tables/Newton_interventions.csv` | 10 种固定权重操作的该县结果，非新 CV 分数 |
| `tables/Newton_intervention_timeline.csv` | 干预逐行结果 |
| `tables/Newton_path_decomposition.csv` | 两路径与单特征两路径的可加分解 |
| `tables/skip_feature_weights.csv` | skip 权重范数及 Newton 标准化值 |

CSV 复读使用 `float_precision="round_trip"`。CPU/CUDA 原始校正绝对容差为 `1e-5`；后处理阈值翻转单独检查，不以容差隐藏。本次阈值翻转为零。

检查点是用户提供且与完成记录哈希匹配的原模型；使用 `torch.load(..., map_location="cpu", weights_only=False)` 读取其数组/元数据。

## 解释约束

- 所有训练 API 在重放脚本中被阻断；模型 eval、固定权重、CPU 确定性推理。
- Newton 干预是计算图敏感性，不能当作新泛化成绩、物理因果证明或可直接部署的修复。
- 不重写原运行 manifest/verification/COMPLETE；不修改上一轮诊断文档。
- 已安装 torch 的环境可继续用于后续授权实验；本次未安装 CUDA 版或开始训练。
