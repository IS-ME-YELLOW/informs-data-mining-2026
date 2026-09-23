# 最终模型：I3 baseline + m2_robust_input GAT 残差

本目录是最终代码入口。它使用 `I3_prediction_package_v1` 中冻结且经过验收的完整 I3 base，使用 `m2_source/gat_model.py` 中从 `code_phase2_compare` 原样带入的 edge-aware residual GAT，并严格保留 m2_robust_input 的：

- 205 维图输入：I3 base、163 列 base 特征、经纬度、7 列 terrain，以及 16 列邻县均值和 16 列邻县差值；
- scope 内拟合的逐列标准化，随后固定 `z` 裁剪到 `[-5, 5]`；
- 两层 edge-aware GAT、4 heads、hidden=24、dropout=0.12、raw-input skip path；
- 原尺度残差 `official OSI - I3_OSI`，残差尺度归一化，MSE，AdamW，220 epochs / 35 patience，CUDA 确定性算法；
- I3 规定的 `S\{r\}` 完整 baseline 依赖、内层 alpha OOF 选择、全 5 折最终训练和四个 horizon 的官方 NaN 尾部。

`final_model.py` 不重新训练或简化 I3 baseline；`PredictionPackage.graph_context()` 直接读取完整 I3 scope 预测。I3 包中的 `reference/` 仅作审计来源，不会直接被当作 GAT 训练数据。

## 远程服务器运行步骤

以下命令均从仓库根目录运行：

```bash
cd /path/to/informs-data-mining-2026
conda activate phase2
```

安装 Python 依赖。PyTorch 请按服务器 CUDA 驱动选择官方兼容版本：

```bash
pip install -r code_phase2_final/requirements.txt
# 另行安装与服务器 CUDA 匹配的 torch
```

先验证冻结 I3 包，不会训练或修改包文件：

```bash
python -B code_phase2_final/I3_prediction_package_v1/verify_package.py
```

再执行不训练的 final 图输入检查。它会读取 I3 parquet、geo DBF/SHP 和 terrain，并确认 302 个 county、3028 条图边和 205 维输入：

```bash
python -B code_phase2_final/final_model.py --preflight --device cuda --seed 42
```

确认通过后，运行完整训练和最终预测：

```bash
python -u code_phase2_final/final_model.py \
  --device cuda \
  --seed 42 \
  --epochs 220 \
  --patience 35 \
  --run-id i3_m2_robust_input_final_seed42
```

如使用另外一套冻结 I3 分折，只能完整替换 seed，不能混用：

```bash
python -u code_phase2_final/final_model.py \
  --device cuda --seed 20260917 \
  --run-id i3_m2_robust_input_final_seed20260917
```

## 输出

运行目录默认是 `code_phase2_final/outputs/runs/<run-id>/`，其中：

- `submission_i3_m2_gat.csv`：按 `data/sample_submission.csv` 原 identifier 和行顺序生成的最终提交文件；
- `checkpoints/gat_t01h_S01234.pt`、`t06h`、`t24h`、`t48h`：全 5 折最终 GAT 及 scope-fitted preprocessing；
- `alpha_selection.json`：每个 horizon 的内层 OOF alpha 网格及选择结果；
- `baseline_scores.json`：强制 alpha=0 时对 frozen I3 cross-fitted base 的训练 OOF 检查；
- `run_manifest.json`：seed、模型配置、I3 包哈希、graph 哈希、checkpoint 哈希和有限值计数。

官方要求的有效预测行数是 `9009 / 8694 / 7560 / 6048`（t01h/t06h/t24h/t48h）；预测窗口之外必须保持 NaN。程序会在写出后再次检查这些计数和 `[0, 0.65]` 范围。

默认相对路径要求仓库中仍保留：

```text
data/geo/c_16ap26.dbf
data/geo/c_16ap26.shp
data/geo/county_terrain.csv
data/sample_submission.csv
```

如果远程服务器路径不同，可用 `--package-dir` 和 `--submission-template` 指定 I3 包与模板；geo/terrain 当前按仓库相对路径读取，建议保持上述目录结构。

