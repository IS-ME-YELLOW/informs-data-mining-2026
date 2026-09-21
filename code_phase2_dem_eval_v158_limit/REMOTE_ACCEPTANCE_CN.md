# DEM v158：给wendy的远程验收与运行说明

日期：2026-09-17。当前代码已加入不可变运行身份、请求/实际轮数、模型级完成记录及恢复、独立进程标签扰动诊断。**本地只执行无训练的替代函数检查；真实LightGBM/GAT的诊断必须按本说明在远程完成。**

诊断沿用原始五折的县归属，默认每折选FIPS排序最前的2县，另选2个测试县；保留144个预测起点、163列冻结特征、7项DEM和173维无邻县摘要 GAT 输入。只缩小诊断样本和训练预算，不改正式分折，不用诊断成绩选模型。

## 1. 交付与环境

请同步完整代码和数据目录，尤其包含新增的：

```text
code_phase2_dem_eval_v158/run_identity.py
code_phase2_dem_eval_v158/model_records.py
code_phase2_dem_eval_v158/diagnostic_support.py
code_phase2_dem_eval_v158/validate_runtime.py
code_phase2_dem_eval_v158/artifact_checks.py
code_phase2_dem_eval_v158/tests/test_p2_contract.py
code_phase2_dem_eval_v158/tests/test_repairs.py
code_phase2_dem_eval_v158/tests/test_artifact_checks.py
versions/xyy/v1.5.6/manifest.json
.gitattributes
```

同时保留现有其余模型代码、五张Parquet、feature_names、组件目标、县折文件和geo资产。不要只拷贝main.py；旧的模型完成记录/manifest不兼容本轮身份格式，不应改写旧记录绕过检查。

以下命令都从 **wendyxu项目根目录** 执行，先激活远程训练环境。这里的 `python` 必须是远程实际用于正式训练的解释器；不要求照搬本机绝对`.venv`路径。

```bash
python -c "import sys,numpy,pandas,pyarrow,lightgbm,shapely,torch; print(sys.executable); print('numpy',numpy.__version__,'pandas',pandas.__version__,'pyarrow',pyarrow.__version__,'lightgbm',lightgbm.__version__,'shapely',shapely.__version__,'torch',torch.__version__); print('CUDA',torch.cuda.is_available(),torch.version.cuda); print(torch.tensor([1.0]).numpy())"
python -B code_phase2_dem_eval_v158/main.py --stage preflight --base-mode component_v158 --device cpu
```

前者确认NumPy/Torch互操作；后者必须打印 `PREFLIGHT_OK`、退出码0。PyTorch需与远程CPU/CUDA平台兼容；不必在本机安装。已有地形CSV时不需要重新下载DEM。本文的测试采用unittest/独立脚本，不要求pytest；若自行运行旧pytest测试文件，再安装pytest。

## 2. 先运行无训练回归

```bash
python -B code_phase2_dem_eval_v158/tests/test_repairs.py
python -B code_phase2_dem_eval_v158/tests/test_artifact_checks.py
python -B code_phase2_dem_eval_v158/tests/test_p2_contract.py
```

三个命令必须均为 `OK`、退出码0。P2测试的“fit”是对标签敏感的算术替代函数，不执行LightGBM优化或GAT反向传播。它会真正经过BaseModelStore、StackContext、图输入和alpha选择代码，检查全部五折位置的外折隔离及多个scope的内层隔离，而不是只检查产物里写的scope字符串。

## 3. 真实CPU快速检查

下面的命令会训练少量诊断模型。输出只能放在 `outputs/diagnostics` 或独立诊断目录；脚本拒绝写入正式 `outputs/runs`。

```bash
python -B code_phase2_dem_eval_v158/validate_runtime.py --backend real --device cpu --modes direct --horizons 1 --outer-folds 0 --counties-per-fold 2 --test-counties 2 --epochs 3 --base-rounds 12 --base-patience 3 --output-dir code_phase2_dem_eval_v158/outputs/diagnostics/cpu_smoke_p2
```

期望结尾：

```json
{"passed": true, "full_coverage": false, "real_acceptance_passed": false}
```

这是正常的快速检查结果：它只覆盖direct/t+1/外折0，不能代表完整验收。出现异常时，查看终端给出的具体`.log`，不要先加大正式训练预算。

## 4. 完整真实CPU诊断

快速检查通过后，运行两条路线、四个horizon、五个外折位置：

```bash
python -B code_phase2_dem_eval_v158/validate_runtime.py --backend real --device cpu --modes direct,component_v158 --horizons 1,6,24,48 --outer-folds 0,1,2,3,4 --counties-per-fold 2 --test-counties 2 --epochs 3 --base-rounds 12 --base-patience 3 --output-dir code_phase2_dem_eval_v158/outputs/diagnostics/cpu_full_p2
```

完整覆盖40个“模式×horizon×外折”组合，每个组合启动10个独立工作进程，最后另有请求/实际轮数边界测试。虽然单个模型很小，完整诊断仍包含较多进程和重复输入构造，不承诺几分钟内结束。终端会持续打印当前组合和案例。

期望结尾：

```json
{"passed": true, "full_coverage": true, "real_acceptance_passed": true}
```

目录中的 `diagnostic_report.json` 必须同时满足上述三个值，且 `backend="real"`。`--backend mock` 无论跑多少案例，都不会把 `real_acceptance_passed` 标成true。

目录已存在时脚本拒绝覆盖。失败后保留日志，修复后用新的诊断目录重跑；诊断调度器本身不做跨命令续跑。这里测试的中断恢复是被测训练核心的模型级恢复能力。

## 5. 每个组合实际检查什么

外折记为k，内层验证折j=(k+1)%5，残差交叉拟合检查折r=(k+2)%5。

| 案例 | 操作 | 通过要求 |
|---|---|---|
| outer / outer_changed | 两个新进程，仅替换外折k的OSI与全部组件监督 | 所有依赖模型、轮数、缩放、base/correction及alpha相同 |
| inner / inner_changed | 在排除k/j的scope中拟合，仅替换j监督 | 该inner模型、缩放和原始base/correction相同；不要求之后选择的alpha不变 |
| crossfit / crossfit_changed | 排除k/j/r生成预测r的base，仅替换r监督 | base预测、模型结构与请求/实际轮数相同 |
| positive | 替换允许使用的训练折r监督 | 必须检测到变化，排除替代对象/比较程序根本不使用标签的无效测试 |
| prepare / resume | 先提交一套完整inner stack，再在下一个base写完模型、尚未提交完成记录时注入中断 | 新进程只复用有完整记录的模型，补齐未完成模型；完成文件字节不变，禁止重新拟合已完成模型 |
| reload | 新进程从连续运行的完成记录恢复，不读取标签、不允许训练接口 | 重载模型参数、base/correction、缩放、alpha对应结果与连续运行相同 |
| rounds | 常量特征refit，请求12轮但实际树数更少 | 合法短模型可保存/重载、预测一致；错误实际轮数或截断模型必须被拒绝 |

标签替换只修改监督对象，并保持四组件与OSI的一致关系、合法NaN尾部；不修改原始特征、地形、观测窗口或五折CSV。数字比较容差固定 `atol=1e-7, rtol=0`；基模型序列化结构摘要要求相同。

真实诊断的“outer标签扰动”覆盖全部五个外折位置；每个外折选一个固定j和r进行更深层检查。本地替代函数测试额外枚举基模型scope。它不是在另一场风暴上的外部验证，不能将诊断预测当作竞赛成绩。

## 6. 如果正式训练使用CUDA

CPU完整检查通过后，至少在目标GPU执行两条路线和四个horizon的设备检查：

```bash
python -B code_phase2_dem_eval_v158/validate_runtime.py --backend real --device cuda --modes direct,component_v158 --horizons 1,6,24,48 --outer-folds 0 --counties-per-fold 2 --test-counties 2 --epochs 3 --base-rounds 12 --base-patience 3 --output-dir code_phase2_dem_eval_v158/outputs/diagnostics/cuda_smoke_p2
```

该命令是GPU补充检查，`full_coverage=false`不替代CPU完整报告；如需在GPU完成同等完整覆盖，把outer-folds改为0,1,2,3,4并使用新目录。

代码启用Torch确定性算法，并在CUDA矩阵计算前设置默认 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。不支持的确定性算子应报错；请保留错误日志，不通过关闭确定性检查或放宽容差把失败改成通过。CPU与CUDA分别检查，不要求两种硬件的输出逐位相同。

## 7. 验收后正式训练和恢复

正式训练建议明确使用目标device，不依赖auto在不同启动时切换设备。以下以component_v158和CUDA为例：

```bash
python -B code_phase2_dem_eval_v158/main.py --stage all --base-mode component_v158 --run-id dem_v158_p2_component_seed42_cuda --device cuda --seed 42 --epochs 220 --patience 35 --time-stride 1 --k 8
```

如果中断，保持代码、依赖、参数、输入和设备一致后执行：

```bash
python -B code_phase2_dem_eval_v158/main.py --stage all --resume --base-mode component_v158 --run-id dem_v158_p2_component_seed42_cuda --device cuda --seed 42 --epochs 220 --patience 35 --time-stride 1 --k 8
```

行为：未完成CV则恢复CV；已有CV_COMPLETE则跳过CV进入final；已有FINAL_READY则不重写最终产物。一个模型训练中断会从固定seed重跑该模型，不做epoch级优化器状态续训。已完成模型需同时具备模型文件及 `.complete.json`，哈希或身份错误会失败，不能静默当作新模型覆盖。

每次启动追加 `attempts.jsonl`；原始run/environment/input/schema/graph记录保持不变。代码或环境变化后恢复被拒绝，并列出不同字段；应保留原版本继续运行或使用新的run_id。代码身份采用保守规则，也包含核验代码，不能修改身份字段绕过检查。README/tests不进入正式训练代码身份。

最终产物生成后，在同一环境的新进程中执行：

```bash
python -B code_phase2_dem_eval_v158/verify_artifacts.py code_phase2_dem_eval_v158/outputs/runs/dem_v158_p2_component_seed42_cuda
```

只有该命令完成数值、依赖、完成记录和无标签重载检查，才会生成正式COMPLETE。诊断报告的通过不等于正式完整数据run已经通过验收。

## 8. 请回传的内容

- CPU完整诊断的 `diagnostic_report.json` 和终端最终状态；若用GPU，附GPU诊断报告。
- 失败时附终端指出的案例`.log`及对应参数，不必先重跑全部正式模型。
- `rounds/rounds.json`：应显示 `actual_rounds < requested_rounds`、重载一致和篡改拒绝。
- 一组 `prepared.json`、`resume_calls.json`，用于确认完成模型未重训。
- 正式训练完成后，附run/environment manifest、verification.json和COMPLETE状态。

本机没有执行真实PyTorch训练或远程命令；本说明中的真实检查由wendy在目标环境执行。只有收到相应真实通过结果后，才能把“真实模型运行验收”一项标记为完成。
