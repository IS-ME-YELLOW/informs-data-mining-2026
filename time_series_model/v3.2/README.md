# v3.2 统一目标小时动态融合模型

本版本以 v1.5.6 四个直接 OSI LightGBM 为候选，把所有预测按真实目标小时对齐，并对每个
`(county,target_hour)` 只生成一个 OSI。主模型用小型因果 GRU 产生非负且和为 1 的动态融合权重；
安全系数允许模型完整退回等权平均。

完整设计见 [Plan_v3.2.md](Plan_v3.2.md)。

截至 2026-09-10，M1～M4 已完成。M3 工程闭环见
[M3_Implementation_Report_2026-09-10.md](M3_Implementation_Report_2026-09-10.md)，严格验证见
[M4_Strict_Nested_CV_Report_2026-09-10.md](M4_Strict_Nested_CV_Report_2026-09-10.md)。

M4 结论是完整 GRU 动态融合没有超过等权平均，静态凸融合的微小点估计收益也不稳定，因此
v3.2 不进入 M5。所有统一模型均通过“同一目标小时一个预测”的逐值验证。

## 当前运行顺序

```powershell
# M1：构建并核验唯一目标轴
python time_series_model\v3.2\build_target_timeline.py --overwrite --verify

# M3：诊断级工程闭环
python time_series_model\v3.2\train_diagnostic.py --max-epochs 30 --overwrite

# 独立重载 M3 的 20 个检查点并复算
python time_series_model\v3.2\verify_artifacts.py --milestone m3

# M4：复用经验证的 v3.1 严格 LightGBM 候选
python time_series_model\v3.2\train_nested_cv.py --max-epochs 50 --overwrite

# 独立重载 M4 的 20 个检查点并复算
python time_series_model\v3.2\verify_artifacts.py --milestone m4
```

M3 使用 v3.1 诊断级 LightGBM OOF，只能验证数据、训练、预测、映射、提交和重载链路。M4
只读复用 v3.1 已独立验证的严格折内 LightGBM 候选，并使用外层五折和内层四折完成正式比较。

所有 v1.5.6 和 v3.1 资产均只读复用；v3.2 的新产物只写入本目录。
