# v1.18 tree-family ensemble

该版本在 v1.16/v1.17 的真实 OOF 上进行严格外折融合。运行与独立重载：

```powershell
C:\Users\Stella\.conda\envs\best-route\python.exe -B versions\v1.18_tree_family_ensemble\run_tree_family_ensemble.py
C:\Users\Stella\.conda\envs\best-route\python.exe -B versions\v1.18_tree_family_ensemble\verify_artifacts.py
```

结果目录包含所有候选的严格嵌套分数；`submission_v1.18_exploratory_best_balanced_v1.csv` 仅表示同一组外折上的探索性最优，不绕过晋级标准。

## 结果

v1.12+ET 在 t+24/t+48 相对 v1.12 退化 0.704%/0.404%；v1.12+RF 退化 0.006%/0.121%；三模型融合退化 0.704%/0.438%。ET 融合虽在 t+24 有 4/5 外折、t+48 有 3/5 外折改善，并改善实际高 OSI 子集，但池化损失与 bootstrap 不支持晋级。探索性最优仍是 v1.12，`promote=false`。

48 个 JSON 模型独立重载后的 OOF/test 最大差为 0，所有哈希通过。
