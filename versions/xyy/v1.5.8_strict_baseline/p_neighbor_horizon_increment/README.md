# P6/P24/P48邻县信息单来源实验

状态：seed42三候选完成，100次拟合、20个模型；独立核验PASS。[结果](/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/p_neighbor_horizon_increment/Results_2026-09-20.md)。点估计均小幅改善、收益区间均跨零；优先确认NB_P24，NB_P06/NB_P48暂缓，当前已确认组合保持不变。

参照是已确认P6 a/b的完整组合。NB_P06仍为a/b加权L2，NB_P24/NB_P48仍为原直接P-Huber。仅增加对应时间口径的20列邻县特征，不联合候选、不训练其他CV。

首次执行顺序：

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-2026
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_neighbor_horizon_increment/prepare_evaluation.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_neighbor_horizon_increment/train_neighbors.py --stage preflight
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_neighbor_horizon_increment/train_neighbors.py --stage train
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_neighbor_horizon_increment/evaluate_neighbors.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_neighbor_horizon_increment/verify_neighbors.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_neighbor_horizon_increment/summarize_results.py
```

完成stage拒绝覆盖；--resume只允许同身份未完成训练。单独运行verify_neighbors.py不拟合模型。

- features/v1：三套183列训练/测试矩阵、白名单视图、固定邻居/坐标和1,043,712条时段依赖；测试矩阵仅供特征覆盖核验。
- reference：冻结输入哈希、时间/因果/作用域预检和评价代码适配记录。
- runs/v18_pneighbor_v1_split42_model42：20个模型及回执、内层预测/曲线、完整新P OOF、候选分量和控制预测、全部指标与独立验收。
- summary：四输出矩阵、P指标、县/折/时段/初始状态、bootstrap和误差变化拆解。
- completion.json、final_integrity_check.json：本轮完成身份与旧来源不变证明。

独立特征验证使用与生产张量不同的按县逐邻居累加，并固定IEEE-754加法顺序；从原始数据核对历史/天气语义，再用冻结缓存的精确值重建183列进行模型重载。原163列继承既有因果审查，不宣称本轮重做全部外部静态数据提取。
