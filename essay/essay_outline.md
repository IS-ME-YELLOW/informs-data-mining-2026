2. Data and Feature Engineering，主要回答“有哪些信息，预测时能用什么”。

- **2.1 Data and prediction setting**：训练/测试县、观测与预测窗口、四个horizon、OSI及P/N/D/R定义。说明官方平滑、时间延迟和数据可用性，避免把目标时刻与可观测时刻混淆。
- **2.2 Data preparation and temporal alignment**：县与时间键对齐、有效评分窗口、缺失值处理、历史特征修复，以及防止使用预测期真实停电信息的规则。
- **2.3 Feature groups**：历史状态、天气及其时间窗口、其他已有县域字段、邻县历史与天气摘要。正文按特征组介绍，完整163/183列清单放附录。
- **2.4 Initial observations**：简短展示误差和初始状态的异质性，为后文设计提供动机。详细极端县研究留到Discussion。

这里要区分：P71、已知历史贡献和邻县摘要怎么计算，属于数据/特征；如何利用它们重建预测，属于Methodology。

3. Methodology中，建议安排五个核心部分：

- **3.1 Component-based forecasting** ：分别预测P/N/D/R，按官方公式组合；直接OSI预测作为另一类来源
- **3.2 Initial-state-conditioned prediction** ：P的a/b分解与重建；D的已知/未知贡献分解及局部使用规则 
- **3.3 Target-time alignment** ： 同县同目标时刻的多horizon分量平均；解释其适用的信息条件
- **3.4 Heterogeneous ensembles and horizon-specific composition** ：六路集成、预测尾部门控，以及当前T/S组合
- **3.5 Training and validation protocol**： 县级外层CV、内层选轮/阈值、三套冻结分折、最终训练规则 

P可以用核心公式呈现：
\widehat P_s=p\,\widehat a_s+(1-p)\widehat b_s,\qquad p=P_{71}.
D则介绍：D_s=K_s+U_s,
但必须写清实际保留的是局部使用规则，不能让读者误以为所有horizon、所有时段都改成U模型。

4. Results，建议以四组结果支撑方法：

- **4.1 Overall performance**：原始严格基线、改进树T、Stella长期方案、整合候选的四horizon成绩，逐套CV报告。
- **4.2 Contributions of retained modifications**：D历史重建、P结构、邻县特征、Stella门控和最终融合的对照结果。标明每轮参考模型；顺序增量实验不能写成完整独立消融，也不能累加改善百分比。
- **4.3 Robustness and error distribution**：跨折、县、目标日稳定性，收益集中度和县轨迹bootstrap区间。
- **4.4 Alternatives not retained**：简表交代P结构迁移、额外信息、N/R两阶段等未采用方案及原因；完整结果放附录。

**最终OSI RMSE是主结果，分量RMSE用于解释机制。** 所有表头都明确评分范围、参考方案、单位和改善方向。Stella报告中的“改善概率”若保留，应改称“bootstrap重采样中改善的比例”，避免被理解为泛化成功概率。

5. Discussion

- 为什么初始状态结构对短时更有效，而长期需要不同的预测组合。
- 为什么P采用较多结构，N/R仍保留直接树：用实验支持这种选择。
- 极端县、相似县和训练—测试支持差异揭示了什么；哪些属于数据支持不足，哪些仍无法区分模型与信息问题。
- 为什么分量精度改善不一定转化为OSI改善，以及集成平均不保证胜过最佳来源。
- 局限：同一事件、有限县数、重复使用OOF作研究决策、空间相关性，以及三套分折并非三次独立外部验证。

具体县名可以作为少量案例，不应把方法描述成针对已知坏县制定规则。

6. Conclusion，收束为三点：提出了什么方法、取得了哪些经验证的收益、适用边界是什么。Wendy尚未完成的结果不写成已实现贡献；最终隐藏测试成绩也要与开发期CV明确区分