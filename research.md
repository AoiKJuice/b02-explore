# B02 开源方案 / 论文 / 数据调研（2026-11-09）

结论：没有现成的“雷达点迹 + 强杂波 + 输出批号”端到端开源方案。可行路线是拼装：
**点级真假打分 + 航迹得分确认 + 学习式关联打分 + 经典批号生命周期管理**。

## 一、优先采用（许可证可直接复用）

| 名称 | 链接 | 许可 | 用途 |
|---|---|---|---|
| SimpleTrack | https://github.com/tusen-ai/SimpleTrack | MIT | 两阶段关联（高分点起批、低分点只续批）、“预测延续”代替删除 → 直接改 tracker.py，几乎零成本 |
| 航迹得分 LLR / SPRT + IPDA 存在概率 | 参考 MATLAB trackScoreLogic / JIPDATracker 文档（只参考公式） | 自实现 | 替换 M/N 确认；杂波图 λ 自适应门限；SNR 似然和分类器 logit 加进得分 |
| gnn_tracking（Object Condensation） | https://github.com/gnn-tracking/gnn_tracking | MIT，活跃 | 把最近 K 圈点迹当一个事件：逐点 β=真假分，隐空间聚类=跨圈航迹段。和后羿“虚警头+注意力关联”同构 |
| MT3v2 | https://github.com/JulianoLagana/MT3v2 | MIT | 多帧点量测 Transformer，设定最接近赛题；借 encoder 做逐点真假头、仿真训练流程；不输出批号 |
| PolarMOT | https://github.com/aleksandrkim61/PolarMOT | MIT | 只用相对几何（极坐标距离/角度/Δt）的 GNN 边分类 → 航迹-点迹关联打分 |
| Stone Soup | https://github.com/dstl/Stone-Soup | MIT，活跃 | 对照基线；JPDA+EHM、多帧分配、IMM 组件。纯 Python 偏慢，不做主干 |
| FilterPy | https://github.com/rlabbe/filterpy | MIT | 快速加 CV/CA/CT 三模型 IMM |
| labeledRFS（LMB/GLMB Python） | https://github.com/linh-gist/labeledRFS | MIT | 标签即批号、自带存在概率；伴飞簇局部用 |
| García-Fernández MTT（PMB/PMBM，MATLAB） | https://github.com/Agarciafernandez/MTT | BSD-2 | 强杂波确认的理论最优框架，PMB 最易移植 |
| fastmurty | https://github.com/motrom/fastmurty | MIT | K-best 分配，做多假设/多帧关联的组件 |
| TrackML 冠军 top-quarks | https://github.com/top-quarks/top-quarks | MIT | 非深度学习：点对→三元组→轨迹逐级打分，可做起批种子打分 |

只能参考思路（无许可证 / GPL / 自定义许可）：Exa.TrkX、DeepMTT、Meyer BP 代码（MTT-PIEEE-18）、pmbm（erikbohnsack）、RadarGNN（GPL-3）、3DMOTFormer / MOTR / MOTRv2（自定义）。

## 二、关键论文

- 后羿：叶宇等，现代雷达 2025,47(8)，DOI 10.16592/j.cnki.1004-7859.2025052701（索引页写 1-9，目录摘要写 27-36，引用前核对）。结构要点见 AGENTS.md 第 7 节。
- 朱新国等《人工智能在雷达目标全自动录取中的应用现状及展望》，空天预警研究学报 2025,39(6)，南京所，出题方的路线图 → 知网下载。
- 《基于 Transformer 网络的机载雷达多目标跟踪方法》，雷达学报 JR22009：https://radars.ac.cn/article/doi/10.12000/JR22009
- Neural Enhanced Belief Propagation（Liang & Meyer）arXiv:2212.08340：经典 BP 中注入学到的虚警/关联权重，和我们的混合路线最一致。
- BP vs JPDA vs MHT 对近邻目标的行为：arXiv:2308.06326（JPDA 合并、MHT 排斥、BP 基本都避免）→ 伴飞问题。
- 自注意力真/假航迹分类航迹起始，太赫兹科学与电子信息学报 2022,20(12):1269 → 对应航迹真假分类器。
- DeepAF（MDPI Aerospace 2025,12(3):194）：杂波下 Transformer 关联 + 滤波，起始仍用传统方法。
- 《回波幅度信息辅助的群目标航迹起始方法》雷达学报 JR19088 → 伴飞起批用 Amp/SNR。
- 《基于 PSO-SVM 算法的雷达点迹真伪鉴别方法研究》雷达科学与技术 2020(4) → 点迹级特征清单。
- 多帧 TBD（Grossi 等 IEEE TSP 2013 点迹列表 DP；Li 等 FUSION 2022）→ 弱目标起批，无开源。

## 三、数据

| 数据 | 链接 | 形态 | 用途 |
|---|---|---|---|
| 金海豚杯（2023）海情科目一“点航迹自动录取跟踪” | https://radars.ac.cn/news/DatasetintheCompetition.htm | 实装雷达点迹 → 航迹，训练集 97.7MB / 测试集 220.7MB；需关注《雷达学报》公众号注册 | **最接近 B02 的公开真实数据**：校准仿真、验证分类器 |
| 国防科大对海监视舰船跟踪与航迹融合数据集 | 中国科学数据 2024,9(1) | 仿真 2D 点迹含虚警，36.8 万点 | 预训练虚警抑制 / 关联 |
| MTAD 多源航迹关联（AIS） | 电子与信息学报数据说明 PDF | 航迹级，有中断和真值 | 断航迹接续 → 零碎度 / 最长跟踪 |
| OpenSky（ADS-B） | https://opensky-network.org | 真实空中航迹 | 真实运动 + 仿真杂波造训练数据 |
| 北理工低空无人机群（信号处理 2025,41(5)） | https://signal.ejournal.org.cn/cn/article/doi/10.12466/xhcl.2025.05.004 | 实测编队点迹，无逐点批号 | 检验伴飞串号；下载方式未知 |
| RadarScenes | https://zenodo.org/records/4559821 | 车载雷达逐点 track_id + 标签 | 结构预训练，场景差异大 |
| 雷达学报对海探测 / 低慢小系列 | https://radars.ac.cn/web/data/getData?dataType=DatasetofRadarDetectingSea | 回波为主 | 统计杂波特性 |
