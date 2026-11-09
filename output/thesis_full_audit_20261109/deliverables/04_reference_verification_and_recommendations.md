# 参考文献逐条交叉核验与补引建议报告（`04_reference_verification_and_recommendations.md`）

> **审计基准**：`HEAD = f550ba60fba5558dc6d3349b1d9d470d066babc4`  
> **核验对象**：[`latex/reference.bib`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/reference.bib)（共 53 条 BibTeX 条目）及 [`latex/docs/chap01.tex`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/docs/chap01.tex) 至 [`chap06.tex`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/docs/chap06.tex) 全文引用上下文  
> **配套机读结果**：[`output/thesis_full_audit_20261109/validation/reference_audit_results.json`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/output/thesis_full_audit_20261109/validation/reference_audit_results.json)  
> **遵循规范**：`nature-ref-verifier` 多源交叉验证与引文-论点匹配规范

---

## 1. 核验总览与自动化校验脚本漏洞法医分析

### 1.1 53 条 BibTeX 条目核验统计

| 分类等级 | 条目数量 | 占比 | 核心判定说明 |
|---|---:|---:|---|
| **A 类：完全通过（Verified Clean）** | **20** | 37.7% | 作者、标题、期刊/会议、年份、卷期、页码、DOI 及正文引文语境均真实准确 |
| **B 类：元数据瑕疵或缺失 DOI（Minor / Missing DOI）** | **16** | 30.2% | 论文真实存在且论点匹配，但缺失 DOI、缺失页码/期号，或 BibTeX key 年份与正式发表年份相差 1 年 |
| **C 类：阻断级异常（Blocking Issues）** | **17** | 32.1% | 包含 **AI 幻觉拼接条目（1 条）**、**真实作者/标题上伪造年份卷期页码（7 条）**、**引文内容与正文论点严重错位（5 条）**、**历史替换遗留的 Key-作者严重冲突（4 条）** |
| **合计** | **53** | 100.0% | 53 条全部在正文中被实际引用（无未引用死条目） |

### 1.2 为什么现有校验脚本 `verify_thesis_references.py` 会漏检？

在 [`scripts/literature/verify_thesis_references.py`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/scripts/literature/verify_thesis_references.py#L76-L89) 中，核验逻辑仅执行了**标题字符串相似度匹配**：

```python
score = SequenceMatcher(None, norm_title, normalize(cand_title)).ratio()
if score > 0.92:
    found = (score, item)
    break
```

该脚本存在三个严重设计盲区：
1. **完全不校验作者、年份、期刊、卷期与页码**：例如 [`wu2021joint`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/reference.bib#L55-L64) 的标题 *Joint trajectory and communication design for multi-UAV enabled wireless networks* 是真实存在的（CrossRef 标题匹配度 `1.0`），因此脚本直接判为 `VERIFIED`；但该文实际上发表于 **2018 年 Vol. 17, No. 3, pp. 2109–2121**，而 `reference.bib` 中写的是伪造的 **2021 年 Vol. 20, No. 3, pp. 1735–1749**！
2. **最近一次提交 `f550ba6` 新增/回退的 9 条无 DOI 条目未重跑核验**：`docs/literature/analysis/reference_verification_final.json` 仅记录了早期 44 条记录，而 `f550ba6` 重新引入的 `rashid2021updet`、`huang2021uav`、`wu2021joint`、`hu2021uav`、`zhang2021energy`、`zhou2020latency`、`liu2020energy`、`bai2021multi`、`de2020sample` 均绕过了严格核验。
3. **不校验正文引用语义（Claim-to-Source Mapping）**：例如 `guo2025mappo` 是一篇关于**道路交叉口智能网联汽车（CAVs）协同决策**的 ECCV Workshop 论文，却在 [`chap01.tex:L88`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/docs/chap01.tex#L88) 被当作“无人机辅助边缘计算动态卸载与轨迹控制”的代表性工作引用。

---

## 2. 阻断级问题条目逐条法医核查与修复方案（17 条）

### 2.1 AI 幻觉拼接与伪造卷期页码类（8 条，必须立即替换或修正）

| 序号 | BibTeX Key | 引用位置 | 当前 `reference.bib` 内容 | 真实文献核查结论（Forensic Finding） | 修复动作 |
|---:|---|---|---|---|---|
| 1 | [`rashid2021updet`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/reference.bib#L422-L429) | [`chap02.tex:L602`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/docs/chap02.tex#L602) | Rashid, Farquhar, Peng, Whiteson (2020). *Attention-based multi-agent reinforcement learning*. ICML, pp. 7958–7968. | **典型 AI 幻觉缝合怪（Franken-citation）**：① Key 中的 `updet` 来自 **Hu, Siyi et al. (ICLR 2021)** *UPDeT: Universal Multi-agent RL via Policy Decoupling with Transformers*；② 作者 `Rashid, Farquhar, Peng, Whiteson` 来自 QMIX 团队（JMLR 2020 *Monotonic Value Function Factorisation...*）；③ ICML 2020 (PMLR v119) pp. 7958–7968 实际上是 Bono et al. 的论文。该条目完全不存在！ | **立即删除**，替换为真实的多智能体注意力奠基论文 **Iqbal & Sha (ICML 2019, MAAC, `iqbal2019actor`)** 与 **Hu et al. (ICLR 2021, UPDeT, `hu2021updet`)** |
| 2 | [`wu2021joint`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/reference.bib#L55-L64) | [`chap01.tex:L73`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/docs/chap01.tex#L73) | Wu, Qingqing, Zeng, Yong, Zhang, Rui (2021). *Joint trajectory and communication design for multi-UAV enabled wireless networks*. IEEE TWC, 20(3):1735–1749. | **真实高被引名作但年份、卷号、页码全错**：该文真实发表信息为 **IEEE TWC, Vol. 17, No. 3, pp. 2109–2121, March 2018, DOI: `10.1109/TWC.2017.2789293`**。 | 修正年份为 `2018`、卷期页码为 `17(3):2109--2121` 并补齐 DOI（建议同步将 key 改为 `wu2018joint`） |
| 3 | [`huang2021uav`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/reference.bib#L44-L53) | [`chap01.tex:L58`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/docs/chap01.tex#L58) | Huang, Liang, Bi, Suzhi, Zhang, Ying-Jun Angela (2021). *DRL based online offloading framework for UAV-MEC systems*. IEEE TWC, 20(5):3025–3039. | **在真实作者组合上伪造标题与期刊**：Huang, Bi, Zhang 三位作者的真实代表作是 **IEEE TMC, Vol. 19, No. 11, pp. 2581–2593, 2020, DOI: `10.1109/TMC.2019.2928811`**（*Deep Reinforcement Learning for Online Computation Offloading in Wireless Powered Mobile-Edge Computing Networks*，即 DROO），并非 IEEE TWC 2021 的 UAV 论文。 | 从 `chap01.tex:L58` 的 UAV 轨迹控制引用中移除，或修正为真实的 DROO (IEEE TMC 2020) 并在卸载段落引用 |
| 4 | [`hu2021uav`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/reference.bib#L66-L75) | [`chap01.tex:L73`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/docs/chap01.tex#L73) | Hu, Xiao, Yang, Kun, Zheng, Zhaohui (2021). *UAV-assisted mobile edge computing: A joint trajectory and resource optimization approach*. IEEE TWC, 20(7):4445–4458. | **对已收录真实条目 `hu2019uav` 的幻觉篡改重复**：真实论文即同文件第 99 行的 `hu2019uav`（Hu, Xiaoyan, Wong, Kai-Kit, Yang, Kun, Zheng, Zhongbin, IEEE TWC 2019, DOI: `10.1109/TWC.2019.2928539`）。`hu2021uav` 将作者名缩写篡改并伪造了 2021 年卷期页码。 | **删除 `hu2021uav`**，在 `chap01.tex:L73` 中直接保留 `\cite{hu2019uav}` |
| 5 | [`zhou2020latency`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/reference.bib#L112-L121) | [`chap01.tex:L73`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/docs/chap01.tex#L73) | Zhou, Fan, Wu, Yongpeng, Hu, Rose Qingyang, Qian, Yu (2020). *Latency optimization for UAV-assisted mobile edge computing*. IEEE TCOM, 68(9):5721–5735. | **作者名、标题与期刊篡改**：该四位作者（第一作者实为 **Zhou, Fuhui** 而非 Zhou, Fan）的真实论文为 *Computation Rate Maximization in UAV-Enabled Wireless-Powered Mobile-Edge Computing Systems*，发表于 **IEEE JSAC, Vol. 36, No. 9, pp. 1927–1941, Sept. 2018, DOI: `10.1109/JSAC.2018.2864426`**。 | 修正为真实的 **Zhou, Fuhui et al. (IEEE JSAC 2018, `zhou2018computation`)** |
| 6 | [`bai2021multi`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/reference.bib#L148-L157) | [`chap01.tex:L73`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/docs/chap01.tex#L73) | Bai, Teng, Wang, Jianping, Ren, Yong, Hanzo, Lajos (2021). *Multi-agent reinforcement learning for UAV-assisted mobile edge computing*. IEEE TWC, 20(12):8184–8198. | **标题/期刊伪造 + 论点自相矛盾**：① Bai, Wang, Wang, Ren, Hanzo 的真实论文为 *Energy-Efficient Computation Offloading for Secure UAV-Edge-Computing Systems*，发表于 **IEEE TVT, Vol. 68, No. 6, pp. 6074–6087, June 2019, DOI: `10.1109/TVT.2019.2912778`**（恰好采用凸优化方法，符合 `chap01.tex:L73` 的凸近似/交替优化语境！）；② 当前伪造的标题写的是 `Multi-agent reinforcement learning`，放在 `chap01.tex:L73` 的“传统数学规划（BCD、SCA）”处完全自相矛盾！ | 修正为真实的 **Bai, Teng et al. (IEEE TVT 2019, `bai2019energy`)** |
| 7 | [`zhang2021energy`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/reference.bib#L77-L86) | [`chap01.tex:L73`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/docs/chap01.tex#L73) | Zhang, Guanglin, Shen, Yanli, Yang, Kun (2021). *Energy-efficient UAV-assisted mobile edge computing*. IEEE TVT, 70(8):8234–8238. | **无 DOI 且卷期页码失真**：`f550ba6` 回退引入的无 DOI 条目，标题过于笼统且与 IEEE TVT 70(8) 真实目录不符。 | 直接从 `chap01.tex:L73` 的密集引用串中删除（该处保留修正后的 `wu2018joint,hu2019uav,zhou2018computation,bai2019energy` 已足够支撑传统优化综述） |
| 8 | [`liu2020energy`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/reference.bib#L123-L132) | [`chap01.tex:L73`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/docs/chap01.tex#L73) | Liu, Yang, Wang, Kezhi, Pan, Cunhua, Aslam, Nauman (2020). *Energy-efficient resource allocation for UAV-assisted mobile edge computing*. IEEE TCOM, 68(11):6963–6977. | **无 DOI 且元数据失真**：Yang, Wang, Pan, Aslam 团队发表的是 IEEE TCOM 2020 *Energy-Efficient Transmission and Computation Resource Allocation inF-RAN/UAV-MEC* 相关文章，但当前标题与页码 `6963--6977` 不匹配且无 DOI。 | 从 `chap01.tex:L73` 的密集引用串（该处已堆砌 8 篇引用）中移除 |

---

### 2.2 引文内容与正文论点严重错位类（5 条）

| 序号 | BibTeX Key | 引用位置与正文宣称 | 实际文献内容 | 修复动作 |
|---:|---|---|---|---|
| 9 | [`guo2025mappo`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/reference.bib#L159-L167) | [`chap01.tex:L88`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/docs/chap01.tex#L88)：“研究者将多智能体深度强化学习引入**无人机辅助边缘计算的动态卸载与轨迹控制**中 \cite{de2020sample,guo2025mappo,zhao2024robust}” | *MAPPO-PIS: A multi-agent proximal policy optimization method with prior intent sharing for **CAVs' cooperative decision-making***（ECCV 2024 Workshops，研究地面自动驾驶车辆路口通行，与无人机和边缘计算毫无关系，且年份为 2024 而 key 写 2025） | 从 `chap01.tex:L88` 的 UAV-MEC 论点处移除，或改在通用 MARL 段落引用 |
| 10 | [`de2020sample`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/reference.bib#L204-L212) | [`chap01.tex:L88`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/docs/chap01.tex#L88)：同上，宣称为“无人机辅助边缘计算的动态卸载与轨迹控制” | *Sample efficient multi-agent reinforcement learning with centralized training*（IEEE CDC 2020，通用多智能体控制会议论文，非 UAV-MEC 卸载论文，且无 DOI） | 从 `chap01.tex:L88` 移除（该句后面紧接着已详细介绍了 `zhao2024robust`、`huang2026madr`、`you2026soft`、`tao2026online` 四篇真正的 2024–2026 UAV-MEC MARL 论文，直接引用这四篇即可！） |
| 11 | [`li2024cacom`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/reference.bib#L243-L250) | [`chap01.tex:L112`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/docs/chap01.tex#L112)：多智能体注意力交互表征 | *Context-aware communication for multi-agent reinforcement learning*（AAMAS 2024，无 DOI；早期提交中曾是伪造的 Computer Networks UAV 论文，后被替换为通用通信论文） | 替换为经过严格验证的多智能体注意力代表作 **Iqbal & Sha (ICML 2019, MAAC, `iqbal2019actor`)** |
| 12 | [`liu2025tmac`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/reference.bib#L252-L259) | [`chap01.tex:L112`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/docs/chap01.tex#L112)：多智能体注意力交互表征 | 第一作者实为 **Li, Xuesi**（Key 却写 `liu2025tmac`），发表于 *PeerJ Computer Science* Vol. 11 (2025)，缺失文章号（`e26xx`）与 DOI | 补齐 DOI 与文章号并将 key 改为 `li2025tmac`，或替换为 **Hu et al. (ICLR 2021, UPDeT, `hu2021updet`)** |
| 13 | [`zhang2025as`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/reference.bib#L261-L270) | [`chap01.tex:L112`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/docs/chap01.tex#L112)：多智能体协同决策 | Zhang, Peng & Li, Guohui (2025), *Discover Artificial Intelligence*, 5(1):153（多无人机动态目标搜索，非边缘计算任务卸载，且缺失 DOI `10.1007/s44163-025-00380-x`） | 补齐 DOI，并在 `chap01.tex:L112` 表述中明确其应用场景为多无人机协同搜索 |

---

### 2.3 历史替换遗留的 BibTeX Key 与真实作者/年份冲突类（4 条）

在之前的文献清理提交中，作者将部分旧虚假条目替换为了真实论文，但为了省事**保留了旧的 BibTeX Key**，导致源码中出现 `\cite{he2019uav}` 实为 Yuan et al. (2022)、`\cite{fan2021uav}` 实为 Hady et al. (2025) 的怪象：

| 序号 | 当前旧 Key | 实际第一作者与年份 | 真实论文状态 | 建议新 Key 与修复动作 |
|---:|---|---|---|---|
| 14 | [`he2019uav`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/reference.bib#L88-L97) | **Yuan, Xinwang** et al. (**2022**) | 真实有效（*Sensors*, 22(10):3854, DOI: `10.3390/s22103854`），[`chap01.tex:L75`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/docs/chap01.tex#L75) 正文已写“Yuan、Xie 和 Tan” | 将 Key 重命名为 `yuan2022computation` |
| 15 | [`fan2021uav`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/reference.bib#L134-L146) | **Hady, Mohamad A.** et al. (**2025**) | 真实有效（*Artificial Intelligence Review*, 58(11):354, DOI: `10.1007/s10462-025-11340-5`），[`chap01.tex:L107`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/docs/chap01.tex#L107) 正文已写“Hady 等” | 将 Key 重命名为 `hady2025marl` |
| 16 | [`zhao2024robust`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/reference.bib#L169-L179) | **Li, Bin** et al. (**2024**) | 真实有效（*IEEE IoT Journal*, 11(3):4775–4786, DOI: `10.1109/JIOT.2023.3302685`），[`chap01.tex:L90`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/docs/chap01.tex#L90) 正文已写“Li 等” | 将 Key 重命名为 `li2024robust` |
| 17 | [`khurshid2021uav`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/reference.bib#L34-L42) | **Khurshid, Tayyaba** et al. (**2023**) | 真实有效（*IEEE Access*, 11:54664–54678），但 Key 年份写 `2021` 且缺失 DOI `10.1109/ACCESS.2023.3281674` | 将 Key 重命名为 `khurshid2023drl` 并补齐 DOI |

---

## 3. 全量 53 条参考文献逐条核验总表

| # | BibTeX Key | 第一作者与年份 | 期刊 / 会议 | 核验状态 | 详细核查结论与补全信息 |
|---:|---|---|---|---|---|
| 1 | `chung2018swarm` | Chung, S.-J. (2018) | *IEEE Trans. Robot.* 34(4):837–855 | **A: Clean** | 已含真实 DOI `10.1109/TRO.2018.2857475`，[`chap01.tex:L58`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/docs/chap01.tex#L58) 引用合理 |
| 2 | `skaltsis2023allocation` | Skaltsis, G. M. (2023) | *Drones* 7(4):264 | **A: Clean** | 已含真实 DOI `10.3390/drones7040264`，[`chap01.tex:L58`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/latex/docs/chap01.tex#L58) 引用合理 |
| 3 | `zeng2016wireless` | Zeng, Y. (2016) | *IEEE Commun. Mag.* 54(5):36–42 | **A: Clean** | 已含真实 DOI `10.1109/MCOM.2016.7470933`，经典综述 |
| 4 | `mozaffari2019tutorial` | Mozaffari, M. (2019) | *IEEE COMST* 21(3):2334–2360 | **A: Clean** | 已含真实 DOI `10.1109/COMST.2019.2902862`，经典综述 |
| 5 | `khurshid2021uav` | Khurshid, T. (2023) | *IEEE Access* 11:54664–54678 | **C: Blocking** | Key 年份(2021)与发表年份(2023)不符；需补 DOI `10.1109/ACCESS.2023.3281674` |
| 6 | `huang2021uav` | Huang, L. (2021*) | *IEEE TWC* 20(5):3025–3039* | **C: Blocking** | **伪造标题与期刊**；真实论文为 IEEE TMC 2020 DROO (`10.1109/TMC.2019.2928811`) |
| 7 | `wu2021joint` | Wu, Q. (2021*) | *IEEE TWC* 20(3):1735–1749* | **C: Blocking** | **伪造年份卷期页码**；真实论文为 IEEE TWC 2018, 17(3):2109–2121 (`10.1109/TWC.2017.2789293`) |
| 8 | `hu2021uav` | Hu, X. (2021*) | *IEEE TWC* 20(7):4445–4458* | **C: Blocking** | **幻觉重复条目**；实为第 11 条 `hu2019uav` 的篡改版，应直接删除 |
| 9 | `zhang2021energy` | Zhang, G. (2021*) | *IEEE TVT* 70(8):8234–8238* | **C: Blocking** | **无 DOI 疑似篡改条目**；建议替换为 `zeng2019energy` (`10.1109/TWC.2019.2902559`) |
| 10 | `he2019uav` | Yuan, X. (2022) | *Sensors* 22(10):3854 | **C: Blocking** | 论文与 DOI (`10.3390/s22103854`) 真实，但 Key `he2019uav` 严重错配，应改名为 `yuan2022computation` |
| 11 | `hu2019uav` | Hu, X. (2019) | *IEEE TWC* 18(10):4738–4752 | **A: Clean** | 已含真实 DOI `10.1109/TWC.2019.2928539` |
| 12 | `zhou2020latency` | Zhou, F. (2020*) | *IEEE TCOM* 68(9):5721–5735* | **C: Blocking** | **篡改标题与期刊**；实为 Zhou, Fuhui et al., IEEE JSAC 2018, 36(9):1927–1941 (`10.1109/JSAC.2018.2864426`) |
| 13 | `liu2020energy` | Liu, Y. (2020*) | *IEEE TCOM* 68(11):6963–6977* | **C: Blocking** | 无 DOI 且页码与标题不匹配，建议从 `chap01.tex:L73` 删除 |
| 14 | `fan2021uav` | Hady, M. A. (2025) | *Artif. Intell. Rev.* 58(11):354 | **C: Blocking** | 论文与 DOI (`10.1007/s10462-025-11340-5`) 真实，但 Key `fan2021uav` 严重错配，应改名为 `hady2025marl` |
| 15 | `bai2021multi` | Bai, T. (2021*) | *IEEE TWC* 20(12):8184–8198* | **C: Blocking** | **伪造标题与期刊**；实为 Bai et al., IEEE TVT 2019, 68(6):6074–6087 (`10.1109/TVT.2019.2912778`) |
| 16 | `guo2025mappo` | Guo, Y. (2024) | *ECCV Workshops* pp. 245–262 | **C: Blocking** | **论点错位**：研究地面自动驾驶汽车 (CAVs) 路口决策，却在 `chap01.tex:L88` 被当作 UAV-MEC 论文引用 |
| 17 | `zhao2024robust` | Li, B. (2024) | *IEEE IoT J.* 11(3):4775–4786 | **C: Blocking** | 论文与 DOI (`10.1109/JIOT.2023.3302685`) 真实，但 Key `zhao2024robust` 与第一作者 `Li` 不符 |
| 18 | `huang2026madr` | Huang, H. (2026) | *Ad Hoc Networks* 181:104029 | **B: Minor** | 已含真实 DOI `10.1016/j.adhoc.2025.104029`，`f550ba6` 已补 `pages={104029}` |
| 19 | `you2026soft` | You, J. (2026) | *PeerJ Comput. Sci.* 12:e3528 | **B: Minor** | 已含真实 DOI `10.7717/peerj-cs.3528`，`f550ba6` 已补 `pages={e3528}` |
| 20 | `de2020sample` | De, A. (2020) | *IEEE CDC* pp. 5958–5963 | **C: Blocking** | **论点错位且无 DOI**：非 UAV-MEC 卸载论文，应从 `chap01.tex:L88` 移除 |
| 21 | `oroojlooyjadid2022review` | Oroojlooy, A. (2023) | *Appl. Intell.* 53(11):13677–13722 | **B: Minor** | 论文真实，建议补齐 DOI `10.1007/s10489-022-03357-y` |
| 22 | `yu2022surprising` | Yu, C. (2022) | *NeurIPS* 35:24611–24624 | **A: Clean** | MAPPO 奠基论文，元数据准确 |
| 23 | `gronauer2022multi` | Gronauer, S. (2022) | *Artif. Intell. Rev.* 55(2):895–943 | **A: Clean** | 已含真实 DOI `10.1007/s10462-021-09996-w` |
| 24 | `li2024cacom` | Li, X. (2024) | *AAMAS* pp. 1156–1164 | **C: Blocking** | 无 DOI 且经过二次替换，建议替换为 `iqbal2019actor` (ICML 2019 MAAC) |
| 25 | `liu2025tmac` | Li, X. (2025) | *PeerJ Comput. Sci.* 11 | **C: Blocking** | Key (`liu`) 与第一作者 (`Li`) 冲突，且缺失页码/文章号与 DOI |
| 26 | `zhang2025as` | Zhang, P. (2025) | *Discover Artif. Intell.* 5(1):153 | **C: Blocking** | 缺失 DOI，且研究多无人机目标搜索而非 MEC 卸载 |
| 27 | `schulman2015high` | Schulman, J. (2016) | *ICLR* 2016 | **B: Minor** | GAE 奠基论文（arXiv 2015 / ICLR 2016），建议补 `note={arXiv:1506.02438}` |
| 28 | `rusu2015distillation` | Rusu, A. A. (2016) | *ICLR* 2016 | **B: Minor** | Policy Distillation 奠基论文（arXiv 2015 / ICLR 2016），建议补 `note={arXiv:1511.06295}` |
| 29 | `yu2020joint` | Yu, Z. (2020) | *IEEE IoT J.* 7(10):9722–9733 | **A: Clean** | 已含真实 DOI `10.1109/JIOT.2020.2996231` |
| 30 | `shi2016edge` | Shi, W. (2016) | *IEEE IoT J.* 3(5):637–646 | **B: Minor** | 边缘计算奠基综述，建议补齐 DOI `10.1109/JIOT.2016.2579198` |
| 31 | `mao2017survey` | Mao, Y. (2017) | *IEEE COMST* 19(4):2322–2358 | **A: Clean** | 已含真实 DOI `10.1109/COMST.2017.2745201` |
| 32 | `wang2020edge` | Wang, X. (2020) | *IEEE COMST* 22(2):869–904 | **B: Minor** | 边缘智能综述，建议补齐 DOI `10.1109/COMST.2020.2970550` |
| 33 | `abbas2018mobile` | Abbas, N. (2018) | *IEEE IoT J.* 5(1):450–465 | **B: Minor** | MEC 综述，建议补齐 DOI `10.1109/JIOT.2017.2750180` |
| 34 | `sutton2018reinforcement` | Sutton, R. S. (2018) | *MIT Press* (2nd ed.) | **A: Clean** | 强化学习经典教材 |
| 35 | `mnih2015human` | Mnih, V. (2015) | *Nature* 518(7540):529–533 | **A: Clean** | 已含真实 DOI `10.1038/nature14236` |
| 36 | `lecun2015deep` | LeCun, Y. (2015) | *Nature* 521(7553):436–444 | **B: Minor** | 深度学习综述，建议补齐 DOI `10.1038/nature14539` |
| 37 | `watkins1992q` | Watkins, C. J. (1992) | *Mach. Learn.* 8(3-4):279–292 | **B: Minor** | Q-learning 奠基论文，建议补齐 DOI `10.1007/BF00992698` |
| 38 | `williams1992simple` | Williams, R. J. (1992) | *Mach. Learn.* 8(3-4):229–256 | **A: Clean** | 已含真实 DOI `10.1007/BF00992696` |
| 39 | `lillicrap2016continuous` | Lillicrap, T. P. (2016) | *ICLR* 2016 | **B: Minor** | DDPG 奠基论文，建议补 `note={arXiv:1509.02971}` |
| 40 | `schulman2017proximal` | Schulman, J. (2017) | *arXiv:1707.06347* | **A: Clean** | PPO 奠基论文 |
| 41 | `busoniu2008comprehensive` | Busoniu, L. (2008) | *IEEE TSMCC* 38(2):156–172 | **B: Minor** | MARL 早期综述，建议补齐 DOI `10.1109/TSMCC.2007.913919` |
| 42 | `littman1994markov` | Littman, M. L. (1994) | *ICML* pp. 157–163 | **B: Minor** | Markov Games 奠基论文，建议补 `publisher={Morgan Kaufmann}` |
| 43 | `lowe2017multi` | Lowe, R. (2017) | *NeurIPS* 30:6379–6390 | **A: Clean** | MADDPG 奠基论文 |
| 44 | `rashid2018qmix` | Rashid, T. (2018) | *ICML* PMLR 80:4295–4304 | **A: Clean** | QMIX 奠基论文 |
| 45 | `sunehag2018value` | Sunehag, P. (2018) | *AAMAS* pp. 2085–2087 | **A: Clean** | VDN 奠基论文 |
| 46 | `vaswani2017attention` | Vaswani, A. (2017) | *NeurIPS* 30:5998–6008 | **A: Clean** | Transformer 奠基论文 |
| 47 | `rashid2021updet` | Rashid, T.* (2020*) | *ICML* pp. 7958–7968* | **C: Blocking** | **AI 幻觉缝合条目（不存在）**，必须替换为 `iqbal2019actor` 与 `hu2021updet` |
| 48 | `shakarami2024beyond` | Shakarami, A. (2024) | *IEEE COMST* 26(3):1849–1886 | **B: Minor** | 综述真实，建议补齐 DOI `10.1109/COMST.2024.3367782` |
| 49 | `ross2011dagger` | Ross, S. (2011) | *AISTATS* PMLR 15:627–635 | **B: Minor** | DAgger 奠基论文，建议补 `volume={15}, series={PMLR}` |
| 50 | `zeng2026uav` | 曾志威 等 (2026) | 南京邮电大学学报(自然科学版) 46(1):87–96 | **A: Clean** | 已含真实 DOI `10.14132/j.cnki.1673-5439.2026.01.008` |
| 51 | `li2026sensing` | 李伟 等 (2026) | 电讯技术 66(2):195–203 | **A: Clean** | 已含真实 DOI `10.20079/j.issn.1001-893x.241219002` |
| 52 | `wang2026ccah` | Wang, Q. (2026) | *J. Supercomput.* 82(7):315 | **A: Clean** | 已含真实 DOI `10.1007/s11227-026-08351-y`，`f550ba6` 已补期号与页码 |
| 53 | `tao2026online` | Tao, Y. (2026) | *Chin. J. Electron.* 35(1):315–326 | **A: Clean** | 已含真实 DOI `10.23919/cje.2024.00.289` |

---

## 4. 阻断级条目的可直接替换 BibTeX 代码块

以下给出修复上述 C 类阻断级条目的完整、真实、带 DOI 的 BibTeX 条目（可直接替换 `latex/reference.bib` 中对应位置）：

```bibtex
@inproceedings{iqbal2019actor,
  title     = {Actor-Attention-Critic for Multi-Agent Reinforcement Learning},
  author    = {Iqbal, Shariq and Sha, Fei},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {97},
  pages     = {2961--2970},
  year      = {2019},
  publisher = {PMLR}
}

@inproceedings{hu2021updet,
  title     = {{UPDeT}: Universal Multi-agent {RL} via Policy Decoupling with Transformers},
  author    = {Hu, Siyi and Zhu, Fengda and Chang, Xiaojun and Liang, Xiaodan},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2021}
}

@article{wu2018joint,
  title     = {Joint Trajectory and Communication Design for Multi-{UAV} Enabled Wireless Networks},
  author    = {Wu, Qingqing and Zeng, Yong and Zhang, Rui},
  journal   = {IEEE Transactions on Wireless Communications},
  volume    = {17},
  number    = {3},
  pages     = {2109--2121},
  year      = {2018},
  publisher = {IEEE},
  doi       = {10.1109/TWC.2017.2789293}
}

@article{huang2020deep,
  title     = {Deep Reinforcement Learning for Online Computation Offloading in Wireless Powered Mobile-Edge Computing Networks},
  author    = {Huang, Liang and Bi, Suzhi and Zhang, Ying-Jun Angela},
  journal   = {IEEE Transactions on Mobile Computing},
  volume    = {19},
  number    = {11},
  pages     = {2581--2593},
  year      = {2020},
  publisher = {IEEE},
  doi       = {10.1109/TMC.2019.2928811}
}

@article{zhou2018computation,
  title     = {Computation Rate Maximization in {UAV}-Enabled Wireless-Powered Mobile-Edge Computing Systems},
  author    = {Zhou, Fuhui and Wu, Yongpeng and Hu, Rose Qingyang and Qian, Yu},
  journal   = {IEEE Journal on Selected Areas in Communications},
  volume    = {36},
  number    = {9},
  pages     = {1927--1941},
  year      = {2018},
  publisher = {IEEE},
  doi       = {10.1109/JSAC.2018.2864426}
}

@article{bai2019energy,
  title     = {Energy-Efficient Computation Offloading for Secure {UAV}-Edge-Computing Systems},
  author    = {Bai, Teng and Wang, Chao and Wang, Jianping and Ren, Yong and Hanzo, Lajos},
  journal   = {IEEE Transactions on Vehicular Technology},
  volume    = {68},
  number    = {6},
  pages     = {6074--6087},
  year      = {2019},
  publisher = {IEEE},
  doi       = {10.1109/TVT.2019.2912778}
}

@article{khurshid2023drl,
  title     = {A {DRL} Strategy for Optimal Resource Allocation Along With {3D} Trajectory Dynamics in {UAV-MEC} Network},
  author    = {Khurshid, Tayyaba andothers},
  journal   = {IEEE Access},
  volume    = {11},
  pages     = {54664--54678},
  year      = {2023},
  publisher = {IEEE},
  doi       = {10.1109/ACCESS.2023.3281674}
}
```

---

## 5. 亟需补充的 8 篇核心支撑文献（回答用户问题 6：“如果还需要加参考文献的话，需要加哪些方面的参考文献”）

结合第二章至第六章的公式来源审计与同行评审意见，当前论文在以下 **6 个核心理论与算法模块**存在明显的“有公式/算法却无源头文献支撑”的空白。强烈建议补充以下 **8 篇真实、权威的奠基文献**：

### 5.1 必须补充的文献清单与插入位置映射

| 补充维度 | 推荐 BibTeX Key | 权威文献信息 | 建议插入的正文位置 | 补充理由 |
|---|---|---|---|---|
| **1. 势函数奖励塑形（PBRS）理论** | `ng1999policy` | Ng, A. Y., Harada, D., & Russell, S. (1999). *Policy invariance under reward transformations: Theory and application to reward shaping*. ICML, pp. 278–287. | 第 2 章 2.4 节；第 5 章式 `(5.10)`–`(5.11)` | 第 5 章核心贡献之一是动作前后双时刻势函数差分引导，必须引用 Ng et al. (1999) 阐明经典 PBRS $\gamma\Phi(s_{t+1})-\Phi(s_t)$ 与本文同一步内动作前后势差 $\Phi_{\text{post}}(s_t^+)-\Phi_{\text{pre}}(s_t)$ 的理论关系 |
| **2. 群组相对策略优化（GRPO）** | `shao2024deepseekmath` | Shao, Z. et al. (2024). *DeepSeekMath: Pushing the limits of mathematical reasoning in open language models*. arXiv:2402.03300. | 第 5 章式 `(5.13)`–`(5.16)` | 代码库 [`scripts/analysis/queued_joint_rl_experiment.py`](file:///c:/Users/woshinibaba/Documents/%E5%AD%A6%E4%B9%A0/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/%E6%88%91%E7%9A%84%E8%AE%BA%E6%96%87/wodexuexi/scripts/analysis/queued_joint_rl_experiment.py) 将该算法命名为 `GRPO-PPO`，第 5 章式 `(5.14)`–`(5.16)` 的组内相对标准化优势直接源于 GRPO，必须正式引用 |
| **3. 留一法（LOO）多采样基线** | `kool2019buy` | Kool, W., van Hoof, H., & Welling, M. (2019). *Buy 4 REINFORCE samples, get a baseline for free!* ICLR Workshop. | 第 5 章式 `(5.13)` | 第 5 章式 `(5.13)` 的 $\frac{1}{G-1}\sum_{j\neq i} Q(s,u^{(j)})$ 是经典的无偏 Leave-One-Out 采样基线（RLOO），引用 Kool et al. (2019) 可直接赋予式 `(5.13)` 坚实的方差缩减理论依据 |
| **4. 多智能体注意力 Actor-Critic** | `iqbal2019actor` | Iqbal, S., & Sha, F. (2019). *Actor-attention-critic for multi-agent reinforcement learning*. ICML, PMLR 97:2961–2970. | 第 1 章 1.2.3 节；第 2 章 2.6.2 节；第 3 章 3.3 节 | 替代幻觉条目 `rashid2021updet`，为第 2、3 章的集中式注意力 Critic 与多头交互表征提供真正的顶会源头支撑 |
| **5. 集合置换不变性/等变性理论** | `zaheer2017deep` | Zaheer, M. et al. (2017). *Deep Sets*. NeurIPS, Vol. 30. | 第 2 章式 `(2.63)`；第 5 章候选等变打分器 | 第 2 章式 `(2.63)` 专门定义了置换不变性（Permutation Invariance），第 5 章双分支网络强调了候选集等变性（Candidate Equivariance），必须引用 Deep Sets 奠基文献 |
| **6. 引导策略搜索与学习引导 MPC** | `levine2013guided` | Levine, S., & Koltun, V. (2013). *Guided policy search*. ICML, PMLR 28(3):1–9. | 第 1 章 1.2.3 节；第 5 章 5.2–5.4 节 | 为第 5 章“将规划器前瞻轨迹蒸馏至神经网络策略并结合在线 Top-6 MPC”提供规划学习（Planning-to-Policy Learning）领域的权威源头文献 |
| **7. 抢占式最早截止期优先（EDF）调度** | `liu1973scheduling` | Liu, C. L., & Layland, J. W. (1973). *Scheduling algorithms for multiprogramming in a hard-real-time environment*. JACM, 20(1):46–61. | 第 5 章式 `(5.3)` 与式 `(5.5)` | 第 5 章全章采用节点抢占式 EDF 计算调度，必须引用实时调度领域的开山之作 Liu & Layland (1973) |
| **8. 深度强化学习小样本Bootstrap统计规范** | `agarwal2021deep` | Agarwal, R. et al. (2021). *Deep reinforcement learning at the edge of the statistical precipice*. NeurIPS, Vol. 34, pp. 29304–29320. | 第 2 章 2.8 节；第 3 章 3.5 节；第 5 章 5.5 节 | 强力支撑本文在小样本训练种子与工作负载上的配对 Bootstrap 95% 置信区间统计方法学 |

### 5.2 推荐新增文献的完整 BibTeX 代码块

```bibtex
@inproceedings{ng1999policy,
  title     = {Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping},
  author    = {Ng, Andrew Y. and Harada, Daishi and Russell, Stuart},
  booktitle = {Proceedings of the 16th International Conference on Machine Learning (ICML)},
  pages     = {278--287},
  year      = {1999},
  publisher = {Morgan Kaufmann}
}

@article{shao2024deepseekmath,
  title   = {{DeepSeekMath}: Pushing the Limits of Mathematical Reasoning in Open Language Models},
  author  = {Shao, Zhihong and Wang, Peiyi and Zhu, Qihao and Xu, Runxin and Song, Junxiao and Bi, Xiao and Zhang, Haowei and Zhang, Mingchuan and Li, Y. K. and Wu, Y. and Guo, Daya},
  journal = {arXiv preprint arXiv:2402.03300},
  year    = {2024}
}

@inproceedings{kool2019buy,
  title     = {Buy 4 {REINFORCE} Samples, Get a Baseline for Free!},
  author    = {Kool, Wouter and van Hoof, Herke and Welling, Max},
  booktitle = {ICLR 2019 Workshop on Deep Reinforcement Learning Meets Structured Prediction},
  year      = {2019}
}

@inproceedings{zaheer2017deep,
  title     = {Deep Sets},
  author    = {Zaheer, Manzil and Kottur, Satwik and Ravanbakhsh, Siamak and Poczos, Barnabas and Salakhutdinov, Ruslan R. and Smola, Alexander J.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {30},
  pages     = {3391--3401},
  year      = {2017}
}

@inproceedings{levine2013guided,
  title     = {Guided Policy Search},
  author    = {Levine, Sergey and Koltun, Vladlen},
  booktitle = {Proceedings of the 30th International Conference on Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {28},
  number    = {3},
  pages     = {1--9},
  year      = {2013},
  publisher = {PMLR}
}

@article{liu1973scheduling,
  title     = {Scheduling Algorithms for Multiprogramming in a Hard-Real-Time Environment},
  author    = {Liu, C. L. and Layland, James W.},
  journal   = {Journal of the ACM},
  volume    = {20},
  number    = {1},
  pages     = {46--61},
  year      = {1973},
  publisher = {ACM},
  doi       = {10.1145/321738.321743}
}

@inproceedings{agarwal2021deep,
  title     = {Deep Reinforcement Learning at the Edge of the Statistical Precipice},
  author    = {Agarwal, Rishabh and Schwarzer, Max and Castro, Pablo Samuel and Courville, Aaron C. and Bellemare, Marc},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {34},
  pages     = {29304--29320},
  year      = {2021}
}
```
