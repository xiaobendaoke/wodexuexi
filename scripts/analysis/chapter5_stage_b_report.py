"""Generate the Stage B evidence report on the experiment server."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys


def read(p):return json.loads(p.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if sys.platform!='linux':raise RuntimeError('Derived scientific summaries run on the server')
    p=read(a.results/'protocol.json');cl=read(a.results/'analysis/closed_loop_summary.json');mech=read(a.results/'analysis/mechanism_summary.json');audit=read(a.results/'analysis/raw_audit.json')
    def ci(z,d=3):return f"{z['mean']:+.{d}f} [{z['ci95'][0]:+.{d}f}, {z['ci95'][1]:+.{d}f}]"
    lines=['# 第五章阶段 B：开发补证与迁移复验（2026-10-27）','',
        '**完成 B0—B3 的开发设计，未启动 C 阶段；未实施可选的6000回合完整迁移复验。** 新数据仅来自五个历史开发上层基座和三个经清单扫描无冲突的新开发负载，不是新训练块独立确认。原20块主证据及封存代码保持不变。','',
        '## 新结果应该怎样理解','',
        '1. 五场景中解析方法相对G的块均差均为正，低算力场景均差最大；高负载和高算力各保留了负向块。各场景解析−二值区间均跨零，不能声称解析方法在五场景都优于二值模块。',
        '2. 零价M0−B0有正向开发信号，共同带价Mλ−Bλ的闭环区间跨零。Mλ−M0的DSR区间也跨零，价格应解释为质量—能耗取舍，不是DSR增益或已证无损。',
        '3. 关键协作边在长回合中并不少见，不能用“关键边稀少”解释信息价值有限。固定快照下模块差异随G/解析源轨迹而变，不应以合并均值掩盖状态分布依赖。',
        '4. 暖态中原公开先验的Brier低于冷态half规则，但其log loss更高，概率误差没有单一全面赢家；本轮未比较新学习模型，不能推断所有学习方法无效。近1概率仍有少量反演残差，已如实保留。','',
        '## 执行与证据身份','',
        '- B1：G和冻结解析方法各15个1000步回合，共30回合；每回合40个25步区间预抽一步，1200主状态。额外step0单列，不给主均值额外权重。',
        '- B2：共同冻结价格下二值/解析模块×零价/带价四组计费闭环，共60回合。Mλ与原mixture_price完全同一策略，复用15个中心回合。',
        '- B3：五块×三个负载×五场景×四方法，共300回合；与B1及B2交叉复用后，实际345个不同回合、345000步。',
        '- 独立描述单位为五个历史上层块。每块先合并计数求S/G、A/G、S/A，再计算配对差；所有区间为各自95%配对t描述区间，非同时区间，不作确认性显著性或非劣效宣称。',
        '- 本轮使用同一服务器，最多32个单线程工作进程，分配至不同物理核（CPU 64—95）。共享服务器存在其他用户负载；并非系统级独占。方法顺序在运行前随机固定，各配对任务束共享CPU。完整时间逐次计费，日志和快照落盘在计时外。不能与历史120核耗时拼接计算加速比。','',
        '## B0：输入、划分与冻结','',
        f"上层基座：{', '.join(map(str,p['blocks']))}；新开发负载：{', '.join(map(str,p['workloads']))}；不新增训练种子。共同λ={p['common_price']['price']:.17g}，归一化常数={p['common_price']['normalizer']:.14g}，沿用现有解析标价，在新效果揭示前登记。原binary_price仍保留其原独立标价0。",
        '',
        '正式原始闭环日志保留逐请求物理时延、逐次完整时间、最终分配、能耗、费用及请求计数，但全部公开rows仅存哈希，候选与分解可经原环境和原决定重放恢复。本轮另保存抽样前完整状态及公开rows，候选/筛选在离线公开重放中补齐；私有缓存单独写入标签文件。五个历史开发模型及旧400状态依赖均在服务器找到，未因本机缺二进制而重训。',
        '',
        '种子核验扫描服务器历史协议、启动清单、manifest与种子目录名，跳过逐时步原始内容。审计清单保留扫描范围和摘要；此为对现存登记资料的冲突核验，不声称检索不存在的外部记录。首次B0仅做元数据准备，随后缩小扫描范围、使用新的development_v2目录冻结实际执行源码，首次准备目录保留，未运行科学样本。','',
        '## B3：五场景零调参迁移','',
        '| 场景 | 解析−G，pp [95%区间] | 二值−G，pp | cold-half−G，pp | 解析/G机队能耗比 | 解析−二值，pp |',
        '|---|---:|---:|---:|---:|---:|']
    names={'center':'100 UE，算力×1.0','ue60':'60 UE，算力×1.0','ue140':'140 UE，算力×1.0','cpu06':'100 UE，算力×0.6','cpu14':'100 UE，算力×1.4'}
    for s in ['ue60','center','ue140','cpu06','cpu14']:
        z=cl['transfer'][s];c=z['comparisons']
        lines.append(f"| {names[s]} | {ci(c['mixture_price-G']['dsr_pp'])} | {ci(c['binary_price-G']['dsr_pp'])} | {ci(c['cold_half_proxy_price-G']['dsr_pp'])} | {c['mixture_price-G']['energy_ratio']['mean']:.4f} | {ci(z['mixture_minus_binary']['dsr_pp'])} |")
    lines+=['','每场景每方法为15回合、15000步；场景分开计数，不跨UE数合并分母。只改变UE总数或UAV CPU容量，不重算训练时观测维度。上层每UAV仍保留30个UE槽，按原覆盖/未覆盖排序截取；140 UE存在信息截断限制。下层100个请求槽保持不变，环境遇到截断会报错，本轮未触发。五场景为十字设计，未检验负载×算力交互。','',
        '### 全方法服务与费用','',
        '| 场景 | 方法 | S/G % | A/G % | S/A % | MBS/A % | 完整均值 ms | 块p95均值 ms |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for s in ['ue60','center','ue140','cpu06','cpu14']:
        for m,z in cl['transfer'][s]['methods'].items():
            lines.append(f"| {names[s]} | {m} | {z['dsr']['mean']*100:.3f} | {z['admission']['mean']*100:.3f} | {z['quality']['mean']*100:.3f} | {z['mbs_fraction']['mean']*100:.3f} | {z['mean_seconds']['mean']*1000:.3f} | {z['p95_seconds']['mean']*1000:.3f} |")
    lines+=['','上表是块率或块时延统计的均值，块p95均值不是合并逐次p95。能耗指完整UAV机队能耗，未实测软件CPU物理能耗。','',
        '## B2：共同价格的评分—匹配模块比较','',
        'B0=二值模块零价；M0=解析混合模块零价；Bλ=二值模块共同λ；Mλ=解析模块共同λ。初值、修复、公开提议顺序、十次匹配总上限、预测硬筛选和反馈一致。二值与混合在配额内标量能耗项尺度上也不同，因此以下是评分及条件指派模块干预，不是仅p公式的因果分解。','',
        '| 闭环比较 | 计费DSR差，pp [95%区间] | 能耗比（左/右） | 接纳率差，pp | 接纳后质量差，pp | MBS占比差，pp |',
        '|---|---:|---:|---:|---:|---:|']
    for key in ['M0-B0','Mlambda-Blambda','Blambda-B0','Mlambda-M0']:
        z=cl['components'][key]
        lines.append(f"| {key} | {ci(z['dsr_pp'])} | {z['energy_ratio']['mean']:.4f} | {z['admission_pp']['mean']:+.3f} | {z['quality_pp']['mean']:+.3f} | {z['mbs_pp']['mean']:+.3f} |")
    lines += ['',f"闭环差之差交互：{ci(cl['components']['interaction_pp'])} pp。",'',
        '| 固定快照共同10 ms比较 | G源轨迹，pp [95%区间] | 解析源轨迹，pp [95%区间] | 等权合并两来源，pp [95%区间] |',
        '|---|---:|---:|---:|']
    for key in ['M0-B0','Mlambda-Blambda','Blambda-B0','Mlambda-M0','interaction_pp']:
        z=mech['fixed_snapshot_common_10ms']
        lines.append(f"| {key} | {ci(z['G'][key])} | {ci(z['mixture_price'][key])} | {ci(z['both'][key])} |")
    lines += ['',
        '固定快照锁定原状态与源动作中的上层motion；对封存的公开分配单独执行原物理环境，按共同10 ms记账。它不是原生计费闭环效果。固定候选池另外把两模块接受的分配并集交给二值/混合评分重排，保留池、公开分数、能耗和排序；不从真实标签中选赢家。',
        '',f"固定池无价格排序：{json.dumps(mech['fixed_pool_changes'],ensure_ascii=False)}。",'',
        '## B1：长回合分布和缓存信息价值','',
        f"主样本{mech['main_states']}状态；额外冷启动{mech['cold_only_states']}状态。完成{mech['physical_source_replays']}个抽样源状态的一步物理重放，逐请求时延与原始记录一致，并用原调用费用复核原成功数。原主分布没有重复计入冷启动。",'',
        '| 开发块 | 主状态 | 合法协作边 | 回退边 | p≥1−10⁻¹⁰边 | G配额关键边/活跃边 % | 解析配额关键边/活跃边 % | 候选配额并集关键边/活跃边 % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for b,groups in mech['per_block'].items():
        z=groups['both'];s=z['sensitivity']
        lines.append(f"| {b} | {z['states']} | {z['edges']} | {z['fallback']} | {z['near_one']} | {s['g']['ratio']*100:.3f} | {s['m']['ratio']*100:.3f} | {s['union']['ratio']*100:.3f} |")
    lines+=['','关键边为固定配额下命中按时、未命中超时的合法协作边；活跃分母仅包含目标配额>0的边，JSON同时保留全部合法协作边分母。候选配额并集对每条请求—目标边只计一次“任一候选配额敏感”；额外记录配额—边重复数但不作独立样本。所选边统计、来源×负载分层、冷暖分层及所有回退原因在原始JSON中保留。','',
        '| 概率规则 | 全边Brier（五块均值） | 暖态Brier（五块均值） | 全边log loss（五块均值） | 状态等权Brier |',
        '|---|---:|---:|---:|---:|']
    for field,label in [('p','原公开先验'),('cold_prior','公开冷态规则＋先验'),('cold_half','公开冷态规则＋half')]:
        vals=[g['both']['calibration'] for g in mech['per_block'].values()]
        mean=lambda st,metric:statistics.mean(z[st][field][metric] for z in vals if z[st][field] is not None)
        lines.append(f"| {label} | {mean('all','brier'):.6f} | {mean('warm','brier'):.6f} | {mean('all','log_loss'):.6f} | {mech['state_balanced_brier'][field]['mean']:.6f} |")
    lines+=['','全边指标在块内对合法协作边计权，状态等权指标先在每状态求误差再等权平均；两者分母不同。全合法边真实缓存覆盖完整，只进入独立离线标签对象。log loss仅为数值评价将概率截到[10⁻¹²,1−10⁻¹²]，不修改在线概率或分支。该分布与旧400状态中step0占25%的预测权重不同，不能直接拼样本。','',
        '### 同分配、同配额换身份与不同配额','',
        '| 块 | 相同分配贡献 pp | 同配额不同身份贡献 pp | 不同配额贡献 pp | 总差 pp |',
        '|---|---:|---:|---:|---:|']
    for b,z in mech['category_contributions']['both'].items():
        c=z['contributions'];lines.append(f"| {b} | {c['same_assignment']:+.6f} | {c['same_quota_different_assignment']:+.6f} | {c['different_quota']:+.6f} | {z['total_pp']:+.6f} |")
    lines+=['','上表比较固定快照共同10 ms的Mλ−B0，三类贡献已逐块精确加回总差；不解释为闭环因果中介。','',
        '### 公开反演近似与近1概率','',
        '公开分解保持原代码近似，不用私有时延“修正”在线分支。对源轨迹选中的合法协作边，用离线真实命中标签核对公开反演与实际时延残差：','',
        '| 组 | 已检查边 | 最大绝对残差 s | 绝对残差>10⁻⁶边 | 中位残差 s |','|---|---:|---:|---:|---:|']
    for name,z in mech['public_inverse_stability'].items():
        fmt=lambda x:'NA' if x is None else f'{x:.6g}'
        lines.append(f"| {name} | {z['selected_edges_checked']} | {fmt(z['max_residual'])} | {z['above_1e6_absolute']} | {fmt(z['median_residual'])} |")
    lines+=['','这项残差只核对源策略实际选中的协作边；未选边不据此声称物理精确。分解失败边未删除，仍按原公开平均时延指示回退。','',
        '## 验收、停止决定与下一步','',
        f"- 全量原始审计通过：{audit['counts']['episodes']}回合、{audit['counts']['steps']}步、{audit['counts']['requests']}条接纳请求；检查费用注入、成功阈值、请求身份与计数、时钟窗口、十次预算、文件摘要、冻结上层和外生流配对。",
        '- 五项针对性验证通过；五场景全部短程通过；固定主样本覆盖40个区间且身份唯一，私有标签与公开候选分别落盘。审查由本次主代理执行，属于自审。',
        '- B1的新缺口已补齐，不因Brier变化启动新预测器或RL训练。B2以开发组件解释结束，不追加“单概率公式”确认性主张。',
        '- B3完成预定五场景开发矩阵。当前证据可报告零调参开发迁移的效果与费用，不能外推未见上层基座、全部负载算力组合或真实机载部署。可选6000回合完整迁移复验本轮停止在开发层；若论文需要升级为确认性跨场景主张，另行事先冻结检验族和资源预算再执行。',
        '- C阶段动态预算、训练、Git提交和推送均未启动。原20块确认结果、旧400状态、旧PPO九格和本轮开发证据分开保留。','',
        '## 可复核文件','',
        '- 协议、输入清单、种子核验、全部回合收据与原始gzip、抽样公开输入与原环境、候选/私有标签、块级CSV及JSON、图形及源表，均保留在本次独立结果目录。',
        '- 本地报告配套路径：`results/chapter5_followup_20261027/development_v2/`；验收记录：`output/chapter5_followup_20261027/validation/`。',
        '- 数值脚本：`scripts/analysis/chapter5_stage_b.py`、`chapter5_stage_b_analysis.py`；图形与报告脚本同目录。全部科研计算和绘图在指定SSH服务器执行。','']
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(lines))
    print(json.dumps({'report':str(a.output),'lines':len(lines)},ensure_ascii=False))


if __name__=='__main__':main()
