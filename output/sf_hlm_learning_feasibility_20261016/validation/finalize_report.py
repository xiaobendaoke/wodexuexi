"""Finalize this task's interpretation without changing frozen raw results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics


def read(p):return json.loads(Path(p).read_text())


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--results-root',type=Path,required=True)
    p.add_argument('--validation-root',type=Path,required=True);p.add_argument('--deliverables',type=Path,required=True)
    a=p.parse_args();os.sched_setaffinity(0,{4});r=a.results_root
    s=read(r/'analysis/summary.json');audit=read(a.validation_root/'raw_record_audit.json')
    assert s['status']=='PASS' and audit['status']=='PASS' and audit['summary_sha256']==sha(r/'analysis/summary.json')
    all_=s['subsets']['all'];bs=all_['blocks'];cs=all_['comparisons'];g=s['decision']
    assert not g['gate_probe_space'] and not any(q['practical'] for q in g['budget'].values())
    decision={'execution_status':'PASS','decision':'STOP_AT_STAGE_0B_FOR_TRUNCATION_AND_BINARY_GATE',
        'reasons':['L8/R8/H8 fail the prespecified measured-fee gain versus G.',
                   'L8 mean raw-quality loss versus L16 exceeds the 0.1pp descriptive margin.',
                   'G-versus-L8/L16 empirical binary oracle mean gains are below 0.5pp before extra gate fees.'],
        'important_open_boundary':'Seven-way empirical oracle exceeds 0.5pp on average; no multi-method gate was fitted or ruled out.',
        'not_a_population_impossibility_proof':True,'no_next_stage_launched':True,
        'summary_sha256':sha(r/'analysis/summary.json'),'raw_audit_sha256':sha(a.validation_root/'raw_record_audit.json'),
        'finalization_source_sha256':sha(__file__)}
    (r/'analysis/gate_decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n')
    def percent(m):return all_['groups'][m]['mean']*100
    def mean(m,k):return statistics.mean(q['methods'][m][k] for q in bs.values())
    def contrast(k):
        q=cs[k];return f"{q['mean']*100:+.3f} pp，95% t区间 [{q['ci95'][0]*100:+.3f}, {q['ci95'][1]*100:+.3f}]，{q['positive']}/{q['n']}块正向"
    lines=['**SF-HLM学习增量诊断：阶段0a/0b已完成**','',
        '**按预定开发门槛，本轮停在0b：预算8尚未兑现相对贪心的费用后服务优势，二选一门控不进入拟合。**',
        '这是指定开发状态及指定服务器测量条件下的筛选决定，不是学习普遍无效或任何门控都无空间的证明。','',
        f"完成五个历史开发块、400状态（学习/贪心来源各200）、6000分支实例和18000次正式计时。采样状态生成计数之和为{sum(q['generated'] for q in bs.values()):,}；它不是独立样本数。",'',
        '| 方法 | 原始成功率/% | 计费后成功率/% | 计费后相对G/pp | 平均调用/ms | 块p95均值/ms | 匹配次数/状态 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m in ['G','L16','L8','R16','R8','H16','H8']:
        delta=0 if m=='G' else cs[m+'-G']['mean']*100
        lines.append(f"| {m} | {mean(m,'raw_rate')*100:.3f} | {percent(m):.3f} | {delta:+.3f} | {mean(m,'mean_seconds')*1000:.2f} | {mean(m,'p95_seconds')*1000:.2f} | {mean(m,'calls_per_state'):.3f} |")
    lines+=['',
        'G为贪心初始化加普通修复；L为冻结学习搜索；R为同框架强配额随机；H为配额启发式。8/16包含初始匹配；预算8仍使用预算16的完整原输入语义，仅外部截断。',
        '这些是采样状态的请求计数汇总。状态包含预定初始时刻，并混合两种策略来源，不能拿本表绝对值替换原整回合DSR或原确认结果。','',
        '| 方法 | 即时UAV机队能耗/状态 | 能耗比G | MBS请求占比/% | 公平性 |',
        '|---|---:|---:|---:|---:|']
    for m in ['G','L16','L8','R16','R8','H16','H8']:
        ratio=statistics.mean(q['methods'][m]['energy']/q['methods']['G']['energy'] for q in bs.values())
        lines.append(f"| {m} | {mean(m,'energy_per_state'):.2f} | {ratio:.4f} | {mean(m,'mbs_share')*100:.3f} | {mean(m,'fairness_per_state'):.4f} |")
    lines+=['',
        '能耗为原仿真的J值，仅作固定快照即时比较，不换算真实机载功率或长期续航；比值先按训练块计算再取均值。MBS按块内分配请求计数汇总，公平性先按状态平均，再按块等权。','',
        '**预算收缩的结果。**',
        f"L8相对G：{contrast('L8-G')}。L16相对G：{contrast('L16-G')}。R8相对G：{contrast('R8-G')}。",'',
        f"L8平均时间从{mean('L16','mean_seconds')*1000:.2f}降至{mean('L8','mean_seconds')*1000:.2f} ms，但其原始质量相对L16为{contrast('raw_L8-L16')}；平均损失超过预定0.1 pp描述门槛。不能称预算减半无损。",'',
        f"L8原始成功率仍比G高{(mean('L8','raw_rate')-mean('G','raw_rate'))*100:.3f} pp；即时代码仿真UAV机队能耗比G为{g['budget']['L8']['actual_energy_ratio']:.4f}。因此这里存在原始质量、即时能耗和软件费用的取舍，不是所有指标均差。",'',
        f"L8相对R8的费用后差为{contrast('L8-R8')}；不足以声称这两个低预算分支存在稳健的总体优劣。五块双侧精确符号翻转检验的离散性必须保留，例如L8−G虽5/5为负，精确双侧p仍为0.0625。这里按预定实用门槛作开发决策，不作独立确认的显著性结论。",'',
        '**门控空间须区分二选一和多方法。**','',
        '| 零额外门控费用的经验预言机 | 相对G/pp | 95% t区间/pp |',
        '|---|---:|---|']
    for label,key in [('G与L8二选一','oracle_L8_G-G'),('G与L16二选一','oracle_L16_G-G'),
                      ('五个非学习分支择优','oracle_nonlearning-G'),('全部七分支择优','oracle_all-G')]:
        q=cs[key];lines.append(f"| {label} | {q['mean']*100:+.3f} | [{q['ci95'][0]*100:+.3f}, {q['ci95'][1]*100:+.3f}] |")
    lines+=['',
        '本轮拟议门控是G与一个学习分支的二选一。其零额外费用经验增益均未达到+0.5 pp，故不进入阶段1。区间较宽，不代表总体上限已被精确证明低于0.5 pp。',
        f"全部七分支经验预言机相对非学习预言机增加{cs['oracle_all-oracle_nonlearning']['mean']*100:.3f} pp。这保留了未排除的多方法选择空间；其事后择优结果不是可部署策略、未计新增特征和门控费用，也没有验证可预测性或闭环收益。不能把二选一停止决定扩写为所有门控均无效。",'',
        '| 额外费用假设/ms | G与L8二选一经验增益/pp | G与L16二选一经验增益/pp |',
        '|---:|---:|---:|']
    for ms in [0,.5,1,2,5]:
        l8=cs[f'oracle_L8_G_overhead_{ms}ms-G']['mean']*100;l16=cs[f'oracle_L16_G_overhead_{ms}ms-G']['mean']*100
        lines.append(f'| {ms} | {l8:+.3f} | {l16:+.3f} |')
    lines+=['',
        '费用假设通过逐请求重新判定截止期得到，不是从DSR中减去毫秒。预先固定的“批次至少16则L8，否则G”规则，即使暂不计新增规则封装费，仍相对G下降'+f"{-cs['rule_L8_G-G']['mean']*100:.3f} pp。这只否定该条固定规则，没有检验其他特征能否预测收益。",'',
        '**两种状态来源。**','',
        '| 状态来源 | G费用后成功率/% | L8费用后成功率/% | L16费用后成功率/% |',
        '|---|---:|---:|---:|']
    for label,key in [('学习轨迹','learned'),('贪心轨迹','greedy')]:
        q=s['subsets'][key]['groups'];lines.append(f"| {label} | {q['G']['mean']*100:.3f} | {q['L8']['mean']*100:.3f} | {q['L16']['mean']*100:.3f} |")
    lines+=['',
        '两种来源下L8、L16均低于G；不是只在某一策略自身状态上出现的排序。仍只支持本次来源混合和四个预定时刻，不外推全部闭环状态。','',
        '**验证及下一步。**','',
        '- 阶段0a：100份原开发回合收据及所登记文件摘要核验通过；复用200学习快照，精确重放50贪心回合采集200快照。',
        '- 8状态烟雾通过；本机及服务器各8项合同测试通过。正式500任务收据、1200项前缀核验、原分配/指标重放及五块冻结模型指纹核验通过。',
        '- 另一条复算程序重算6000个分支的逐请求原始/费用后成功数、18000个时间样本，以及全部块级差、区间、精确符号翻转和预言机；核对1600个状态文件摘要。此为同一主代理的另一实现复算，不冒充独立代理审查。',
        '- 继续停在阶段0b，不训练门控、不追加RL训练、不启动独立确认、不自动修改论文。若考虑七方法选择器或更低预算，需要另行冻结新候选与实用目标，不能放宽本轮门槛后称通过。',
        '- 软件等待仍只作事后截止期评分，没有回写位置、缓存、电量、持续队列或下一状态；结果不证明实际部署或等待反馈后的闭环效果。',
        '- 多步教师候选尚未在本轮比较，不能由本结果宣布该候选或跨时隙学习已被否定。','',
        '完整统计与逐状态数据在结果目录的analysis/summary.json、analysis/state_summary.json；任务原始记录在tasks/；复算验收在validation/raw_record_audit.json；冻结协议与源码摘要见对应protocol.json及release_manifest.json。','']
    a.deliverables.mkdir(parents=True,exist_ok=True);(a.deliverables/'analysis.md').write_text('\n'.join(lines))
    validation={'status':'PASS','scope':'stage0a/0b completed; next stages not started','summary_sha256':sha(r/'analysis/summary.json'),
        'report_sha256':sha(a.deliverables/'analysis.md'),'gate_decision_sha256':sha(r/'analysis/gate_decision.json'),
        'raw_audit_sha256':sha(a.validation_root/'raw_record_audit.json'),'states':400,'tasks':500,'instances':6000,'timing_samples':18000,
        'review':'explicit self-review plus independent record recomputation by same agent',
        'unrelated_layout_failure':'Existing root file 修改总结报告.md is outside allowlist; unchanged.'}
    (a.validation_root/'completion.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(validation,ensure_ascii=False))


if __name__=='__main__':main()
