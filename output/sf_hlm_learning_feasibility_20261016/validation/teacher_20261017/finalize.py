"""Format the validated remote result; never change scientific result JSONs."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())


ap=argparse.ArgumentParser();ap.add_argument('--results-root',type=Path,required=True)
ap.add_argument('--validation-root',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
r=a.results_root;v=a.validation_root;s=read(r/'analysis/summary.json');p=read(r/'protocol.json');audit=read(v/'raw_record_audit.json')
assert audit['status']=='PASS' and audit['summary_sha256']==sha(r/'analysis/summary.json')
d=s['decision'];all_=s['subsets']['all'];comp=all_['comparisons']


def effect(stat):
    return f"{stat['mean']*100:+.3f} pp，95% t区间[{stat['ci95'][0]*100:+.3f}, {stat['ci95'][1]*100:+.3f}]，{stat['positive']}/5块正向"


quality='通过原开发质量候选门槛，仍需另行独立确认' if d['quality_candidate'] else '未通过原开发质量候选门槛'
cost='通过费用侧开发门槛，仍需重做开发闭环与独立确认' if d['cost_candidate'] else '未解决计全费用后相对贪心的实用收益问题'
lead=f"**决定：多步PPO{quality}；{cost}。**\n\n本轮只评价已有模型，未启动重训、门控、开发闭环或独立确认，论文未修改。\n\n"
text=a.report.read_text();assert '**完成核验与结果解释。**' not in text
text=lead+text
text=text.replace('代替交互。\n-','代替交互。\n\n-')
text+='\n**完成核验与结果解释。**\n\n'
text+=f"- 最终教师干预（多步PPO−基线PPO）：原始{effect(comp['multistep_ppo-baseline_ppo']['raw'])}；费用后{effect(comp['multistep_ppo-baseline_ppo']['fee'])}。\n"
text+=f"- 教师初始化（多步BC−基线BC）：原始{effect(comp['multistep_bc-baseline_bc']['raw'])}；费用后{effect(comp['multistep_bc-baseline_bc']['fee'])}。\n"
text+=f"- 基线初始化后的PPO更新：原始{effect(comp['baseline_ppo-baseline_bc']['raw'])}。多步初始化后的PPO更新：原始{effect(comp['multistep_ppo-multistep_bc']['raw'])}。\n"
text+=f"- 多步PPO相对强随机：原始{effect(comp['multistep_ppo-R16']['raw'])}；费用后{effect(comp['multistep_ppo-R16']['fee'])}。\n"
text+=f"- 多步PPO相对贪心：费用后{effect(comp['multistep_ppo-G']['fee'])}。全四模型费用侧晋级情况："+', '.join(k+('通过' if flag else '未过') for k,flag in d['all_four_cost_candidates'].items())+'。\n'
bc=all_['groups']['multistep_bc'];ppo=all_['groups']['multistep_ppo']
text+=f"- 多步初始化后的PPO更新使平均匹配次数从{bc['calls']:.3f}降至{ppo['calls']:.3f}，平均完整调用时间从{bc['mean_seconds']*1000:.2f}降至{ppo['mean_seconds']*1000:.2f} ms。对应费用后差为{effect(comp['multistep_ppo-multistep_bc']['fee'])}；区间跨零，不能称已确认稳定的费用收益。\n"
text+='\n本轮留下两类不同证据：多步BC有原始质量增益信号，同时付出更高软件费用；其后PPO的描述性作用包括减少搜索调用，原始质量没有给出清晰的额外增益。不能把“教师完全无效”或“PPO完全没有作用”当成本轮结论。原冻结主候选多步PPO未过闸门，BC的正向原始差只记录为质量—费用取舍，不临时改门槛或替换主候选。\n'
text+='\n上述同状态比较不复现原开发整回合轨迹上的绝对DSR；原开发多步PPO−基线PPO约+0.209 pp的闭环发现保留。状态固定后去除了策略改变后续状态的作用，四个采样时刻也不能代表1000步完整回合，不能用本轮正负方向改写原闭环事实。\n'
text+='\n| 状态来源 | 比较 | 原始差/pp | 费用后差/pp |\n|---|---|---:|---:|\n'
for source in ('greedy','learned'):
    for name in ('multistep_ppo-baseline_ppo','multistep_ppo-multistep_bc','multistep_ppo-R16','multistep_ppo-G'):
        q=s['subsets'][source]['comparisons'][name]
        text+=f"| {source} | {name} | {q['raw']['mean']*100:+.3f} | {q['fee']['mean']*100:+.3f} |\n"
text+='\n| 方法 | MBS请求占比/% | 公平性 |\n|---|---:|---:|\n'
for name,g in all_['groups'].items():text+=f"| {name} | {g['mbs_share']*100:.3f} | {g['fairness']:.5f} |\n"
text+='\nMBS以已接纳请求为分母，先按块汇总；公平性为块内状态均值再对块等权。能耗为原仿真即时机队能耗，不换算实际机载功率或长期续航。\n'
text+='\n| 开发块 | 多步教师训练图数 | 额外离线匹配调用 | 路径标签数 |\n|---|---:|---:|---:|\n'
for b in p['offline_teacher']:text+=f"| {b['block']} | {b['graphs']} | {b['matching_calls']} | {b['labels']} |\n"
text+='\n标签可能来自同图的多个路径步骤，不能充当独立训练图；监督样本数和PPO抽样机会虽配对，多步教师额外离线搜索仍须计入训练成本。\n'
text+='\n**验收。** 本机19项轻量合同测试通过，服务器新增5项测试通过；8状态烟雾仅验证兼容性。正式400状态、20个BC/PPO检查点、1600项归档搜索等价核验、2800项原控制分支复现及13200次计时全部完成。另一实现逐请求复算全部费用、块级统计、交互与Holm敏感性，核对模型指纹和输入摘要；这是主代理自审，不称独立代理审查。首次启动清单路径错误发生在输入清点，已用摘要一致的原发布清单修正并保留日志，没有产生需排除的实验样本。\n'
text+='\n仓库布局检查仍因既存根目录`修改总结报告.md`失败，本轮未修改该文件或白名单。原确认、原结果和论文未改写。所有结论限于五个复用开发块和已冻结状态集合。\n'
text+='\n可复核文件：[机器协议](../../../results/sf_hlm_learning_feasibility_20261016/teacher_20261017/release1/protocol.json)、[完整统计](../../../results/sf_hlm_learning_feasibility_20261016/teacher_20261017/release1/analysis/summary.json)、[逐状态汇总](../../../results/sf_hlm_learning_feasibility_20261016/teacher_20261017/release1/analysis/states.json)、[输入血缘清点](../validation/teacher_20261017/inventory.json)、[复算验收](../validation/teacher_20261017/raw_record_audit.json)、[执行计划](../../../docs/plans/sf_hlm_learning_feasibility_20261016.md)。\n'
a.report.write_text(text)
files=[r/'protocol.json',r/'status.json',r/'analysis/summary.json',r/'analysis/states.json',v/'inventory.json',v/'raw_record_audit.json',a.report]
verdict=dict(status='COMPLETED_AND_VALIDATED',states=400,branch_instances=4400,timing_samples=13200,
    quality_candidate=d['quality_candidate'],cost_candidate=d['cost_candidate'],no_new_training=True,
    no_automatic_next_stage=True,independent_confirmation=False,independent_agent_review=False,
    thesis_unchanged=True,original_results_unchanged=True,raw_record_audit=audit,
    files={str(path):sha(path) for path in files})
(v/'completion.json').write_text(json.dumps(verdict,indent=2)+'\n')
print(json.dumps({k:verdict[k] for k in ('status','quality_candidate','cost_candidate','no_automatic_next_stage')}))
