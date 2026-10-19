import argparse,json
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--launch',type=Path,required=True);arg=parser.parse_args();a=json.loads(arg.launch.read_text())['args'];root=Path(a['results_root']);v=Path(a['validation_root']);s=json.loads((root/'analysis/summary.json').read_text());audit=json.loads((v/'raw_record_audit.json').read_text());assert audit['status']=='PASS'
lines=['# 公开全miss规则与条件缓存校准：开发预测筛选','','本轮保留原18维full_logit_context特征、模型C=1及相同25折；不新增观测或模型容量。公开批次有本地option可见且所有可见本地缓存均miss时，测试概率置0；其余状态由模型预测。每折192个未触发规则的训练状态用于模型及标准化，原64个触发状态排除；16个测试状态仍全部覆盖，guard由公开option决定，不用时间索引或私有缓存。','',
'该规则与条件拟合作为整体方法测试。有效训练权重/标准化人口随筛选改变，不能把结果单独归为某一系数或架构。旧混合模型与原失败均保留。','',
'| 方法 | 全体Brier | 全体Log loss |','|---|---:|---:|']
for name,x in s['groups'].items():lines.append(f"| {name} | {x['brier']:.8f} | {x['log_loss']:.8f} |")
lines+=['','| 方法 | 未guard组Brier | 未guard组Log loss |','|---|---:|---:|']
for name,x in s['subsets']['non_guarded']['groups'].items():lines.append(f"| {name} | {x['brier']:.8f} | {x['log_loss']:.8f} |")
best=s['best_label_free_control'];b=s['brier_improvements'][best];w=s['non_guarded_vs_prior'];lines+=['',
f"相对六个无标签概率规则中最好者{best}，全体Brier减小{b['mean_improvement']:.8f}，五历史块描述性95% t区间[{b['ci95'][0]:.8f},{b['ci95'][1]:.8f}]，{b['positive']}/5块改善。",'',
f"在未guard组内，相对原prior的Brier减小{w['mean_improvement']:.8f}，描述性95% t区间[{w['ci95'][0]:.8f},{w['ci95'][1]:.8f}]，{w['positive']}/5块改善。因此本轮不只依靠原25%冷态权重获得筛选优势。",'',
f"预测筛选判断：{s['decision']['conditional_target_promising_development_only']}。这些数值是Brier差，不是DSR百分点。所有数据仍为复用开发状态，尚无条件匹配、实际软件费用、完整闭环或独立确认收益。",'',
'三个guard规则也提供给prior/own/half，保持同信息强对照；未见批次中全miss并不必然意味着所有邻居为空，本启发式仍可能误触发，必须闭环检验。','',
'全部原特征/标签/权重逐元素相同，公开guard逐状态重建；25个条件配方重拟合复现预测，标准化只看未guard训练行，全部留出行一次预测、规则元数据及按块统计复核通过。审计为主代理自审，不是独立审查。模型声明需要显式支持该context rule的适配器；当前校验器拒绝旧默认调用方式，不能用旧冻结推理协议直接加载新模型。','',
f"协议SHA `{audit['protocol_sha256']}`；统计SHA `{audit['summary_sha256']}`。",'',
'持续目标未完成。后续必须保留全部强规则对照，重新校准训练期价格、核验原生匹配与软件费用，再做全新负载闭环和独立确认。原十块确认、论文和Git发布不改动。','',
'[主统计](../../../results/sf_hlm_learning_feasibility_20261016/cache_regime_20261018/training_v1/analysis/summary.json) · [原始审计](../validation/cache_regime_20261018/raw_record_audit.json)']
out=v.parent.parent/'deliverables/cache_regime.md';out.write_text('\n'.join(lines)+'\n');print(out)
