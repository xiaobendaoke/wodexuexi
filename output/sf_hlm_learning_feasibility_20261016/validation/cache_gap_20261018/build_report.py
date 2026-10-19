"""Render the audited one-feature prediction experiment on the server."""
import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--launch',type=Path,required=True);o=p.parse_args();args=json.loads(o.launch.read_text())['args'];r=Path(args['results_root']);v=Path(args['validation_root']);read=lambda p:json.loads(p.read_text());s=read(r/'analysis/summary.json');a=read(v/'raw_record_audit.json');assert a['status']=='PASS' and a['summary_sha256']==hashlib.sha256((r/'analysis/summary.json').read_bytes()).hexdigest()
lines=['# 公开fetch时延差：单一特征的条件校准检验','','在原18维后仅增加log1p(公开miss_base−hit_base)，不读取服务ID、实际文件大小或私有缓存。额外时延受文件大小和链路共同影响，不能与文件大小混同。原前18维/标签/权重/划分及公开guard逐元素不变；同C=1逻辑回归，每折192非guard训练状态，25折各16留出状态。旧18维运行器显式拒绝新19维schema。','',
'| 方法 | 留出Brier | 留出log loss |','|---|---:|---:|']
for m,x in s['groups'].items():lines.append(f"| {m} | {x['brier']:.8f} | {x['log_loss']:.8f} |")
e=s['legacy_eighteen_dimensional_improvement'];lines+=['',f"相对原18维Brier改善={e['mean_improvement']:.8f}，五历史块描述性95% t区间[{e['ci95'][0]:.8f},{e['ci95'][1]:.8f}]，{e['positive']}/5块正。预定晋级还要求改善至少0.001，而不能只凭方向一致或区间为正晋级。",'',
f"预测筛选通过：{s['decision']['public_gap_feature_promising_development_only']}。新特征带来的改善较小；未进行任务匹配、原生计时或闭环，不将预测改善改称实际DSR收益。",'',
'| 固定子集 | 新19维Brier | 原18维Brier | cold_prior Brier |','|---|---:|---:|---:|']
for name,x in s['subsets'].items():
 g=x['groups'];lines.append(f"| {name} | {g['learned']['brier']:.8f} | {g['legacy_learned']['brier']:.8f} | {g['cold_guard_prior']['brier']:.8f} |")
lines+=['','敏感集合来自上一轮已封存配额，未按新预测改变；完整五块与log loss保留在JSON。原公开反演在先验近1时的精度限制保留，未用私有直接分量替代。全部13,459条新公开特征重构、原数组核对、25次配方重拟合、标准化和组别统计自审通过。五个历史上层块只作开发描述，不能把折或边当独立训练重复。','',
f"协议SHA `{a['protocol_sha256']}`；统计SHA `{a['summary_sha256']}`。",'',
'[完整预测统计](../../../results/sf_hlm_learning_feasibility_20261016/cache_gap_20261018/training_v1/analysis/summary.json) · [原始自审](../validation/cache_gap_20261018/raw_record_audit.json)']
if not a['prediction_screen']:lines+=['','本单特征候选按预定门槛停止，不追加特征或改阈值求正。原确认和论文不改，持续研究目标尚未完成。']
out=v.parent.parent/'deliverables/cache_gap.md';out.write_text('\n'.join(lines)+'\n');print(out)
