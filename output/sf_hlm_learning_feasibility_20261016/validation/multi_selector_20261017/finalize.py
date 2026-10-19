"""Attach verified execution details to the remote report, preserve metrics."""
import argparse
import hashlib
import json
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('--results-root',type=Path,required=True)
ap.add_argument('--validation-root',type=Path,required=True);ap.add_argument('--report',type=Path,required=True)
a=ap.parse_args();summary=json.loads((a.results_root/'summary.json').read_text())
audit=json.loads((a.validation_root/'raw_record_audit.json').read_text())
assert audit['status']=='PASS'
assert not summary['decision']['proceed_to_full_path_timing']
text=a.report.read_text();assert '**完成验证与归因核对。**' not in text
text+='''
**完成验证与归因核对。**

- 本机原8项加新增6项合同测试全部通过；服务器新增6项全部通过。
- 另一实现核对1,200项benchmark/state/input摘要，独立构造每状态63维特征；重拟合100模型，并从6,000分支和18,000次原始时间重算计费。25折身份、训练折随机比例及1,575项块级分数全部核对通过。该复算由同一主代理完成，不称独立代理审查。
- 含学习小树相对非学习小树仅+0.021 pp，95% t区间[−0.003,+0.045]；含学习Ridge相对非学习Ridge为−0.049 pp，区间[−0.160,+0.062]。不能把加入PPO分支称为已证实的正贡献。
- 把含学习选择器选中的L8/L16换成对应R8/R16后，原选择器相对替换对照：Ridge −0.017 pp，小树 −0.014 pp，区间均跨零。当前没有支持“PPO搜索不可替代”的证据。
- 表现最接近G的含学习小树仍调用搜索10.25%的状态，其中PPO占全部状态2.25%；相对经验择优累计损失89.07个期望成功请求。这是对平均随机/计时结果求得的计数，可以为非整数，不是独立样本量。
- 原分支完整费用已计入；本轮未新增模拟或完整选择器计时。预言机在额外2 ms下仍约+0.519 pp，而四候选在零额外费用下均未实现正收益。本轮主要未兑现的是选择能力，不能靠仅声称门控很快来解决。
- 仓库布局检查仍因既存根目录`修改总结报告.md`失败，本任务未修改、移动或扩充白名单；本轮新增文件均在约定子目录。

本轮探索到此结束，保留全部负结果。继续研究的可审查候选是已有多步教师检查点的同状态、计全费用比较；它需要另行冻结基线BC/PPO与多步BC/PPO的归因设计，不能由本轮门控失败推断结果，也不在本次自动启动。复杂门控、进一步调阈值或更换网络不作为当前结果下的优先投入。

可复核文件：[冻结机器协议](../../../results/sf_hlm_learning_feasibility_20261016/multi_selector_20261017/release1/protocol.json)、[完整统计](../../../results/sf_hlm_learning_feasibility_20261016/multi_selector_20261017/release1/summary.json)、[25折模型与身份](../../../results/sf_hlm_learning_feasibility_20261016/multi_selector_20261017/release1/folds.json)、[逐状态结果](../../../results/sf_hlm_learning_feasibility_20261016/multi_selector_20261017/release1/states.json)、[复算验收](../validation/multi_selector_20261017/raw_record_audit.json)、[执行计划](../../../docs/plans/sf_hlm_learning_feasibility_20261016.md)。
'''
a.report.write_text(text)
files=[p for p in a.results_root.iterdir() if p.is_file()]+[a.report,a.validation_root/'raw_record_audit.json']
completion=dict(status='COMPLETED_AND_VALIDATED',stage='multi-method selector development screen',
    decision='STOP_NO_FULL_PATH_TIMING_OR_CLOSED_LOOP',states=400,folds=25,models=100,
    original_results_unchanged=True,thesis_unchanged=True,no_RL_training=True,independent_confirmation=False,
    independent_agent_review=False,raw_record_audit=audit,
    files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
(a.validation_root/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
print(json.dumps(dict(status=completion['status'],decision=completion['decision'],report=str(a.report))))
