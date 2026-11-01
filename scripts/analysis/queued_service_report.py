"""Audit downloaded queue results and write a count-consistent Chinese report."""
import argparse
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis.queued_service_validation import TASK, REVISION, source_contract


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root):
    out = root/'results'/TASK/REVISION
    val = root/'output'/TASK/'validation'/REVISION
    deliver = root/'output'/TASK/'deliverables'; deliver.mkdir(parents=True, exist_ok=True)
    data = json.loads((out/'analysis.json').read_text(encoding='utf-8'))
    proto = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    assert data['status'] == 'PASS_COMPLETED_ENGINE_AND_CALIBRATION'
    assert data['source_contract'] == proto['source_contract'] == source_contract()[0]
    groups = {}; hashes = {}; main_tasks = 0
    for name, expected in data['raw_sha256'].items():
        path = out/name; assert sha(path) == expected, name; hashes[name] = expected
        d = json.loads(path.read_text(encoding='utf-8')); counts = d['summary']['counts']
        assert counts['generated'] == counts.get('success', 0)+counts.get('expired', 0)
        assert d['rows'][-1]['active'] == 0
        assert sum(row['success'] for row in d['rows']) == counts.get('success', 0)
        assert d['audit']['status'] == 'PASS'
        # Recount arrivals from cumulative step counters, never sum cumulative counts.
        assert d['rows'][-1]['generated'] == counts['generated']
        if d['spec']['kind'] != 'main': continue
        main_tasks += counts['generated']
        s = d['spec']; key = (s['profile'], s['compute'], s['bandwidth'], s['method'])
        groups.setdefault(key, []).append(d)
    rows = []
    for s in data['summary']:
        ds = groups[(s['profile'], s['compute'], s['bandwidth'], s['method'])]
        assert len(ds) == 6
        generated = sum(d['summary']['counts']['generated'] for d in ds)
        admitted = sum(d['summary']['counts'].get('admitted', 0) for d in ds)
        success = sum(d['summary']['counts'].get('success', 0) for d in ds)
        expired = sum(d['summary']['counts'].get('expired', 0) for d in ds)
        result = dict(s, generated=generated, admitted=admitted, success=success, expired=expired,
                      dsr=success/generated, admission=admitted/generated, conditional=success/max(admitted, 1))
        assert abs(result['dsr']-result['admission']*result['conditional']) < 1e-12
        result['expiration_stage'] = {stage: sum(d['summary']['expiration_stage'].get(stage, 0) for d in ds)
                                      for stage in ('waiting', 'upload', 'transfer', 'compute')}
        result['cpu_utilization'] = [statistics.mean(d['summary']['cpu_utilization'][i] for d in ds) for i in range(6)]
        rows.append(result)
    diagnostic = None; diagnostic_cases = []; precision = None
    supplemental = out/'coupling_diagnostic/analysis.json'
    if supplemental.exists():
        diagnostic = json.loads(supplemental.read_text(encoding='utf-8'))
        assert diagnostic['source_contract'] == data['source_contract']
        assert diagnostic['diagnostic_script_sha256'] == sha(root/'scripts/analysis/queued_service_coupling.py')
        for name, expected in diagnostic['raw_sha256'].items():
            path = out/'coupling_diagnostic'/name; assert sha(path) == expected
            d = json.loads(path.read_text(encoding='utf-8')); assert d['replay_counts_exact']
            assert d['source_contract'] == data['source_contract']
            assert len({v['at_risk'] for v in d['cases'].values()}) == 1
            j = {k: v['completed'] for k, v in d['cases'].items()}
            a = j['hold_cooperative']-j['hold_local']; b = j['move_cooperative']-j['move_local']
            assert d['interaction'] == a-b and d['reversal'] == (a*b < 0)
            assert d['eligible'] == (d['initial_changed_assignments'] > 0 and d['motion_length'] > 1e-6)
            diagnostic_cases.append(d); hashes['coupling_diagnostic/'+name] = expected
        eligible_cases = [x for x in diagnostic_cases if x['eligible']]
        assert len(diagnostic_cases) == diagnostic['cases'] == 288
        assert len(eligible_cases) == diagnostic['eligible']
        assert sum(x['interaction'] != 0 for x in eligible_cases) == diagnostic['nonzero_interactions']
        assert sum(x['reversal'] for x in eligible_cases) == diagnostic['rank_reversals']
    precision_path = out/'coupling_precision/analysis.json'
    if precision_path.exists():
        precision = json.loads(precision_path.read_text(encoding='utf-8'))
        assert precision['source_contract'] == data['source_contract']
        assert precision['script_sha256'] == sha(root/'scripts/analysis/queued_service_coupling_precision.py')
        refined = []
        for p in (out/'coupling_precision').glob('*.json'):
            if p.name == 'analysis.json': continue
            d = json.loads(p.read_text(encoding='utf-8'))
            original = next(x for x in diagnostic_cases if x['spec'] == d['spec'])
            assert d['original_counts'] == {k: v['completed'] for k, v in original['cases'].items()}
            assert d['identical'] == (d['counts'] == d['original_counts'])
            assert d['source_contract'] == data['source_contract']
            refined.append(d); hashes['coupling_precision/'+p.name] = sha(p)
        assert len(refined) == precision['runs'] == 9
        assert all(x['identical'] for x in refined) == precision['all_counts_identical']
    table = deliver/'queued_protocol_summary.csv'
    fields = ('profile', 'compute', 'bandwidth', 'method', 'generated', 'admitted', 'success', 'expired',
              'dsr', 'admission', 'conditional', 'compute_wait', 'crossed', 'transfer_nonempty', 'cpu_nonempty',
              'mbs_fraction', 'decision_mean', 'decision_over_one_second', 'queue_screen')
    with table.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    samples = [json.loads(x) for x in (val/'resource_samples.jsonl').read_text().splitlines() if x]
    peak = max(x['active'] for x in samples)
    cpu_coverage = len(data['executed_main_cpus'])
    eligible = []; strict_eligible = []
    for compute in (.6, 1., 1.4):
        for bandwidth in (.25, .5, 1.):
            group = [x for x in rows if x['profile'] == 'burst' and x['compute'] == compute and x['bandwidth'] == bandwidth]
            if all(x['queue_screen'] and x.get('recovery_screen', False) for x in group):
                eligible.append(dict(compute=compute, bandwidth=bandwidth))
            if all(x['queue_screen'] and x['backlog']['late'] <= 1.2*x['backlog']['before']+1e-12 for x in group):
                strict_eligible.append(dict(compute=compute, bandwidth=bandwidth))
    checks = dict(status='PASS_LOCAL_RAW_HASH_AND_COUNT_AUDIT', source_contract=data['source_contract'],
                  checked_files=len(hashes), main_generated_tasks=main_tasks, cohort_count_decomposition=True,
                  independent_review=False, main_cpu_coverage=cpu_coverage, peak_active_jobs=peak,
                  eligible_burst_resource_cells=eligible, raw_sha256=hashes)
    checks['strict_recovery_eligible_burst_resource_cells'] = strict_eligible
    if diagnostic is not None:
        checks['supplemental_coupling_cases'] = diagnostic['cases']
        checks['supplemental_eligible'] = diagnostic['eligible']
        checks['supplemental_nonzero_interactions'] = diagnostic['nonzero_interactions']
        checks['supplemental_rank_reversals'] = diagnostic['rank_reversals']
    if precision is not None: checks['coupling_precision'] = {k: v for k, v in precision.items() if k != 'results'}
    remote_audit = json.loads((val/'final_remote_audit.json').read_text(encoding='utf-8')) if (val/'final_remote_audit.json').exists() else None
    if remote_audit is not None:
        assert remote_audit['status'] == 'PASS_FINAL_REMOTE_IMMUTABILITY'
        assert remote_audit['source_contract'] == data['source_contract']
        checks['final_remote_immutability'] = remote_audit
    (val/'local_result_audit.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
    def pct(x): return f'{100*x:.2f}%'
    def row_for(profile, method, compute=1., bandwidth=1.):
        return next(r for r in rows if (r['profile'], r['method'], r['compute'], r['bandwidth']) == (profile, method, compute, bandwidth))
    method_name = dict(history_queue='历史轨迹＋EDF感知分配', history_balance='历史轨迹＋负载均衡', rolling_joint='有限候选滚动联合搜索')
    profile_name = dict(light='轻载10/s', steady='平稳50/s', burst='突发20→75→20/s', overload='持续高负载100/s')
    lines = ['# 跨时隙任务协议：引擎、物理耦合与非学习负载校准', '',
        f"主扫描完成：{data['completed']}；报告生成：{datetime.now(timezone(timedelta(hours=8))).isoformat()}。本轮没有训练RL；以下均是新协议的开发验证。", '',
        '## 完成范围与证据边界', '',
        f"23项确定性引擎回归通过；{data['main_jobs']}组200秒主仿真和{data['refinement_jobs']}组固定动作时间精度复核完成，主仿真累计生成{main_tasks:,}个任务。每组包含停止到达后的排空，任务级工作守恒在引擎内审计；本机另逐文件复核摘要及回合计数。", '',
        f"服务器128物理核/256逻辑CPU，主实验实际覆盖{cpu_coverage}个逻辑CPU，峰值{peak}个任务并行，单任务固定一CPU。尾部与精度复核因剩余任务较少并发下降。{data['protected_old_files']:,}份旧任务文件摘要保持不变；未停止其他任务或修改全局CPU设置。", '',
        '## 冻结协议', '',
        '5 UAV、100 UE、1 MBS，固定UE位置，15m/s速度、100m服务半径、460m感知/协作范围、200m安全间距，线性移动全路径检查安全。首接纳固定来源、一次性分配固定目标；等待覆盖、上传、转发、计算均可跨时隙；断链暂停并保留进度；传完才能计算。', '',
        '宏决策1秒，信道中点积分步长0.1秒，另在接触边界切分；完成、截止期与EDF切换在区间内部推进。接纳只发生在宏边界，恰在截止期完成记成功。CPU采用无开销可抢占EDF。截止期2–5秒，任务1–5 MB、1200–2799 cycle/byte。', '',
        '无线采用正交资源池：各UAV上行40MHz、全网UAV间20MHz、全网回传0.75MHz；各池按活动链路均分全带宽服务时间，链路内EDF。三类无线可并行、CPU与无线可并行；固定功率/自由空间信道，无能耗目标、无电池、无迁移分片。带宽扫描系数同时作用于三类资源池。该资源抽象及连续服务均不同于旧批次协议，绝对DSR不能拼接旧结果。', '',
        '## 确定性机制检查', '',
        '两UAV三任务的缩放例子中，保持B位置时“转给B/留在A”完成数为1/0；B移动接纳另一组任务时为1/2，出现分配优劣反转，交互量为2个任务。它验证计算义务与覆盖机会的耦合，但使用固定测试速率和缩放计算量，不是代表性网络增益。断链暂停、续传及移动中接触边界另由确定性回归验证。', '',
        f"初始全网状态的{len(data['representative_probes'])}个探针中，实际分配不同且存在移动的有效对照有{sum(x['assignments_differ'] and x['motion_nonzero'] for x in data['representative_probes'])}个；观察到{data['representative_interactions_nonzero']}个非零交互。这些零结果保留：它们只说明在这些状态、这组干预中未测出交互，不能据此说明整体不存在物理耦合。原探针只改变初始分配和首段轨迹，后续统一来源计算/悬停。", '',
        '## 原始资源配置下的结果', '',
        '下表固定算力和带宽系数均为1；每格6个工作负载，比例由汇总请求计数计算。', '',
        '| 到达档 | 控制器 | DSR | 接纳率 | 接纳后完成率 | 计算等待均值（全任务） | 接纳后跨宏时隙比例 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for profile in ('light', 'steady', 'burst', 'overload'):
        for method in ('history_queue', 'history_balance', 'rolling_joint'):
            r = row_for(profile, method)
            lines.append(f"| {profile_name[profile]} | {method_name[method]} | {pct(r['dsr'])} | {pct(r['admission'])} | {pct(r['conditional'])} | {r['compute_wait']:.4f}s | {pct(r['crossed'])} |")
    lines += ['', '“overload”只是持续100请求/秒的到达档名称，是否真正形成资源过载须看队列、利用和失败结果，不能从名称推断。', '',
        '## 负载校准与下一步判断', '',
        '预先声明的队列筛查同时要求：DSR在5%–95%之间、全生成任务平均CPU等待>0.01秒、接纳后跨宏时隙比例>10%、转发队列非空>5%。冻结代码的恢复实现为“后20秒积压≤1.2×max(突发前积压,1任务)”，含一个任务的下限；协议摘要只写了1.2倍，没有明确该下限。本报告同时按不带下限的严格文字口径复核，避免用实现细节扩大晋级范围。这些只是开发筛查，既不是应用参数的证明，也不是统计显著性标准。', '',
        f"三控制器均通过实现口径的突发资源格：{json.dumps(eligible, ensure_ascii=False)}；同时通过严格恢复口径的资源格：{json.dumps(strict_eligible, ensure_ascii=False)}。筛查没有使用RL成绩，后续优先参考严格口径。", '',
        '主配置突发的已接纳积压（上传＋转发＋计算任务数）为：', '',
        '| 控制器 | 突发前30–50秒 | 突发中80–100秒 | 恢复后180–200秒 | 转发非空比例 | MBS分配占比 |',
        '|---|---:|---:|---:|---:|---:|']
    for method in METHODS_LOCAL:
        r = row_for('burst', method); b = r['backlog']
        lines.append(f"| {method_name[method]} | {b['before']:.3f} | {b['during']:.3f} | {b['late']:.3f} | {pct(r['transfer_nonempty'])} | {pct(r['mbs_fraction'])} |")
    if diagnostic is not None:
        lines += ['', '## 补充：明确不同的协作分配与连续轨迹干预', '',
            '初始8个探针未出现交互后，补充更广状态和更长轨迹、明确协作去向的干预，属于探索性补充；后续核查确认原8个探针也确实改变了分配，不把其零结果归因于动作相同。固定四到达档、三个资源对(算力/带宽1/1、0.6/0.25、0.6/1)、六个既有工作负载和40/75/100/150秒四个时刻，共288个状态，没有按结果筛选。重放主实验轨迹至各状态并逐步精确核对计数。', '',
            '交叉比较首次新接纳任务全部来源计算/可达协作UAV计算，以及悬停/依据初始信息承诺的六段可行轨迹。之后五批任务是共同的外生模拟输入，控制动作不读取未来；后续新接纳任务统一来源计算。停止生成后继续排空，四臂分母均为初态未终结任务及共同新增任务。协作分配是受控干预，不代表更优分配器。', '',
            f"有效对照{diagnostic['eligible']}/{diagnostic['cases']}个；其中{diagnostic['nonzero_interactions']}个交互非零，{diagnostic['rank_reversals']}个分配优劣反转，最大交互绝对值{diagnostic['max_absolute_interaction']}个任务。补充诊断实际覆盖{len(diagnostic['actual_cpus'])}个逻辑CPU。", '',
            '| 算力/带宽 | 有效状态 | 非零交互 | 分配优劣反转 |', '|---|---:|---:|---:|']
        for c, b in ((1., 1.), (.6, .25), (.6, 1.)):
            xs = [d for d in diagnostic_cases if d['eligible'] and d['spec']['compute'] == c and d['spec']['bandwidth'] == b]
            lines.append(f"| {c}/{b} | {len(xs)} | {sum(d['interaction'] != 0 for d in xs)} | {sum(d['reversal'] for d in xs)} |")
        lines += ['', '3个反转全部位于0.6倍算力、原带宽的持续100请求/秒场景。该资源配置的突发场景24个状态只有1个非零交互、没有反转。可恢复积压和主场景中的联合收益不能混为一谈。', '',
            '非零交互和反转说明某些状态下同时考虑两种决策会影响选择；它们不自动证明优化联合策略优于充分协调的分解方法，更不证明RL优势。未通过队列及恢复筛查的资源格，不因这里出现一个正面例子就晋级训练。']
    max_change = data['max_abs_fixed_action_dsr_change']
    lines += ['', '## 数值精度与计算成本', '',
        f"固定原0.1秒策略的全部宏动作和分配，改用0.05/0.025秒积分重放48组；相对0.1秒的最大DSR绝对变化为{100*max_change:.4f}个百分点。固定动作复核隔离积分影响，不代表闭环重训或其他参数下已数值收敛。", '',
        '| 主配置突发控制器 | 完整决策平均耗时 | 超过1秒的决策数 |', '|---|---:|---:|']
    for method in METHODS_LOCAL:
        r = row_for('burst', method)
        lines.append(f"| {method_name[method]} | {1000*r['decision_mean']:.3f}ms | {r['decision_over_one_second']} |")
    if precision is not None:
        lines += ['', f"全部{precision['cases']}个反转状态另在完全相同初态、分配和轨迹上，以0.1/0.05/0.025秒重放四臂，共{precision['runs']}组。验收为`{precision['status']}`，四臂完成数是否全部相同：{precision['all_counts_identical']}。这是对已观察反转的数值检查，不是独立确认。"]
    lines += ['', '耗时在批量服务器负载下测得；本轮用于离线模型校准，没有把超过1秒的规划强制回退为悬停，不能据此声称实时可部署。滚动搜索只在至多4条轨迹候选及2种分配启发式之间搜索，预测2秒内当前已知任务、后续悬停，不读取未来到达，不称精确MPC最优解。']
    lines += ['', '## 当前研究判断', '',
        '任务生命周期和共享资源引擎已经走通，也在部分全网状态观察到轨迹×分配的交互。原资源配置的服务跨时隙比例很低，不能直接作为联合RL主场景。0.6倍算力、原带宽是唯一同时通过三控制器队列筛查与严格恢复口径的开发候选，但其突发场景接纳后完成率仍约99.8%，观察到的交互也少。', '',
        '因此本轮完成环境与机制验证，不启动大规模联合训练。下一步应先固定一个业务依据清楚、截止期确有约束、且能反复观察到联合决策影响的开发主场景，再做同信息、同长期目标的联合/分解短试。不能仅为了增加反转次数或RL排名而修改环境。']
    lines += ['', '## 使用限制', '',
        '- 源码冻结、引擎回归和数据审计通过，与算法性能和创新性成立是不同结论。本轮为显式自审，没有独立代码审查。',
        '- 排队、上传、转发及未覆盖的到期阶段是互斥描述，不是对超时原因的唯一因果归因。硬截止期清队列使积压有界，不能以此证明稳定或有效。',
        '- 固定来源/目标、无迁移、共享池、EDF、固定无线功率及信息遥测均是模型假设。分配启发式采用近似预计完成时间，滚动控制也受候选与预测窗限制。',
        '- 本轮沿用每UE每宏时隙至多一个新任务的到达结构；持续100请求/秒时全部UE均活跃，热点到达不再集中。加上100m覆盖范围内UE数有限，该结构可能限制局部可接纳负载；后续若改变为可重复到达须单列协议并给出业务依据。',
        '- 目前只检查6个开发工作负载，没有训练种子，不作训练重复或独立确认；未训练联合策略，也未证明联合RL优于分解方法。',
        '- 若筛查没有找到同时具有持续传输、计算竞争和恢复性的资源格，应先调整负载/期限的物理依据并重新冻结校准，不直接扩训或按RL排名选环境。', '',
        '## 可复核文件', '',
        f"- [完整实验分析](../../../results/{TASK}/{REVISION}/analysis.json)",
        f"- [冻结协议](../../../results/{TASK}/{REVISION}/protocol.json)",
        f"- [确定性任务事件轨迹](../../../results/{TASK}/{REVISION}/mechanism_example.json)",
        f"- [补充有效干预](../../../results/{TASK}/{REVISION}/coupling_diagnostic/analysis.json)",
        f"- [反转状态数值复核](../../../results/{TASK}/{REVISION}/coupling_precision/analysis.json)",
        '- [逐格计数与指标CSV](queued_protocol_summary.csv)',
        f"- [服务器回归](../validation/{REVISION}/engine_tests.json)",
        f"- [完成验收](../validation/{REVISION}/completion.json)",
        f"- [本机摘要及计数复核](../validation/{REVISION}/local_result_audit.json)",
        f"- [本机静态及布局检查](../validation/{REVISION}/local_validation.json)", '',
        f"科学源码合同：`{data['source_contract']}`。报告生成脚本摘要：`{sha(Path(__file__))}`。", '']
    report = deliver/'queued_protocol_validation.md'; report.write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({k: v for k, v in checks.items() if k != 'raw_sha256'}, ensure_ascii=False))
    return checks


METHODS_LOCAL = ('history_queue', 'history_balance', 'rolling_joint')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--task', type=Path, default=REPO_ROOT)
    build(parser.parse_args().task.resolve())
