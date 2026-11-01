"""Locally verify business raw files and deliver the calibrated parameter report."""
import argparse
import csv
from datetime import datetime,timezone,timedelta
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis.queued_service_business import TASK,REVISION,read,sha,contract,locations


def verify_aggregates(records, rows):
    """Reconcile published cells with original per-episode counters."""
    groups = {}
    for spec, summary in records:
        if spec['phase'] not in ('screen', 'holdout'):
            continue
        key = tuple(spec[k] for k in ('phase', 'profile', 'sla', 'scale', 'method'))
        groups.setdefault(key, []).append(summary)
    assert len(groups) == len(rows)
    for r in rows:
        key = tuple(r[k] for k in ('phase', 'profile', 'sla', 'scale', 'method'))
        xs = groups[key]
        assert len(xs) == r['replicates']
        sums = {k: sum(x['counts'].get(k, 0) for x in xs) for k in ('generated', 'admitted', 'success', 'service_crossed')}
        for k in ('generated', 'admitted', 'success'):
            assert r[k] == sums[k]
        a = max(sums['admitted'], 1)
        duration = sum(x['elapsed'] for x in xs)
        cohort = sum(x['burst_cohort_size'] for x in xs)
        assert r['cohort_size'] == cohort
        expected = dict(
            dsr=sums['success']/sums['generated'], admission=sums['admitted']/sums['generated'],
            conditional=sums['success']/a, service_crossed=sums['service_crossed']/a,
            cpu_wait_per_admitted=sum(x['cpu_wait_seconds_total'] for x in xs)/a,
            transfer_nonempty=sum(x['nonempty_fraction']['transfer']*x['elapsed'] for x in xs)/duration,
            mbs_fraction=sum(x['assignment_by_node'].get('5', 0) for x in xs)/a,
            cohort_completion=sum(x['burst_cohort_completed'] for x in xs)/cohort if cohort else None)
        for k, v in expected.items():
            assert r[k] is None if v is None else math.isclose(r[k], v, abs_tol=1e-12), (key, k)
        for k in ('before', 'during', 'late'):
            assert math.isclose(r['backlog'][k], statistics.mean(x['backlog'][k] for x in xs), abs_tol=1e-12)
        criteria = dict(dsr=.05<=r['dsr']<=.8, conditional=.4<=r['conditional']<=.95,
            crossed=r['service_crossed']>=.1, cpu_wait=r['cpu_wait_per_admitted']>=.05,
            transfer=r['transfer_nonempty']>=.05, mbs=r['mbs_fraction']<.8,
            recovery=r['backlog']['late']<=1.2*r['backlog']['before']+1.,
            cohort=r['cohort_completion'] is not None and r['cohort_completion']>=.5)
        assert r['criteria'] == criteria and r['gate'] == all(criteria.values())


def precision_files(out, precision):
    """Match numerical replays to their original per-run records."""
    checked = {}
    cases = set()
    for result in precision['results']:
        s = result['spec']
        key = (s['profile'], s['workload'], result['time'], result['dt'])
        assert key not in cases
        cases.add(key)
        name = f"reversal_precision/{s['profile']}_w{s['workload']}_t{result['time']}_dt{result['dt']}.json"
        assert read(out/name) == result
        checked[name] = sha(out/name)
    states = {key[:3] for key in cases}
    assert len(states) == precision['reversal_states']
    assert precision['runs'] == len(cases) == 3*len(states)
    assert all({key[3] for key in cases if key[:3] == state} == {.1, .05, .025} for state in states)
    return checked


def recovery_addendum(task):
    """Verify the second protocol without activating its experiment runtime."""
    import hashlib
    revision='business_recovery_20261031';out=task/'results'/TASK/revision;val=task/'output'/TASK/'validation'/revision
    if not (out/'analysis.json').exists():return None
    data=read(out/'analysis.json');proto=read(out/'protocol.json');files=proto['source_files']
    assert all(sha(task/name)==h for name,h in files.items())
    digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    assert digest==data['source_contract']==proto['source_contract']
    hashes=dict(data['screen']['raw_sha256'],**data['holdout']['raw_sha256'],**data['precision_raw_sha256'])
    paired={};records=[]
    for name,h in hashes.items():
        path=out/name;assert sha(path)==h;d=read(path);s=d['spec'];c=d['summary']['counts']
        assert d['source_contract']==digest and d['audit']['status']=='PASS'
        assert c['generated']==c.get('success',0)+c.get('expired',0) and d['rows'][-1]['active']==0
        assert sum(r['success'] for r in d['rows'])==c.get('success',0)
        assert sum(d['summary']['expiration_stage'].values())==c.get('expired',0)
        k=(s['profile'],s['scale'],s['workload'])
        assert k not in paired or paired[k]==d['arrival_fingerprint'];paired[k]=d['arrival_fingerprint']
        records.append((s,d['summary']))
    rows=[dict(phase=phase,**r) for phase in ('screen','holdout') for r in data[phase]['summary']]
    verify_aggregates(records, rows)
    for r in rows:assert math.isclose(r['dsr'],r['admission']*r['conditional'],abs_tol=1e-12)
    for p in data['holdout']['interventions']:
        j={k:v['success'] for k,v in p['cases'].items()};a=j['hold_cooperative']-j['hold_source'];b=j['move_cooperative']-j['move_source']
        assert a-b==p['interaction'] and p['reversal']==(a*b<0)
        assert len({v['generated'] for v in p['cases'].values()})==1
    precision=read(out/'reversal_precision/analysis.json') if (out/'reversal_precision/analysis.json').exists() else None
    precision_hashes = {}
    if precision:
        assert precision['source_contract']==digest
        assert precision['script_sha256']==sha(task/'scripts/analysis/queued_service_business_precision.py')
        for r in precision['results']:
            parent=next(x for x in data['holdout']['interventions'] if x['spec']==r['spec'] and x['time']==r['time'])
            assert r['original']==parent['cases'] and r['identical']==(r['cases']==r['original'])
        assert precision['all_counts_identical']==all(r['identical'] for r in precision['results'])
        precision_hashes = precision_files(out, precision)
    deliver=task/'output'/TASK/'deliverables'
    fields=('phase','profile','sla','scale','method','generated','admitted','success','dsr','admission','conditional',
            'cpu_wait_per_admitted','service_crossed','transfer_nonempty','mbs_fraction','cohort_size','cohort_completion',
            'decision_mean','decision_over_one_second','gate','strict_recovery')
    with (deliver/'queued_business_recovery_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    selected=data['selection']['selected'];peak=selected['burst_peak'];hold=data['holdout']['summary']
    burst=[r for r in hold if r['profile']=='burst'];passed=all(r['gate'] for r in burst)
    audit=dict(status='PASS_LOCAL_RECOVERY_RAW_COUNT_AUDIT',source_contract=digest,checked_raw_files=len(hashes),
        scientific_gate=data['scientific_gate'],holdout_all_controller_gates=passed,
        actual_screen_cpu_count=len(data['actual_screen_cpus']),old_files_preserved=data['protected_files'],
        changed_old_files=data['changed_old_files'],reversal_precision=precision['status'] if precision else None,
        raw_sha256=hashes,precision_raw_sha256=precision_hashes,
        checked_precision_files=len(precision_hashes),independent_review=False)
    (val/'local_result_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    method={'history_queue':'历史轨迹＋队列感知分配','history_balance':'历史轨迹＋负载均衡','rolling_joint':'有限候选滚动联合搜索'}
    profile={'light':'轻载','steady':'平稳','burst':'突发','overload':'持续高到达'};pct=lambda x:f'{100*x:.2f}%'
    lines=['# 业务负载与截止期核定：最终阶段结果','',
        f"第二轮完成：{data['completed']}。科学筛查状态：**{data['scientific_gate']}**。本轮没有训练RL。",'',
        '## 当前核定参数','',
        '| 项目 | 选定开发设置 |','|---|---|',
        '| 业务抽象 | 非闭环图像巡检／告警事件的合成代理负载 |',
        '| 输入大小 | 均匀1.5–3 MB，十进制 |',
        '| 计算密度 | 均匀整数500–1000 cycle/bit；平均13.5 Gcycle/任务 |',
        '| 截止期 | 到达后2–3秒 |',
        '| 轻载／平稳／持续高到达 | Poisson平均7.5／30／90任务每秒 |',
        f'| 突发 | 平均15→{peak}→15任务每秒，高峰[50,100)秒 |',
        '| 空间到达 | 热点加背景、有放回选UE；同UE同秒可多任务 |',
        '| 决策与服务 | 1秒宏决策、0.1秒信道积分，完成和到期事件区间内推进 |',
        '| 资源 | 五UAV原预算75/60/70/85/50 Gcycle/s、MBS200 Gcycle/s；无线预算不变 |','',
        '参数来源、bit/byte换算及不能外推的范围见[依据JSON](queued_business_sources.json)及后面的第一轮记录。数据量和计算密度是文献仿真量级，2–3秒为本研究声明的事件级SLA；没有真实业务日志和目标模型CPU剖析，不称实测业务标定。','',
        '## 为什么保留首轮失败并修订峰值','',
        '第一轮2–3秒、15→75→15/s在开发数据通过，但6个新工作负载上，三控制器100秒边界积压最终完成比例仅约42.6%–47.3%，未保持预先设定的50%恢复门槛。该失败与原选择均保留。没有在原验证数据上换一个看起来好的设置。','',
        '第二轮只做一次有上限的修订：期限、任务量级和硬件不变，固定基线15/s，测试45/60/s峰值、75/s作失败边界参照。用24个新开发工作负载12–35，按相同物理筛查优先45；45未满足而60满足才选60。随后冻结设置，用12个全新工作负载36–47复核。没有根据方法排名、RL成绩或交互频次选择。','',
        '峰值45/s在参考覆盖占比0.321下，计算到达约为五UAV预算0.573倍；若接纳占比0.5则约0.893倍。这只是容量量纲核算，实际空间不均和传输瓶颈仍由实验检查。','',
        '| 峰值 | 开发共同筛查 | 用途 | 未通过指标并集 |','|---|---|---|---|']
    for c in data['selection']['candidates']:
        failed=sorted({x for f in c['failures'].values() for x in f})
        lines.append(f"| {c['peak']}/s | {c['passed']} | {'仅参照' if c['reference_only'] else '候选'} | {', '.join(failed) or '无'} |")
    lines+=['','## 新工作负载结果','',
        '432个开发回合、144个新工作负载回合和48个精度回合完成。以下每格12个新工作负载，比例由合并请求计数计算；通过指该组合并口径通过，不代表每个单独工作负载均通过。','',
        '| 到达档 | 控制器 | DSR | 接纳后完成率 | 接纳后跨宏时隙 | 每接纳任务CPU等待 |',
        '|---|---|---:|---:|---:|---:|']
    for p in profile:
        for m in method:
            r=next(x for x in hold if x['profile']==p and x['method']==m)
            lines.append(f"| {profile[p]} | {method[m]} | {pct(r['dsr'])} | {pct(r['conditional'])} | {pct(r['service_crossed'])} | {r['cpu_wait_per_admitted']:.3f}s |")
    lines+=['','突发积压的恢复机制：','',
        '| 控制器 | 突发前积压 | 突发中积压 | 后期积压 | 边界积压最终完成率 | 全部物理筛查 |',
        '|---|---:|---:|---:|---:|---|']
    for r in burst:
        b=r['backlog'];q=pct(r['cohort_completion']) if r['cohort_completion'] is not None else '无边界积压'
        lines.append(f"| {method[r['method']]} | {b['before']:.2f} | {b['during']:.2f} | {b['late']:.2f} | {q} | {r['gate']} |")
    lines+=['',f"新工作负载中三控制器是否均保持所有预定门槛：**{passed}**。门槛原样沿用：DSR5%–80%、接纳后完成率40%–95%、跨时隙≥10%、CPU等待≥0.05秒、转发非空≥5%、MBS分配<80%、后期积压≤1.2×前期+1任务、边界积压真正完成≥50%。硬截止期淘汰不能单独作为恢复证据。",'',
        '## 联合耦合、数值和成本边界','',
        f"192个预定状态中，有效对照{data['interventions']['eligible']}个，非零交互{data['interventions']['nonzero']}个，分配优劣反转{data['interventions']['reversals']}个。状态数是诊断计数，不是独立训练重复。四臂使用相同初态和外生任务；协作分配与承诺轨迹是机制干预，不代表优化联合策略。",'',
        '| 到达档 | 有效状态 | 非零交互 | 分配优劣反转 |','|---|---:|---:|---:|']
    for p in profile:
        xs=[x for x in data['holdout']['interventions'] if x['eligible'] and x['spec']['profile']==p]
        lines.append(f"| {profile[p]} | {len(xs)} | {sum(x['interaction']!=0 for x in xs)} | {sum(x['reversal'] for x in xs)} |")
    lines+=['',f"48个固定动作精度回合最大DSR变化为{100*data['max_abs_fixed_action_dsr_change']:.4f}个百分点。"]
    if precision:
        lines+=['',f"全部{precision['reversal_states']}个反转状态另以0.1/0.05/0.025秒复核，共{len(precision['results'])}组四臂轨迹，状态`{precision['status']}`。"]
    lines+=['','突发档48个状态只有3个非零交互、1个反转；耦合存在的证据不能代替平均联合收益。后续应固定本设置检验联合学习与同信息强对照，不继续为提高交互频次或方法排名挑参数。',
        '',f"原始结果已同步完成。本机逐项核验第一轮768个、第二轮{len(hashes)}个仿真文件及{len(precision_hashes)}个反转精度文件；按原始回合重新合并计数、等待、跨时隙、恢复指标并核对逐格统计和筛查。来源合同、配对任务指纹、唯一终局计数和尾部排空均通过。布局检查仅剩两个既有根目录问题，详见[本机验收](../validation/business_recovery_20261031/local_validation.json)。"]
    lines+=['','本轮没有联合RL训练，不能从队列成立、筛查通过或局部交互推断联合学习优于充分协调的分解方法。滚动对照只有有限候选和2秒已知任务预测窗；其推理成本记录在CSV，超时没有触发部署回退，不能声称实时可部署。来源锁定、无迁移/分片、资源正交池、EDF和秒边界到达均为模型假设。','',
        '第二轮CSV的scale是到达参数：突发峰值为100×scale，非突发三档固定为0.75对应7.5/30/90请求每秒；它不是算力系数，硬件预算始终保持原值。','',
        f"科学源码合同：`{digest}`；37项回归通过。开发扫描实际覆盖{len(data['actual_screen_cpus'])}个逻辑CPU；服务器配置128物理核/256逻辑CPU。{data['protected_files']:,}份旧文件未变，包括第一轮失败结果。显式自审，不冒称独立审查。",'',
        '## 本轮成果','',
        '- [第二轮逐格CSV](queued_business_recovery_summary.csv)',
        f'- [第二轮冻结协议](../../../results/{TASK}/{revision}/protocol.json)',
        f'- [第二轮完整分析](../../../results/{TASK}/{revision}/analysis.json)',
        f'- [第二轮本机复核](../validation/{revision}/local_result_audit.json)',
        f'- [第二轮回归](../validation/{revision}/tests.json)',
        f'- [第二轮完成记录](../validation/{revision}/completion.json)',
        '- [第一轮逐格CSV：保留失败](queued_business_summary.csv)',
        '- [文献和单位依据](queued_business_sources.json)','']
    return lines


def report(task):
    out,val=locations(task);d=read(out/'analysis.json');assert d['status']=='PASS_COMPLETED_BUSINESS_CALIBRATION'
    assert d['source_contract']==contract()[0]
    checked={};streams={};records=[]
    expected=dict(d['screen']['raw_sha256'],**d['holdout']['raw_sha256'],**d['precision_raw_sha256'])
    for name,h in expected.items():
        p=out/name;assert sha(p)==h;raw=read(p);s=raw['spec'];c=raw['summary']['counts']
        assert raw['source_contract']==contract()[0] and raw['audit']['status']=='PASS'
        assert c['generated']==c.get('success',0)+c.get('expired',0) and raw['rows'][-1]['active']==0
        assert sum(r['success'] for r in raw['rows'])==c.get('success',0)
        assert c.get('expired',0)==sum(raw['summary']['expiration_stage'].values())
        pair=(s['profile'],s['scale'],s['workload'])
        assert pair not in streams or streams[pair]==raw['arrival_fingerprint'];streams[pair]=raw['arrival_fingerprint']
        assert math.isclose(raw['summary']['dsr'],raw['summary']['admission']*raw['summary']['conditional'],abs_tol=1e-12)
        checked[name]=h
        records.append((s,raw['summary']))
    allrows=[dict(phase=phase,**r) for phase in ('screen','holdout') for r in d[phase]['summary']]
    verify_aggregates(records, allrows)
    for r in allrows:
        assert math.isclose(r['dsr'],r['success']/r['generated'],abs_tol=1e-12)
        assert math.isclose(r['dsr'],r['admission']*r['conditional'],abs_tol=1e-12)
    for p in d['holdout']['interventions']:
        j={k:v['success'] for k,v in p['cases'].items()};a=j['hold_cooperative']-j['hold_source'];b=j['move_cooperative']-j['move_source']
        assert p['interaction']==a-b and p['reversal']==(a*b<0)
        assert len({v['generated'] for v in p['cases'].values()})==1
    precision=read(out/'reversal_precision/analysis.json') if (out/'reversal_precision/analysis.json').exists() else None
    precision_hashes = {}
    if precision:
        assert precision['source_contract']==contract()[0]
        assert precision['script_sha256']==sha(task/'scripts/analysis/queued_service_business_precision.py')
        for result in precision['results']:
            assert result['identical']==(result['cases']==result['original'])
            parent=next(x for x in d['holdout']['interventions'] if x['spec']==result['spec'] and x['time']==result['time'])
            assert result['original']==parent['cases']
        assert precision['all_counts_identical']==all(x['identical'] for x in precision['results'])
        precision_hashes = precision_files(out, precision)
    deliver=task/'output'/TASK/'deliverables';deliver.mkdir(parents=True,exist_ok=True)
    fields=('phase','profile','sla','scale','method','generated','admitted','success','dsr','admission','conditional',
            'cpu_wait_per_admitted','service_crossed','transfer_nonempty','mbs_fraction','cohort_size','cohort_completion',
            'decision_mean','decision_over_one_second','gate','strict_recovery')
    with (deliver/'queued_business_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(allrows)
    selected=d['selection']['selected'];hold=d['holdout']['summary'];burst=[r for r in hold if r['profile']=='burst']
    selected_sla=read(out/'protocol.json')['sla'][selected['sla']];rate=selected['scale']
    frozen_pass=all(r['gate'] for r in burst)
    resources=[json.loads(x) for x in (val/'resources.jsonl').read_text().splitlines() if x]
    peak=max(x['active'] for x in resources)
    audit=dict(status='PASS_LOCAL_BUSINESS_RAW_COUNT_AUDIT',checked_raw_files=len(checked),source_contract=contract()[0],
        old_files_preserved=d['protected_files'],changed_old_files=d['changed_old_files'],peak_active=peak,
        actual_screen_cpu_count=len(d['actual_screen_cpus']),selected=selected,holdout_all_controller_gates=frozen_pass,
        reversal_precision=precision['status'] if precision else None,raw_sha256=checked,
        precision_raw_sha256=precision_hashes,checked_precision_files=len(precision_hashes),independent_review=False)
    (val/'local_result_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    method={'history_queue':'历史轨迹＋队列感知分配','history_balance':'历史轨迹＋负载均衡','rolling_joint':'有限候选滚动联合搜索'}
    profile={'light':'轻载','steady':'平稳','burst':'突发','overload':'持续高到达'}
    pct=lambda x:f'{100*x:.2f}%'
    lines=['# 业务负载、截止期核定与新工作负载验证','',
        f"主实验完成：{d['completed']}；本报告生成：{datetime.now(timezone(timedelta(hours=8))).isoformat()}。本轮没有训练RL。",'',
        '## 核定结论','',
        f"声明的首选是2–3秒截止期、到达倍率1；开发筛查冻结的设置为：**{selected['sla']}期限族、到达倍率{selected['scale']}**。选择状态为`{d['selection']['status']}`。新工作负载上三控制器是否全部保持预定业务筛查：**{frozen_pass}**。没有依据新工作负载的表现重新挑参数。",'',
        f"选定设置的具体参数：截止期{selected_sla[0]}–{selected_sla[1]}秒；轻载平均{10*rate:g}/秒、平稳{40*rate:g}/秒、突发{20*rate:g}→{100*rate:g}→{20*rate:g}/秒、持续高到达{120*rate:g}/秒。到达率是Poisson均值，不是每秒固定任务数。",'',
        '这里核定的是可复现、具有文献量级依据的研究业务协议。尚无真实应用日志或目标图像模型的CPU剖析，不能称为实测业务标定、真实系统SLA或现实部署验证。业务锚点是非闭环的图像巡检/告警事件处理。','',
        '## 参数及来源','',
        '| 项目 | 本轮采用值 | 依据与边界 |','|---|---|---|',
        '| 数据量 | 均匀1.5–3 MB，1 MB=10^6 byte | EP-MUSTO的仿真小任务量级；十进制单位为本研究声明 |',
        '| 计算密度 | 均匀整数500–1000 cycle/bit | 位于文献比较量级之间；均匀分布为本研究假设 |',
        '| 总计算量 | `8 × bytes × cycle/bit`，均值13.5 Gcycle/任务 | 不是将旧cycle/byte数值直接换标签 |',
        '| 截止期族 | 紧1–2秒、主2–3秒、宽3–5秒，从到达时计时 | 秒级文献范围内的研究SLA；不是交互视频或安全闭环要求 |',
        '| 到达 | 每秒Poisson批到达，UE按热点权重有放回抽样 | 允许同UE同秒多任务，未模拟秒内连续到达 |',
        '| 资源 | 原五UAV及MBS算力、原无线池 | 未降硬件；340 Gcycle/s为五UAV总预算 |','',
        '数据量和计算密度的量级来源是[EP-MUSTO，IEEE IoT Journal 2025，Fig.6](https://www.cs.newpaltz.edu/~lik/publications/Longxin-Zhang-IEEE-IoTJ-2025)。其安全机制与能耗目标不在本研究协议内，也没有把这些参数称为实测图像模型开销。','',
        '1–5秒与500–1000 cycle/bit范围可见[HOOA，arXiv预印本](https://arxiv.org/html/2512.24659v1)的仿真表；该文输入量是Mb，不能用于证明MB大小。[ALERT，USENIX ATC 2020](https://www.usenix.org/system/files/atc20-wan.pdf)讨论的交互推理期限通常更短，不能拿它证明2–5秒适用于实时视频控制。','',
        '[Huo等的UAV-MEC仿真](https://file.techscience.com/files/onlinefirst/2025/12.5/TSP_CMC_72681/TSP_CMC_72681.pdf)提供Poisson到达的建模先例；下面的具体到达率、热点权重和突发倍率是本研究按资源量纲确定的实验设定，并非该论文的实测值。来源及限制逐字段记录在[依据JSON](queued_business_sources.json)。','',
        '## 负载的量纲核算','',
        '平均任务需要13.5 Gcycle，五UAV合计340 Gcycle/s，理想上限约25.19任务/s。以5π·100²/700²≈0.321作几何参考覆盖占比，20/40/100/120任务每秒对应约0.255/0.509/1.273/1.527倍五UAV预算。真实接纳率取决于位置、UE分布及策略，不等于该参考占比；MBS算力受回传限制，不能无条件加进可达容量。','',
        '| 到达档 | 倍率1下的平均请求率 | 解释 |','|---|---|---|',
        '| 轻载 | 10/s | 检查基础服务与非退化 |',
        '| 平稳 | 40/s | 检查常态资源竞争 |',
        '| 突发 | 20→100→20/s；高峰[50,100)秒 | 检查可恢复拥塞 |',
        '| 持续高到达 | 120/s | 检查持续压力，不预先断言每节点均过载 |','',
        '另交叉0.75/1/1.25倍率。最大强度150/s的共同外生Poisson流通过独立mark稀疏化，UE有放回抽取；各控制器及期限族使用相同实际请求，跨强度保留共同任务属性。改变期限不会改变输入任务或计算量。','',
        '## 实验及选择规则','',
        f"开发扫描648组，冻结后新工作负载72组，固定动作精度48组；开发扫描累计生成{d['main_generated_tasks']:,}个任务（含各方法重放，不充当独立统计样本）。服务器实际覆盖{len(d['actual_screen_cpus'])}个逻辑CPU、128个物理核，峰值{peak}任务并行。32项回归通过。",'',
        '环境、无线预算、EDF、来源/目标锁定、断链暂停、唯一计数及排空规则沿用上一轮队列引擎。保留旧生成器、源码和结果。每个回合产生请求200秒，此后继续到所有任务完成或到期。','',
        '三个非学习控制器必须共同满足：突发DSR在5%–80%，接纳后完成率40%–95%，接纳后跨宏边界比例≥10%，每接纳任务CPU等待≥0.05秒，转发非空≥5%，MBS分配占比<80%；末20秒积压≤1.2×突发前均值+1任务；100秒边界已接纳积压最终真正完成的比例≥50%。后一个条件用于排除单靠到期清队列的假恢复。这些是开发筛查，不是统计显著性标准或业务SLA。','',
        '优先首选2–3秒/倍率1；若未通过，在共同通过筛查的设置中按预先固定的期限/倍率距离选择，绝不按哪个控制器获胜或交互频次选择。设置保存后才运行新工作负载6–11。','',
        '| 期限族 | 到达倍率 | 开发共同通过 | 未通过指标（控制器并集） |','|---|---:|---|---|']
    for c in d['selection']['candidates']:
        failures=sorted({k for xs in c['failures'].values() for k in xs})
        lines.append(f"| {c['sla']} | {c['scale']} | {c['passed']} | {', '.join(failures) or '无'} |")
    lines += ['', '## 冻结设置的新工作负载结果','',
        '每格6个新工作负载，三比例由合并的成功数、接纳数和生成数计算，满足DSR=接纳率×接纳后完成率。它们不是训练种子重复。','',
        '| 到达档 | 控制器 | DSR | 接纳率 | 接纳后完成率 | 接纳后跨宏时隙 | 每接纳任务CPU等待 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for p in ('light','steady','burst','overload'):
        for m in method:
            r=next(x for x in hold if x['profile']==p and x['method']==m)
            lines.append(f"| {profile[p]} | {method[m]} | {pct(r['dsr'])} | {pct(r['admission'])} | {pct(r['conditional'])} | {pct(r['service_crossed'])} | {r['cpu_wait_per_admitted']:.3f}s |")
    lines += ['', '突发恢复与主设置筛查：','',
        '| 控制器 | 突发前积压 | 突发中积压 | 后期积压 | 边界积压完成比例 | 新工作负载筛查 |',
        '|---|---:|---:|---:|---:|---|']
    for r in burst:
        b=r['backlog'];q=pct(r['cohort_completion']) if r['cohort_completion'] is not None else '无边界积压'
        lines.append(f"| {method[r['method']]} | {b['before']:.2f} | {b['during']:.2f} | {b['late']:.2f} | {q} | {r['gate']} |")
    lines += ['', '## 联合物理耦合与时间精度','',
        f"在冻结设置、新工作负载的96个预定状态上做两轨迹×两分配干预，有效状态{d['interventions']['eligible']}个，其中{d['interventions']['nonzero']}个交互非零、{d['interventions']['reversals']}个出现分配优劣反转。各臂初态与随后任务相同，先改变新接纳任务的来源/协作去向与承诺轨迹，之后统一来源计算，停止到达后排空。它是局部机制诊断，不是优化联合策略或RL增益。",'',
        '| 到达档 | 有效状态 | 非零交互 | 分配优劣反转 |','|---|---:|---:|---:|']
    for p in profile:
        ps=[x for x in d['holdout']['interventions'] if x['spec']['profile']==p and x['eligible']]
        lines.append(f"| {profile[p]} | {len(ps)} | {sum(x['interaction']!=0 for x in ps)} | {sum(x['reversal'] for x in ps)} |")
    lines += ['',f"48组固定动作/分配的0.05/0.025秒重放，相对0.1秒的最大DSR绝对差为{100*d['max_abs_fixed_action_dsr_change']:.4f}个百分点；不代表所有参数和策略已数值收敛。"]
    if precision:
        lines += ['',f"全部{precision['reversal_states']}个反转状态另按0.1/0.05/0.025秒复核，共{len(precision['results'])}组四臂轨迹，状态`{precision['status']}`，完成数是否全部相同：{precision['all_counts_identical']}。这是对观察结果的数值检查，不作独立确认。"]
    lines += ['', '## 研究判断与限制','',
        '参数选择已有来源映射、量纲核算、冻结规则和新工作负载结果。是否适合后续联合学习，应同时看截止期约束、真实排队、恢复机制和干预结果，不能只看队列非空或某一最高DSR。若新工作负载未保持共同筛查，就保留该失败，不回头改选一个好看的设置。','',
        '本轮没有联合RL训练，因此不能声称联合学习已优于强对照。滚动控制仅搜索有限候选、预测当前已知任务2秒，不读未来到达；其近似分配估计及短预测窗不构成最优性保证。所有控制器仅使用共同公开观测。','',
        '决策时间包含观测构建、动作选择及分配校验，干预诊断另计。批量服务器负载下可能超过1秒；本轮校准没有施加部署超时回退，不能据此认定实时可部署。到期阶段只是互斥状态描述，不是唯一因果归因。来源固定、无迁移/分片、共享无线池、EDF和秒边界到达均为模型抽象。','',
        f"旧数据保护共{d['protected_files']:,}份文件，改变数为{len(d['changed_old_files'])}。本轮为显式自审，不冒称独立审查；布局检查仍单独列出既有根目录问题。",'',
        '## 可复核文件','',
        '- [逐格结果CSV](queued_business_summary.csv)',
        '- [文献字段和参数依据](queued_business_sources.json)',
        f'- [冻结业务协议](../../../results/{TASK}/{REVISION}/protocol.json)',
        f'- [选择记录](../../../results/{TASK}/{REVISION}/selection.json)',
        f'- [完整实验分析](../../../results/{TASK}/{REVISION}/analysis.json)',
        f'- [本机原始摘要及计数复核](../validation/{REVISION}/local_result_audit.json)',
        f'- [回归检查](../validation/{REVISION}/tests.json)',
        f'- [完成验收](../validation/{REVISION}/completion.json)',
        f'- [布局和引用检查](../validation/{REVISION}/local_validation.json)', '',
        f"科学源码合同：`{contract()[0]}`。报告脚本摘要：`{sha(Path(__file__))}`。",'']
    addendum=recovery_addendum(task)
    if addendum:
        historical=[('### '+line[3:]) if line.startswith('## ') else line for line in lines[1:]]
        lines=addendum+['','## 第一轮完整记录：未通过新工作负载恢复门槛','']+historical
    target=deliver/'queued_business_calibration.md';target.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k!='raw_sha256'},ensure_ascii=False));return audit


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',type=Path,default=REPO_ROOT);a=p.parse_args();report(a.task.resolve())
