"""Render the confirmed SF-HLM evidence chain for the master's thesis.

All quantitative panels use archived block summaries, verified training logs,
or a prespecified frozen-policy replay. Mechanism examples are labelled as
schematics. PDF/SVG/600-dpi PNG share one Matplotlib source.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from paths import REPO_ROOT
from thesis_figure_style import FONT_CANDIDATES, setup_thesis_style

BUNDLED_QA = REPO_ROOT/'tools/figure_qa'
SKILL = BUNDLED_QA if BUNDLED_QA.is_dir() else Path.home()/'.codex/skills/nature-figure/scripts'
sys.path.insert(0,str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

DATA = REPO_ROOT/'docs/figures/data/sf_hlm_20261004'
OUT = REPO_ROOT/'docs/figures'
QA = REPO_ROOT/'output/thesis_writing_20261004/validation/figures'
METHODS = ['learned','random','heuristic','greedy','original_frozen','original_adapted','upper_random_offload','full_random']
LABELS = dict(learned='SF-HLM 学习配额',random='强配额随机',heuristic='启发式配额',greedy='贪心＋修复',
    original_frozen='冻结请求级下层＋修复',original_adapted='适配请求级下层＋修复',
    upper_random_offload='相同上层＋随机卸载',full_random='全随机（质量掩码内）')
SHORT = dict(learned='SF-HLM',random='强随机',heuristic='启发式',greedy='贪心修复',
    original_frozen='冻结请求级下层',original_adapted='适配请求级下层',upper_random_offload='随机卸载',full_random='全随机')
COLORS = dict(zip(METHODS,['#3569AC','#89909A','#65A69A','#CCAB69','#8790BE','#AF98B5','#B9A58A','#BCBCBC']))
BLUE, TEAL, GRAY = COLORS['learned'],COLORS['heuristic'],'#7B818A'
NAMES = {
1:'SFHLM总体架构与决策时序',2:'配额与容量槽匹配',3:'二部图网络与搜索状态',4:'训练确认数据划分',
5:'基座与配额训练曲线',6:'UAV轨迹与覆盖',7:'实际前向注意力',8:'八方法DSR比较',
9:'主比较训练块配对差',10:'请求结果与接纳后质量',11:'UAV能耗与MBS依赖',
12:'公平性与隔离软件耗时',13:'强框架机制配对差',14:'九格冻结策略DSR',15:'九格主对手DSR差',
16:'自由与固定MBS快照诊断',17:'墙钟预算质量时间与超限',18:'匹配调用与停止行为',
19:'集群任务执行分工'}


def read(path): return json.loads(path.read_text())


def save(fig, number, axes=None):
    name=f'图5-{number}_{NAMES[number]}'
    QA.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw()
    groups = {'column_groups': [['a','b']]} if number == 7 else {}
    require_matplotlib_panel_alignment(fig,axes=axes,json_out=QA/f'{name}.alignment.json',
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True,**groups)
    fig.savefig(OUT/f'{name}.pdf',facecolor='white',bbox_inches=None)
    fig.savefig(OUT/f'{name}.svg',facecolor='white',bbox_inches=None)
    fig.savefig(OUT/f'{name}.png',dpi=600,facecolor='white',bbox_inches=None)
    plt.close(fig)
    print(name,flush=True)


def panels(rows=1,cols=2,height=3.6):
    fig,axs=plt.subplots(rows,cols,figsize=(7.2,height),squeeze=False)
    fig.subplots_adjust(left=.10,right=.97,bottom=.17,top=.88,wspace=.38,hspace=.63)
    for i,ax in enumerate(axs.flat):
        ax.text(-.14,1.06,chr(97+i),transform=ax.transAxes,fontweight='bold',fontsize=11,va='bottom')
    return fig,axs


def box(ax,x,y,w,h,text,color=BLUE):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.007',
        facecolor=color+'18',edgecolor=color,lw=.85))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)


def arrow(ax,a,b,color=GRAY):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color=color,lw=1))


def schematic(number):
    fig,ax=plt.subplots(figsize=(7.2,5.2 if number==1 else 4.4))
    fig.subplots_adjust(left=.025,right=.975,bottom=.10,top=.94)
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    if number==1:
        box(ax,.06,.80,.91,.15,'当前状态：位置／缓存／UE电量／当前需求\n两层均使用当前信息准备动作',GRAY)
        box(ax,.06,.55,.40,.17,'冻结上层：分散轨迹决策\n预接纳需求＋邻机观测\n五机连续轨迹动作',BLUE)
        box(ax,.55,.55,.42,.17,'自然覆盖接纳：100 m\n依据当前无人机位置\n形成请求—执行节点图',TEAL)
        arrow(ax,(.26,.79),(.26,.73));arrow(ax,(.76,.79),(.76,.73))
        box(ax,.06,.30,.40,.17,'轨迹动作准备完毕\n等待当前请求处理结束',BLUE)
        box(ax,.55,.30,.42,.17,'集中下层：学习配额搜索\n条件匹配＋最好方案保留',TEAL)
        arrow(ax,(.26,.54),(.26,.48));arrow(ax,(.76,.54),(.76,.48))
        box(ax,.06,.05,.40,.17,'随后应用轨迹动作\n位移／边界／机间距修复',BLUE)
        box(ax,.55,.05,.42,.17,'先执行当前请求批次\n共享算力／判定截止期',TEAL)
        arrow(ax,(.26,.29),(.26,.23));arrow(ax,(.76,.29),(.76,.23))
        arrow(ax,(.54,.135),(.47,.135))
        # Close the state transition outside all text-bearing boxes.
        ax.plot([.06,.025,.025],[.135,.135,.875],color=GRAY,lw=1)
        arrow(ax,(.025,.875),(.055,.875))
        fig.text(.5,.027,'执行时序：当前请求先完成批量处理，移动后进入下一时隙；上层权重冻结',
            ha='center',fontsize=9)
    elif number==2:
        ax.text(.5,.94,'机制示例：4个请求，配额为（2，1，0，0，0，1）',ha='center',fontsize=11)
        ys=[.76,.57,.38,.19]
        for i,y in enumerate(ys):box(ax,.04,y-.05,.17,.10,f'请求 {i+1}',TEAL)
        for i,(y,t) in enumerate(zip(ys,['UAV1 槽1','UAV1 槽2','UAV2 槽1','MBS 槽1'])):
            box(ax,.70,y-.05,.25,.10,t,BLUE)
        mapping=[1,0,3,2]
        for i,j in enumerate(mapping):arrow(ax,(.22,ys[i]),(.69,ys[j]),BLUE)
        ax.text(.46,.87,'给定配额的合法指派',ha='center',fontsize=10,color=BLUE)
        ax.text(.46,.07,'成本：优先预测超时数，随后预测分配能耗\n每个请求占用一个槽；槽总数等于请求总数',ha='center',fontsize=10)
    elif number==3:
        box(ax,.03,.73,.27,.18,'请求嵌入（4维输入）\n节点嵌入（16维输入）',TEAL)
        box(ax,.37,.73,.27,.18,'合法边（14维输入）\n当前／最好方案标记',TEAL)
        box(ax,.71,.73,.26,.18,'两轮二部图消息传递\n128维隐藏表示',BLUE)
        arrow(ax,(.31,.82),(.36,.82));arrow(ax,(.65,.82),(.70,.82))
        box(ax,.71,.42,.26,.18,'请求与节点均值池化\n当前状态＋剩余预算',BLUE)
        arrow(ax,(.84,.72),(.84,.61))
        box(ax,.37,.42,.27,.18,'STOP／迁移策略头\n独立价值输出头',BLUE)
        arrow(ax,(.70,.51),(.65,.51))
        box(ax,.03,.42,.27,.18,'91类动作及有效掩码\n转移1／2／4个任务',TEAL)
        arrow(ax,(.36,.51),(.31,.51))
        box(ax,.03,.08,.94,.18,'当前方案可以暂时变差；执行始终选择已找到的最好方案\n教师标签、PPO采样概率、评估结果使用各自独立的数据路径',GRAY)
    else:
        box(ax,.03,.70,.29,.20,'10个新独立训练块\n基座200回合×1000步\n上层训练负载1080001-10',BLUE)
        box(ax,.37,.70,.29,.20,'独立下层训练图\n25回合×8预定时刻\n采图负载1090001-10',TEAL)
        box(ax,.71,.70,.26,.20,'冻结最终检查点\n教师5轮＋PPO100更新\n每更新32图／8轮',BLUE)
        arrow(ax,(.33,.80),(.36,.80));arrow(ax,(.67,.80),(.70,.80))
        box(ax,.03,.36,.43,.20,'确认：10块×5负载×6回合×6方法\n1800质量回合＋1800隔离计时回合\n确认负载1070001-5',BLUE)
        box(ax,.54,.36,.43,.20,'冻结跨场景：UE×算力九格\n8个新格，共14400质量回合\n主格引用原确认结果',TEAL)
        box(ax,.03,.06,.94,.17,'随机补充：复用确认负载，两组各300质量＋300隔离计时回合\n推断单位为训练块／配对随机流重复，工作负载和请求不增加独立训练数',GRAY)
    save(fig,number,[ax])


def block_values(pack,method,metric):
    vals=[]
    for block in range(1060001,1060011):
        vals.append(pack['random']['block_metrics'][f'{method}:{block}'][metric])
    return np.asarray(vals,float)


def interval(vals):
    return float(np.mean(vals)),float(2.262157*np.std(vals,ddof=1)/np.sqrt(len(vals)))


def ranking(ax,pack,metric,factor=1,xlabel='',order=METHODS):
    y=np.arange(len(order))
    for i,m in enumerate(order):
        v=block_values(pack,m,metric)*factor
        mean,err=interval(v)
        ax.barh(i,mean,color=COLORS[m],alpha=.66,height=.59)
        ax.errorbar(mean,i,xerr=err,color='#333333',capsize=2.4,lw=.9,fmt='none')
        ax.scatter(v,i+np.linspace(-.17,.17,len(v)),s=12,color=COLORS[m],edgecolors='white',linewidths=.25,zorder=3)
    ax.set_yticks(y,[LABELS[m] for m in order],fontsize=9)
    ax.invert_yaxis();ax.set_xlabel(xlabel);ax.set_ylim(len(order)-.5,-.5)


def paired(ax,comparison,title):
    vals=np.asarray(comparison['values'])*100
    ax.scatter(vals,np.arange(1,11),s=28,color=BLUE)
    mean=comparison['mean']*100;lo,hi=np.asarray(comparison['ci95'])*100
    ax.errorbar(mean,12,xerr=[[mean-lo],[hi-mean]],fmt='D',color=TEAL,capsize=4,markersize=6)
    ax.axvline(0,lw=.8,ls='--',color=GRAY)
    ax.set_yticks([1,3,5,7,9,12],['1','3','5','7','9','均值'])
    ax.set_ylim(13,0);ax.set_title(title,pad=14);ax.set_xlabel('DSR差值 / 百分点')


def training():
    data=read(DATA/'training_behavior.json')
    fig,axs=panels(2,3,6.4)
    fig.subplots_adjust(left=.095,right=.985,wspace=.60,hspace=.72)
    specifications=[('foundation','reward','基座平均步奖励','训练回合'),('foundation','dsr','基座DSR','训练回合'),
        ('quota','ppo_loss','配额PPO目标损失','PPO更新次数'),('quota','value_loss','配额价值损失','PPO更新次数')]
    for ax,(kind,metric,title,xlabel) in zip(axs.flat,specifications):
        series=[]
        for rows in data[kind].values():
            rows=[r for r in rows if metric in r]
            v=np.asarray([r[metric] for r in rows]);series.append(v)
            ax.plot(np.arange(1,len(v)+1),v,color=GRAY,alpha=.25,lw=.6)
        values=np.asarray(series)
        mean=values.mean(0);sd=values.std(0,ddof=1);x=np.arange(1,len(mean)+1)
        ax.fill_between(x,mean-sd,mean+sd,color=BLUE,alpha=.14)
        ax.plot(x,mean,color=BLUE,lw=1.3)
        ax.set(title=title,xlabel=xlabel,ylabel='比例' if metric=='dsr' else '记录值')
    for ax,metric,title in zip(list(axs.flat)[4:],['supervised_loss','teacher_samples'],['教师训练记录损失','教师监督样本数']):
        vals=[rows[0][metric] for rows in data['quota'].values()]
        ax.scatter(range(1,11),vals,color=TEAL,s=21)
        ax.set(title=title,xlabel='独立训练块序号',ylabel='记录值')
        ax.set_xticks([1,3,5,7,9])
    fig.text(.5,.027,'前四面板：全部10块原始曲线；粗线为均值，带为块间±1 SD；未平滑',ha='center',fontsize=9)
    fig.text(.5,.006,'后两面板：每块一次教师监督记录；未保存逐轮损失，不构造教师收敛曲线',ha='center',fontsize=9)
    save(fig,5,list(axs.flat))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def write_csv(path, fields, rows):
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def motion_diagnostics(data):
    """Derive motion from consecutive recorded positions, never raw actions.

    The existing prespecified replay records the initial UE positions only.
    These coordinates must not be treated as a dynamic coverage heatmap.
    The geometry check is descriptive for this episode, not a safety proof.
    """
    assert (data['block'],data['workload'],data['episode'])==(1060001,1070001,0)
    tolerance=1e-6
    result={
        'source':'docs/figures/data/sf_hlm_20261004/replay.json',
        'source_sha256':sha256(DATA/'replay.json'),
        'protocol_sha256':data['protocol_sha256'],
        'selection':data['selection'],
        'block':data['block'],'workload':data['workload'],'episode':data['episode'],
        'units':{'position':'m','path_length':'m','displacement':'m','time':'s'},
        'constraints':{'maximum_step_displacement_m':15.0,'minimum_pairwise_separation_m':200.0,
            'position_domain_m':[50.0,650.0],'time_slot_s':1.0,'numeric_tolerance_m':tolerance},
        'scope':'One prespecified episode per method; descriptive post-repair positions, no inferential test.',
        'methods':{}}
    summary_rows=[];time_rows=[]
    for method in ('learned','greedy'):
        record=data['methods'][method];rows=record['steps']
        assert len(rows)==1000
        assert [r['step'] for r in rows]==list(range(1000))
        before=np.asarray([r['uav_before'] for r in rows],float)
        after=np.asarray([r['uav_after'] for r in rows],float)
        assert before.shape==after.shape==(1000,5,2)
        assert np.isfinite(before).all() and np.isfinite(after).all()
        discontinuity=float(np.max(np.abs(before[1:]-after[:-1])))
        assert discontinuity<=tolerance
        paths=np.concatenate([before,after[-1:]],axis=0)
        displacement=np.linalg.norm(after-before,axis=-1)
        separation=np.stack([np.linalg.norm(paths[:,i]-paths[:,j],axis=-1)
            for i in range(5) for j in range(i+1,5)],axis=-1).min(axis=-1)
        checks={
            'continuous':discontinuity<=tolerance,
            'step_budget':bool(np.all(displacement<=15.0+tolerance)),
            'minimum_separation':bool(np.all(separation>=200.0-tolerance)),
            'position_domain':bool(np.all((paths>=50.0-tolerance)&(paths<=650.0+tolerance)))}
        assert all(checks.values()),f'Motion geometry failed for {method}: {checks}'
        lengths=displacement.sum(axis=0)
        result['methods'][method]={
            'archived_episode_sha256':record['archived_episode_sha256'],
            'all_assignments_and_metrics_match':record['all_assignments_and_metrics_match'],
            'steps':len(rows),'uavs':5,'checks':checks,
            'maximum_continuity_error_m':discontinuity,
            'maximum_step_displacement_m':float(displacement.max()),
            'minimum_pairwise_separation_m':float(separation.min()),
            'minimum_coordinate_m':float(paths.min()),'maximum_coordinate_m':float(paths.max()),
            'path_length_per_uav_m':lengths.tolist(),'fleet_path_length_m':float(lengths.sum()),
            'time_s':list(range(1001)),
            'min_pairwise_separation_m':separation.tolist(),
            'actual_step_displacement_m':displacement.tolist()}
        for i,length in enumerate(lengths):
            summary_rows.append({'method':method,'uav':i+1,'path_length_m':float(length),
                'maximum_step_displacement_m':float(displacement[:,i].max())})
        for t,value in enumerate(separation):
            item={'method':method,'time_s':t,'minimum_pairwise_separation_m':float(value)}
            item.update({f'uav_{i+1}_step_displacement_m':float(displacement[t-1,i]) if t else ''
                for i in range(5)})
            time_rows.append(item)
    assert result['methods']['learned']['all_assignments_and_metrics_match']
    assert result['methods']['greedy']['all_assignments_and_metrics_match']
    write_csv(DATA/'swarm_motion_summary.csv',
        ['method','uav','path_length_m','maximum_step_displacement_m'],summary_rows)
    write_csv(DATA/'swarm_motion_timeseries.csv',
        ['method','time_s','minimum_pairwise_separation_m']+
        [f'uav_{i+1}_step_displacement_m' for i in range(5)],time_rows)
    result['csv_sha256']={name:sha256(DATA/name) for name in
        ('swarm_motion_summary.csv','swarm_motion_timeseries.csv')}
    write_json(DATA/'swarm_motion_diagnostics.json',result)
    validation={
        'status':'PASS','scope':result['scope'],'source_sha256':result['source_sha256'],
        'constraints':result['constraints'],
        'methods':{method:{key:value for key,value in record.items() if key not in
            ('time_s','min_pairwise_separation_m','actual_step_displacement_m')}
            for method,record in result['methods'].items()},
        'derived_sha256':{name:sha256(DATA/name) for name in
            ('swarm_motion_summary.csv','swarm_motion_timeseries.csv','swarm_motion_diagnostics.json')}}
    write_json(QA.parent/'swarm_motion_check.json',validation)
    return result


def trajectories():
    data=read(DATA/'replay.json');derived=motion_diagnostics(data)
    fig,axs=panels(2,2,7.6)
    fig.subplots_adjust(left=.10,right=.97,bottom=.15,top=.90,wspace=.42,hspace=.50)
    colors=['#3569AC','#65A69A','#B18D5B','#A486B0','#CD7466']
    for ax,m in zip(axs[0],['learned','greedy']):
        rows=data['methods'][m]['steps'];users=np.array(rows[0]['ue_positions'])
        ax.scatter(users[:,0],users[:,1],s=8,color='#9EA2A8',alpha=.65)
        paths=np.array([r['uav_before'] for r in rows]+[rows[-1]['uav_after']])
        for i in range(5):
            ax.plot(paths[:,i,0],paths[:,i,1],color=colors[i],lw=.95,alpha=.9,label=f'UAV{i+1}')
            ax.scatter(*paths[0,i],s=30,marker='o',color=colors[i],edgecolor='white',lw=.5,zorder=4)
            ax.scatter(*paths[-1,i],s=37,marker='^',color=colors[i],edgecolor='white',lw=.5,zorder=4)
            angle=np.linspace(0,2*np.pi,721)
            circle=paths[-1,i]+100*np.column_stack([np.cos(angle),np.sin(angle)])
            # Materialize only the visible arc: PDF geometry auditors cannot infer
            # Matplotlib's clip path when checking axis tick-label intersections.
            circle[np.any((circle<0)|(circle>700),axis=1)]=np.nan
            ax.plot(circle[:,0],circle[:,1],color=colors[i],ls='--',lw=.6)
        ax.set(xlim=(0,700),ylim=(0,700),xlabel='x / m',ylabel='y / m',title=SHORT[m])
        ax.set_aspect('equal')
        ax.set_xticks([0,350,700]);ax.set_yticks([0,350,700])
    for ax in axs.flat:
        ax.set_box_aspect(1)
    ax=axs[1,0];x=np.arange(1,6)
    for offset,m in zip([-.18,.18],['learned','greedy']):
        lengths=np.asarray(derived['methods'][m]['path_length_per_uav_m'])/1000
        ax.bar(x+offset,lengths,width=.33,color=COLORS[m],alpha=.8,label=SHORT[m])
    ax.set(xlabel='无人机编号',ylabel='实际累计航程 / km',title='各机实际机动航程')
    ax.set_xticks(x,[f'{i}' for i in x]);ax.set_ylim(0,11.8)
    ax.legend(loc='upper center',ncol=2,frameon=False,fontsize=8.5)
    ax=axs[1,1]
    for m in ['learned','greedy']:
        row=derived['methods'][m]
        ax.plot(row['time_s'],row['min_pairwise_separation_m'],color=COLORS[m],
            lw=1 if m=='learned' else .8,alpha=.9,label=SHORT[m])
    ax.axhline(200,color=GRAY,ls='--',lw=.9,label='既定200 m约束')
    upper=max(max(row['min_pairwise_separation_m']) for row in derived['methods'].values())
    ax.set(xlabel='回合时间 / s',ylabel='最小机间距 / m',title='环境修复后的几何约束',
        xlim=(0,1000),ylim=(190,upper+8))
    ax.set_xticks([0,500,1000])
    ax.legend(loc='upper right',frameon=False,fontsize=8)
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=5,loc='lower center',bbox_to_anchor=(.5,.035),frameon=False,fontsize=9)
    fig.text(.5,.976,'预定首块1060001／负载1070001／回合0：两方法各1000步，描述性诊断',ha='center',fontsize=9)
    fig.text(.5,.950,'灰点仅为初始UE位置；圆点起点，三角终点，虚线为终点100 m服务半径',ha='center',fontsize=9)
    fig.text(.5,.012,'航程由实际位置差累加；位移≤15 m／步，位置域为[50,650] m；几何约束由环境修复',
        ha='center',fontsize=8.5)
    save(fig,6,list(axs.flat))


def attention():
    record=read(DATA/'replay.json')['methods']['learned']['attention']['500']
    fig,axs=panels(2,1,5.8)
    fig.subplots_adjust(left=.13,right=.88,hspace=.66)
    for ax,name,title in zip(axs.flat,['ue','neighbor'],['UE观测槽位：八头平均','邻机观测槽位：八头平均']):
        d=record[name];weights=np.mean(np.asarray(d['weights']),axis=1)
        mask=np.asarray(d['mask']);masked=np.ma.masked_where(~mask,weights)
        cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#E5E5E5')
        image=ax.imshow(masked,aspect='auto',cmap=cmap,vmin=0,vmax=max(.05,float(weights.max())))
        ax.set_yticks(range(5),[f'UAV{i+1}' for i in range(5)])
        ax.set_xticks(np.arange(weights.shape[1]),np.arange(1,weights.shape[1]+1),fontsize=8)
        ax.set(xlabel='观测槽位序号',title=title)
        fig.colorbar(image,ax=ax,fraction=.035,pad=.025)
    fig.text(.5,.02,'预定回合第500步；灰色为掩码槽；按真实SDPA的每头维度8归一化，权重仅作描述',ha='center',fontsize=9)
    save(fig,7,list(axs.flat))


def performance(pack):
    fig,ax=plt.subplots(figsize=(7.2,4.6));fig.subplots_adjust(left=.31,right=.97,bottom=.16,top=.91)
    ranking(ax,pack,'dsr',100,'DSR / %');ax.set_xlim(0,40)
    ax.set_title('相同自然覆盖主场景的八方法比较',pad=12)
    fig.text(.5,.025,'点为10个块，条为均值，误差线为块均值95% t区间；全随机块是策略随机流重复',ha='center',fontsize=8.5)
    save(fig,8,[ax])
    fig,axs=panels(1,3,4.2)
    for ax,m in zip(axs.flat,['greedy','original_frozen','original_adapted']):
        paired(ax,pack['confirmation']['comparisons']['learned-'+m],SHORT[m])
    axs[0,0].set_ylabel('训练块序号')
    save(fig,9,list(axs.flat))
    fig,axs=panels(1,2,4.7);fig.subplots_adjust(left=.26,wspace=.62)
    ax=axs[0,0];y=np.arange(8)
    adm=np.array([block_values(pack,m,'admission').mean() for m in METHODS])*100
    succ=np.array([block_values(pack,m,'dsr').mean() for m in METHODS])*100
    ax.barh(y,succ,color=BLUE,label='按时成功')
    ax.barh(y,adm-succ,left=succ,color='#A8BFCC',label='接纳后失败')
    ax.barh(y,100-adm,left=adm,color='#DEDFE2',label='未接纳')
    ax.set_yticks(y,[SHORT[m] for m in METHODS],fontsize=9);ax.invert_yaxis();ax.set(xlabel='实际生成服务请求占比 / %',xlim=(0,100))
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.19),ncol=1,fontsize=9,frameon=False)
    for i,m in enumerate(METHODS):
        v=block_values(pack,m,'conditional')*100;mean,err=interval(v)
        axs[0,1].errorbar(mean,i,xerr=err,fmt='o',capsize=3,color=COLORS[m])
    axs[0,1].set_yticks(y,[SHORT[m] for m in METHODS],fontsize=9);axs[0,1].invert_yaxis()
    axs[0,1].set_xlabel('接纳后按时完成率 / %')
    fig.subplots_adjust(bottom=.25)
    save(fig,10,list(axs.flat))
    fig,axs=panels(1,2,4.6);fig.subplots_adjust(left=.25,wspace=.62)
    ranking(axs[0,0],pack,'energy_per_step',.001,'UAV总能耗 / (kJ/步)')
    ranking(axs[0,1],pack,'mbs_share',100,'MBS占已接纳服务请求 / %')
    for ax in axs.flat:ax.set_yticklabels([SHORT[m] for m in METHODS],fontsize=9)
    save(fig,11,list(axs.flat))
    fig,axs=panels(1,2,4.6);fig.subplots_adjust(left=.25,wspace=.62)
    ranking(axs[0,0],pack,'fairness',1,'Jain公平性')
    ranking(axs[0,1],pack,'p95',1000,'隔离完整路径p95 / ms')
    for ax in axs.flat:ax.set_yticklabels([SHORT[m] for m in METHODS],fontsize=9)
    axs[0,0].set_xlim(.74,.96)
    save(fig,12,list(axs.flat))
    fig,axs=panels(1,2,4.2)
    for ax,m in zip(axs.flat,['heuristic','random']):paired(ax,pack['confirmation']['comparisons']['learned-'+m],SHORT[m])
    axs[0,0].set_ylabel('训练块序号')
    save(fig,13,list(axs.flat))


def scenario(pack,ue,scale):
    r=pack['robustness']
    return r['nominal_reference'] if (ue,scale)==(100,100) else r['scenario_results'][f'ue_{ue}_compute_{scale}']


def scenario_comparison(pack,ue,scale,method):
    values=scenario(pack,ue,scale)['comparisons']
    suffix='learned-'+method
    return values[suffix] if suffix in values else values[f'ue_{ue}_compute_{scale}:'+suffix]


def heat(ax,values,title,fmt='.2f',vmin=None,vmax=None,cmap='Blues'):
    image=ax.imshow(values,aspect='auto',cmap=cmap,vmin=vmin,vmax=vmax)
    ax.set_xticks([0,1,2],['0.6','1.0','1.4'])
    ax.set_yticks([0,1,2],['60','100','140']);ax.set(xlabel='UAV算力倍率',ylabel='UE数量',title=title)
    low,high=image.get_clim()
    for i in range(3):
        for j in range(3):ax.text(j,i,format(values[i,j],fmt),ha='center',va='center',fontsize=10,
            color='white' if values[i,j] > low+.57*(high-low) else '#222222')
    return image


def robustness(pack):
    fig,axs=panels(1,2,3.9);fig.subplots_adjust(right=.88,wspace=.48)
    for ax,metric,title in zip(axs.flat,['dsr','conditional'],['SF-HLM DSR / %','接纳后完成率 / %']):
        values=np.array([[scenario(pack,u,s)['groups']['learned'][metric]*100 for s in (60,100,140)] for u in (60,100,140)])
        image=heat(ax,values,title,vmin=0,vmax=85)
    fig.colorbar(image,ax=list(axs.flat),fraction=.025,pad=.04)
    save(fig,14,list(axs.flat))
    fig,axs=panels(1,3,3.7);fig.subplots_adjust(left=.085,right=.94,wspace=.42,bottom=.22)
    for ax,m in zip(axs.flat,['greedy','original_frozen','original_adapted']):
        vals=np.array([[scenario_comparison(pack,u,s,m)['mean']*100 for s in (60,100,140)] for u in (60,100,140)])
        heat(ax,vals,SHORT[m],vmin=0,vmax=4)
    fig.text(.5,.03,'数值为SF-HLM减对手的DSR差（百分点）；同10套冻结权重，各格均含五负载×六回合',ha='center',fontsize=9)
    save(fig,15,list(axs.flat))
    fig,axs=panels(1,2,3.8)
    ms=['learned','heuristic','random','greedy']
    for ax,key,title in zip(axs.flat,['free','fixed_mbs'],['自由MBS数量','固定共同MBS数量']):
        d=pack['robustness']['snapshot_diagnostics'][key]
        for i,m in enumerate(ms):ax.bar(i,d['groups'][m]['actual_misses'],color=COLORS[m],width=.65)
        ax.set_xticks(range(4),[SHORT[m] for m in ms],fontsize=9)
        ax.set(ylabel='每快照实际超时请求数',title=title,ylim=(0,13))
    save(fig,16,list(axs.flat))
    fig,axs=panels(1,3,3.7);fig.subplots_adjust(left=.085,right=.98,wspace=.54,bottom=.21)
    budgets=[20,50,100]
    for m in ms:
        rows=[pack['robustness']['snapshot_diagnostics'][f'wall_{b}ms']['groups'][m] for b in budgets]
        for ax,metric,factor in zip(axs.flat,['actual_misses','mean_seconds','overrun_fraction'],[1,1000,100]):
            ax.plot(budgets,[d[metric]*factor for d in rows],'-o',color=COLORS[m],ms=4,lw=1.1,label=SHORT[m])
    for ax,ylabel in zip(axs.flat,['每快照实际超时数','实际完整路径均值 / ms','超出软预算的快照 / %']):
        ax.set(xlabel='墙钟软预算 / ms',ylabel=ylabel);ax.set_xticks(budgets)
    fig.legend(*axs[0,0].get_legend_handles_labels(),ncol=4,loc='lower center',bbox_to_anchor=(.5,.002),frameon=False,fontsize=9)
    save(fig,17,list(axs.flat))


def behavior():
    d=read(DATA/'training_behavior.json')['behavior'];fig,axs=panels(1,2,3.8)
    totals={};terms={}
    for block in d.values():
        for k,v in block['matching_calls'].items():totals[int(k)]=totals.get(int(k),0)+v
        for k,v in block['termination'].items():terms[k]=terms.get(k,0)+v
    calls=sorted(totals);axs[0,0].bar(calls,[totals[k]/3000 for k in calls],color=BLUE)
    axs[0,0].set(xlabel='每步条件匹配调用数',ylabel='全部确认环境步占比 / %');axs[0,0].set_xticks([0,4,8,12,16])
    aliases={'policy_stop':'策略STOP','stop':'策略STOP','budget':'预算用尽','matching_budget':'预算用尽',
        'no_actions':'无有效动作','no_valid_action':'无有效动作','empty':'空批次','empty_batch':'空批次',
        'infeasible_initial':'初始匹配不可行','no_feasible_match':'无可行匹配',
        'empty_or_zero_budget':'空批次或\n零预算','unrecorded':'未单独记录'}
    keys=sorted(terms,key=terms.get,reverse=True)
    axs[0,1].barh(range(len(keys)),[terms[k]/3000 for k in keys],color=TEAL)
    axs[0,1].set_yticks(range(len(keys)),[aliases.get(k,k.replace('_','\n')) for k in keys],fontsize=8.5)
    axs[0,1].invert_yaxis();axs[0,1].set_xlabel('全部确认环境步占比 / %')
    fig.text(.5,.015,'300000个环境步的描述分布；包含空请求批次，环境步不作为独立训练重复',ha='center',fontsize=9)
    save(fig,18,list(axs.flat))


def task_division():
    """Show all learned-policy training blocks without pseudo-replication.

    Ratios use the sum of the three recorded execution branches in each
    block. The arithmetic mean of block ratios and pooled count ratio are
    stored separately. Neither quantity is a successful-completion rate.
    """
    data=read(DATA/'training_behavior.json')
    behavior=data['behavior'];keys=['local','cooperative','mbs']
    assert sorted(behavior)==[str(i) for i in range(1060001,1060011)]
    records=[]
    for block in sorted(behavior):
        source=behavior[block]['destinations']
        assert set(source)==set(keys)
        assert behavior[block]['steps']==30000
        assert all(np.isfinite(source[k]) and source[k]>=0 and source[k]==int(source[k]) for k in keys)
        counts={k:int(source[k]) for k in keys};denominator=sum(counts.values())
        assert denominator>0
        fractions={k:counts[k]/denominator for k in keys}
        assert abs(sum(fractions.values())-1)<1e-12
        records.append({'block':int(block),'environment_steps':30000,'execution_branch_total':denominator,
            **{f'{k}_count':counts[k] for k in keys},**{f'{k}_fraction':fractions[k] for k in keys}})
    values=np.asarray([[r[f'{k}_fraction'] for k in keys] for r in records])
    means=values.mean(axis=0)
    pooled={k:sum(r[f'{k}_count'] for r in records) for k in keys}
    total=sum(pooled.values())
    source_receipts={path:digest for path,digest in data['source_files'].items()
        if '/confirmation_eval/learned/' in path}
    assert len(source_receipts)==300
    result={
        'source':'docs/figures/data/sf_hlm_20261004/training_behavior.json',
        'source_sha256':sha256(DATA/'training_behavior.json'),
        'protocol_sha256':data['protocol_sha256'],
        'method':'learned','independent_unit':'training block','blocks':10,
        'workloads_per_block':5,'episodes_per_workload':6,'environment_steps_per_episode':1000,
        'denominator':'local_count + cooperative_count + mbs_count, separately in each block',
        'source_metric_mapping':{k:f'service_offloads_{k}' for k in keys},
        'meaning':{'local':'Execution at the admitting/source UAV',
            'cooperative':'Execution at another UAV','mbs':'Execution at the MBS'},
        'scope':'Recorded execution-branch fractions; not successful-completion fractions, per-UAV loads or cross-method effects.',
        'summary':'Arithmetic mean of all ten block fractions; individual block points are shown; no confidence intervals or hypothesis tests.',
        'records':records,'mean_block_fractions':dict(zip(keys,means.tolist())),
        'pooled_counts':pooled,'pooled_execution_branch_total':total,
        'pooled_fractions':{k:pooled[k]/total for k in keys},
        'archived_source_receipts_count':len(source_receipts),
        'archived_source_receipts_sha256':hashlib.sha256(json.dumps(source_receipts,sort_keys=True).encode()).hexdigest()}
    fields=list(records[0])
    write_csv(DATA/'swarm_task_division.csv',fields,records)
    result['csv_sha256']=sha256(DATA/'swarm_task_division.csv')
    write_json(DATA/'swarm_task_division.json',result)
    write_json(QA.parent/'swarm_task_division_check.json',{
        'status':'PASS','blocks':10,'recorded_execution_branches':total,
        'denominator':result['denominator'],'source_sha256':result['source_sha256'],
        'mean_block_fractions':result['mean_block_fractions'],'pooled_fractions':result['pooled_fractions'],
        'source_data_sha256':{name:sha256(DATA/name) for name in
            ('swarm_task_division.csv','swarm_task_division.json')},
        'checks':{'all_expected_blocks':True,'nonnegative_integer_counts':True,
            'positive_denominators':True,'block_fraction_sums_equal_one':True,
            'confirmation_episode_receipts':300}})
    fig,ax=plt.subplots(figsize=(7.2,4.4))
    fig.subplots_adjust(left=.12,right=.97,bottom=.20,top=.86)
    labels=['接纳源UAV执行','协作UAV执行','MBS执行']
    colors=[BLUE,TEAL,COLORS['greedy']]
    jitter=np.linspace(-.13,.13,10)
    for i in range(3):
        ax.bar(i,means[i]*100,width=.54,color=colors[i],alpha=.45,edgecolor='none')
        ax.scatter(i+jitter,values[:,i]*100,s=32,color=colors[i],edgecolor='white',linewidth=.55,zorder=3)
        ax.text(i,values[:,i].max()*100+2,f'{means[i]*100:.2f}%',ha='center',va='bottom',fontsize=10)
    ax.set_xticks(range(3),labels)
    ax.set(ylabel='记录的执行分支占比 / %',xlabel='服务请求的执行节点类别',
        title='SF-HLM集群任务执行分工：全部10个训练块',ylim=(0,60),xlim=(-.6,2.6))
    fig.text(.5,.069,'点：每块五负载×六回合的计数比例；柱与数字：十块比例的算术均值',ha='center',fontsize=9)
    fig.text(.5,.024,'分母为三类执行分支计数之和；包含超时任务，未进行显著性检验',ha='center',fontsize=9)
    save(fig,19,[ax])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--figures',nargs='*',type=int,choices=range(1,20),default=list(range(1,20)))
    args=p.parse_args();font=setup_thesis_style(10)
    # Preserve the school-compatible Chinese serif face selected by the shared
    # style; declare both families here so source preflight can inspect them.
    plt.rcParams.update({'font.family':'serif','font.serif':[font]+FONT_CANDIDATES,
        'font.sans-serif':[font]+FONT_CANDIDATES,'font.size':10,'axes.titlesize':10.5,'axes.labelsize':10,
        'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'svg.fonttype':'none','pdf.fonttype':42})
    pack=read(DATA/'analysis_data.json')
    for number in args.figures:
        if number in (1,2,3,4):schematic(number)
        elif number==5:training()
        elif number==6:trajectories()
        elif number==7:attention()
        elif number==8:performance(pack)
        elif number==14:robustness(pack)
        elif number==18:behavior()
        elif number==19:task_division()
        elif number in (9,10,11,12,13,15,16,17):
            continue  # emitted with the corresponding evidence group
        else:raise ValueError('Unknown figure')


if __name__=='__main__':main()
