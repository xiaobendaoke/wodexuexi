"""Plot all paired confirmation contrasts only from completed, verified data."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT

METHODS=('G','R10','R16','binary_price','mixture','cold_prior_price','cold_half_proxy_price','HGB_probability_price')
LABELS=dict(G='Greedy (primary)',R10='Random 10',R16='Random 16',binary_price='Binary (primary)',mixture='Unpriced mixture',cold_prior_price='Cold prior',cold_half_proxy_price='Cold half',HGB_probability_price='HGB probability')


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--evidence',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--qa-tools',type=Path,required=True);a=parser.parse_args()
    if sys.platform!='linux':raise RuntimeError('All scientific plotting runs remotely')
    done=read(a.evidence/'completion.json');index=read(a.evidence/'evidence_index.json');contract=read(a.evidence/'protocol.json')
    if done['status']!='COMPLETED' or done['smoke'] or not done['formal_confirmation_completed'] or index['smoke'] or not index['formal_confirmation_completed']:raise ValueError('Only completed formal evidence may be plotted')
    for name,digest in done['files'].items():
        if sha(a.evidence/name)!=digest:raise ValueError('Completed evidence export changed')
    summary=Path(index['source_summary'])
    if str(summary) not in contract['input_files'] or sha(summary)!=contract['input_files'][str(summary)]:raise ValueError('Figure input summary is not the sealed audited result')
    data=read(summary)
    if data['runner_smoke'] or len(data['blocks'])!=20 or data['episodes']!=2700 or data['steps']!=2700000 or set(data['comparisons'])!=set(METHODS):raise ValueError('All prespecified block/control observations required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    sys.path.insert(0,str(a.qa_tools))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    matplotlib.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7.5,'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7.5,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False})
    a.output.mkdir(parents=True,exist_ok=True);fig,ax=plt.subplots(figsize=(183/25.4,106/25.4));fig.subplots_adjust(left=.25,right=.78,bottom=.21,top=.88);records=[];extent=[0.]
    for i,method in enumerate(METHODS):
        effect=data['comparisons'][method]['dsr'];values=np.asarray(effect['values'],float)*100;ci=np.asarray(effect['ci95'],float)*100;mean=float(effect['mean'])*100
        if effect['n']!=20 or len(values)!=20 or not np.isfinite(values).all() or not np.isfinite(ci).all():raise ValueError('Invalid paired block/interval population')
        if not np.isclose(np.mean(values),mean,rtol=0,atol=1e-12):raise ValueError('Plot mean differs from all paired values')
        expected=[(row['mixture_price']['dsr']-row[method]['dsr'])*100 for row in data['blocks'].values()]
        if not np.allclose(values,expected,rtol=0,atol=1e-12):raise ValueError('Plot block identities do not reproduce the source paired differences')
        primary=method in ('G','binary_price');color='#31688E' if primary else '#737B83';jitter=np.linspace(-.13,.13,20)
        ax.scatter(values,i+jitter,s=8,color=color,alpha=.32,edgecolors='none',zorder=2)
        ax.errorbar(mean,i,xerr=[[mean-ci[0]],[ci[1]-mean]],fmt='D' if primary else 'o',color=color,markersize=4,capsize=2,linewidth=1.1,zorder=3)
        if primary:
            p=float(data['primary_family'][method]['holm_p']);label='Holm p < 0.0001' if p<.0001 else f'Holm p = {p:.3g}';ax.text(1.04,i,label,transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=7,color=color)
        extent.extend(values.tolist()+ci.tolist())
        for block,value in zip(data['blocks'],values.tolist()):records.append(dict(comparator=method,training_block=block,paired_dsr_difference_pp=value,mean_difference_pp=mean,ci95_low_pp=float(ci[0]),ci95_high_pp=float(ci[1]),primary_comparison=primary))
    low,high=min(extent),max(extent);margin=max((high-low)*.12,.1);ax.set_xlim(low-margin,high+margin);ax.set_ylim(len(METHODS)-.6,-.6);ax.set_yticks(range(len(METHODS)),[LABELS[m] for m in METHODS]);ax.tick_params(axis='y',length=0,pad=6);ax.axvline(0,color='#93999F',linewidth=.7,linestyle='--',zorder=1);ax.set_xlabel('Priced mixture minus comparator: paid DSR difference (pp)',labelpad=7)
    fig.text(.25,.965,'Loaded-server independent confirmation',fontsize=8.5,fontweight='bold',ha='left');fig.text(.25,.925,'20 training blocks; paired mean and 95% t confidence interval',fontsize=7,ha='left');fig.text(.25,.055,'Blue: two prespecified primary comparisons. Grey: descriptive comparisons.',fontsize=7,ha='left');fig.text(.25,.02,'Each small point is one training block; all eight controls and all 20 blocks retained.',fontsize=7,ha='left')
    fig.canvas.draw();require_matplotlib_panel_alignment(fig,axes=[ax],json_out=a.output/'framework_paid_dsr.alignment.json',tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(a.output/'framework_paid_dsr.pdf',facecolor='white',bbox_inches=None)
    fig.savefig(a.output/'framework_paid_dsr.svg',facecolor='white',bbox_inches=None)
    fig.savefig(a.output/'framework_paid_dsr.tiff',dpi=600,facecolor='white',bbox_inches=None,pil_kwargs={'compression':'tiff_lzw'})
    fig.savefig(a.output/'framework_paid_dsr.png',dpi=600,facecolor='white',bbox_inches=None);plt.close(fig)
    with (a.output/'framework_paid_dsr_source.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    caption='固定解析候选相对全部八个参照的计费DSR配对差。每个小点为一个独立上层训练块（n=20），每块合并5个新负载各3个1000步回合；请求和回合不增加独立n。大点及线为块均差和未作同时区间校正的95%配对t区间。蓝色两项为预定主比较，右侧给出Holm双侧校正p；其余为描述性区间。全部调用时间来自最多120个物理核的已登记高并发服务器条件，不与旧8核费用拼接。该图仅展示计费DSR比较，综合晋级还须核对方向数、UAV能耗和逐次p95门槛；不证明PPO独有增量。\n'
    (a.output/'framework_paid_dsr_caption.md').write_text(caption)
    manifest=dict(status='RENDERED_AWAITING_QA',summary_sha256=sha(summary),evidence_completion_sha256=sha(a.evidence/'completion.json'),plot_source_sha256=sha(Path(__file__)),backend='python/matplotlib',width_mm=183,height_mm=106,training_blocks=20,controls=8,source_rows=len(records),excluded_rows=0,main_intervals='paired t 95%, unadjusted intervals; two primary Holm p values',files={p.name:sha(p) for p in a.output.glob('framework_paid_dsr*') if p.is_file()})
    (a.output/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:v for k,v in manifest.items() if k!='files'}),flush=True)


if __name__=='__main__':main()
