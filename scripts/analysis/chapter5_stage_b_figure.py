"""Server-only Stage B figures from completed, audited block summaries.

Contract: show the range of zero-retuning development transfer and the actual
long-episode calibration distribution. Quantitative single-panel figures,
183 mm wide, editable PDF/SVG, 600 dpi raster and complete source tables.
No confirmatory claim, multiplicity-adjusted interval or sample exclusion.
"""
import argparse
import csv
import json
from pathlib import Path
import sys


def read(p):return json.loads(p.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--qa-tools',type=Path,required=True);a=parser.parse_args()
    if sys.platform!='linux':raise RuntimeError('Scientific figures are generated on the server only')
    assert read(a.results/'analysis/raw_audit.json')['status']=='PASS'
    cl=read(a.results/'analysis/closed_loop_summary.json');mech=read(a.results/'analysis/mechanism_summary.json')
    assert mech['status']=='PASS' and mech['main_states']==1200
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import t
    sys.path.insert(0,str(a.qa_tools))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    matplotlib.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':8,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False})
    a.output.mkdir(parents=True,exist_ok=True)
    def save(fig,name,rows,caption):
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig,json_out=a.output/(name+'.alignment.json'),strict=True)
        fig.savefig(a.output/(name+'.pdf'),facecolor='white',bbox_inches=None)
        fig.savefig(a.output/(name+'.svg'),facecolor='white',bbox_inches=None)
        fig.savefig(a.output/(name+'.png'),dpi=600,facecolor='white',bbox_inches=None)
        fig.savefig(a.output/(name+'.tiff'),dpi=600,facecolor='white',bbox_inches=None,pil_kwargs={'compression':'tiff_lzw'})
        plt.close(fig)
        with (a.output/(name+'_source.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        (a.output/(name+'_caption.md')).write_text(caption+'\n')
    names=['ue60','center','ue140','cpu06','cpu14'];labels=['60 UE, CPU ×1.0','100 UE, CPU ×1.0','140 UE, CPU ×1.0','100 UE, CPU ×0.6','100 UE, CPU ×1.4']
    methods=[('mixture_price-G','Mixture − G','#3676A8',-.19),('binary_price-G','Binary − G','#85919A',0.),('cold_half_proxy_price-G','Cold-half − G','#B67F48',.19)]
    fig,ax=plt.subplots(figsize=(183/25.4,105/25.4));fig.subplots_adjust(left=.25,right=.96,bottom=.22,top=.82)
    rows=[];extent=[0.]
    for i,s in enumerate(names):
        for key,label,color,offset in methods:
            e=cl['transfer'][s]['comparisons'][key]['dsr_pp'];mean=e['mean'];lo,hi=e['ci95'];ys=i+offset
            ax.plot(e['values'],ys+np.linspace(-.045,.045,5),'o',ms=2.4,color=color,alpha=.35,zorder=2)
            ax.errorbar(mean,ys,xerr=[[mean-lo],[hi-mean]],fmt='o',ms=4.3,color=color,lw=1.05,capsize=2,zorder=3,label=label if i==0 else None)
            extent.extend([lo,hi,*e['values']])
            for block,value in zip(range(620001,620006),e['values']):rows.append(dict(scenario=s,comparison=key,block=block,dsr_difference_pp=value,mean_pp=mean,ci95_low_pp=lo,ci95_high_pp=hi))
    low,high=min(extent),max(extent);pad=max((high-low)*.1,.1)
    ax.set_xlim(low-pad,high+pad);ax.set_ylim(4.5,-.5);ax.set_yticks(range(5),labels);ax.tick_params(axis='y',length=0,pad=6)
    ax.axvline(0,color='#A0A4A8',lw=.7,ls='--',zorder=1);ax.set_xlabel('Paid DSR difference from G (percentage points)',labelpad=8)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.60,.91),ncol=3,fontsize=7.5,handletextpad=.5,columnspacing=1.2)
    fig.text(.25,.965,'Frozen-policy transfer: five historical development blocks',fontsize=8.5,fontweight='bold')
    fig.text(.25,.08,'Small points: paired blocks. Large points and lines: mean and marginal 95% t interval.',fontsize=7)
    fig.text(.25,.035,'Three new workloads per block; actual per-call charges; no retuning or simultaneous-interval claim.',fontsize=7)
    save(fig,'stage_b_transfer',rows,'五场景零调参开发迁移。每方法每块合并三个新负载各一个1000步回合的成功与生成计数，再求S/G。小点为五个历史上层块的配对差，大点与线为块均差和描述性95%配对t区间（非同时区间）。冻结价格与观测槽保持不变，140 UE场景仍可能受30槽上层观测截断限制。所有方法在本轮最多32核条件下重新计入逐次完整费用；不与旧120核耗时拼接，不代表新训练块确认。')
    fig,ax=plt.subplots(figsize=(183/25.4,115/25.4));fig.subplots_adjust(left=.14,right=.94,bottom=.22,top=.86)
    styles=[('p','Original prior','#3676A8','o'),('cold_prior','Cold prior','#85919A','s'),('cold_half','Cold-half','#B67F48','^')]
    rows=[];limits=[0.,1.]
    ax.plot([0,1],[0,1],color='#B2B6BA',lw=.8,ls='--',zorder=1)
    for field,label,color,marker in styles:
        points=[]
        for bin_id in range(10):
            selected=[]
            for block,data in mech['per_block'].items():
                z=data['both']['calibration']['all'][field]
                if z is None:continue
                rec=z['bins'][bin_id]
                rows.append(dict(block=block,method=field,bin=bin_id,edges=rec['n'],mean_probability=rec['predicted'],observed_hit_rate=rec['observed']))
                if rec['n']:
                    selected.append((rec['predicted'],rec['observed']))
                    ax.plot(rec['predicted'],rec['observed'],marker=marker,ms=2.6,color=color,alpha=.3,ls='none',zorder=2)
            if selected:
                x=statistics_mean([z[0] for z in selected]);ys=[z[1] for z in selected];y=statistics_mean(ys);n=len(ys)
                radius=float(t.ppf(.975,n-1))*float(np.std(ys,ddof=1))/np.sqrt(n) if n>1 else 0.
                ax.errorbar(x,y,yerr=radius,fmt=marker,ms=5,color=color,capsize=2,lw=1.1,zorder=3,label=label if not points else None)
                points.append((x,y));limits.extend([y-radius,y+radius])
    ax.set_xlim(-.04,1.04);ax.set_ylim(min(limits)-.04,max(limits)+.04)
    ax.set_xticks(np.linspace(0,1,6));ax.set_yticks(np.linspace(0,1,6))
    ax.set_xlabel('Mean public probability within bin',labelpad=7);ax.set_ylabel('Observed cooperative-cache hit fraction',labelpad=7)
    ax.legend(loc='lower right',fontsize=7.5)
    fig.text(.14,.95,'Long-episode calibration on all legal cooperative edges',fontsize=8.5,fontweight='bold')
    fig.text(.14,.10,'1,200 stratified main states; cold-start extras excluded; private truth used only offline.',fontsize=7)
    fig.text(.14,.055,'Small points: available block-bin means. Error bars: marginal 95% t interval across blocks.',fontsize=7)
    fig.text(.14,.02,'All ten bins retained in Source Data; empty bins unplotted; a single block has no interval.',fontsize=7)
    save(fig,'stage_b_calibration',rows,'当前G与解析方法长回合状态中的全合法协作边校准。主样本为30回合×40等长区间各随机一步，共1200状态；额外step0不计入主均值。每个10等宽概率分箱先在块内合并边，再绘制可用块的均值与描述性95%t区间；小点为各块分箱均值，空箱在源表保留。各分箱可用块数见源数据，单块不画区间；边与时步不增加独立n。对角线只作校准参照，不拟合新预测器。')


def statistics_mean(values):return sum(values)/len(values)


if __name__=='__main__':main()
