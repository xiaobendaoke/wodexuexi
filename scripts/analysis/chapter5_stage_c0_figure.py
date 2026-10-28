"""C0 joint quality/time/energy screen: three complementary evidence panels.

Contract: is there a fixed prefix that meets the predeclared development tradeoff?
All six methods and all five paired blocks are retained; marginal intervals,
not simultaneous confidence or formal noninferiority. Server-only Python.
"""
import argparse
import csv
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--qa-tools',type=Path,required=True)
    a=parser.parse_args()
    if sys.platform!='linux':raise RuntimeError('Scientific rendering executes on server')
    data=json.loads((a.results/'analysis/summary.json').read_text())
    assert data['status']=='COMPLETE_C0_DEVELOPMENT'
    assert json.loads((a.results/'analysis/raw_audit.json').read_text())['status']=='PASS'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    sys.path.insert(0,str(a.qa_tools))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    matplotlib.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7.5,
        'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.7,'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False})
    a.output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(183/25.4,88/25.4))
    fig.subplots_adjust(left=.085,right=.98,bottom=.35,top=.82,wspace=.63)
    methods=['G','B2','B4','B6','B10','native_B10'];positions=[0,2,4,6,10,12.5];labels=['G','2','4','6','10','N10']
    panels=[('dsr_difference_pp','Paid DSR difference (pp)',-.1,'Paid quality'),
        ('full_time_ratio','Full-call mean ratio',.8,'Complete software time'),
        ('energy_ratio','Fleet-energy ratio',1.05,'Energy constraint')]
    rows=[]
    for ax,(key,ylabel,gate,title),letter in zip(axes,panels,'abc'):
        extent=[];means=[]
        for i,method in enumerate(methods):
            e=data['closed_loop'][method][key];mean=e['mean'];low,high=e['ci95'];x=positions[i]
            color='#3676A8' if method!='native_B10' else '#8A939B'
            ax.plot(x+np.linspace(-.13,.13,5),e['values'],'o',ms=2.3,color=color,alpha=.35,zorder=2)
            ax.errorbar(x,mean,yerr=[[mean-low],[high-mean]],fmt='s' if method=='native_B10' else 'o',
                ms=4,color=color,lw=1,capsize=2,zorder=3)
            means.append(mean);extent.extend([low,high,*e['values'],gate])
            for block,value in zip(range(620001,620006),e['values']):
                rows.append(dict(panel=letter,endpoint=key,method=method,budget=0 if method=='G' else 10 if method=='native_B10' else int(method[1:]),
                    block=block,value=value,mean=mean,ci95_low=low,ci95_high=high,development_threshold=gate))
        ax.plot(positions[:5],means[:5],color='#3676A8',lw=.7,alpha=.7,zorder=1)
        ax.axhline(gate,color='#9C7662',ls='--',lw=.8,zorder=1)
        low,high=min(extent),max(extent);pad=max((high-low)*.13,.015)
        ax.set_ylim(low-pad,high+pad);ax.set_xlim(-.7,13.2)
        ax.set_xticks(positions,labels);ax.set_xlabel('Matching budget',labelpad=7);ax.set_ylabel(ylabel,labelpad=5)
        ax.set_title(title,fontsize=7.5,pad=10)
        ax.annotate(letter,xy=(0,1),xycoords='axes fraction',xytext=(-23,10),textcoords='offset points',fontsize=9,fontweight='bold')
    fig.text(.085,.955,'C0 fixed prefixes: joint development screen',fontsize=9,fontweight='bold')
    fig.text(.085,.16,'Five historical blocks; means and marginal 95% paired t intervals. All methods use current full-call charges.',fontsize=7)
    fig.text(.085,.105,'Dashed lines: predeclared development limits. N10: original batched B10; blue budgets share the lazy implementation.',fontsize=7)
    fig.text(.085,.05,'Budget includes initialization and failed matches. This development screen does not establish formal noninferiority.',fontsize=7)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=a.output/'c0_tradeoff.alignment.json',overlay_svg=a.output/'c0_tradeoff.alignment.svg',require_panel_labels=True,strict=True)
    fig.savefig(a.output/'c0_tradeoff.pdf',facecolor='white',bbox_inches=None)
    fig.savefig(a.output/'c0_tradeoff.svg',facecolor='white',bbox_inches=None)
    fig.savefig(a.output/'c0_tradeoff.png',dpi=600,facecolor='white',bbox_inches=None)
    fig.savefig(a.output/'c0_tradeoff.tiff',dpi=600,facecolor='white',bbox_inches=None,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    with (a.output/'c0_tradeoff_source.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    caption='C0固定预算的质量—完整时间—机队能耗取舍。六方法各复用五个历史上层开发块，每块合并三个已用开发负载各1000步回合。小点为块值，大点和误差线为均值及逐项95%配对t描述区间；请求、回合、快照和计时重复不增加独立n。a为相对同实现B10的计费DSR百分点差，b/c为块内完整调用均值比和机队能耗比。虚线分别为−0.1 pp、0.80和1.05开发限值。N10保留原生批量B10，工程差异不归为停止规则收益；该图不构成正式非劣效或自适应策略证据。'
    (a.output/'c0_tradeoff_caption.md').write_text(caption+'\n')


if __name__=='__main__':main()
