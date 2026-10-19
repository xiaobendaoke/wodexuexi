"""Conceptual diagrams for the cost-aware chapter; no quantitative evidence.

Brief fixed by docs/plans/chapter5_cost_aware_rewrite_20261018.md:
Frozen upper policy, analytical lower assignment and measured-cost feedback;
fixed-quota slots with actual mixture scalar costs. All boxes/edges are
illustrations, not empirical states. Existing scientific paired plot is copied
byte-for-byte elsewhere; this script performs no analysis or statistics.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from paths import REPO_ROOT
os.environ.setdefault('MPLCONFIGDIR', str(REPO_ROOT / '.tmp/chapter5_cost_aware_rewrite_20261018/mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from thesis_figure_style import setup_thesis_style


def box(ax, x, y, w, h, text, accent=False):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.005,rounding_size=0.012',
        facecolor='#E7F0F7' if accent else '#F4F5F6',edgecolor='#3D6FB6' if accent else '#7B8490',linewidth=.8))
    ax.text(x,y,text,ha='center',va='center',fontsize=8.5,color='#22252A',linespacing=1.6)


def arrow(ax, a, b, feedback=False):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,
        linewidth=.85,color='#6F7D8B',linestyle='--' if feedback else '-'))


def save(fig, ax, name, out, qa, tools):
    sys.path.insert(0,str(tools))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    out.mkdir(parents=True,exist_ok=True);qa.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,axes=[ax],json_out=qa/(name+'.alignment.json'))
    fig.savefig(out/(name+'.pdf'),facecolor='white')
    fig.savefig(out/(name+'.svg'),facecolor='white')
    fig.savefig(out/(name+'.png'),dpi=600,facecolor='white')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=REPO_ROOT/'docs/figures')
    p.add_argument('--validation',type=Path,default=REPO_ROOT/'output/chapter5_cost_aware_rewrite_20261018/validation/figures')
    p.add_argument('--qa-tools',type=Path,default=Path.home()/'.codex/skills/nature-figure/scripts')
    a=p.parse_args();family=setup_thesis_style(8.5)
    plt.rcParams.update({'font.family':'serif','font.serif':[family,'DejaVu Serif'],'font.sans-serif':['DejaVu Sans'],'svg.fonttype':'none','pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(7.2,4.6));fig.subplots_adjust(left=.015,right=.985,bottom=.015,top=.985)
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    # A single process map; rows are stages rather than comparable data panels.
    for x,text in ((.18,'移动前公开观测'),(.50,'冻结需求感知上层\nAttention-MAPPO'),(.82,'轨迹动作\n当前请求之后应用')):
        box(ax,x,.84,.26,.14,text)
    arrow(ax,(.315,.84),(.365,.84));arrow(ax,(.635,.84),(.685,.84))
    for x,text in ((.18,'当前已接纳请求\n允许目标与公开预测'),(.50,'共同贪心与修复\n最多128次普通评价'),(.82,'公开配额提议\n当前配额＋至多8个迁移')):
        box(ax,x,.62,.26,.14,text)
    arrow(ax,(.18,.765),(.18,.695));arrow(ax,(.315,.62),(.365,.62));arrow(ax,(.635,.62),(.685,.62))
    for x,text in ((.82,'缓存混合评分与匹配\n含初始化最多10次'),(.50,'预测硬筛选后\n候选间能耗定价'),(.18,'完整分配与动作映射\n返回轨迹和卸载动作')):
        box(ax,x,.40,.26,.14,text,accent=x in (.82,.50))
    arrow(ax,(.82,.545),(.82,.475));arrow(ax,(.685,.40),(.635,.40));arrow(ax,(.365,.40),(.315,.40))
    for x,text in ((.18,'当前批次物理执行'),(.50,'注入实测软件费用\n随后应用轨迹动作'),(.82,'指标、奖励与下一观测\n缓存、电量与位置反馈')):
        box(ax,x,.18,.26,.14,text)
    arrow(ax,(.18,.325),(.18,.255));arrow(ax,(.315,.18),(.365,.18));arrow(ax,(.635,.18),(.685,.18))
    # Dashed is exclusively feedback, routed outside all labels and boxes.
    ax.plot([.82,.82,.98,.98,.18],[.105,.055,.055,.96,.96],color='#6F7D8B',lw=.8,ls='--')
    arrow(ax,(.18,.96),(.18,.915),True)
    save(fig,ax,'ch5_cost_aware_architecture',a.output,a.validation,a.qa_tools)
    fig,ax=plt.subplots(figsize=(7.2,3.6));fig.subplots_adjust(left=.02,right=.98,bottom=.03,top=.97)
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ax.text(.5,.95,'给定配额：示意 N[0]=2，N[1]=1，其余节点为0',ha='center',fontsize=9)
    ax.text(.16,.82,'请求身份',ha='center',fontsize=8.5)
    ax.text(.76,.82,'节点容量槽',ha='center',fontsize=8.5)
    for y,t in ((.68,'请求 a'),(.48,'请求 b'),(.28,'请求 c')):box(ax,.16,y,.20,.11,t)
    for y,t in ((.68,'UAV 0 · 槽1'),(.48,'UAV 0 · 槽2'),(.28,'UAV 1 · 槽1')):box(ax,.76,y,.27,.11,t)
    # Only selected illustrative matching edges; illegal edges are omitted.
    for y in (.68,.48,.28):arrow(ax,(.267,y),(.617,y))
    ax.text(.44,.745,'固定配额后确定边成本',ha='center',fontsize=8)
    ax.text(.44,.565,'1 - 混合成功分数',ha='center',fontsize=8)
    ax.text(.44,.365,'＋ 1e-9 × 归一化预测能耗',ha='center',fontsize=8)
    ax.text(.50,.09,'每请求一槽；每槽一请求；只使用允许边；求解后再做预测能耗硬筛选',ha='center',fontsize=8)
    save(fig,ax,'ch5_cost_aware_slots',a.output,a.validation,a.qa_tools)


if __name__=='__main__':main()
