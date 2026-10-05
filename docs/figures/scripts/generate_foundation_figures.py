"""Vector schematics for the thesis technology foundations (no empirical data)."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from paths import REPO_ROOT
from thesis_figure_style import setup_thesis_style
BUNDLED_QA = REPO_ROOT/'tools/figure_qa'
SKILL = BUNDLED_QA if BUNDLED_QA.is_dir() else Path.home()/'.codex/skills/nature-figure/scripts'
sys.path.insert(0,str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

OUT=REPO_ROOT/'docs/figures'
QA=REPO_ROOT/'output/thesis_writing_20261004/validation/figures'
BLUE,TEAL,GRAY='#3569AC','#65A69A','#7B818A'

def box(ax,x,y,w,h,text,color=BLUE):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008',
        facecolor=color+'16',edgecolor=color,lw=.9))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)

def arrow(ax,a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,lw=1,color=GRAY))

def render(number):
    fig,ax=plt.subplots(figsize=(7.2,3.9))
    fig.subplots_adjust(left=.025,right=.975,bottom=.07,top=.94)
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    if number==1:
        name='MEC三层架构示意图'
        box(ax,.31,.76,.38,.18,'中心服务／云\n程序与内容来源',GRAY)
        box(ax,.03,.40,.28,.19,'移动边缘节点\nUAV计算与缓存',BLUE)
        box(ax,.36,.40,.28,.19,'固定边缘节点\nMBS计算与回传',BLUE)
        box(ax,.69,.40,.28,.19,'其他边缘设备\n通用MEC协作层',TEAL)
        for x in [.17,.50,.83]:arrow(ax,(.50,.75),(x,.61))
        box(ax,.15,.04,.70,.19,'用户终端／传感设备\n产生输入数据与服务需求',TEAL)
        for x in [.17,.50,.83]:arrow(ax,(.50,.24),(x,.39))
        fig.text(.5,.013,'通用层次示意；本文实际三条服务路径见第三章',ha='center',fontsize=9)
    elif number==2:
        name='UAV辅助通信应用场景'
        box(ax,.03,.68,.28,.22,'灾后或临时热点\n快速形成空中接入',TEAL)
        box(ax,.36,.68,.28,.22,'巡检与感知需求\n数据进入计算链路',TEAL)
        box(ax,.69,.68,.28,.22,'移动用户与覆盖变化\n持续调整服务位置',TEAL)
        box(ax,.20,.27,.60,.22,'共同决策关系\n位置影响接纳与链路，卸载影响负载与完成时间',BLUE)
        for x in [.17,.50,.83]:arrow(ax,(x,.67),(.50,.50))
        box(ax,.10,.025,.80,.13,'本文评价：请求能否自然接纳，以及接纳后能否按时完成',GRAY)
        arrow(ax,(.50,.26),(.50,.17))
        fig.text(.5,.01,'应用关系示意，不表示实地部署验证',ha='center',fontsize=9)
    else:
        name='强化学习智能体环境交互框架'
        box(ax,.03,.34,.34,.29,'智能体／参数化策略\n根据观测选择动作',BLUE)
        box(ax,.63,.34,.34,.29,'环境／状态转移\n执行动作并产生反馈',TEAL)
        arrow(ax,(.38,.52),(.62,.52));ax.text(.5,.58,'动作 $a_t$',ha='center',fontsize=11)
        arrow(ax,(.79,.33),(.79,.13));arrow(ax,(.79,.13),(.20,.13));arrow(ax,(.20,.13),(.20,.33))
        ax.text(.5,.20,'下一观测 $o_{t+1}$、奖励 $r_t$、终止标记',ha='center',fontsize=10)
        box(ax,.25,.75,.50,.14,'训练：采样轨迹 → 价值估计与策略更新',GRAY)
        fig.text(.5,.018,'概念示意；奖励是训练信号，论文评价指标须另行统计',ha='center',fontsize=9)
    title=f'图2-{number}_{name}'
    require_matplotlib_panel_alignment(fig,axes=[ax],json_out=QA/f'{title}.alignment.json',strict=True)
    fig.savefig(OUT/f'{title}.pdf',facecolor='white',bbox_inches=None)
    fig.savefig(OUT/f'{title}.svg',facecolor='white',bbox_inches=None)
    fig.savefig(OUT/f'{title}.png',dpi=600,facecolor='white',bbox_inches=None)
    plt.close(fig)
    print(title)

if __name__=='__main__':
    setup_thesis_style(10)
    QA.mkdir(parents=True,exist_ok=True)
    for number in (1,2,3):render(number)
