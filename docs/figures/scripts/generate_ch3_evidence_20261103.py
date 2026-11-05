#!/usr/bin/env python3
"""Render third-chapter evidence figures on the original experiment server.

This script only visualizes frozen, audited summaries. It does not train models
or recompute inferential estimates. The original/new cohorts remain separate.
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
from matplotlib import font_manager as fm
from matplotlib.text import Text
from fontTools.ttLib import TTFont
import numpy as np
from audit_panel_alignment import require_matplotlib_panel_alignment

BLUE = '#3D6FB6'
GRAY = '#717784'
TEXT = '#22252A'
GRID = '#E5E7EB'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def initialize_font(path):
    fm.fontManager.addfont(str(path))
    name = fm.FontProperties(fname=str(path)).get_name()
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': [name, 'DejaVu Serif'],
        'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
        'axes.unicode_minus': False, 'axes.spines.right': False,
        'axes.spines.top': False, 'axes.linewidth': 0.6,
        'axes.edgecolor': TEXT, 'axes.labelcolor': TEXT,
        'text.color': TEXT, 'xtick.color': TEXT, 'ytick.color': TEXT,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'mathtext.fontset': 'dejavuserif', 'savefig.dpi': 600,
    })
    return name, set(TTFont(path).getBestCmap())


def paired_forest(ax, records, labels, xlim, ticks):
    y = np.arange(len(records))
    for j, row in enumerate(records):
        mean = row['mean_difference'] * 100
        low, high = np.array(row['t_ci95']) * 100
        ax.errorbar(mean, j, xerr=[[mean-low], [high-mean]], fmt='o',
                    color=BLUE, markersize=4, capsize=2, elinewidth=1.1,
                    markeredgecolor='white', markeredgewidth=0.4, zorder=3)
    ax.axvline(0, color=GRAY, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
    ax.set(yticks=y, yticklabels=labels, xlim=xlim, xticks=ticks,
           ylim=(len(records)-0.45, -0.55), xlabel='DSR差值 / 百分点')
    ax.grid(axis='x', color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6, pad=4)


def seed_panel(ax, record, xlabel, xlim, ticks):
    values = np.array(record['seed_mean_differences']) * 100
    ax.scatter(values, np.arange(10), s=19, color=BLUE,
               edgecolors='white', linewidths=0.4, zorder=3)
    mean = record['mean_difference'] * 100
    low, high = np.array(record['t_ci95']) * 100
    ax.errorbar(mean, 10.7, xerr=[[mean-low], [high-mean]], fmt='D',
                color=BLUE, markersize=4, capsize=2, elinewidth=1.2, zorder=4)
    ax.axhline(10.15, color=GRID, linewidth=0.6)
    ax.axvline(0, color=GRAY, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
    ax.set(yticks=list(range(10))+[10.7],
           yticklabels=[str(k+1) for k in range(10)]+['均值'],
           ylim=(11.45, -0.65), xlim=xlim, xticks=ticks, xlabel=xlabel)
    ax.grid(axis='x', color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6, pad=4)


def export(fig, name, output, cmap):
    fig.canvas.draw()
    missing = sorted({c for item in fig.findobj(match=Text)
                      for c in item.get_text() if ord(c)>127 and
                      '\u4e00' <= c <= '\u9fff' and ord(c) not in cmap})
    if missing:
        raise ValueError('Font lacks Chinese glyphs: '+''.join(missing))
    require_matplotlib_panel_alignment(
        fig, json_out=output/(name+'.alignment.json'),
        overlay_svg=output/(name+'.alignment.svg'),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(output/(name+'.pdf'), facecolor='white')
    fig.savefig(output/(name+'.svg'), facecolor='white')
    fig.savefig(output/(name+'.png'), facecolor='white', dpi=600)
    plt.close(fig)


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def effect_row(name, record, n):
    return {'contrast':name, 'mean_pp':record['mean_difference']*100,
            't_low_pp':record['t_ci95'][0]*100,
            't_high_pp':record['t_ci95'][1]*100, 'n_training_seeds':n}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--font', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data=read_json(args.review)
    original=read_json(args.original)
    assert data['status']=='PASS' and data['completion']['status']=='COMPLETE'
    font_name, cmap=initialize_font(args.font)
    pairs=data['comparisons']
    fig, axes=plt.subplots(1, 2, figsize=(7.4, 3.45))
    fig.subplots_adjust(left=0.175, right=0.975, bottom=0.17, top=0.88, wspace=0.65)
    names=['primary','covered_architecture','attention_information','mlp_information']
    labels=['B-D：扩展观测','A-C：仅已覆盖','B-A：注意力','D-C：MLP']
    records=[pairs[k]['DSR'] for k in names]
    paired_forest(axes[0], records, labels, (-0.65,7.1), [0,2,4,6])
    axes[0].set_title('(a) 网络与需求信息对照', loc='left', pad=10)
    seed_panel(axes[1], pairs['primary']['DSR'], 'DSR差值 / 百分点',
               (-1.2,2.8), [-1,0,1,2])
    axes[1].set_title('(b) 主比较：扩展观测', loc='left', pad=10)
    export(fig, 'ch3_factorial_evidence_20261103', args.output, cmap)
    write_csv(args.output/'factorial_effects.csv',
              [effect_row(k,pairs[k]['DSR'],10) for k in names])
    write_csv(args.output/'factorial_cell_means.csv',
              [{'cell':k, **v} for k,v in data['means'].items()])
    fig, axes=plt.subplots(1, 2, figsize=(7.4,3.45))
    fig.subplots_adjust(left=0.12,right=0.975,bottom=0.23,top=0.88,wspace=0.60)
    old=original['dsr_identity_decomposition']
    new=pairs['attention_information']['algebraic_decomposition']
    components=[(old['admission_component']*100,old['conditional_success_component']*100),
                (new['admission_component']*100,new['conditional_component']*100)]
    ax=axes[0]
    for k,(admission,quality) in enumerate(components):
        for offset,value,color,label in [(-0.15,admission,BLUE,'接纳项'),
                                         (0.15,quality,GRAY,'条件完成项')]:
            ax.barh(k+offset,value,height=0.26,color=color,
                    label=label if k==0 else None,zorder=3)
            ax.text(value+(0.16 if value>=0 else -0.16),k+offset,
                    f'{value:+.2f}',ha='left' if value>=0 else 'right',
                    va='center',fontsize=9)
    ax.axvline(0,color=GRAY,linewidth=0.8)
    ax.set(yticks=[0,1],yticklabels=['原确认','新四格'],ylim=(1.65,-0.65),
           xlim=(-3.6,6.9),xticks=[-2,0,2,4,6],xlabel='DSR变动项 / 百分点')
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5,width=0.6,pad=4)
    ax.set_title('(a) 接纳与条件完成项',loc='left',pad=10)
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',
               bbox_to_anchor=(0.3,0.015),ncol=2,frameon=False,
               handlelength=1.1,columnspacing=1.2)
    seed_panel(axes[1],data['common_admitted_eta_change'],
               '完成率差值 / 百分点',(-2.8,9.5),[-2,0,2,4,6,8])
    axes[1].set_title('(b) 共同接纳请求',loc='left',pad=10)
    export(fig,'ch3_admission_diagnostic_20261103',args.output,cmap)
    write_csv(args.output/'admission_decomposition.csv',
              [{'cohort':cohort,'admission_pp':row[0],'conditional_pp':row[1]}
               for cohort,row in zip(['original_confirmation','new_factorial_B-A'],components)])
    seed_rows=[]
    for key,record in [('primary_B-D',pairs['primary']['DSR']),
                       ('common_admitted_eta_B-A',data['common_admitted_eta_change'])]:
        seed_rows.extend({'contrast':key,'seed_ordinal':k+1,'difference_pp':v*100}
                         for k,v in enumerate(record['seed_mean_differences']))
    write_csv(args.output/'paired_seed_differences.csv',seed_rows)
    fig, axes=plt.subplots(1,2,figsize=(7.4,3.45))
    fig.subplots_adjust(left=0.13,right=0.98,bottom=0.17,top=0.88,wspace=0.65)
    groups=[(['main','ue60','ue140','cpu06','cpu14','uniform'],
             ['基准','60 UE','140 UE','0.6倍算力','1.4倍算力','无热点']),
            (['radius100','radius230','delay1','delay5','drop20','drop50'],
             ['半径100 m','半径230 m','观测延后1','观测延后5','Drop 20%','Drop 50%'])]
    boundary_rows=[]
    for ax,(keys,labels),title in zip(axes,groups,['(a) 场景条件','(b) 观测条件']):
        records=[data['boundary'][k]['B_minus_A']['DSR'] for k in keys]
        paired_forest(ax,records,labels,(-13,12),[-10,-5,0,5,10])
        ax.set_title(title,loc='left',pad=10)
        boundary_rows.extend(effect_row(k,row,3) for k,row in zip(keys,records))
    export(fig,'ch3_boundary_evidence_20261103',args.output,cmap)
    write_csv(args.output/'boundary_effects.csv',boundary_rows)
    meta={'script_sha256':digest(Path(__file__)),
          'review_sha256':digest(args.review),'original_sha256':digest(args.original),
          'font_sha256':digest(args.font),'font_family':font_name,
          'source_hash':data['source_hash'],'backend':matplotlib.get_backend(),
          'matplotlib_version':matplotlib.__version__,
          'inference':'Frozen summary estimates; no re-estimation or cohort pooling',
          'execution_location':'original Linux experiment server',
          'primary_passed':pairs['primary']['DSR']['criterion_met'],
          'capacity_stage':'not triggered',
          'seed_identification':'ordinal follows frozen protocol order',
          'command':sys.argv}
    (args.output/'figure_provenance.json').write_text(
        json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'COMPLETE','figures':3,'font':font_name}))


if __name__=='__main__':
    main()
