"""Diagnostic figures for the finite-snapshot two-branch oracle.

Contract: expose selection headroom without claiming an implementable policy.
Each single-panel figure has one role: constrained headroom, or heterogeneity.
Python exports use editable text, all observations, and external captions.
"""
import argparse
import csv
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--qa-tools', type=Path, required=True)
    p.add_argument('--features', type=Path)
    a = p.parse_args()
    if sys.platform != 'linux':
        raise RuntimeError('Scientific rendering executes on server')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    sys.path.insert(0, str(a.qa_tools))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    data = json.loads((a.results/'summary.json').read_text())
    assert json.loads((a.results/'input_audit.json').read_text())['status'] == 'PASS'
    assert json.loads((a.results/'completion.json').read_text())['inputs_unchanged']
    with (a.results/'state_deltas.csv').open() as f:
        rows = list(csv.DictReader(f))
    out = a.results/'figures'
    out.mkdir(exist_ok=True)
    matplotlib.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
        'svg.fonttype': 'none', 'legend.frameon': False, 'axes.linewidth': .7})

    def export(fig, name):
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig, json_out=out/(name+'.alignment.json'),
            overlay_svg=out/(name+'.alignment.svg'), strict=True)
        fig.savefig(out/(name+'.pdf'), facecolor='white')
        fig.savefig(out/(name+'.svg'), facecolor='white')
        fig.savefig(out/(name+'.png'), dpi=600, facecolor='white')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(183/25.4, 105/25.4))
    fig.subplots_adjust(left=.19, right=.96, bottom=.19, top=.89)
    for full, name, color in (('native_B10', 'Native full', '#3676A8'), ('B10', 'Prefix B10 full', '#9B7760')):
        for relaxed, style in ((False, '-'), (True, '--')):
            curve = sorted([r for r in data['curves'] if r['full'] == full and r['relaxed'] == relaxed],
                           key=lambda r: r['allowed_loss_pp'])
            assert all(r['feasible_route'] for r in curve)
            assert all(r['optimal'] for r in curve if relaxed), 'The relaxed bound must be certified'
            ax.plot([r['allowed_loss_pp'] for r in curve], [r['objective_time_ratio'] for r in curve],
                    color=color, ls=style, lw=1.25 if not relaxed else .8,
                    label=name+(' (binary route)' if not relaxed else ' (relaxed bound)'))
            unfinished = [r for r in curve if not r['optimal']]
            if unfinished:
                ax.plot([r['allowed_loss_pp'] for r in unfinished], [r['objective_time_ratio'] for r in unfinished],
                        's', ms=3, mfc='white', mec=color, mew=.7)
    ax.axhline(.8, color='#777777', lw=.8, ls=':')
    ax.axvline(.1, color='#777777', lw=.8, ls=':')
    ax.set(xlabel='Allowed paid DSR loss (pp)', ylabel='Full-call time ratio',
           xlim=(0, .2), title='Offline routing headroom')
    ax.set_xticks([0, .05, .1, .15, .2])
    ax.legend(loc='best', fontsize=7)
    export(fig, 'oracle_tradeoff')

    fig, ax = plt.subplots(figsize=(183/25.4, 105/25.4))
    fig.subplots_adjust(left=.17, right=.96, bottom=.19, top=.89)
    vals = [float(r['delta_success']) for r in rows if r['full'] == 'native_B10']
    low, high = np.floor(min(vals))-.5, np.ceil(max(vals))+.5
    bins = np.arange(low, high+1., 1.)
    for source, label, color in (('G', 'G source', '#8497A5'), ('mixture_price', 'Analytical source', '#3676A8')):
        x = [float(r['delta_success']) for r in rows if r['full'] == 'native_B10' and r['source'] == source]
        ax.hist(x, bins=bins, histtype='step', lw=1.25, color=color, label=label)
    ax.axvline(0, color='#777777', ls=':', lw=.8)
    ax.set(xlabel='Paid successes: native full minus G', ylabel='Fixed states',
           title='Heterogeneity of extra computation value')
    ax.legend(fontsize=7)
    export(fig, 'oracle_gain_distribution')
    captions = (
        '图1为既定1200个开发固定状态上的离线二分支选择空间。横轴为允许的DSR损失，纵轴为满足质量和两项机队能耗约束时的最小完整时间比。'
        '实线为找到的逐状态二值可行路由，虚线为连续松弛下界；空心方块标记限时内未证最优的二值点。两参照分别求解。参考值为原0.10个百分点与0.80时间比。'
        '五历史块内先汇总请求计数，再平均块率或块比值；无总体置信带。门控费用为零，不是可部署或闭环性能。\n\n'
        '图2为原生完整方法相对G的计费成功数增量分布，两种源轨迹各600状态。每状态三次实测费用分别用于截止期判断后取均值，'
        '没有挑最快重复；分箱宽度为1个成功请求。全部1200状态保留，计时重复和状态数不增加独立n=5。\n')
    (out/'captions.md').write_text(captions)

    if a.features:
        with (a.features/'feature_bins.csv').open() as f:
            grouped = list(csv.DictReader(f))
        summary = json.loads((a.features/'summary.json').read_text())
        names = {'batch_size': 'Batch size', 'deadline_q25': 'Deadline lower quartile',
                 'G_pred_success_fraction': 'G predicted success fraction',
                 'G_margin_q25': 'G margin lower quartile',
                 'G_max_uav_load_fraction': 'G maximum UAV load fraction',
                 'mean_legal_options': 'Mean legal options'}
        fig, axes = plt.subplots(2, 3, figsize=(183/25.4, 123/25.4), sharey=True)
        fig.subplots_adjust(left=.10, right=.98, bottom=.15, top=.90, wspace=.48, hspace=.77)
        for ax, feature, letter in zip(axes.flat, summary['features'], 'abcdef'):
            subset = [r for r in grouped if r['feature'] == feature]
            bins = sorted({int(r['bin']) for r in subset})
            block_ids = sorted({int(r['block']) for r in subset})
            means = []
            for b in bins:
                values = [float(r['dsr_gain_pp']) for r in subset if int(r['bin']) == b and r['dsr_gain_pp']]
                means.append(float(np.mean(values)))
                offsets = np.linspace(-.10, .10, len(block_ids))
                for offset, block in zip(offsets, block_ids):
                    r = next(r for r in subset if int(r['bin']) == b and int(r['block']) == block)
                    if r['dsr_gain_pp']:
                        ax.plot(b+offset, float(r['dsr_gain_pp']), 'o', ms=2.2, color='#8497A5', alpha=.65)
            ax.plot(bins, means, 'o-', ms=3.2, lw=1.05, color='#3676A8')
            ax.axhline(0, lw=.7, color='#777777', ls=':')
            ax.set_xticks(bins)
            ax.set_xlim(.65, max(bins)+.35)
            ax.set_title(names[feature], fontsize=7.3, pad=12)
            ax.set_xlabel('Feature bin (low to high)', fontsize=7, labelpad=6)
            ax.tick_params(labelsize=7)
            ax.annotate(letter, xy=(0, 1), xycoords='axes fraction', xytext=(-20, 12),
                        textcoords='offset points', fontsize=8, fontweight='bold')
        axes[0, 0].set_ylabel('Paid DSR gain (pp)', fontsize=7.5)
        axes[1, 0].set_ylabel('Paid DSR gain (pp)', fontsize=7.5)
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig, json_out=out/'public_feature_bins.alignment.json',
            overlay_svg=out/'public_feature_bins.alignment.svg', require_panel_labels=True, strict=True)
        fig.savefig(out/'public_feature_bins.pdf', facecolor='white')
        fig.savefig(out/'public_feature_bins.svg', facecolor='white')
        fig.savefig(out/'public_feature_bins.png', dpi=600, facecolor='white')
        plt.close(fig)
        with (out/'captions.md').open('a') as f:
            f.write('\n图3为六项预先列定公开特征的分组关联。灰点为五个历史块，蓝点线为可用块差的均值；'
                    '块内合并加权成功数和实际生成数后计算原生完整方法减G的DSR差。每特征按全开发状态分位点分组，'
                    '重复分界合并，全部1200状态保留。没有总体置信带或显著性检验，分组不是部署阈值，关联不证明留出预测能力。\n')


if __name__ == '__main__':
    main()
