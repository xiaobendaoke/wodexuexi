"""Frozen threshold grid on old measured snapshot labels; not online speedup.

Contract: show why observed feature associations did not yield a qualifying
threshold rule. One quantitative panel, all 120 pooled development rules.
"""
import argparse
from pathlib import Path
import sys

from chapter5_oracle_gate import read, csv_write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--oracle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--qa-tools', type=Path, required=True)
    a = p.parse_args()
    if sys.platform != 'linux':
        raise RuntimeError('Scientific rendering executes on server')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    sys.path.insert(0, str(a.qa_tools))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    grid = read(a.selection/'threshold_grid.json')
    oracle = next(r for r in read(a.oracle/'summary.json')['curves']
                  if r['full'] == 'native_B10' and not r['relaxed'] and r['allowed_loss_pp'] == .1)
    matplotlib.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
        'svg.fonttype': 'none', 'legend.frameon': False})
    a.output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(183/25.4, 105/25.4))
    fig.subplots_adjust(left=.14, right=.97, bottom=.20, top=.89)
    ax.add_patch(Rectangle((-.04, .53), .14, .27, facecolor='#E5EEE9', edgecolor='none', zorder=0))
    source = []
    for mode, label, marker, color in [('C', 'C: Two features (100)', 's', '#B39B84'),
                                      ('A', 'A: Batch size (10)', '^', '#3676A8'),
                                      ('B', 'B: G predicted quality (10)', 'o', '#8C7BA8')]:
        rows = grid[f'pooled_{mode}']
        for i, z in enumerate(rows):
            m = z['metrics']
            resources = all(z['screen'][k] for k in ('energy', 'energy_G', 'mbs'))
            ax.scatter(-m['dsr_difference_pp'], m['time_ratio'], s=22, marker=marker,
                       facecolors=color if resources else 'none', edgecolors=color,
                       linewidths=.6, alpha=.70, label=label if i == 0 else None, zorder=3)
            source.append(dict(mode=mode, n0=z['rule']['n0'], q0=z['rule']['q0'],
                               loss_pp=-m['dsr_difference_pp'], time_ratio=m['time_ratio'],
                               resources_pass=resources, energy_ratio=m['energy_ratio'],
                               energy_ratio_G=m['energy_ratio_G'], mbs_difference_pp=m['mbs_difference_pp']))
    ax.scatter(-oracle['metrics']['dsr_difference_pp'], oracle['objective_time_ratio'],
               marker='*', s=75, color='#333333', label='Offline outcome oracle', zorder=4)
    ax.axvline(.1, color='#777777', ls=':', lw=.8)
    ax.axhline(.8, color='#777777', ls=':', lw=.8)
    ax.set(xlabel='Paid DSR loss relative to native full (pp)', ylabel='Full-call time ratio',
           xlim=(-.04, .65), ylim=(.53, 1.03), title='Frozen threshold rules: development proxy')
    ax.legend(loc='upper right', fontsize=7.5)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=a.output/'entry_gate_grid.alignment.json',
        overlay_svg=a.output/'entry_gate_grid.alignment.svg', strict=True)
    fig.savefig(a.output/'entry_gate_grid.pdf', facecolor='white')
    fig.savefig(a.output/'entry_gate_grid.svg', facecolor='white')
    fig.savefig(a.output/'entry_gate_grid.png', dpi=600, facecolor='white')
    plt.close(fig)
    csv_write(a.output/'entry_gate_grid_source.csv', source)
    (a.output/'caption.md').write_text(
        '固定阈值网格的开发代理取舍。展示全部五历史块上的A10、B10、C100个预定规则，重复点保留。'
        '质量先在块内合并成功数与生成数，时间先在块内取比，再平均五块。使用原C0两分支实测标签，'
        '不包含新门控开销，不是新门控完整费用或闭环结果。阴影表示原0.10个百分点和0.80时间比的联合区域；'
        '实心点通过两项能耗及新增MBS开发约束，空心点未通过。星号为同一固定标签上的历史离线结果oracle。'
        '各折阈值均回退Full，图中网格仅解释失败，不重新选择部署规则；没有总体置信带或非劣效结论。\n')


if __name__ == '__main__':
    main()
