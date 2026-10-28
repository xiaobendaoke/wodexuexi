"""Explain failed frozen threshold screens; never changes the selected rules."""
import argparse
from pathlib import Path
import sys

from chapter5_oracle_gate import read, write, csv_write, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if sys.platform != 'linux':
        raise RuntimeError('Scientific diagnosis executes on server')
    if a.output.exists():
        raise FileExistsError('Preserve diagnostics')
    a.output.mkdir(parents=True)
    grid = read(a.selection/'threshold_grid.json')
    rows = []
    for key, values in grid.items():
        resources = [z for z in values if all(z['screen'][k] for k in ('energy', 'energy_G', 'mbs'))]
        fast = [z for z in resources if z['screen']['time']]
        quality_original = [z for z in resources if z['metrics']['dsr_difference_pp'] >= -.10
                            and z['screen']['block_quality']]
        best_quality = max(fast, key=lambda z: z['metrics']['dsr_difference_pp']) if fast else None
        fastest_safe = min(quality_original, key=lambda z: z['metrics']['time_ratio']) if quality_original else None
        rows.append(dict(fold_mode=key, tested=len(values), selected_count=sum(z['eligible'] for z in values),
            original_target_feasible=sum(z['metrics']['dsr_difference_pp'] >= -.10 and z['screen']['block_quality'] for z in fast),
            best_dsr_pp_at_20pct=best_quality['metrics']['dsr_difference_pp'] if best_quality else None,
            best_dsr_rule=str(best_quality['rule']) if best_quality else None,
            fastest_time_at_original_quality=fastest_safe['metrics']['time_ratio'] if fastest_safe else None,
            fastest_safe_rule=str(fastest_safe['rule']) if fastest_safe else None))
    csv_write(a.output/'screen_diagnostics.csv', rows)
    write(a.output/'summary.json', dict(status='DESCRIPTIVE_FROZEN_GRID_DIAGNOSIS', rows=rows,
        rules_unchanged=True, selection_sha256=sha(a.selection/'rules.json'),
        interpretation='No threshold is reselected; original target comparisons explain the stricter training screen'))
    print(__import__('json').dumps(rows), flush=True)


if __name__ == '__main__':
    main()
