import argparse
import csv
from collections import defaultdict
from math import ceil, floor, isclose
from pathlib import Path
from statistics import mean

from . import run, tilts


def read(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def screening(rows):
    for r in rows:
        if int(r['retained_per_sleeve']) != run.CONFIG['shortlist_per_sleeve']:
            continue
        full_lower, full, short_lower, short = (
            float(r[k]) for k in ('full_opt_lower_bound', 'full_opt_cost',
                                   'short_opt_lower_bound', 'short_opt_cost')
        )
        if not (0 <= full - full_lower <= 2e-6 and 0 <= short - short_lower <= 2e-6):
            raise RuntimeError(f"Uncertified screening comparison: {r['instance_seed']}")
        yield {
            'case': r['family'], 'seed': r['instance_seed'],
            'full_cost': f'{full:.6f}', 'shortlist_cost': f'{short:.6f}',
            'gap_lower': f'{floor(max(0., short_lower - full) * 1e6) / 1e6:.6f}',
            'gap_upper': f'{ceil(max(0., short - full_lower) * 1e6) / 1e6:.6f}',
        }


def sampler_table(rows, follow_up=False):
    groups = defaultdict(list)
    for r in rows:
        groups[r['method']].append(r)
    expected_runs = (len(tilts.P['instances']) * tilts.P['repeats'] if follow_up else
                     len(run.CONFIG['evaluation']) * run.CONFIG['proposal_repeats'])
    new_budget = tilts.P['additional_calls'] if follow_up else run.CONFIG['additional_calls']
    raw_cap = tilts.P['raw_proposal_cap'] if follow_up else run.CONFIG['proposal_cap']
    for method, group in groups.items():
        if len(group) != expected_runs:
            raise RuntimeError(f'Incomplete run count for {method}')
        for r in group:
            if not (int(r['initial_calls']) == run.CONFIG['initial_calls'] and
                    0 <= int(r['new_calls']) <= new_budget and
                    int(r['raw_examined']) <= int(r['raw_generated']) <= raw_cap):
                raise RuntimeError(f'Budget mismatch: {method}, seed {r["instance_seed"]}')
            if method.startswith('xy'):
                fits = (tilts.P['total_xy_evaluations'] if follow_up else
                        len(run.CONFIG['xy_gamma_grid']) * len(run.CONFIG['xy_beta_grid']))
                if (int(r['circuit_evaluations']), int(r['simulated_readout_draws'])) != (fits, raw_cap):
                    raise RuntimeError(f'Circuit budget mismatch: {method}')

        def avg(key, digits=3):
            return f'{mean(float(r[key]) for r in group):.{digits}f}'

        yield {
            'method': method, 'runs': len(group), 'mean_new_calls': avg('new_calls', 2),
            'mean_normalized_regret': avg('normalized_regret', 6),
            'near_best_hit_rate_5pct': avg('hit_near_opt'),
            'mean_profiles_5pct': avg('near_opt_exposure_bins'),
            'mean_profiles_15pct': avg('near_opt_exposure_bins_15pct'),
            **({} if follow_up else {'beats_shortlist_rate': avg('beats_shortlist')}),
        }


def check(actual, reference):
    a, b = read(actual), read(reference)
    if len(a) != len(b) or not a or list(a[0]) != list(b[0]):
        raise SystemExit(f'{actual.name}: table shape changed')
    for i, (got, want) in enumerate(zip(a, b), 2):
        for k, value in got.items():
            if value == want[k]:
                continue
            if k not in ('case', 'seed', 'method', 'runs') and isclose(
                float(value), float(want[k]), rel_tol=0, abs_tol=5e-6
            ):
                continue
            raise SystemExit(f'{actual.name}:{i}: {k} = {value}; expected {want[k]}')


def main():
    parser = argparse.ArgumentParser(description='Reproduce the synthetic ETF shortlist pilot.')
    parser.add_argument('--output', type=Path, default=Path('runs'))
    parser.add_argument('--check', action='store_true', help='Compare with the checked-in CSVs.')
    args = parser.parse_args()
    raw, out = args.output / 'raw', args.output / 'results'
    out.mkdir(parents=True, exist_ok=True)
    common = ['--output', str(raw), '--checkpoints', str(args.output / 'checkpoints')]
    run.main([*common, '--split', 'evaluation'])
    tilts.main(common)
    tables = {
        'screening.csv': list(screening(read(raw / 'screening.csv'))),
        'samplers.csv': list(sampler_table(read(raw / 'metrics.csv'))),
        'tilts.csv': list(sampler_table(read(raw / 'tilts.csv'), follow_up=True)),
    }
    for name, rows in tables.items():
        run.atomic(out / name, rows)
        if args.check:
            check(out / name, run.HERE.parent / 'results' / name)
    print(f'Results: {out}' + (' (matched the checked-in CSVs)' if args.check else ''))


if __name__ == '__main__':
    main()
