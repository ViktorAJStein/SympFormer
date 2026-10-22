"""Aggregate complete paper studies, preserving all declared seeds and endpoints.
GPU allocation time is optional external scheduler evidence, NEVER trainer time.
"""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from run import presets, STUDIES
from sympformer.training import evaluation_updates


def key(c):
    return c['dataset'], c['method'], c['seed'], c.get('core')


def collect(study, root):
    expected = {key(c): c for c in presets(study)}
    observed = {}
    for path in Path(root).glob('*/summary.json'):
        summary = json.loads(path.read_text())
        c = summary['config']
        if c['study'] != study:
            raise ValueError('Mixed studies')
        k = key(c)
        if k not in expected or c != expected[k] or k in observed:
            raise ValueError('Unknown/changed/duplicate scientific cell')
        manifest = json.loads((path.parent/'run_manifest.json').read_text())
        if not summary['complete'] or not summary['paper_protocol'] or not manifest['paper_protocol']:
            raise ValueError('Incomplete or smoke-test result')
        if summary['updates'] != 6103 or summary['tokens'] != 99991552:
            raise ValueError('Different training budget')
        if not math.isfinite(summary['final_nll']) or summary['final_nll'] <= 0:
            raise ValueError('Invalid endpoint')
        if (path.parent/'failure.json').exists() or manifest['config'] != c:
            raise ValueError('Failed/mismatched run')
        with (path.parent/'metrics.csv').open() as f:
            metrics = list(csv.DictReader(f))
        if [int(v['update']) for v in metrics] != evaluation_updates(c):
            raise ValueError('Incomplete/misaligned fixed evaluations')
        if float(metrics[-1]['val_nll']) != summary['final_nll']:
            raise ValueError('Endpoint does not match final evaluation')
        for v in metrics:
            if int(v['tokens']) != int(v['update'])*16384 or not math.isfinite(float(v['val_nll'])) or float(v['val_nll']) <= 0:
                raise ValueError('Invalid metric/work record')
        with (path.parent/'damping.csv').open() as f:
            damping = list(csv.DictReader(f))
        if [(int(v['update']), int(v['layer'])) for v in damping] != [(u,l) for u in evaluation_updates(c) for l in range(c['layers'])]:
            raise ValueError('Incomplete logical-layer history')
        if c['method'] != 'B':
            for v in damping:
                for half in ('attn','mlp'):
                    beta = float(v['beta_'+half])
                    if not 0 < beta < 1 or not math.isclose(float(v['nominal_rate_'+half]), -math.log(beta), rel_tol=1e-12, abs_tol=1e-14):
                        raise ValueError('Invalid damping record')
        observed[k] = dict(path=path.parent, config=c, summary=summary, manifest=manifest)
    if set(observed) != set(expected):
        raise ValueError(f'Require all {len(expected)} declared cells, found {len(observed)}')
    for dataset in ('tinystories', 'openwebtext'):
        block = [v for k, v in observed.items() if k[0] == dataset]
        first = block[0]['manifest']
        for v in block:
            for name in ('data_identity', 'source_hashes', 'precision', 'python', 'torch', 'gpu'):
                if v['manifest'][name] != first[name]:
                    raise ValueError('Mixed corpus/source/software/hardware cohort: '+name)
    return observed


def aggregate(study, root):
    runs = collect(study, root)
    grouped = defaultdict(list)
    for r in runs.values():
        grouped[r['config']['dataset'], r['config']['method'], r['config'].get('layout', 'untied')].append(r)
    rows = []
    for (dataset, method, layout), block in sorted(grouped.items()):
        vals = [r['summary']['final_nll'] for r in block]
        parameters = {r['summary']['parameters'] for r in block}
        if len(parameters) != 1:
            raise ValueError('Different parameter counts within arm')
        allocations = []
        for r in block:
            path = r['path']/'allocation.json'
            if not path.exists():
                continue
            a = json.loads(path.read_text())
            if a['gpu_count'] != 1 or not math.isfinite(a['allocated_seconds']) or a['allocated_seconds'] <= 0:
                raise ValueError('Bad scheduler allocation record')
            if a['gpu'] != r['manifest']['gpu']:
                raise ValueError('Allocation hardware mismatch')
            allocations.append(a['allocated_seconds']/3600)
        mem = [r['summary']['peak_allocated_mib'] for r in block]
        rows.append(dict(study=study, dataset=dataset, method=method, layout=layout, n=len(vals),
            mean_nll=statistics.mean(vals), sample_sd=statistics.stdev(vals), parameters=parameters.pop(),
            mean_allocated_gpu_hours=statistics.mean(allocations) if len(allocations)==len(block) else None,
            max_peak_allocated_mib=max(mem) if all(m is not None for m in mem) else None,
            seed_values={str(r['config']['seed']):r['summary']['final_nll'] for r in block}))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', choices=STUDIES, required=True)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    rows = aggregate(a.study, a.runs)
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    fields = [k for k in rows[0] if k != 'seed_values']
    with (a.out/'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k:r[k] for k in fields} for r in rows)
    for r in rows:
        cost = 'unavailable' if r['mean_allocated_gpu_hours'] is None else f'{r["mean_allocated_gpu_hours"]:.4f}'
        print(f'{r["dataset"]:12s} {r["method"]:2s} {r["layout"]:6s} '
              f'{r["mean_nll"]:.4f} ({r["sample_sd"]:.4f}) GPUh={cost}')


if __name__ == '__main__':
    main()
