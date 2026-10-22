"""Exact shared-evaluation log-NLL, all-layer coefficients and r/t+c diagnostics."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import collect
from run import STUDIES


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def band(ax, times, values, label):
    v = np.asarray(values, dtype=float)
    m, s = v.mean(0), v.std(0, ddof=1)
    ax.plot(times, m, label=label)
    ax.fill_between(times, m-s, m+s, alpha=.15)
    return m, s


def save(fig, prefix):
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(str(prefix)+'.'+ext, dpi=150)
    plt.close(fig)


def plot_runs(study, runs, out):
    grouped = defaultdict(list)
    for r in collect(study, runs).values():
        c = r['config']
        grouped[c['dataset'], c['method'], c.get('layout', 'untied')].append(r)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    trajectory = []
    for dataset in ('tinystories', 'openwebtext'):
        fig, ax = plt.subplots(figsize=(8, 5))
        for (ds, method, layout), block in sorted(grouped.items()):
            if ds != dataset:
                continue
            tables = [{int(v['update']):float(v['val_nll']) for v in read(r['path']/'metrics.csv')} for r in block]
            times = sorted(set.intersection(*(set(t) for t in tables)))
            if not times or any(set(t) != set(times) for t in tables):
                raise ValueError('Missing/misaligned evaluation events; no interpolation')
            m, s = band(ax, times, [[t[u] for u in times] for t in tables], method+' '+layout)
            trajectory.extend(dict(dataset=ds, method=method, layout=layout, update=u, mean_nll=m[i], sample_sd=s[i]) for i,u in enumerate(times))
        ax.set_yscale('log')
        ax.set_xlabel('Completed optimizer updates (exact evaluations)')
        ax.set_ylabel('Validation NLL (log axis)')
        ax.set_title(study+' / '+dataset)
        ax.legend()
        save(fig, out/(dataset+'_nll'))
    write(out/'nll.csv', trajectory)
    raw, fits = [], []
    for (ds, method, layout), block in sorted(grouped.items()):
        layers = block[0]['config']['layers']
        tables = [{(int(v['update']), int(v['layer'])):v for v in read(r['path']/'damping.csv')} for r in block]
        if any(set(t) != set(tables[0]) for t in tables):
            raise ValueError('Incomplete logical-layer diagnostics')
        times = sorted({u for u,l in tables[0]})
        name = ds+'_'+method+'_'+layout
        fig, axes = plt.subplots(layers, 4, figsize=(16, 2*layers), sharex=True, squeeze=False)
        for l in range(layers):
            for col, field in enumerate(('beta', 'gain', 'nominal_rate', 'forward_step')):
                ax = axes[l,col]
                for half in ('attn', 'mlp'):
                    if method == 'B' and field != 'forward_step':
                        ax.text(.5,.5,'Not applicable: native B',ha='center',transform=ax.transAxes)
                        break
                    if field == 'forward_step' and half == 'mlp':
                        continue
                    k = field if field == 'forward_step' else field+'_'+half
                    band(ax, times, [[float(t[u,l][k]) for u in times] for t in tables], half)
                    if field == 'nominal_rate':
                        ax.axhline(3/(l+1),color='k',ls='--',label='3/t')
                if l==0:
                    ax.set_title(field)
                    if ax.get_legend_handles_labels()[0]:
                        ax.legend(fontsize=7)
                if col==0:
                    ax.set_ylabel('Logical layer '+str(l))
                if l==layers-1:
                    ax.set_xlabel('Optimizer update')
        save(fig, out/(name+'_layers'))
        for r,t in zip(block,tables):
            for (u,l),v in sorted(t.items()):
                raw.append(dict(dataset=ds, method=method, layout=layout, seed=r['config']['seed'], update=u, layer=l,
                    forward_step=float(v['forward_step']), internal_time=float(v['internal_time']),
                    beta_attn=v.get('beta_attn',''), beta_mlp=v.get('beta_mlp',''),
                    gain_attn=v.get('gain_attn',''), gain_mlp=v.get('gain_mlp',''),
                    nominal_rate_attn=v.get('nominal_rate_attn',''), nominal_rate_mlp=v.get('nominal_rate_mlp',''),
                    damping_applicable=method!='B'))
            if method=='B':
                continue
            design = np.column_stack((1/np.arange(1,layers+1), np.ones(layers)))
            for u in times:
                for half in ('attn','mlp'):
                    rates = np.array([float(t[u,l]['nominal_rate_'+half]) for l in range(layers)])
                    co = np.linalg.lstsq(design,rates,rcond=None)[0]
                    fits.append(dict(dataset=ds, method=method, layout=layout, seed=r['config']['seed'], update=u,
                        branch=half, r=co[0], c=co[1], rmse=np.sqrt(np.mean((design@co-rates)**2))))
        if method=='B':
            continue
        fig,axes = plt.subplots(2,3,figsize=(13,7))
        for row,half in enumerate(('attn','mlp')):
            for col,k in enumerate(('r','c','rmse')):
                values=[[next(f[k] for f in fits if f['dataset']==ds and f['method']==method and f['layout']==layout and f['seed']==r['config']['seed'] and f['update']==u and f['branch']==half) for u in times] for r in block]
                band(axes[row,col],times,values,half)
                if k in ('r','c'):
                    axes[row,col].axhline(3 if k=='r' else 0,color='k',ls='--')
                axes[row,col].set_title(k+' in descriptive r/t+c')
                axes[row,col].set_xlabel('Optimizer update')
        save(fig,out/(name+'_fit_evolution'))
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        t=np.arange(1,layers+1)
        for ax,half in zip(axes,('attn','mlp')):
            band(ax,t,[[float(v[times[-1],l]['nominal_rate_'+half]) for l in range(layers)] for v in tables],'final learned rate')
            co=[next(f for f in fits if f['dataset']==ds and f['method']==method and f['layout']==layout and f['seed']==r['config']['seed'] and f['update']==times[-1] and f['branch']==half) for r in block]
            band(ax,t,[v['r']/t+v['c'] for v in co],'descriptive r/t+c')
            ax.plot(t,3/t,'k--',label='3/t')
            ax.set_xlabel('Internal forward time')
            ax.set_title(half)
            ax.legend(fontsize=8)
        save(fig,out/(name+'_final_law'))
    write(out/'damping_raw.csv',raw)
    if fits:
        write(out/'damping_fits.csv',fits)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',choices=STUDIES,required=True)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    plot_runs(a.study,a.runs,a.out)
