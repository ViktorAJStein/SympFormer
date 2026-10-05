"""Exact-shared-update log-y NLL and all-layer nominal damping plots.
No smoothing/interpolation. Input is trusted completed run directories, not checkpoints.
Fits r/t+c are descriptive, not physical LN damping or a learned Nesterov law.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def csvrows(p):
    with p.open() as f:return list(csv.DictReader(f))


def write(p,rows):
    if not rows:return
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def save(fig,p):
    fig.tight_layout();fig.savefig(str(p)+'.pdf');fig.savefig(str(p)+'.png',dpi=180);plt.close(fig)


def read_runs(paths):
    runs=[];blocks=set();seen=set()
    for folder in paths:
        m=json.loads((folder/'run_manifest.json').read_text());s=json.loads((folder/'summary.json').read_text())
        looped='protocol' in m
        if looped:
            c=m['protocol'];a=dict(dataset=c['dataset'],seed=c['seed'],n_layer=8,n_head=c['heads'],n_embd=c['width'],block_size=c['context'],optimizer='adamw',peak_lr=c['peak_lr'],scalar_lr_mult=c['scalar_lr_mult'],eval_interval=c['eval_interval'],eval_batches=c['eval_batches'])
            assert s['complete'] and s['update']==6103
            family='looped';method=c['method']+'_'+c['layout'];baseline=c['method']=='B'
        else:
            a=m['args'];assert s['final_step']==6103
            family='linear' if a.get('e062_method') else 'softmax'
            method=a.get('e062_method') or ('B' if a['arch']=='baseline' else a['hb_mode'])
            if a.get('hb_eta_learnable'):method+='_eta'
            baseline=method in ('B','B_L','B_S')
        assert s['tokens']==99991552 and math.isfinite(s['final_val'])
        key=(method,a['seed']);assert key not in seen;seen.add(key)
        blocks.add((family,a['dataset'],a['n_layer'],a['n_head'],a['n_embd'],a['block_size'],a['optimizer'],a['peak_lr'],a['scalar_lr_mult'],a['eval_interval'],a['eval_batches']))
        vals={}
        metrics=csvrows(folder/'metrics.csv')
        for r in metrics:
            if looped or r['val_loss']:
                u=int(r['update']) if looped else int(r['tokens_cum'])//int(r['tokens_step'])
                v=float(r['val_nll'] if looped else r['val_loss']);assert v>0 and math.isfinite(v) and u not in vals
                vals[u]=v
        assert max(vals)==6103
        dynamics={}
        path=folder/('dynamics.csv' if looped else 'hb_dynamics.csv')
        if not baseline:
            assert path.exists()
            for r in csvrows(path):
                u=int(r['update']) if looped else int(r['tokens_cum'])//16384
                layer=int(r['layer']);h=float(r.get('forward_step') or 1.);t=float(r.get('depth_time') or r.get('t_start') or layer+1)
                assert h>0 and t>0
                data={}
                for half in ('attn','mlp'):
                    b=float(r['beta_'+half]);g=float(r['gain_'+half]);assert 0<b<1 and g>0
                    data[half]=(b,g,-math.log(b)/h)
                k=(u,layer)
                value=dict(update=u,layer=layer,forward_step=h,internal_time=t,attn=data['attn'],mlp=data['mlp'])
                if k in dynamics:assert value==dynamics[k]
                dynamics[k]=value
        runs.append(dict(method=method,seed=a['seed'],nll=vals,dynamics=dynamics,path=str(folder),dynamics_file=path.name))
    assert len(blocks)==1,'Do not mix dataset/shape/context/protocol blocks'
    return runs


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--runs',type=Path,nargs='+',required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);runs=read_runs(a.runs)
    methods=sorted({r['method'] for r in runs});groups={m:[r for r in runs if r['method']==m] for m in methods}
    assert len({tuple(sorted(r['seed'] for r in rs)) for rs in groups.values()})==1,'Paired seedsets must match'
    common=sorted(set.intersection(*(set(r['nll']) for r in runs)))
    assert common and common[-1]==6103
    fig,ax=plt.subplots(figsize=(7,4));plotted=[];raw=[]
    for method,rs in groups.items():
        values=np.array([[r['nll'][u] for u in common] for r in rs]);mean=values.mean(0);lo=values.min(0);hi=values.max(0)
        ax.plot(common,mean,label=method);ax.fill_between(common,lo,hi,alpha=.16)
        for j,u in enumerate(common):
            plotted.append(dict(method=method,update=u,mean=mean[j],seed_min=lo[j],seed_max=hi[j],n=len(rs)))
            raw.extend(dict(method=method,seed=r['seed'],update=u,nll=r['nll'][u],origin=r['path']+'/metrics.csv') for r in rs)
    ax.set_yscale('log');ax.set_xlabel('Completed optimizer updates');ax.set_ylabel('Validation NLL (nats/token; log scale)');ax.legend();save(fig,a.out/'nll')
    write(a.out/'nll_plotted.csv',plotted);write(a.out/'nll_origins.csv',raw)
    coeff=[];fits=[];notapp=[]
    for method,rs in groups.items():
        if not rs[0]['dynamics']:notapp.append(dict(method=method,damping='not applicable'));continue
        shared=sorted(set.intersection(*(set(r['dynamics']) for r in rs)))
        times=sorted({u for u,l in shared});layers=sorted({l for u,l in shared})
        assert shared==[(u,l) for u in times for l in layers]
        fig,axes=plt.subplots(2,3,figsize=(12,6))
        fitted={}
        for half in ('attn','mlp'):
            fitted[half]=[]
            for r in rs:
                bytime={}
                for u in times:
                    records=[r['dynamics'][u,l] for l in layers]
                    design=np.array([[1/v['internal_time'],1.] for v in records]);rates=np.array([v[half][2] for v in records]);rc=np.linalg.lstsq(design,rates,rcond=None)[0]
                    fits.append(dict(method=method,seed=r['seed'],update=u,half=half,fitted_r=rc[0],fitted_constant=rc[1],rmse=float(np.sqrt(np.mean((design@rc-rates)**2))),scope='posthoc nominal rate; not physical/learned damping law'))
                    bytime[u]=rc
                fitted[half].append(bytime)
        for row,half in enumerate(('attn','mlp')):
            for layer in layers:
                values=np.array([[r['dynamics'][u,layer][half] for u in times] for r in rs]);color=None
                for j,label in enumerate(('beta','force gain','nominal -log(beta)/h')):
                    mu=values[:,:,j].mean(0);line=axes[row,j].plot(times,mu,label=f'layer {layer}')[0];color=line.get_color()
                    axes[row,j].fill_between(times,values[:,:,j].min(0),values[:,:,j].max(0),alpha=.12)
                    axes[row,j].set_xlabel('Completed optimizer updates');axes[row,j].set_ylabel(half+' '+label)
                predicted=np.array([[f[u][0]/r['dynamics'][u,layer]['internal_time']+f[u][1] for u in times] for f,r in zip(fitted[half],rs)])
                reference=np.array([[3/r['dynamics'][u,layer]['internal_time'] for u in times] for r in rs]).mean(0)
                axes[row,2].plot(times,predicted.mean(0),'--',color=color,alpha=.65,label=f'layer {layer} fitted r/t+c')
                axes[row,2].plot(times,reference,color=color,linestyle=':',alpha=.5,label=f'layer {layer} 3/t')
                for r in rs:
                    for u in times:
                        v=r['dynamics'][u,layer];b,g,rate=v[half]
                        coeff.append(dict(method=method,seed=r['seed'],update=u,layer=layer,half=half,beta=b,force_gain=g,forward_step=v['forward_step'],internal_time=v['internal_time'],nominal_rate=rate,nesterov_reference=3/v['internal_time'],origin=r['path']+'/'+r['dynamics_file']))
            for ax in axes[row]:ax.legend(fontsize=6)
        save(fig,a.out/(method+'_damping'))
        fig,axes=plt.subplots(2,2,figsize=(9,6))
        for row,half in enumerate(('attn','mlp')):
            for col,key in enumerate(('fitted_r','fitted_constant')):
                values=np.array([[next(f[key] for f in fits if f['method']==method and f['seed']==r['seed'] and f['half']==half and f['update']==u) for u in times] for r in rs])
                axes[row,col].plot(times,values.mean(0));axes[row,col].fill_between(times,values.min(0),values.max(0),alpha=.2)
                axes[row,col].set_xlabel('Completed optimizer updates');axes[row,col].set_ylabel(half+' '+key+' (posthoc)')
        save(fig,a.out/(method+'_nominal_fits'))
    write(a.out/'damping_origins.csv',coeff);write(a.out/'nominal_fit_origins.csv',fits)
    (a.out/'not_applicable.json').write_text(json.dumps(notapp,indent=2)+'\n')
    print('PASS exactshared NLL log-y; raworigins; all-layer coefficients/forwardsteps/nominal rates/reference andfittedr,c; no smoothing')


if __name__=='__main__':main()
