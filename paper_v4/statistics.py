"""Recompute the paper's primary summaries and complete linear Holm20 family.
Requires scipy: uv run --with scipy python paper_v4/statistics.py --help
Use trusted, independently audited completed directories. This is not a replacement
for scheduler/source/checkpoint provenance audit, and never selects seeds/endpoints.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats


def paired(values):
    v=np.asarray(values,dtype=float);assert len(v) in (3,5) and np.isfinite(v).all()
    mean=float(v.mean());sem=float(stats.sem(v))
    if sem==0:ci=(mean,mean);p=1. if mean==0 else 0.
    else:ci=stats.t.interval(.95,len(v)-1,loc=mean,scale=sem);p=float(stats.ttest_1samp(v,0).pvalue)
    return dict(mean=mean,ci_low=float(ci[0]),ci_high=float(ci[1]),p=p,wins=int((v<0).sum()),n=len(v),differences=v.tolist())


def holm(ps):
    order=np.argsort(ps);adj=np.minimum(1,np.maximum.accumulate(np.asarray(ps)[order]*np.arange(len(ps),0,-1)))
    out=np.empty(len(ps));out[order]=adj;return out.tolist()


def analyze(paths,family):
    linear=family=='linear';methods=('B_S','Y_S','B_L','Y_L','H0_L','H1_L','R2_L') if linear else ('B','Y','H0','H1','R2')
    expected={(d,m,s) for d,start in (('tinystories',95020 if linear else 70020),('openwebtext',95020 if linear else 71020)) for m in methods for s in range(start,start+(3 if linear else 5))}
    rows={};data={}
    for path in paths:
        s=json.loads((path/'summary.json').read_text());manifest=json.loads((path/'run_manifest.json').read_text());a=manifest['args']
        method=a.get('e062_method') if linear else ('B' if a['arch']=='baseline' else a['hb_mode']);key=(a['dataset'],method,a['seed'])
        assert key in expected and key not in rows,'Unexpected/duplicate seed, method or dataset'
        assert s['tokens']==99991552 and s['final_step']==6103 and math.isfinite(s['final_val']) and s['final_val']>0
        assert (a['n_layer'],a['n_head'],a['n_embd'],a['block_size'])==(4,4,64,512 if linear else 128)
        assert a['amp_dtype']=='float32' and a['optimizer']=='adamw' and a['peak_lr']==.0006 and a['scalar_lr_mult']==10
        assert a['batch_size']*a['grad_accum_steps']*a['block_size']==16384
        assert a['eval_batches']*a['batch_size']*a['block_size']==32768
        if linear:assert a['e062_validation_seed']==95019 and a['e062_chunk']==512
        count=(3449408 if method in ('B_S','B_L') else 6701912 if method in ('Y_S','Y_L') else 6701908) if linear else (3424832 if method=='B' else 6652760 if method=='Y' else 6652756)
        assert s['trainable_parameters']==count
        rows[key]=float(s['final_val'])
    assert set(rows)==expected,'All50primarysoftmax or all42linear runs required before inference'
    summaries=[];comparisons=[]
    for d in ('tinystories','openwebtext'):
        seeds=sorted(s for dataset,m,s in expected if dataset==d and m==methods[0])
        for method in methods:
            v=np.array([rows[d,method,s] for s in seeds]);summaries.append(dict(dataset=d,method=method,mean=float(v.mean()),sample_sd=float(v.std(ddof=1)),n=len(v)))
        pairs=[(m,ref) for m in ('H1_L','R2_L') for ref in ('H0_L','B_L','Y_L','B_S','Y_S')] if linear else [('R2','Y'),('R2','H0')]
        for candidate,reference in pairs:
            comparisons.append(dict(dataset=d,candidate=candidate,reference=reference,**paired([rows[d,candidate,s]-rows[d,reference,s] for s in seeds])))
    if linear:
        assert len(comparisons)==20
        for r,p in zip(comparisons,holm([r['p'] for r in comparisons])):r.update(p_holm=p,family='Holm20',family_size=20)
    else:
        assert len(comparisons)==4
        for r in comparisons:r.update(family='nominal paired tests as displayed; not pooled with diagnostic families',p_holm=None)
    return dict(summaries=summaries,contrasts=comparisons,scope='Recomputation of trusted100M endpoints; not a new strictcluster audit or headline claim',complete_endpoint_set=True)


def analyze_looped(paths):
    methods=('B','Y','H0','R2');layouts=('tied','untied');datasets=('tinystories','openwebtext');seeds=(97020,97021,97022)
    expected={(d,l,m,s) for d in datasets for l in layouts for m in methods for s in seeds};rows={};summaries=[];comparisons=[]
    params={('B','tied'):14583040,('B','untied'):19304704,('Y','tied'):27592460,('Y','untied'):32315696}
    for method in ('H0','R2'):params[method,'tied']=27592458;params[method,'untied']=32315688
    for path in paths:
        s=json.loads((path/'summary.json').read_text());m=json.loads((path/'run_manifest.json').read_text());c=m['protocol'];key=(c['dataset'],c['layout'],c['method'],c['seed'])
        assert key in expected and key not in rows
        assert (c['width'],c['heads'],c['context'],c['vocab'])==(256,8,512,50304)
        assert c['updates']==6103 and c['batch']*c['accumulation']*c['context']==16384 and c['validation_seed']==97018
        assert c['eval_batches']*c['batch']*c['context']==32768 and c['scalar_lr_mult']==10 and c['peak_lr']==.0006
        assert s['complete'] and s['update']==6103 and s['tokens']==99991552
        assert s['parameters']==params[c['method'],c['layout']] and math.isfinite(s['final_val']) and s['final_val']>0
        rows[key]=s['final_val']
    assert set(rows)==expected,'All48looped/untied endpoints required'
    for dataset in datasets:
        for layout in layouts:
            for method in methods:
                v=np.array([rows[dataset,layout,method,s] for s in seeds]);summaries.append(dict(dataset=dataset,layout=layout,method=method,mean=float(v.mean()),sample_sd=float(v.std(ddof=1)),n=3))
            for control in ('H0','B','Y'):
                comparisons.append(dict(dataset=dataset,layout=layout,comparison='R2-'+control,**paired([rows[dataset,layout,'R2',s]-rows[dataset,layout,control,s] for s in seeds])))
        values=[(rows[dataset,'tied','R2',s]-rows[dataset,'tied','H0',s])-(rows[dataset,'untied','R2',s]-rows[dataset,'untied','H0',s]) for s in seeds]
        comparisons.append(dict(dataset=dataset,layout='interaction',comparison='(R2-H0)_tied-(R2-H0)_untied',**paired(values)))
    assert len(comparisons)==14
    for r,p in zip(comparisons,holm([r['p'] for r in comparisons])):r.update(p_holm=p,family='Holm14',family_size=14)
    return dict(summaries=summaries,contrasts=comparisons,complete_endpoint_set=True,scope='Trusted100Mlooping endpoints; no1B/tying-superiority or new strictcluster-audit claim')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--family',choices=('softmax_primary','linear','softmax_looped'),required=True);ap.add_argument('--runs',type=Path,nargs='+',required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();result=analyze_looped(a.runs) if a.family=='softmax_looped' else analyze(a.runs,a.family);a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('PASS completeendpoint set; nominal CIs; originaltestfamily preserved')


if __name__=='__main__':main()
