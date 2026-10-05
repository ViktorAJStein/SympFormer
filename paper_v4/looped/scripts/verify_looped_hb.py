"""Independent E063 weight-sharing, causal/state and trainer-resume checks."""
import argparse
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_looped_hb import Protocol, make_model, run
from scripts.looped_hb_model import METHODS, LAYOUTS
from train import build_optimizer, configure_precision
import normalized_hb


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def formulas(device):
    records=[];configure_precision('float32',device)
    ids=torch.tensor([[1,3,5,2,4,6],[2,5,1,3,6,4]],device=device)
    for method in METHODS:
        c=Protocol(method=method,width=8,heads=2,context=6,vocab=19,batch=2,accumulation=1,updates=4,eval_interval=1,eval_batches=2)
        tied=make_model(c).to(device=device,dtype=torch.float64)
        untied=make_model(replace(c,layout='untied')).to(device=device,dtype=torch.float64)
        tied.eval();untied.eval()
        assert len({id(b) for b in tied.net.blocks})==2
        assert len({id(b) for b in untied.net.blocks})==8
        a=tied(ids,ids);b=untied(ids,ids)
        assert torch.equal(a[0],b[0]),method
        a[1].backward();b[1].backward()
        gradient_max=0.
        for i in range(2):
            for name,p in tied.net.blocks[i].named_parameters():
                copies=[dict(untied.net.blocks[j].named_parameters())[name] for j in range(i,8,2)]
                assert p.grad is not None and all(q.grad is not None for q in copies),(method,name)
                expected=sum(q.grad for q in copies)
                gradient_max=max(gradient_max,float((p.grad-expected).abs().max()))
                torch.testing.assert_close(p.grad,expected,atol=1e-10,rtol=1e-9)
        for name,p in tied.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(),(method,name)
            if not name.startswith('net.blocks.'):
                torch.testing.assert_close(p.grad,dict(untied.named_parameters())[name].grad,atol=1e-10,rtol=1e-9)
        opt=build_optimizer(tied,.0006,optimizer_name='adamw',scalar_lr_mult=10)
        params=[p for g in opt.param_groups for p in g['params']]
        assert len(params)==len({id(p) for p in params})==len(list(tied.parameters()))
        # Independent old full-model forward including velocity initialization.
        legacy=copy.deepcopy(tied.net)
        legacy.blocks=nn.ModuleList([legacy.blocks[j%2] for j in range(8)])
        torch.testing.assert_close(legacy(ids)[0],a[0],atol=0,rtol=0)
        leakage=0.
        with torch.no_grad():
            for loops in (1,2,4,8,16):
                out=tied(ids,loops=loops)[0]
                assert torch.isfinite(out).all()
                count=sum(p.numel() for p in tied.parameters())
                for cut in range(1,ids.shape[1]):
                    other=ids.clone();other[:,cut:]=(other[:,cut:]+7)%19
                    changed=tied(other,loops=loops)[0]
                    error=float((out[:,:cut]-changed[:,:cut]).abs().max())
                    leakage=max(leakage,error)
                    assert error<=1e-10,(method,loops,cut,error)
                measured=tied(ids,loops=loops,capture=True)[0]
                assert torch.equal(out,measured)
                assert len(tied.last_trace)==2*loops and [r['layer'] for r in tied.last_trace]==list(range(2*loops))
                assert sum(p.numel() for p in tied.parameters())==count
                if method!='B':
                    for left,right in zip(tied.last_trace,tied.last_trace[1:]):
                        assert left['next_velocity_norm']==right['velocity_norm']
                        assert left['t_end']==right['t_start']
            untied(ids,capture=True)
            assert len(untied.last_trace)==8
        for model,loops in ((untied,2),(tied,3)):
            try:model(ids,loops=loops)
            except ValueError:pass
            else:raise AssertionError('Bad inference depth accepted')
        tied.train()
        try:tied(ids,loops=8)
        except ValueError:pass
        else:raise AssertionError('Variable training depth accepted')
        records.append(dict(method=method,gradient_sum_max_error=gradient_max,suffix_error=leakage,
                            initial_logits='bitwise equal',legacy_forward='bitwise equal',loop_counts=5))
    c=Protocol(method='H0',width=8,heads=2,context=6,vocab=19)
    off=make_model(c).to(device=device,dtype=torch.float64)
    on=make_model(replace(c,method='R2')).to(device=device,dtype=torch.float64)
    assert equal(off.state_dict(),on.state_dict())
    original=normalized_hb.attention_connection
    def zero(*args,**kwargs):
        force,connection,proposal=original(*args,**kwargs)
        return force,torch.zeros_like(connection),proposal
    with patch.object(normalized_hb,'attention_connection',zero):
        a=off(ids,ids);b=on(ids,ids)
        assert torch.equal(a[0],b[0]);a[1].backward();b[1].backward()
    for (n,p),(m,q) in zip(off.named_parameters(),on.named_parameters()):
        assert n==m
        torch.testing.assert_close(p.grad,q.grad,atol=1e-10,rtol=1e-9)
    with torch.no_grad():on.net.blocks[0].beta1.raw.fill_(100.)
    try:on.check_coefficients()
    except FloatingPointError:pass
    else:raise AssertionError('Saturated beta accepted')
    return records


def resumes(out,device):
    fixture=out/'fixture';fixture.mkdir()
    rng=np.random.default_rng(97019)
    for split in ('train','val'):
        rng.integers(0,31,size=4096,dtype=np.uint16).tofile(fixture/f'tinystories_{split}.bin')
    records=[]
    for method in METHODS:
        for layout in LAYOUTS:
            c=Protocol(method=method,layout=layout,width=8,heads=2,context=6,vocab=31,
                       batch=2,accumulation=2,updates=4,eval_interval=1,eval_batches=2,seed=97019)
            tag=method+'_'+layout
            run(c,fixture,out/(tag+'_full'),device,inference=True)
            run(c,fixture,out/(tag+'_half'),device,stop_after=2)
            run(c,fixture,out/(tag+'_resume'),device,resume=out/(tag+'_half/checkpoint.pt'))
            a=torch.load(out/(tag+'_full/checkpoint.pt'),map_location='cpu',weights_only=False)
            b=torch.load(out/(tag+'_resume/checkpoint.pt'),map_location='cpu',weights_only=False)
            for key in a:
                if key == 'observer':
                    left, right = copy.deepcopy(a[key]), copy.deepcopy(b[key])
                    for ledger in (left, right):
                        for row in ledger['updates']:
                            assert row.pop('optimizer_seconds') > 0
                    assert equal(left,right),(tag,key)
                else:
                    assert equal(a[key],b[key]),(tag,key)
            rejections=0
            for key,value in (('seed',97018),('layout','untied' if layout=='tied' else 'tied'),
                              ('updates',5),('validation_seed',97017),('peak_lr',.001)):
                try:run(replace(c,**{key:value}),fixture,out/(tag+'_bad'),device,resume=out/(tag+'_half/checkpoint.pt'))
                except ValueError as e:
                    assert 'Incompatible E063 resume' in str(e);rejections+=1
                else:raise AssertionError((tag,key))
                assert not (out/(tag+'_bad')).exists()
            records.append(dict(method=method,layout=layout,resume='bitwise model/optimizer/RNG/iterator/trace/counters',rejections=rejections))
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--formulas-only',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);a.out.mkdir(parents=True,exist_ok=False)
    result=dict(status='PASS',device=a.device,formulas=formulas(a.device),
                resumes=[] if a.formulas_only else resumes(a.out,a.device),quality_result=False)
    (a.out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
