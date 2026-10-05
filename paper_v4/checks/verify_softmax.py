#!/usr/bin/env python3
"""E061 independent equations, gradients, route/parity and causal suffix tests."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

MODES = ('B','Y','H0','H1','R2','H1_LA','R2_LA')


def setup(source):
    sys.path[:0] = [str(source), str(source/'scripts')]
    import scripts.attention_lookahead_model as la
    from model import ModelConfig, GPTModel
    from train import initialize_learned_v0_tables, build_optimizer, configure_precision
    from verify_normalized_hb import reference_attention
    return la, ModelConfig, GPTModel, initialize_learned_v0_tables, build_optimizer, configure_precision, reference_attention


def make(source, mode, width=8, layers=2, context=6, vocab=19):
    la, Config, GPT, initialize, *_ = setup(source)
    cfg = Config(n_embd=width,n_layer=layers,n_head=4,block_size=context,vocab_size=vocab,
                 bias=False,dropout=0.,yurii_force_gain_floor=0. if mode == 'B' else 1e-6)
    if mode == 'B':
        return GPT(cfg)
    version = la.VERSION if mode.endswith('_LA') else 'e048_r2_dual_ln_v1' if mode == 'R2' else 'e046_identity_v1'
    m = la.NormalizedHBModel(cfg,la.NormalizedHBConfig(mode=mode,version=version))
    initialize(m,torch.initial_seed())
    return m


def compare(a, b, ids, forward=None, common_only=False):
    a.zero_grad(set_to_none=True); b.zero_grad(set_to_none=True)
    xa = a(ids)[0]; xb = b(ids)[0] if forward is None else forward(b,ids)
    weights = torch.linspace(.3,1.3,xa.numel(),device=xa.device,dtype=xa.dtype).reshape_as(xa)
    (xa*weights).sum().backward(); (xb*weights).sum().backward()
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    if not common_only:
        assert pa.keys() == pb.keys()
    differences = []
    for name in pa.keys() & pb.keys():
        assert pa[name].grad is not None and pb[name].grad is not None, name
        assert torch.isfinite(pa[name].grad).all() and torch.isfinite(pb[name].grad).all()
        differences.append(float((pa[name].grad-pb[name].grad).abs().max()))
    return float((xa-xb).detach().abs().max()), max(differences)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();source=a.source.resolve()
    la, Config, GPT, initialize, optimizer, precision, reference_attention = setup(source)
    torch.set_num_threads(2);precision('float32',a.device)
    spec=importlib.util.spec_from_file_location('e061_legacy_hb',source/'normalized_hb.py')
    legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)
    records=[]; parities=[]; nulls=[]; jacobians=[]
    def ref(m,ids):
        pos=torch.arange(ids.shape[1],device=ids.device)
        x=m.tok_emb(ids)+m.pos_emb(pos)[None];v=m.tok_v0_emb(ids)+m.pos_v0_emb(pos)[None]
        for b in m.blocks:
            z=b.ln_x_attn(x+b.mu1()*v)
            f,_=reference_attention(b.attn,z,v)
            W=b.beta1()*v+(b.gamma1()+b.force_gain_floor)*f
            _,C=reference_attention(b.attn,z,W if m.hb.mode == 'R2_LA' else v)
            va=b.ln_v(W-C)
            xa=x+(b.ln_v(W-.5*C) if m.hb.mode == 'R2_LA' else va)
            fm=b.mlp(b.ln_x_mlp(xa+b.mu2()*va))
            v=b.ln_v(b.beta2()*va+(b.gamma2()+b.force_gain_floor)*fm);x=xa+v
        return m.lm_head(m.ln_f(x))
    for dtype in (torch.float32,torch.float64):
        tol=3e-4 if dtype == torch.float32 else 1e-10
        ids=torch.tensor([[1,3,2,7,8,4],[4,5,6,2,1,3]],device=a.device)
        for mode in MODES:
            torch.manual_seed(95019);m=make(source,mode)
            rng=torch.get_rng_state().clone()
            if mode != 'B':
                torch.manual_seed(95019)
                old_mode='Y' if mode.endswith('_LA') else mode
                old=legacy.NormalizedHBModel(m.cfg,legacy.NormalizedHBConfig(mode=old_mode,
                      version='e048_r2_dual_ln_v1' if old_mode == 'R2' else 'e046_identity_v1'))
                initialize(old,95019)
                assert torch.equal(rng,torch.get_rng_state())
                assert m.state_dict().keys() == old.state_dict().keys()
                assert all(torch.equal(value,old.state_dict()[n]) for n,value in m.state_dict().items())
                if not mode.endswith('_LA'):
                    e,g=compare(m.to(device=a.device,dtype=dtype),old.to(device=a.device,dtype=dtype),ids)
                    assert e == g == 0,(mode,dtype,e,g)
                    parities.append(dict(mode=mode,dtype=str(dtype),forward=e,gradient=g))
            m=m.to(device=a.device,dtype=dtype)
            # Every candidate/control has causal prefix logits, including learned V0.
            original=m(ids)[0].detach()
            for cut in range(1,6):
                changed=ids.clone();changed[:,cut:]=(changed[:,cut:]+3)%19
                assert torch.equal(m(changed)[0][:,:cut].detach(),original[:,:cut])
                assert (m(ids[:,:cut])[0]-original[:,:cut]).abs().max() < tol
            if mode.endswith('_LA'):
                for perturbed in (False,True):
                    model=copy.deepcopy(m)
                    if perturbed:
                        with torch.no_grad():
                            for n,v in model.named_parameters():
                                if n.endswith('.raw'):v.add_(.15*torch.randn_like(v))
                    expected=copy.deepcopy(model)
                    e,g=compare(model,expected,ids,ref)
                    assert max(e,g)<tol,(mode,dtype,perturbed,e,g)
                    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
                    assert any(float(b.mu1.raw.grad.abs())>0 for b in model.blocks)
                    records.append(dict(mode=mode,dtype=str(dtype),perturbed=perturbed,forward=e,gradient=g))
                # Zero connection recovers the full Y formula and gradients.
                real=la.attention_connection
                def no_correction(*args,**kwargs):
                    values=real(*args,**kwargs)
                    return values[0],values[1]*0,*values[2:]
                torch.manual_seed(95019);new=make(source,mode).to(device=a.device,dtype=dtype)
                torch.manual_seed(95019);y=make(source,'Y').to(device=a.device,dtype=dtype)
                with patch.object(la,'attention_connection',no_correction):
                    e,g=compare(new,y,ids)
                assert max(e,g)<tol,(mode,'C0',e,g)
                nulls.append(dict(mode=mode,null='C0_full_Y',dtype=str(dtype),forward=e,gradient=g))
                # Mu=0 recovers the appropriate old H1/R2, not a different retraction.
                torch.manual_seed(95019);new=make(source,mode).to(device=a.device,dtype=dtype)
                torch.manual_seed(95019);off=make(source,mode[:-3]).to(device=a.device,dtype=dtype)
                for b in new.blocks:
                    b.mu1.forward=lambda b=b: b.mu1.raw*0
                e,g=compare(new,off,ids,common_only=True)
                assert max(e,g)<tol,(mode,'mu0',e,g)
                nulls.append(dict(mode=mode,null='mu0_historical',dtype=str(dtype),forward=e,gradient=g))
                for scale in (0.,1e-4,1.,100.):
                    x=torch.randn(1,4,8,device=a.device,dtype=dtype,requires_grad=True)
                    v=(scale*torch.randn_like(x)).requires_grad_()
                    out,_=m.blocks[0](x,v)
                    grads=torch.autograd.grad(out[:,:2].square().sum(),(x,v))
                    assert all(torch.isfinite(q).all() and torch.count_nonzero(q[:,2:]) == 0 for q in grads)
                    jacobians.append(dict(mode=mode,dtype=str(dtype),scale=scale,suffix_derivative=0))
                opt=optimizer(m,.0006,optimizer_name='adamw',scalar_lr_mult=10)
                groups={id(q):g for g in opt.param_groups for q in g['params']}
                for n,q in m.named_parameters():
                    assert abs(groups[id(q)]['lr']-(.006 if n.endswith('.raw') else .0006))<1e-12
                    assert groups[id(q)]['weight_decay']==(.1 if any(k in n for k in ('tok_emb','pos_emb','tok_v0_emb','pos_v0_emb')) else 0.)
                for b in m.blocks:b.capture_diagnostics=True
                m(ids)
                dyn=m.hb_dynamics()
                assert all(r['mu_attn']>0 and r['connection_enabled']==1 for r in dyn)
                assert all(r['forward_step']==1 and r['post_ln_velocity_correction_norm']>=0 for r in dyn)
    # Changed modes must be active, not CLI aliases, under common initialization.
    logits={}
    for mode in ('Y','H1','R2','H1_LA','R2_LA'):
        torch.manual_seed(95019);logits[mode]=make(source,mode)(torch.ones((1,6),dtype=torch.long))[0].detach()
    for mode in ('H1_LA','R2_LA'):
        assert not torch.equal(logits[mode],logits['Y'])
        assert not torch.equal(logits[mode],logits[mode[:-3]])
    result=dict(status='PASS',device=a.device,references=records,legacy_bitwise_parity=parities,
                null_equivalence=nulls,causal_jacobians=jacobians,all_seven_suffix_checks='PASS',
                parameter_initialization='PASS',optimizer_groups='PASS',active_mu_gradients='PASS',
                distinct_routes='PASS',diagnostics='PASS')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS',a.device,len(records),'references',len(parities),'bitwise controls',len(nulls),'null pairs',len(jacobians),'causal JVP cases')


if __name__=='__main__':main()
